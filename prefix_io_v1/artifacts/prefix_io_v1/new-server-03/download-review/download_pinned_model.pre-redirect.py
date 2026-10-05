#!/usr/bin/env python3
"""Bounded fixed-model payload download; no GPU, implicit retry or Range resume."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import ssl
import time
import urllib.parse
import urllib.request
import uuid

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GIB = 1024 ** 3
RESERVE_DISK = 2 * GIB
# Only public provider endpoints verified for this task. No wildcard CDN hosts.
DOWNLOAD_HOSTS = {"modelscope.cn", "www.modelscope.cn", "huggingface.co"}
AUX = {"config.json", "generation_config.json", "tokenizer.json",
       "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
       "special_tokens_map.json", "chat_template.jinja",
       "model.safetensors.index.json", "model.safetensors"}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def atomic_json(path, obj):
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def valid_url(url):
    p = urllib.parse.urlsplit(url)
    require(p.scheme == "https" and p.hostname in DOWNLOAD_HOSTS
            and p.port in (None, 443) and p.username is None and p.password is None
            and not p.fragment, "download URL outside verified HTTPS provider hosts")
    return p


class VerifiedRedirect(urllib.request.HTTPRedirectHandler):
    max_redirections = 5
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib's default http_error_302 consumes fp.read() before following.
        # That unbounded body bypasses this downloader's payload accounting.
        # No redirect route has been qualified for this fixed provider request.
        try:
            valid_url(newurl)
        finally:
            fp.close()
        raise RuntimeError("download redirect requires explicit route review; no automatic redirect")


def make_opener():
    return urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        VerifiedRedirect())


def validate_manifest(plan):
    require(plan.get("model_id") == MODEL_ID, "fixed model identity required")
    rev = plan.get("revision")
    require(isinstance(rev, str) and re.fullmatch("[0-9a-f]{40}", rev),
            "real immutable 40-hex provider revision required")
    source = plan.get("source")
    require(source in ("https://modelscope.cn", "https://huggingface.co"),
            "unsupported model provenance")
    require(plan.get("trust_remote_code") is False, "remote code must be disabled")
    files = plan.get("files")
    require(isinstance(files, list) and 1 <= len(files) <= 32, "bounded file list required")
    seen = set()
    for rec in files:
        name = rec.get("path")
        require(isinstance(name, str) and name not in seen
                and (name in AUX or re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name)),
                "unsafe, duplicate or unapproved model file")
        seen.add(name)
        require(type(rec.get("bytes")) is int and rec["bytes"] > 0,
                "exact positive integer size required")
        algorithm = rec.get("hash_algorithm")
        require(algorithm in ("sha256", "git_blob_sha1"), "unsupported hash")
        length = 64 if algorithm == "sha256" else 40
        require(isinstance(rec.get("hash"), str)
                and re.fullmatch("[0-9a-f]{" + str(length) + "}", rec["hash"]),
                "exact hash required")
        require(not name.endswith(".safetensors") or algorithm == "sha256",
                "weights require SHA256")
        u = valid_url(rec["url"])
        if source == "https://modelscope.cn":
            q = urllib.parse.parse_qs(u.query, strict_parsing=True)
            require(u.hostname in ("modelscope.cn", "www.modelscope.cn")
                    and u.path == "/api/v1/models/" + MODEL_ID + "/repo"
                    and q == {"Revision": [rev], "FilePath": [name]},
                    "ModelScope file URL must match pinned model, revision and file")
        else:
            require(u.hostname == "huggingface.co" and not u.query
                    and u.path == "/" + MODEL_ID + "/resolve/" + rev + "/" + name,
                    "Hugging Face file URL must match pinned model, revision and file")
    require({"config.json", "tokenizer_config.json", "tokenizer.json",
             "model.safetensors.index.json"}.issubset(seen), "required local model files absent")
    require(plan.get("total_bytes") == sum(r["bytes"] for r in files),
            "manifest byte total mismatch")
    require(plan["total_bytes"] <= 20 * GIB, "manifest exceeds fixed 20 GiB ceiling")
    return files


def verify_file(path, rec):
    require(path.is_file() and not path.is_symlink(), "regular model file required")
    require(path.stat().st_size == rec["bytes"], "model file size mismatch")
    h = hashlib.sha256() if rec["hash_algorithm"] == "sha256" else hashlib.sha1()
    if rec["hash_algorithm"] == "git_blob_sha1":
        h.update(b"blob " + str(rec["bytes"]).encode("ascii") + b"\0")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 ** 2), b""):
            h.update(chunk)
    require(h.hexdigest() == rec["hash"], "model file hash mismatch: " + rec["path"])


def real_project_path(root, path):
    # Reject symlink components, including links that resolve back into the root.
    target = Path(os.path.abspath(path))
    require(target.is_relative_to(root) and target != root, "path must stay inside project")
    cursor = target
    while cursor != root:
        require(not cursor.is_symlink(), "symlink path is not allowed")
        cursor = cursor.parent
    return target


def execute(root, plan_path, target, label):
    root = root.resolve(strict=True)
    plan_path = real_project_path(root, plan_path)
    target = real_project_path(root, target)
    require(re.fullmatch("[a-zA-Z0-9_-]+", label), "safe unique label required")
    raw_plan = plan_path.read_bytes()
    require(len(raw_plan) <= 2 * 1024 ** 2, "manifest too large")
    plan = json.loads(raw_plan)
    files = validate_manifest(plan)
    from prefix_io_control.config import read_yaml, validate_permissions
    permissions = validate_permissions(read_yaml(root / "experiments/prefix_io_v1/configs/permissions.yaml"))
    require(permissions["allow_model_downloads"], "model download is not authorized")
    limit_gib = permissions["max_model_download_gib"]
    require(limit_gib is not None, "finite model budget required")
    ceiling = min(int(limit_gib * GIB), 20 * GIB)
    require(Path(permissions["approved_dependency_root"]).resolve() == root,
            "approved dependency root mismatch")
    ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    out = root / "artifacts/prefix_io_v1/new-server-03/downloads" / label
    opener = make_opener()
    with ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads(ledger.read_text())
        require(budget.get("active_reservation") is None, "unfinished reservation; inspect and reconcile first")
        used = budget.get("model_download_bytes")
        require(type(used) is int and used >= 0, "invalid cumulative model charge")
        require(type(budget.get("model_payload_received_bytes", used)) is int, "invalid actual payload ledger")
        out.mkdir(parents=True, exist_ok=False)
        target.mkdir(parents=True, exist_ok=True)
        # Reuse only complete, previously verified regular files.
        pending = []
        reused = []
        for rec in files:
            dest = target / rec["path"]
            if dest.exists() or dest.is_symlink():
                verify_file(dest, rec)
                reused.append(rec["path"])
            else:
                pending.append(rec)
        planned = sum(r["bytes"] + 1 for r in pending)  # One-byte overrun probe per response.
        require(used + planned <= ceiling, "insufficient cumulative budget for all remaining files")
        require(shutil.disk_usage(target).free >= sum(r["bytes"] for r in pending) + RESERVE_DISK,
                "insufficient disk space with explicit 2 GiB reserve")
        job = {"label": label, "model_id": MODEL_ID, "revision": plan["revision"],
               "source": plan["source"], "manifest": str(plan_path),
               "manifest_sha256": hashlib.sha256(raw_plan).hexdigest(),
               "target": str(target), "reused_verified_files": reused,
               "files": [], "status": "RUNNING", "gpu_execution": False,
               "accounting": "response payload bytes; failed requests conservatively charged full reservation"}
        atomic_json(out / "result.json", job)
        budget.setdefault("download_events", [])
        budget.setdefault("model_payload_received_bytes", used)
        for index, rec in enumerate(pending):
            require(shutil.disk_usage(target).free >= rec["bytes"] + RESERVE_DISK,
                    "insufficient free disk before file request")
            dest = target / rec["path"]
            partial = target / (rec["path"] + "." + label + ".partial")
            require(not partial.exists() and not partial.is_symlink() and not dest.exists(),
                    "partial or destination already exists; no implicit resume/overwrite")
            amount = rec["bytes"] + 1
            require(budget["model_download_bytes"] + amount <= ceiling, "remaining download budget exceeded")
            reservation = {"kind": "model_download", "id": uuid.uuid4().hex,
                           "label": label, "file": rec["path"], "url": rec["url"],
                           "reserved_bytes": amount, "runner_pid": os.getpid(),
                           "started_unix": time.time(), "manifest_sha256": job["manifest_sha256"]}
            budget["active_reservation"] = reservation
            atomic_json(ledger, budget)  # Durable before opening the network request.
            actual = 0
            error = None
            try:
                req = urllib.request.Request(rec["url"], headers={
                    "Accept-Encoding": "identity", "User-Agent": "prefix-io-v1-fixed-model"})
                with opener.open(req, timeout=60) as response:
                    valid_url(response.geturl())
                    require(response.status == 200, "only complete HTTP 200 responses accepted")
                    require(response.headers.get("Content-Encoding", "identity").lower() == "identity",
                            "compressed responses are not allowed")
                    length = response.headers.get("Content-Length")
                    require(length is None or int(length) == rec["bytes"],
                            "response Content-Length differs from frozen manifest")
                    with partial.open("xb") as f:
                        while actual < amount:
                            chunk = response.read(min(8 * 1024 ** 2, amount - actual))
                            if not chunk:
                                break
                            actual += len(chunk)
                            f.write(chunk)
                            require(actual <= rec["bytes"], "response exceeds frozen size")
                        f.flush()
                        os.fsync(f.fileno())
                require(actual == rec["bytes"], "short response")
                verify_file(partial, rec)
                # Same-directory hard-link publication fails if a destination appeared.
                os.link(partial, dest)
                partial.unlink()
                fd = os.open(str(target), os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except BaseException as exc:
                error = type(exc).__name__ + ": " + str(exc)
            charge = actual if error is None else amount
            budget["model_download_bytes"] += charge
            budget["model_payload_received_bytes"] += actual
            event = dict(reservation, actual_payload_bytes=actual, charged_bytes=charge,
                         ended_unix=time.time(), error=error, published=error is None)
            budget["download_events"].append(event)
            budget["active_reservation"] = None
            atomic_json(ledger, budget)
            job["files"].append(event)
            job["status"] = "FAILED" if error else "RUNNING"
            atomic_json(out / "result.json", job)
            print(json.dumps({"file": rec["path"], "actual_bytes": actual,
                              "charged_bytes": charge, "error": error}), flush=True)
            if error:
                raise RuntimeError(error)
        # Reverify the complete dataset before marking the download complete.
        for rec in files:
            verify_file(target / rec["path"], rec)
        job["status"] = "VERIFIED_LOCAL_FILES"
        job["model_payload_received_bytes_cumulative"] = budget["model_payload_received_bytes"]
        job["model_charged_bytes_cumulative"] = budget["model_download_bytes"]
        atomic_json(out / "result.json", job)
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError("signal " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    result = execute(Path(__file__).resolve().parents[3], args.manifest, args.model_dir, args.label)
    print(json.dumps({k: result[k] for k in ("status", "target", "revision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
