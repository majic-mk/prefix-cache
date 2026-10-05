#!/usr/bin/env python3
"""Explicit one-file continuation of a recorded failed download; no implicit retry."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import time
import urllib.error
import urllib.request
import uuid

_spec = importlib.util.spec_from_file_location(
    "prefix_pinned_download", Path(__file__).with_name("download_pinned_model.py"))
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 ** 2), b""):
            h.update(chunk)
    return h.hexdigest()


def resume(root, plan_path, target, label, failed_label, filename, partial_path,
           offset, partial_sha256):
    root = root.resolve(strict=True)
    plan_path = dl.real_project_path(root, plan_path)
    target = dl.real_project_path(root, target)
    partial_path = dl.real_project_path(root, partial_path)
    for value in (label, failed_label):
        dl.require(isinstance(value, str) and re.fullmatch("[a-zA-Z0-9_-]+", value),
                   "safe explicit labels required")
    dl.require(label != failed_label, "resume requires a new label")
    raw = plan_path.read_bytes()
    dl.require(len(raw) <= 2 * 1024 ** 2, "manifest too large")
    plan = json.loads(raw)
    files = dl.validate_manifest(plan)
    dl.require(plan["source"] == "https://modelscope.cn",
               "resume limited to the qualified official ModelScope route")
    matches = [item for item in files if item["path"] == filename]
    dl.require(len(matches) == 1, "one exact manifest file required")
    rec = matches[0]
    dl.require(filename.endswith(".safetensors") and rec["hash_algorithm"] == "sha256",
               "resume is limited to a SHA256 pinned weight")
    dl.require(type(offset) is int and 0 < offset < rec["bytes"],
               "offset must be inside the exact manifest file")
    dl.require(isinstance(partial_sha256, str)
               and re.fullmatch("[0-9a-f]{64}", partial_sha256),
               "explicit partial SHA256 required")
    manifest_sha = hashlib.sha256(raw).hexdigest()
    from prefix_io_control.config import read_yaml, validate_permissions
    permissions = validate_permissions(read_yaml(
        root / "experiments/prefix_io_v1/configs/permissions.yaml"))
    dl.require(permissions["allow_model_downloads"], "model download is not authorized")
    dl.require(permissions["max_model_download_gib"] is not None,
               "finite model budget required")
    ceiling = min(int(permissions["max_model_download_gib"] * dl.GIB), 20 * dl.GIB)
    dl.require(Path(permissions["approved_dependency_root"]).resolve() == root,
               "approved dependency root mismatch")
    ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    base = root / "artifacts/prefix_io_v1/new-server-03/downloads"
    out = dl.real_project_path(root, base / label)
    source_result = dl.real_project_path(root, base / failed_label / "result.json")
    expected_partial = target / (filename + "." + failed_label + ".partial")
    dl.require(partial_path == expected_partial,
               "partial path must belong to the exact failed label and file")
    destination = target / filename
    new_partial = dl.real_project_path(root, target / (filename + "." + label + ".partial"))
    remaining = rec["bytes"] - offset
    amount = remaining + 1
    with ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads(ledger.read_text())
        dl.require(budget.get("active_reservation") is None,
                   "unfinished reservation; inspect and reconcile first")
        used = budget.get("model_download_bytes")
        received = budget.get("model_payload_received_bytes")
        dl.require(type(used) is int and used >= 0 and type(received) is int
                   and received >= 0, "invalid cumulative accounting")
        dl.require(used + amount <= ceiling, "insufficient cumulative resume budget")
        failed = json.loads(source_result.read_text())
        dl.require(failed.get("status") == "FAILED"
                   and failed.get("label") == failed_label
                   and failed.get("target") == str(target)
                   and failed.get("manifest_sha256") == manifest_sha
                   and failed.get("revision") == plan["revision"]
                   and failed.get("source") == plan["source"],
                   "failed result does not bind this target and manifest")
        events = [event for event in failed.get("files", [])
                  if event.get("file") == filename and event.get("error")]
        dl.require(len(events) == 1, "one recorded failed file event required")
        event = events[0]
        ledger_matches = [old for old in budget.get("download_events", [])
                          if old.get("id") == event.get("id")]
        dl.require(len(ledger_matches) == 1 and ledger_matches[0] == event,
                   "failed event must exactly match settled ledger history")
        dl.require(event.get("kind") == "model_download"
                   and event.get("label") == failed_label
                   and event.get("manifest_sha256") == manifest_sha
                   and event.get("url") == rec["url"]
                   and event.get("published") is False
                   and event.get("reserved_bytes") == rec["bytes"] + 1
                   and event.get("charged_bytes") == rec["bytes"] + 1
                   and type(event.get("actual_payload_bytes")) is int
                   and offset <= event["actual_payload_bytes"] <= rec["bytes"] + 1,
                   "failed event is not a conservatively settled source download")
        dl.require(partial_path.is_file() and not partial_path.is_symlink()
                   and partial_path.stat().st_size == offset,
                   "failed partial is not a regular file of the exact offset size")
        dl.require(not destination.exists() and not destination.is_symlink()
                   and not new_partial.exists() and not new_partial.is_symlink(),
                   "destination or resume partial already exists; no overwrite")
        dl.require(dl.shutil.disk_usage(target).free >= rec["bytes"] + dl.RESERVE_DISK,
                   "insufficient disk for preserved source plus complete new file")
        out.mkdir(parents=True, exist_ok=False)
        job = {"label": label, "status": "PREPARING_LOCAL_COPY", "kind": "model_resume",
               "source_label": failed_label, "source_event_id": event["id"],
               "source_partial": str(partial_path), "source_partial_sha256": partial_sha256,
               "offset": offset, "remaining_bytes": remaining,
               "target": str(target), "file": filename, "revision": plan["revision"],
               "source": plan["source"], "manifest": str(plan_path),
               "manifest_sha256": manifest_sha, "gpu_execution": False,
               "accounting": "application-level returned response bytes; not network wire bytes"}
        dl.atomic_json(out / "result.json", job)
        before = partial_path.stat()
        copied = 0
        h = hashlib.sha256()
        with partial_path.open("rb") as source, new_partial.open("xb") as copied_file:
            while copied <= offset:
                chunk = source.read(min(8 * 1024 ** 2, offset + 1 - copied))
                if not chunk:
                    break
                copied += len(chunk)
                dl.require(copied <= offset, "source partial grew during copy")
                h.update(chunk)
                copied_file.write(chunk)
            copied_file.flush()
            os.fsync(copied_file.fileno())
        after = partial_path.stat()
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        dl.require(copied == offset and identity(before) == identity(after)
                   and h.hexdigest() == partial_sha256,
                   "source partial changed or partial SHA256 mismatch")
        dl.require(dl.shutil.disk_usage(target).free >= remaining + dl.RESERVE_DISK,
                   "insufficient disk before Range request")
        reservation = {"kind": "model_resume", "id": uuid.uuid4().hex,
                       "label": label, "file": filename, "url": rec["url"],
                       "reserved_bytes": amount, "runner_pid": os.getpid(),
                       "started_unix": time.time(), "manifest_sha256": manifest_sha,
                       "source_event_id": event["id"], "source_label": failed_label,
                       "offset": offset, "source_partial_sha256": partial_sha256}
        budget["active_reservation"] = reservation
        dl.atomic_json(ledger, budget)  # Durable before any network open.
        actual = 0
        error = None
        try:
            req = urllib.request.Request(rec["url"], headers={
                "Accept-Encoding": "identity", "User-Agent": "prefix-io-v1-fixed-model",
                "Range": "bytes=" + str(offset) + "-"})
            req._prefix_download_expected = dict(rec, revision=plan["revision"])
            with dl.make_opener().open(req, timeout=60) as response:
                dl.valid_url(response.geturl())
                dl.require(response.status == 206, "Range requires HTTP 206; body rejected")
                dl.require(response.headers.get("Content-Encoding", "identity").lower()
                           == "identity", "compressed Range response rejected")
                expected_range = f"bytes {offset}-{rec['bytes'] - 1}/{rec['bytes']}"
                dl.require(response.headers.get("Content-Range") == expected_range,
                           "Content-Range does not exactly bind offset/end/total")
                length = response.headers.get("Content-Length")
                dl.require(isinstance(length, str) and re.fullmatch("[0-9]+", length)
                           and int(length) == remaining,
                           "Range requires exact Content-Length")
                with new_partial.open("ab") as output:
                    dl.require(output.tell() == offset, "local resume offset changed")
                    while actual < amount:
                        chunk = response.read(min(8 * 1024 ** 2, amount - actual))
                        if not chunk:
                            break
                        actual += len(chunk)
                        output.write(chunk)
                        dl.require(actual <= remaining, "Range response exceeds frozen remainder")
                    output.flush()
                    os.fsync(output.fileno())
            dl.require(actual == remaining, "short Range response")
            dl.verify_file(new_partial, rec)
            os.link(new_partial, destination)
            new_partial.unlink()
            fd = os.open(str(target), os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()  # Never read an error body.
            error = type(exc).__name__ + ": " + str(exc)
        charge = actual if error is None else amount
        budget["model_download_bytes"] += charge
        budget["model_payload_received_bytes"] += actual
        settled = dict(reservation, actual_payload_bytes=actual, charged_bytes=charge,
                       ended_unix=time.time(), error=error, published=error is None)
        budget["download_events"].append(settled)
        budget["active_reservation"] = None
        dl.atomic_json(ledger, budget)
        job["event"] = settled
        job["status"] = "FAILED" if error else "VERIFIED_RESUMED_FILE_ONLY"
        job["original_failed_partial_preserved"] = True
        dl.atomic_json(out / "result.json", job)
        if error:
            raise RuntimeError(error)
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--failed-label", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--partial", required=True, type=Path)
    parser.add_argument("--offset", required=True, type=int)
    parser.add_argument("--partial-sha256", required=True)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError("signal " + str(signum))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    result = resume(Path(__file__).resolve().parents[3], args.manifest, args.model_dir,
                    args.label, args.failed_label, args.file, args.partial,
                    args.offset, args.partial_sha256)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
