#!/usr/bin/env python3
"""Plan or copy a bounded Prefix I/O delivery with ephemeral URL/cookie redaction.

Default operation is read-only plan generation. Never edits source evidence.
No network, GPU, imports from project, archive creation, or credential output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

SCOPES = (
    "docs/prefix_io_v1", "src/prefix_io_control",
    "tests/prefix_io_v1", "tests/prefix_io_v1_progress",
    "tests/prefix_io_v1_observer", "tests/prefix_io_v1_runner",
    "tests/prefix_io_v1_vllm_platform", "tests/prefix_io_v1_native_prefix",
    "tests/prefix_io_v1_download", "patches/prefix_io_v1",
    "experiments/prefix_io_v1/scripts", "experiments/prefix_io_v1/configs",
    "experiments/prefix_io_v1/locks", "experiments/prefix_io_v1/runs",
    "artifacts/prefix_io_v1",
)
RAW_DIR = "artifacts/prefix_io_v1/new-server-03/redirect-review"
RAW_NAMES = ("redirect-review.json", "requests-started.jsonl", "requests-completed.jsonl")
RAW_EXCLUSIONS = frozenset(RAW_DIR + "/" + name for name in RAW_NAMES)
MAX_BYTES = 64 * 1024 * 1024
AUTH = re.compile(r"(?i)([?&]auth_key=)([^&#\s\"\\<>]+)")
REPLACEMENT = "REDACTED"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_bounded(path):
    if path.is_symlink() or path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("source must be a regular non-symlink file: " + str(path))
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("source exceeds per-file 64 MiB ceiling: " + str(path))
    return data

def escaped_variants(value):
    variants = {value}
    frontier = {value}
    # Also catch JSON strings nested inside transcript JSON values.
    for _ in range(8):
        following = set()
        for item in frontier:
            for ascii_only in (False, True):
                following.add(json.dumps(item, ensure_ascii=ascii_only)[1:-1])
        following -= variants
        if not following:
            break
        variants |= following
        frontier = following
    return variants

def redactor_from_pair(raw, public):
    secrets = {}
    def remember(value, category):
        if not value or value == REPLACEMENT:
            raise ValueError("invalid empty or already-redacted source secret")
        secrets[value] = category
    def compare(before, after):
        if isinstance(before, dict):
            if not isinstance(after, dict) or not before.keys() <= after.keys():
                raise ValueError("public/raw structure mismatch")
            for key, value in before.items():
                compare(value, after[key])
        elif isinstance(before, list):
            if not isinstance(after, list) or len(before) != len(after):
                raise ValueError("public/raw list mismatch")
            for a, b in zip(before, after):
                compare(a, b)
        elif isinstance(before, str) and isinstance(after, str):
            if before == after:
                return
            if after == REPLACEMENT:
                # Public evidence redacts complete Set-Cookie values.
                remember(before, "known_cookie_value")
                return
            found = list(AUTH.finditer(before))
            if not found or AUTH.sub(lambda m: m.group(1) + REPLACEMENT, before) != after:
                raise ValueError("unexpected public/raw string difference")
            for match in found:
                remember(match.group(2), "known_auth_key")
        elif before != after:
            raise ValueError("unexpected public/raw non-string difference")
    compare(raw, public)
    if not secrets:
        raise ValueError("no source secrets extracted; refusing empty redaction")
    variants = {}
    for secret, category in secrets.items():
        for variant in escaped_variants(secret):
            if variant in variants and variants[variant] != category:
                # Categories are descriptive only; replacement is identical.
                variants[variant] = "known_secret"
            else:
                variants[variant] = category
    return Redactor(variants, len(secrets))

class Redactor:
    def __init__(self, variants, secret_count):
        self.variants = sorted(variants.items(), key=lambda item: (-len(item[0]), item[0]))
        self.secret_count = secret_count

    def redact(self, text):
        counts = {}
        for value, category in self.variants:
            count = text.count(value)
            if count:
                text = text.replace(value, REPLACEMENT)
                counts[category] = counts.get(category, 0) + count
        def generic(match):
            if match.group(2) == REPLACEMENT:
                return match.group(0)
            counts["additional_auth_key"] = counts.get("additional_auth_key", 0) + 1
            return match.group(1) + REPLACEMENT
        text = AUTH.sub(generic, text)
        if any(value in text for value, _ in self.variants):
            raise ValueError("residual known secret after redaction")
        if any(m.group(2) != REPLACEMENT for m in AUTH.finditer(text)):
            raise ValueError("residual auth_key after redaction")
        return text, counts

    def transform(self, data, relative_path):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            if Path(relative_path).suffix != ".bin":
                raise ValueError("unsupported non-UTF-8 source: " + relative_path)
            # Known CPU probe .bin files are preserved only after byte screening.
            if any(v.encode("utf-8") in data for v, _ in self.variants):
                raise ValueError("known secret in binary source: " + relative_path)
            if re.search(rb"(?i)[?&]auth_key=", data):
                raise ValueError("auth_key marker in binary source: " + relative_path)
            return data, {}, "binary-bin-screened"
        redacted, counts = self.redact(text)
        return redacted.encode("utf-8"), counts, "utf-8"

def load_redactor(project):
    directory = project / RAW_DIR
    raw_data = read_bounded(directory / RAW_NAMES[0])
    public_data = read_bounded(directory / "redirect-review-public.json")
    raw, public = json.loads(raw_data), json.loads(public_data)
    manifest = public.get("redaction", {}).get("raw_sources", [])
    if {entry.get("path") for entry in manifest} != set(RAW_NAMES):
        raise ValueError("public evidence does not bind all three raw sources")
    sources = []
    for entry in manifest:
        data = read_bounded(directory / entry["path"])
        if sha(data) != entry["sha256"] or len(data) != entry["bytes"]:
            raise ValueError("raw source differs from public provenance: " + entry["path"])
        sources.append({"path": RAW_DIR + "/" + entry["path"],
                        "sha256": sha(data), "bytes": len(data)})
    sources.append({"path": RAW_DIR + "/redirect-review-public.json",
                    "sha256": sha(public_data), "bytes": len(public_data)})
    return redactor_from_pair(raw, public), sources

def source_entries(project, plan_path):
    found = {}
    exclusions = []
    for scope in SCOPES:
        base = project / scope
        if not base.is_dir() or base.is_symlink():
            raise ValueError("missing or symlink scope: " + scope)
        for directory, directories, files in os.walk(base, followlinks=False):
            for name in directories:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("symlink directory in delivery scope")
            for name in files:
                file = Path(directory) / name
                rel = file.relative_to(project).as_posix()
                if rel in RAW_EXCLUSIONS:
                    exclusions.append({"path": rel, "reason": "raw ephemeral provider evidence; server only"})
                elif "__pycache__" in file.parts or file.suffix == ".pyc":
                    exclusions.append({"path": rel, "reason": "generated Python bytecode"})
                elif file.absolute() == plan_path.absolute():
                    exclusions.append({"path": rel, "reason": "plan control file; outside payload"})
                else:
                    found[rel] = file
    parent_log = project.parent / "commands.jsonl"
    found["server-root/commands.jsonl"] = parent_log
    return sorted(found.items()), sorted(exclusions, key=lambda e: e["path"])

def make_plan(project, plan_path):
    redactor, evidence = load_redactor(project)
    entries, exclusions = source_entries(project, plan_path)
    files, errors = [], []
    counts = {}
    for rel, source in entries:
        try:
            data = read_bounded(source)
            output, replacements, encoding = redactor.transform(data, rel)
            for category, count in replacements.items():
                counts[category] = counts.get(category, 0) + count
            files.append({"path": rel, "source_path": str(source), "bytes": len(data),
                          "source_sha256": sha(data), "copy_bytes": len(output),
                          "copy_sha256": sha(output), "encoding": encoding,
                          "redaction_count": sum(replacements.values()),
                          "redactions": replacements})
        except (OSError, ValueError) as exc:
            # Error messages contain only paths/fixed diagnostics, never source contents.
            errors.append({"path": rel, "error": str(exc)})
    return {"schema_version": 1, "mode": "copy-time-redaction-plan",
            "project_root": str(project), "source_scopes": list(SCOPES),
            "parent_log_mapping": "server-root/commands.jsonl",
            "source_evidence": evidence, "known_secret_count": redactor.secret_count,
            "files": files, "exclusions": exclusions, "errors": errors,
            "totals": {"files": len(files), "source_bytes": sum(x["bytes"] for x in files),
                       "changed_files": sum(bool(x["redaction_count"]) for x in files),
                       "redaction_count": sum(counts.values()), "redactions": counts},
            "ready": not errors,
            "notes": ["Original evidence is never modified.",
                      "Source hashes are verified again before copies are created.",
                      "Regenerate after live command logs or canonical files change.",
                      "This plan is omitted from payload to avoid a self hash cycle."]}

def copy_plan(plan, destination):
    if not plan.get("ready") or plan.get("errors") or plan.get("schema_version") != 1:
        raise ValueError("plan is not ready")
    project = Path(plan["project_root"]).resolve()
    if plan.get("source_scopes") != list(SCOPES):
        raise ValueError("unexpected source scopes")
    destination = destination.absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("destination must not exist")
    if destination == project or destination.is_relative_to(project):
        raise ValueError("destination must be outside original project")
    if not destination.parent.is_dir() or destination.parent.resolve() != destination.parent:
        raise ValueError("destination parent must exist and contain no symlinks")
    redactor, evidence = load_redactor(project)
    if evidence != plan["source_evidence"]:
        raise ValueError("redaction evidence changed after plan")
    pending = []
    seen = set()
    for entry in plan["files"]:
        rel = entry["path"]
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts or rel in seen:
            raise ValueError("invalid or duplicate delivery path")
        seen.add(rel)
        if rel == "server-root/commands.jsonl":
            source = project.parent / "commands.jsonl"
        elif rel in RAW_EXCLUSIONS or not any(relative.is_relative_to(Path(s)) for s in SCOPES):
            raise ValueError("source outside allowed delivery scope")
        else:
            source = project / relative
        if str(source) != entry["source_path"]:
            raise ValueError("source path mismatch")
        data = read_bounded(source)
        if sha(data) != entry["source_sha256"] or len(data) != entry["bytes"]:
            raise ValueError("source changed after plan: " + rel)
        output, counts, encoding = redactor.transform(data, rel)
        if (sha(output) != entry["copy_sha256"] or len(output) != entry["copy_bytes"]
                or counts != entry["redactions"] or encoding != entry["encoding"]):
            raise ValueError("redacted copy differs from plan: " + rel)
        pending.append((rel, output))
    # All source/evidence checks finish before creating the new staging tree.
    destination.mkdir(mode=0o700)
    for rel, data in pending:
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    manifest = {"status": "COPIED_NOT_ARCHIVED", "files": plan["files"],
                "exclusions": plan["exclusions"], "source_evidence": evidence,
                "totals": plan["totals"]}
    (destination / "DELIVERY_REDACTION_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--copy-to", type=Path, help="optional new staging directory; no archive is produced")
    args = parser.parse_args()
    project, plan_path = args.project.resolve(), args.plan.absolute()
    try:
        if args.copy_to is None:
            plan = make_plan(project, plan_path)
            allowed_plan = project / "artifacts/prefix_io_v1/new-server-03/delivery-redaction/delivery-plan.json"
            if plan_path != allowed_plan or plan_path.is_symlink():
                raise ValueError("plan output must be the dedicated delivery-redaction/delivery-plan.json")
            if plan_path.exists():
                old = json.loads(read_bounded(plan_path))
                if old.get("schema_version") != 1 or old.get("mode") != "copy-time-redaction-plan":
                    raise ValueError("refusing to overwrite an unrelated existing file")
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"status": "PLAN_READY" if plan["ready"] else "PLAN_BLOCKED",
                              "plan": str(plan_path), "plan_sha256": sha(plan_path.read_bytes()),
                              "totals": plan["totals"], "errors": plan["errors"]}, ensure_ascii=False))
            return 0 if plan["ready"] else 2
        plan = json.loads(read_bounded(plan_path))
        if Path(plan["project_root"]).resolve() != project:
            raise ValueError("--project differs from plan")
        result = copy_plan(plan, args.copy_to)
        print(json.dumps({"status": result["status"], "destination": str(args.copy_to),
                          "totals": result["totals"]}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, ensure_ascii=False))
        return 2

if __name__ == "__main__":
    sys.exit(main())
