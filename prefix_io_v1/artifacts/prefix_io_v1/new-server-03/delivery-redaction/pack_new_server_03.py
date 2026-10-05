#!/usr/bin/env python3
"""Create the explicitly scoped, redacted new-server-03 evidence archive."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import importlib.util
import json
import zipfile

root = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
base = root / "artifacts/prefix_io_v1/new-server-03"
control = base / "delivery-redaction/delivery-plan.json"
stage = root.parent / "delivery-new-server-03-sanitized"
archive = root.parent / "prefix-io-v1-new-server-03-20260926.zip"
assert not stage.exists() and not archive.exists()
assert json.loads((root / "experiments/prefix_io_v1/gpu-budget-ledger.json").read_text())["active_reservation"] is None
spec = importlib.util.spec_from_file_location("redact_delivery", base / "delivery-redaction/redact_delivery.py")
redact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(redact)
plan = redact.make_plan(root, control)
assert plan["ready"], plan["errors"]
# Working prose drafts are not final evidence. Originals remain on the server.
removed = [x for x in plan["files"] if x["path"].endswith("_DRAFT.md")]
plan["files"] = [x for x in plan["files"] if x not in removed]
plan["exclusions"].extend({"path": x["path"], "reason": "stale working prose draft; original retained"} for x in removed)
counts = {}
for record in plan["files"]:
    for key, value in record["redactions"].items():
        counts[key] = counts.get(key, 0) + value
plan["totals"] = {
    "files": len(plan["files"]),
    "source_bytes": sum(x["bytes"] for x in plan["files"]),
    "changed_files": sum(bool(x["redaction_count"]) for x in plan["files"]),
    "redaction_count": sum(counts.values()), "redactions": counts,
}
plan["notes"].append("The archive caller adds a separate exact-file allowlist for state, ledger and root build metadata.")
control.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
redact.copy_plan(plan, stage)
transformer, evidence = redact.load_redactor(root)
# Explicit supplement: the generic directory scanner does not cover these files.
extra_names = [
    "experiments/prefix_io_v1/execution_state.json",
    "experiments/prefix_io_v1/gpu-budget-ledger.json",
    "pyproject.toml", "setup.cfg", "setup.py", "README.md", ".gitignore", ".gitattributes",
]
extra = []
for rel in extra_names:
    source = root / rel
    data = redact.read_bounded(source)
    output, replacements, encoding = transformer.transform(data, rel)
    target = stage / rel
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(output)
    extra.append({
        "path": rel, "source_path": str(source), "source_sha256": redact.sha(data),
        "copy_sha256": redact.sha(output), "bytes": len(data), "copy_bytes": len(output),
        "encoding": encoding, "redaction_count": sum(replacements.values()),
        "redactions": replacements,
    })
(stage / "DELIVERY_EXTRA_FILES.json").write_text(json.dumps({
    "allowlist": extra_names, "files": extra,
    "scope": "Only named project state/budget/build metadata; originals unchanged",
}, indent=2) + "\n")
records = []
metadata = {"DELIVERY_REDACTION_MANIFEST.json", "DELIVERY_EXTRA_FILES.json"}
for path in sorted(stage.rglob("*")):
    if not path.is_file():
        continue
    assert not path.is_symlink()
    rel = path.relative_to(stage).as_posix()
    arc = rel if rel.startswith("server-root/") or rel in metadata else "project/" + rel
    data = path.read_bytes()
    records.append({"path": rel, "archive_path": arc, "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest()})
assert len({r["archive_path"] for r in records}) == len(records)
required = ["project/experiments/prefix_io_v1/execution_state.json",
            "project/experiments/prefix_io_v1/gpu-budget-ledger.json",
            "project/docs/prefix_io_v1/NEW_SERVER_03_REPORT.md",
            "project/experiments/prefix_io_v1/runs/native-prefix-04/details/smoke-result.json"]
assert all(any(r["archive_path"] == name for r in records) for name in required)
manifest = {
    "schema_version": 1, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "server": "connect.westc.seetacloud.com:24801", "project_root": str(root),
    "phase": "P1_native_GPU_Prefix_passed_SSD_blocked",
    "files": records, "file_count": len(records),
    "source_redaction_manifest": "DELIVERY_REDACTION_MANIFEST.json",
    "additional_exact_file_manifest": "DELIVERY_EXTRA_FILES.json",
    "snapshot_note": "Parent commands log ends before this packaging invocation; original raw evidence stays on server.",
    "excludes": ["model weights and failed partials", "virtual environments", "SDK and runtime JIT cache",
                 "full third-party repository bodies and .git", "three raw ephemeral redirect evidence files",
                 "working prose drafts", "Python bytecode"],
}
manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
    for record in records:
        zipped.write(stage / record["path"], record["archive_path"])
    zipped.writestr("delivery-manifest.json", manifest_bytes)
with zipfile.ZipFile(archive) as zipped:
    assert zipped.testzip() is None
    for record in records:
        data = zipped.read(record["archive_path"])
        assert len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"]
check = {
    "archive": str(archive), "archive_bytes": archive.stat().st_size,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "manifest_records": len(records), "zip_entries": len(records) + 1,
    "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    "all_file_sizes_and_sha256_verified": True,
    "redaction_totals": plan["totals"],
    "additional_file_count": len(extra),
    "gpu_execution": False,
}
(base / "delivery-manifest.json").write_bytes(manifest_bytes)
(base / "package-check.json").write_text(json.dumps(check, indent=2) + "\n")
print(json.dumps(check, indent=2))
