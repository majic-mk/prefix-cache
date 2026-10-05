"""Append a regular-file-only KV evidence archive; never decide system PASS.

Reads the three actual diagnostic jobs plus their capture/launcher deliveries.
Runtime caches, model aliases and all unrelated runs are excluded. No GPU or
backend imports occur. The current inactive budget ledger is copied separately.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tarfile

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
OUT = "artifacts/prefix_io_v1/server10-kv-system-delivery-v1-20261003"
JOBS = ("server10-kv-byte-populate-01", "server10-kv-byte-populate-02", "server10-kv-byte-paired-02")
PRODUCER_JOBS = JOBS[:2]
RUNS = "experiments/prefix_io_v1/runs"
SCOPES = tuple(RUNS + "/" + job for job in JOBS) + (
    "artifacts/prefix_io_v1/server10-kv-capture-v1-20261003",
    "artifacts/prefix_io_v1/server10-kv-diagnostic-v1-20261003",
    "artifacts/prefix_io_v1/server10-kv-diagnostic-v2-20261003",
    OUT,
)
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server10.reference.yaml"
TEXT_EXTENSIONS = {".json", ".jsonl", ".log", ".txt", ".md", ".py", ".pyi", ".yaml", ".yml", ".xml", ".out", ".err", ".csv", ".tsv", ".toml", ".ini", ".cfg", ".patch", ".diff", ".sh", ".ps1", ".c", ".h", ".cpp", ".hpp", ".cu", ".cuh", ".sha256"}
TEXT_NAMES = {"README", "LICENSE", "Makefile", "stdout", "stderr"}
SKIP_DIRECTORIES = {"runtime-cache", "__pycache__", ".git", ".pytest_cache"}
FILE_BYTES = 917504
MAX_BINARY_FILES = 48
MAX_FILE_BYTES = 8 * 1024**2
MAX_TOTAL_BYTES = 100 * 1024**2
FLOOR_BYTES = 8 * 1024**3
NAMES = ("GPU_BUDGET_LEDGER_SNAPSHOT.json", "KV_EVIDENCE_MANIFEST.json", "KV_EVIDENCE.tar", "KV_EVIDENCE_RECEIPT.json")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe_name(relative):
    require(type(relative) is str and relative and not any(c in relative for c in ("\\", ":", "\x00"))
        and not relative.startswith("/") and not Path(relative).is_absolute()
        and all(p not in ("", ".", "..") for p in relative.split("/")), "safe evidence relative path")


def safe(root, relative):
    safe_name(relative)
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "symlink in selected evidence path: " + relative)
        cursor = cursor.parent
    require(root in path.resolve().parents, "selected evidence outside project")
    return path


def read_regular(path, maximum=MAX_FILE_BYTES):
    """No following selected links; compare the opened regular descriptor too."""
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= maximum, "bounded regular evidence file: " + str(path))
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        require(stat.S_ISREG(opened.st_mode) and (opened.st_dev, opened.st_ino, opened.st_size) ==
                (before.st_dev, before.st_ino, before.st_size), "evidence changed while opening")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    require(len(data) == before.st_size and len(data) <= maximum
            and (after.st_size, after.st_mtime_ns, after.st_ctime_ns) ==
            (before.st_size, before.st_mtime_ns, before.st_ctime_ns), "evidence changed while reading")
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected(relative):
    path = Path(relative)
    if "runtime-cache" in path.parts or not any(relative.startswith(scope + "/") for scope in SCOPES):
        return False
    if relative in {OUT + "/" + name for name in NAMES}:
        return False
    if path.suffix.lower() in TEXT_EXTENSIONS or path.name in TEXT_NAMES:
        return True
    return any(path.parent.as_posix() == RUNS + "/" + job + "/details/kv-capture" for job in JOBS) and bool(
        re.fullmatch(r"producer-[0-9a-f]{2,128}\.bin", path.name))


def collect(root):
    paths, omitted = [], []
    for relative in SCOPES:
        base = safe(root, relative)
        require(base.exists(), "required actual job/artifact directory missing: " + relative)
        require(base.is_dir(), "fixed evidence scope must be a directory")
        for directory, dirs, files in os.walk(base, topdown=True, followlinks=False):
            keep = []
            for name in sorted(dirs):
                item = Path(directory) / name
                rel = item.relative_to(root).as_posix()
                if name in SKIP_DIRECTORIES:
                    omitted.append(dict(path=rel, reason="excluded_runtime_or_tool_cache_directory"))
                else:
                    require(not item.is_symlink(), "symlink directory in evidence scope: " + rel)
                    safe(root, rel)
                    keep.append(name)
            dirs[:] = keep
            for name in sorted(files):
                item = Path(directory) / name
                rel = item.relative_to(root).as_posix()
                if selected(rel):
                    safe(root, rel)
                    require(stat.S_ISREG(item.lstat().st_mode), "selected evidence must be regular")
                    paths.append(rel)
                else:
                    omitted.append(dict(path=rel, reason="outside_regular_text_or_bounded_producer_payload_allowlist"))
    require(len(paths) == len(set(paths)) and len(paths) <= 4096, "finite unique evidence files")
    binaries = [name for name in paths if name.endswith(".bin")]
    require(len(binaries) <= MAX_BINARY_FILES and all(safe(root, name).stat().st_size == FILE_BYTES for name in binaries),
            "at most48 complete917504-byte producer payloads")
    require(all(any(name.startswith(RUNS + "/" + job + "/details/kv-capture/") for job in PRODUCER_JOBS) for name in binaries),
            "producer payloads belong only to actual populate jobs; paired must refer to them")
    require(all(sum(name.startswith(RUNS + "/" + job + "/") for name in binaries) <= 24 for job in PRODUCER_JOBS),
            "at most24 captured producer payloads in each populate run")
    safe(root, PERMISSIONS)
    paths.append(PERMISSIONS)
    return sorted(paths), omitted


def inactive_ledger(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate ledger JSON key")
            result[key] = value
        return result
    ledger = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    require(type(ledger) is dict and "active_reservation" in ledger and ledger["active_reservation"] is None,
            "packaging requires no active original GPU reservation")
    require(type(ledger.get("events")) is list, "actual original ledger events")
    return ledger


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def archive_member(archive, relative, raw):
    safe_name(relative)
    member = tarfile.TarInfo(relative)
    member.type, member.size, member.mode = tarfile.REGTYPE, len(raw), 0o600
    member.uid = member.gid = member.mtime = 0
    member.uname = member.gname = ""
    archive.addfile(member, io.BytesIO(raw))


def package(root):
    root = Path(root).absolute()
    require(root.resolve(strict=True) == root and root.is_dir(), "canonical real project root without symlink ancestors")
    output = safe(root, OUT)
    require(not output.exists() or output.is_dir(), "regular delivery output directory")
    require(all(not (output / name).exists() and not (output / name).is_symlink() for name in NAMES),
            "append-only delivery; a previous package must not be overwritten")
    ledger_path = safe(root, LEDGER)
    ledger_raw = read_regular(ledger_path)
    ledger = inactive_ledger(ledger_raw)
    selected_paths, omitted = collect(root)
    total = sum(safe(root, name).stat().st_size for name in selected_paths) + len(ledger_raw)
    require(total <= MAX_TOTAL_BYTES, "bounded complete evidence package")
    require(shutil.disk_usage(root).free - total - 4*1024**2 >= FLOOR_BYTES,
            "archive plus metadata must preserve original8GiB free-space floor")
    output.mkdir(parents=True, exist_ok=True)
    snapshot_relative = OUT + "/" + NAMES[0]
    write_new(output / NAMES[0], ledger_raw)
    rows = []
    for relative in selected_paths:
        raw = read_regular(safe(root, relative))
        rows.append(dict(path=relative, bytes=len(raw), sha256=sha(raw), origin="actual_regular_source_file"))
    rows.append(dict(path=snapshot_relative, bytes=len(ledger_raw), sha256=sha(ledger_raw), origin="independent_current_ledger_snapshot"))
    require(sum(row["bytes"] for row in rows) <= MAX_TOTAL_BYTES, "actual collected bytes exceed100MiB")
    events = {job: [event for event in ledger["events"] if type(event) is dict and event.get("label") == job] for job in JOBS}
    manifest = dict(schema_version=1, status="EVIDENCE_PACKAGE_ONLY_NOT_A_SYSTEM_VERDICT",
        observed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=rows,
        source_scopes=list(SCOPES), required_jobs=list(JOBS), missing_scopes=[], omitted=omitted,
        ledger_source=dict(path=LEDGER, bytes=len(ledger_raw), sha256=sha(ledger_raw)), active_reservation=None,
        job_ledger_event_counts={job: len(values) for job, values in events.items()},
        job_ledger_exits={job: [dict(exit=value.get("exit"), child_exit=value.get("child_exit"),
            timed_out=value.get("timed_out"), session_drained=value.get("session_drained")) for value in values]
            for job, values in events.items()},
        file_count=len(rows), total_bytes=sum(row["bytes"] for row in rows),
        producer_binary_files=sum(row["path"].endswith(".bin") for row in rows),
        producer_binary_bytes=sum(row["bytes"] for row in rows if row["path"].endswith(".bin")),
        archive_manifest_self_excluded_from_files=True, partial_or_failed_runs_can_be_packaged=True,
        qualification_decided=False, production_qualified=False, performance_claim=False, GPU_operations=0,
        runtime_cache_included=False, symlinks_in_archive=False, hardlinks_in_archive=False)
    manifest_raw = encoded(manifest)
    require(len(manifest_raw) <= MAX_FILE_BYTES, "manifest exceeds8MiB")
    archive_bound = (sum(512 + ((row["bytes"] + 511) // 512) * 512 for row in rows)
        + 512 + ((len(manifest_raw) + 511) // 512) * 512 + 10240)
    require(archive_bound <= MAX_TOTAL_BYTES, "archive including metadata/padding exceeds100MiB")
    require(shutil.disk_usage(root).free - archive_bound - len(manifest_raw) - 65536 >= FLOOR_BYTES,
            "actual archive/manifest/receipt reservation must preserve8GiB floor")
    write_new(output / NAMES[1], manifest_raw)
    # Manual TarInfo/addfile guarantees no automatic symlink/hardlink entries.
    with (output / NAMES[2]).open("xb") as archive_stream:
        with tarfile.open(fileobj=archive_stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for row in rows:
                raw = read_regular(safe(root, row["path"]))
                require(len(raw) == row["bytes"] and sha(raw) == row["sha256"], "source changed after manifest")
                archive_member(archive, row["path"], raw)
            archive_member(archive, OUT + "/" + NAMES[1], manifest_raw)
    require(read_regular(ledger_path) == ledger_raw, "budget ledger changed during packaging")
    require(shutil.disk_usage(root).free >= FLOOR_BYTES, "post-package actual free disk below8GiB")
    archive_path = output / NAMES[2]
    require(archive_path.stat().st_size <= MAX_TOTAL_BYTES, "actual archive exceeds100MiB")
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    receipt = dict(status="PACKAGED_CURRENT_KV_EVIDENCE_NOT_SYSTEM_PASS", file_count=len(rows),
        archive_entries=len(rows)+1, source_bytes=manifest["total_bytes"],
        producer_binary_files=manifest["producer_binary_files"], producer_binary_bytes=manifest["producer_binary_bytes"],
        missing_scopes=[], required_jobs=list(JOBS), active_reservation=None, ledger_unchanged_during_package=True,
        archive=dict(path=str(archive_path), bytes=archive_path.stat().st_size, sha256=digest.hexdigest()),
        manifest=dict(path=str(output / NAMES[1]), bytes=len(manifest_raw), sha256=sha(manifest_raw)),
        ledger_snapshot=dict(path=str(output / NAMES[0]), bytes=len(ledger_raw), sha256=sha(ledger_raw)),
        GPU_operations=0, qualification_decided=False, runtime_cache_included=False)
    write_new(output / NAMES[3], encoded(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    args = parser.parse_args()
    require(args.project.absolute() == ROOT and args.project.resolve(strict=True) == ROOT, "fixed actual server10 project")
    print(json.dumps(package(args.project), ensure_ascii=False))


if __name__ == "__main__":
    main()
