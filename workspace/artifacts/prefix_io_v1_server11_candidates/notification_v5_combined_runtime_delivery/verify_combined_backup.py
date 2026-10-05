"""Validate a bounded CPU-only tar backup completely before safe new extraction.

The separately supplied receipt is the trust anchor. No extractall, symlinks,
overwrites, GPUs, or network calls. Validation and writing use the same bytes.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

MIB = 1024 * 1024
MAX_ARCHIVE = 20 * MIB
MAX_RAW = 30 * MIB
MAX_FILE = 10 * MIB
MAX_FILES = 300
MAX_TAR = MAX_RAW + (MAX_FILES + 1) * 10240 + MIB
MANIFEST = "ARCHIVE_CONTENTS_MANIFEST.json"
ROOTS = frozenset((
    "server11-c5-combined-runtime-cpu-20261004",
    "server11-c5-combined-runtime-review-cpu-20261004",
    "server11-c5-combined-runtime-resource-cpu-20261004",
    "server11-c5-combined-runtime-delivery-cpu-20261004",
))
DEVICE = re.compile(r"^(?:CON|PRN|AUX|NUL|CLOCK\$|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])$", re.I)
HEX = re.compile(r"^[0-9a-f]{64}$")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(data):
    def reject_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(data.decode("utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def integer(value, maximum, label):
    require(type(value) is int and 0 <= value <= maximum, "bounded integer: " + label)
    return value


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_name(name, *, archive_member=True):
    require(type(name) is str and 0 < len(name) <= 240, "bounded raw member name")
    require("\\" not in name and not name.startswith("/"), "backslash/absolute member name")
    parts = name.split("/")
    for part in parts:
        require(part not in ("", ".", ".."), "empty/traversal member component")
        require(not part.endswith((" ", ".")), "Windows trailing space/dot")
        require(not any(ord(c) < 32 or ord(c) == 127 or c in '<>:"|?*' for c in part),
                "Windows illegal character/ADS")
        require(not DEVICE.fullmatch(part.split(".", 1)[0].rstrip(" .")), "Windows reserved device component")
    if archive_member:
        require(name == MANIFEST or (len(parts) >= 2 and parts[0] in ROOTS),
                "member outside four exact CPU directories")
    return tuple(parts)


def bounded_read(path, maximum):
    with Path(path).open("rb") as stream:
        data = stream.read(maximum + 1)
    require(len(data) <= maximum, "input exceeds byte cap: " + str(path))
    return data


def verify_reference(reference, data, label):
    require(type(reference) is dict and set(reference) == {"file", "bytes", "sha256"},
            "exact receipt reference: " + label)
    safe_name(reference["file"], archive_member=False)
    require("/" not in reference["file"], "receipt file must be basename")
    require(type(reference["sha256"]) is str and HEX.fullmatch(reference["sha256"]),
            "canonical SHA256: " + label)
    require(type(reference["bytes"]) is int and reference["bytes"] == len(data),
            "receipt byte mismatch: " + label)
    require(reference["sha256"] == digest(data), "receipt SHA mismatch: " + label)


def validate_backup(archive_bytes, receipt, *, archive_name):
    """Return fully verified member bytes; no filesystem mutations."""
    require(type(archive_bytes) is bytes and len(archive_bytes) <= MAX_ARCHIVE, "archive size cap")
    require(type(receipt) is dict, "receipt object")
    require(receipt.get("status") == "PASS_SEALED_CPU_COMBINED_RUNTIME_RESOURCE_LIMITED", "CPU seal receipt status")
    require(all(type(receipt.get(key)) is int and receipt[key] == 0
                for key in ("GPU_runs", "formal_benchmark_runs")), "receipt must declare zero new GPU/benchmark runs")
    verify_reference(receipt["archive"], archive_bytes, "archive")
    require(receipt["archive"]["file"] == archive_name, "archive basename differs from receipt")
    expected_count = integer(receipt["data_files"], MAX_FILES, "receipt files")
    expected_raw = integer(receipt["raw_bytes"], MAX_RAW, "receipt raw bytes")
    # Bound decompression before tarfile parses metadata, including PAX/longname
    # headers. Never let a compressed archive inflate without an explicit cap.
    if archive_bytes.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(archive_bytes), mode="rb") as stream:
            tar_bytes = stream.read(MAX_TAR + 1)
    else:
        tar_bytes = archive_bytes
    require(len(tar_bytes) <= MAX_TAR, "decompressed tar cap")
    payloads = {}
    raw_sum = 0
    folded = set()
    parent_dirs = set()
    original_spellings = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as archive:
        for member in archive:
            require(len(payloads) < MAX_FILES + 1, "archive file-count cap")
            require(member.type in (tarfile.REGTYPE, tarfile.AREGTYPE), "only regular files allowed")
            require(not member.linkname and not member.sparse, "links/sparse members forbidden")
            # tarfile applies PAX path/size metadata before exposing a member.
            # Those authoritative values must still satisfy every check below.
            parts = safe_name(member.name)
            for n in range(1, len(parts) + 1):
                spelling = "/".join(parts[:n])
                canonical = spelling.casefold()
                require(canonical not in original_spellings or original_spellings[canonical] == spelling,
                        "case-colliding path component")
                original_spellings[canonical] = spelling
            key = member.name.casefold()
            require(key not in folded, "duplicate or case-colliding file")
            ancestors = {"/".join(parts[:n]).casefold() for n in range(1, len(parts))}
            require(not (ancestors & folded) and key not in parent_dirs,
                    "file/parent-path collision")
            integer(member.size, MAX_FILE, "single member bytes")
            stream = archive.extractfile(member)
            require(stream is not None, "regular member cannot be read")
            data = stream.read(MAX_FILE + 1)
            require(len(data) == member.size, "truncated member")
            if member.name != MANIFEST:
                raw_sum += len(data)
                require(raw_sum <= MAX_RAW, "raw data cap")
            payloads[member.name] = data
            folded.add(key)
            parent_dirs.update(ancestors)
    require(MANIFEST in payloads, "top-level manifest missing")
    manifest_bytes = payloads[MANIFEST]
    verify_reference(receipt["manifest"], manifest_bytes, "manifest")
    require(receipt["manifest"]["file"] == MANIFEST, "exact manifest basename")
    manifest = read_json(manifest_bytes)
    require(type(manifest) is dict and type(manifest.get("files")) is list, "manifest schema")
    require(all(type(manifest.get(key)) is int and manifest[key] == 0
                for key in ("GPU_runs", "formal_benchmark_runs")), "manifest must declare zero new GPU/benchmark runs")
    count = integer(manifest["data_file_count"], MAX_FILES, "manifest files")
    raw = integer(manifest["raw_bytes"], MAX_RAW, "manifest raw bytes")
    require(count == expected_count == len(manifest["files"]) == len(payloads) - 1,
            "file counts disagree")
    require(raw == expected_raw == raw_sum, "raw byte counts disagree")
    declared = set()
    present_roots = set()
    for row in manifest["files"]:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact manifest row")
        parts = safe_name(row["path"])
        require(row["path"] != MANIFEST and row["path"] not in declared, "duplicate/recursive manifest row")
        require(row["path"] in payloads, "declared member missing")
        data = payloads[row["path"]]
        require(type(row["bytes"]) is int and row["bytes"] == len(data), "member byte mismatch")
        require(type(row["sha256"]) is str and HEX.fullmatch(row["sha256"]) and
                row["sha256"] == digest(data), "member SHA mismatch")
        declared.add(row["path"])
        present_roots.add(parts[0])
    require(declared == set(payloads) - {MANIFEST}, "unmanifested member")
    require(present_roots == ROOTS, "all four exact CPU directories must be represented")
    return payloads, dict(data_files=count, raw_bytes=raw, archive_sha256=digest(archive_bytes),
                          archive_bytes=len(archive_bytes), manifest_sha256=digest(manifest_bytes),
                          manifest_bytes=len(manifest_bytes), verified_members=len(payloads))


def validate_destination(path):
    path = Path(os.path.abspath(path))
    # Do not resolve through links before checking them. A Windows junction is
    # another reparse point and is rejected alongside ordinary symlinks.
    for part in (path,) + tuple(path.parents):
        require(not part.is_symlink() and not (hasattr(part, "is_junction") and part.is_junction()),
                "output path crosses link/junction")
    return path


def run(args):
    output = validate_destination(args.output_dir)
    result_path = validate_destination(args.result)
    require(not os.path.lexists(output), "output directory must be new")
    require(not os.path.lexists(result_path), "result file must be new")
    require(result_path != output and output not in result_path.parents,
            "result must be outside extracted output directory")
    archive_path = Path(args.archive).resolve(strict=True)
    receipt_path = Path(args.receipt).resolve(strict=True)
    receipt_bytes = bounded_read(receipt_path, MIB)
    archive_bytes = bounded_read(archive_path, MAX_ARCHIVE)
    payloads, summary = validate_backup(archive_bytes, read_json(receipt_bytes), archive_name=archive_path.name)
    # Every path, count, byte and SHA has now passed. No reread of mutable input.
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_destination(output)
    output.mkdir(exist_ok=False)
    for name, data in payloads.items():
        target = output.joinpath(*safe_name(name))
        require(output in target.parents, "output containment")
        target.parent.mkdir(parents=True, exist_ok=True)
        validate_destination(target)
        with target.open("xb") as stream:
            stream.write(data)
    # Rehash only the newly written files, not the input archive a second time.
    for name, data in payloads.items():
        target = output.joinpath(*name.split("/"))
        written = bounded_read(target, MAX_FILE)
        require(written == data, "written member differs from validated bytes")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    validate_destination(result_path)
    result = dict(status="PASS_SHA_VERIFIED_NEW_CPU_COMBINED_RUNTIME_BACKUP", gpu_runs=0,
                  archive=str(archive_path), receipt=str(receipt_path), output_dir=str(output),
                  receipt_sha256=digest(receipt_bytes), result=str(result_path),
                  allowed_roots=sorted(ROOTS), same_verified_bytes_written=True,
                  post_write_verified=True, **summary)
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()


