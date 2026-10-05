"""Bounded CPU audit delivery: archive small files and verify literal bytes."""
import argparse, gzip, hashlib, io, json, math, os
from pathlib import Path
import tarfile

REL = "artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004"
PREFIX = REL + "/"
STEM = "CPU_AUDIT_DELIVERY"
MAX = 32 * 1024**2
MAX_TAR = 40 * 1024**2
LEDGER = "experiments/prefix_io_v1/gpu-budget-ledger.json"
LEDGER_SHA = "60368b94c494d15227eb528273e7b4a1a20119c6f371767ffde5cc96aa8f52f9"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def document(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def name(value):
    require(type(value) is str and value.startswith(PREFIX) and len(value) < 250,
            "exact bounded audit namespace")
    require(all(part not in ("", ".", "..") for part in value.split("/")) and
            not any(c in value for c in ("\\", ":", "\x00")) and value.isascii(),
            "literal relative member name")
    return value


def ref(path, relative):
    require(path.is_file() and not path.is_symlink(), "regular byte source")
    raw = path.read_bytes()
    require(len(raw) <= MAX, "bounded individual file")
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def parse(raw):
    def pairs(rows):
        out = {}
        for k, v in rows:
            require(k not in out, "duplicate JSON key")
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: require(False, "nonfinite JSON"))


def literal_tar(path):
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_TAR + 1)
    require(len(raw) <= MAX_TAR and len(raw) % 512 == 0, "bounded aligned tar")
    rows = []; seen = set(); cursor = 0
    while cursor < len(raw):
        h = raw[cursor:cursor + 512]
        if h == b"\0" * 512:
            require(len(raw) - cursor >= 1024 and raw[cursor:] == b"\0" * (len(raw) - cursor),
                    "two terminal blocks and no hidden trailing payload")
            return rows
        require(h[156:157] == b"0" and h[257:263] == b"ustar\0" and
                h[263:265] == b"00", "literal regular USTAR, no extension/symlink")
        info = tarfile.TarInfo.frombuf(h, encoding="ascii", errors="strict")
        n = name(info.name)
        require(n not in seen and 0 <= info.size <= MAX, "unique bounded member")
        seen.add(n)
        lo = cursor + 512; hi = lo + info.size; end = lo + ((info.size + 511) // 512) * 512
        require(end <= len(raw) and raw[hi:end] == b"\0" * (end - hi), "exact member/padding")
        rows.append((n, raw[lo:hi])); cursor = end
        require(len(rows) <= 256, "bounded member count")
    raise ValueError("missing tar termination")


def pack(root):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "actual CPU-only pack")
    root = root.resolve(strict=True); audit = root / REL
    before = parse((audit / "SOURCE_PROTECTION_BEFORE.json").read_bytes())
    after = parse((audit / "SOURCE_PROTECTION_AFTER.json").read_bytes())
    for key in ("source_lock_ref", "source_map", "actual_completed_GPU_archive_ref", "original_ledger_ref"):
        require(before[key] == after[key], "complete protected source/ledger match")
    ledger = (root / LEDGER).read_bytes()
    require(hashlib.sha256(ledger).hexdigest() == LEDGER_SHA and
            ledger == (audit / "CPU_AUDIT_START_LEDGER_SNAPSHOT.json").read_bytes() and
            parse(ledger)["active_reservation"] is None, "unchanged original idle budget")
    for required in ("COST_EVIDENCE_AUDIT.json", "STARTUP_EVIDENCE_AUDIT.json",
                     "PROGRESS_DEPENDENCY_AUDIT.json", "CPU_NEXT_PHASE_DECISION.json", "FINAL_CPU_AUDIT_REPORT.md"):
        require((audit / required).is_file(), "actual complete audit evidence: " + required)
    rows = []
    for p in sorted(audit.iterdir()):
        require(not p.is_symlink(), "no audit symlink")
        if p.is_dir():
            require(p.name == "__pycache__", "unexpected audit subdirectory")
            continue
        if p.name.startswith(STEM):
            continue
        require(p.suffix in (".py", ".json", ".md", ".log", ".txt"), "small auditable files only")
        rows.append(ref(p, name(p.relative_to(root).as_posix())))
    require(1 <= len(rows) <= 255 and sum(r["bytes"] for r in rows) <= MAX, "bounded payload")
    mf = audit / (STEM + "_MANIFEST.json"); archive = audit / (STEM + ".tar.gz")
    result_path = audit / (STEM + "_RESULT.json")
    require(not any(p.exists() for p in (mf, archive, result_path)), "append-only delivery")
    manifest = {"schema": "fixed_evidence_CPU_byte_manifest_v1", "files": rows, "count": len(rows),
                "payload_bytes": sum(r["bytes"] for r in rows), "complete_original_source_rows_server_verified": 4816,
                "GPU_operations": 0, "new_runtime_qualification": False, "model_or_SDK_payload_embedded": False}
    manifest_raw = document(manifest)
    with mf.open("xb") as stream: stream.write(manifest_raw)
    manifest_ref = ref(mf, mf.relative_to(root).as_posix())
    with archive.open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, compresslevel=6) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as tar:
                for row in rows + [manifest_ref]:
                    p = root / row["path"]
                    require(ref(p, row["path"]) == row, "input drift before insertion")
                    data = p.read_bytes(); info = tarfile.TarInfo(row["path"])
                    info.size = len(data); info.type = tarfile.REGTYPE; info.mode = 0o644
                    tar.addfile(info, io.BytesIO(data))
    actual = literal_tar(archive)
    expected = rows + [manifest_ref]
    require(len(actual) == len(expected), "exact server archive member count")
    for (n, raw), row in zip(actual, expected):
        require(n == row["path"] and len(raw) == row["bytes"] and
                hashlib.sha256(raw).hexdigest() == row["sha256"], "actual server archive bytes")
        require(ref(root / n, n) == row, "source bytes unchanged after pack")
    require((root / LEDGER).read_bytes() == ledger, "original ledger changed during pack")
    result = {"status": "PASS_ACTUAL_CPU_AUDIT_ARCHIVE_BYTES", "archive": ref(archive, archive.relative_to(root).as_posix()),
              "manifest": manifest_ref, "payload_files": len(rows), "archive_members": len(actual),
              "payload_bytes": manifest["payload_bytes"], "original_idle_ledger_sha256": LEDGER_SHA,
              "GPU_operations": 0, "data_deletions": 0, "runtime_qualification": False, "P4_effect_verified": False}
    with result_path.open("xb") as stream: stream.write(document(result))
    return result


def verify(archive, manifest_path, result_path, expected_result_sha, output):
    require(hashlib.sha256(result_path.read_bytes()).hexdigest() == expected_result_sha,
            "actual server-trusted result byte pin")
    result = parse(result_path.read_bytes()); manifest_raw = manifest_path.read_bytes()
    manifest = parse(manifest_raw)
    require(result["status"] == "PASS_ACTUAL_CPU_AUDIT_ARCHIVE_BYTES" and
            result["GPU_operations"] == 0 and result["runtime_qualification"] is False,
            "CPU byte receipt cannot promote GPU qualification")
    require(ref(archive, result["archive"]["path"]) == result["archive"] and
            ref(manifest_path, result["manifest"]["path"]) == result["manifest"], "downloaded byte pins")
    rows = manifest["files"]
    require(manifest["count"] == len(rows) == result["payload_files"] and
            result["archive_members"] == len(rows) + 1 and
            sum(r["bytes"] for r in rows) == manifest["payload_bytes"] == result["payload_bytes"],
            "exact payload totals")
    expected = rows + [result["manifest"]]
    actual = literal_tar(archive)
    require(len(actual) == len(expected), "exact local literal archive members")
    for (n, raw), row in zip(actual, expected):
        require(n == name(row["path"]) and len(raw) == row["bytes"] and
                hashlib.sha256(raw).hexdigest() == row["sha256"], "every actual local member hash")
    require(actual[-1][1] == manifest_raw, "literal terminal manifest")
    require(not output.exists() and output.parent.is_dir(), "fresh output directory")
    output.mkdir(); recorded = output / "recorded_files"; recorded.mkdir()
    mappings = []
    for (n, raw), row in zip(actual, expected):
        filename = hashlib.sha256(n.encode()).hexdigest()[:12] + "_" + Path(n).name[:64]
        target = recorded / filename
        with target.open("xb") as stream: stream.write(raw)
        require(ref(target, n) == row, "actual flat file SHA after writing")
        mappings.append(dict(row, local_path=str(target)))
    receipt = dict(result, status="PASS_ACTUAL_CPU_AUDIT_LOCAL_ARCHIVE_BYTE_VERIFICATION",
                   server_result_sha256=expected_result_sha, files=mappings,
                   local_model_or_SDK_payload_read=False, local_complete4816_assets_rehashed=False)
    with (output / "LOCAL_CPU_AUDIT_BYTE_VERIFICATION.json").open("xb") as stream:
        stream.write(document(receipt))
    return {k: v for k, v in receipt.items() if k != "files"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("pack"); a.add_argument("--project-root", type=Path, required=True)
    a = sub.add_parser("verify")
    for n in ("archive", "manifest", "result", "output-dir"): a.add_argument("--" + n, type=Path, required=True)
    a.add_argument("--result-sha256", required=True)
    args = p.parse_args()
    result = pack(args.project_root) if args.action == "pack" else verify(
        args.archive, args.manifest, args.result, args.result_sha256, args.output_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())

