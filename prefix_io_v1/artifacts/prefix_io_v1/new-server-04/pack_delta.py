"""Package this CPU-only continuation as an additive delta to server-03."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "artifacts/prefix_io_v1/new-server-04"
DEST = ROOT.parent / "prefix-io-v1-new-server-04-20260927-delta.zip"
BASE_SHA = "55a7ef22048073c4ffc386eb62eb5a855b5c95b3dc8388b5096bb4c877714698"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def main():
    assert ROOT.name == "project" and EVIDENCE.is_dir()
    assert not DEST.exists(), "Never overwrite a previous delivery."
    selected = list(EVIDENCE.rglob("*"))
    selected += list((ROOT / "tests/prefix_io_v1_calibration").rglob("*"))
    selected += [ROOT / p for p in [
        "experiments/prefix_io_v1/scripts/prepare_native_calibration_plan.py",
        "experiments/prefix_io_v1/execution_state.json",
        "experiments/prefix_io_v1/configs/permissions.yaml",
        "docs/prefix_io_v1/NEW_SERVER_04_REPORT.md",
        "docs/prefix_io_v1/REPRODUCE_NEW_SERVER_04.md",
        "docs/prefix_io_v1/CAPABILITY_MATRIX.md",
        "docs/prefix_io_v1/PATCH_MAP.md",
        "docs/prefix_io_v1/GPU_STAGE_REPORT.md",
        "docs/prefix_io_v1/PLATFORM_IO_URING_REQUEST.md",
        "artifacts/prefix_io_v1/new-server-03/platform-probe/io_uring_setup_probe.py",
    ]]
    records = []
    with zipfile.ZipFile(DEST, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(selected)):
            assert p.exists(), str(p)
            if not p.is_file() or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
                continue
            if p.name in ("delivery-manifest.json", "package-check.json"):
                continue
            assert not p.is_symlink(), str(p)
            p.resolve().relative_to(ROOT.resolve())
            data = p.read_bytes()
            name = "project/" + p.relative_to(ROOT).as_posix()
            records.append(dict(path=p.relative_to(ROOT).as_posix(), archive_path=name,
                                bytes=len(data), sha256=digest(data)))
            z.writestr(name, data)
        manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                        delivery_type="incremental CPU-only continuation",
                        base_archive="prefix-io-v1-new-server-03-20260926.zip",
                        base_archive_sha256=BASE_SHA, phase="P1_BLOCKED",
                        gpu_execution=False, files=records, file_count=len(records))
        payload=(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n").encode("utf-8")
        z.writestr("delivery-manifest.json",payload)
    with zipfile.ZipFile(DEST) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(records)+1
        for row in records:
            data=z.read(row["archive_path"])
            assert len(data)==row["bytes"] and digest(data)==row["sha256"]
    (EVIDENCE/"delivery-manifest.json").write_bytes(payload)
    result=dict(archive=str(DEST), archive_bytes=DEST.stat().st_size,
                archive_sha256=digest(DEST.read_bytes()), manifest_records=len(records),
                zip_entries=len(records)+1, manifest_sha256=digest(payload),
                all_file_sizes_and_sha256_verified=True, gpu_execution=False,
                base_archive_sha256=BASE_SHA)
    (EVIDENCE/"package-check.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))

if __name__ == "__main__":
    main()
