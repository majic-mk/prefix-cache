"""Package P313 delivery on approved PRIMARY storage; never include payloads.

Run only after final source/version locks, report and patches exist. The original
failed CPU evidence is included alongside the final qualified matrix.
"""
import hashlib
import json
import time
import zipfile
from pathlib import Path

from experiment_storage import ROOT, permission, preflight

OUT = ROOT / "artifacts/prefix_io_v1/server08-p3-13"
AUX = Path("/root/prefix-io-v1-validation")
PRIMARY = ROOT / "experiments/prefix_io_v1/runs"
DESTINATION = PRIMARY / "deliveries/server08-p3-dispatch-shadow.zip"
RESERVATION_BYTES = 256 * 1024**2
PACKAGE_SCRIPT = "experiments/prefix_io_v1/scripts/package_dispatch_shadow_delivery.py"
EXCLUDED_DELIVERY_NAMES = {
    "delivery-manifest.json", "delivery-receipt.json",
    "delivery-local-verification.json", "delivery-final-check.json",
}
PAYLOAD_SUFFIXES = {".bin", ".safetensors", ".pt", ".pth", ".ckpt", ".zip"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def assert_idle(ledger_path):
    ledger = json.loads(ledger_path.read_text())
    assert ledger["active_reservation"] is None, "GPU budget must be idle before packaging"
    return ledger


def main():
    ledger_path = ROOT / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    assert_idle(ledger_path)
    ledger_digest = sha(ledger_path)
    authorized = permission(ROOT)
    assert Path(authorized["approved_experiment_root"]) == PRIMARY
    assert DESTINATION.is_relative_to(PRIMARY)
    assert not DESTINATION.exists(), "delivery ZIP is append-only"
    storage_before = preflight(DESTINATION, RESERVATION_BYTES)
    assert Path(storage_before["root"]) == PRIMARY
    assert storage_before["reserved_new_bytes"] == RESERVATION_BYTES

    lock_path = OUT / "source-lock.json"
    version_path = OUT / "version-lock.json"
    lock = json.loads(lock_path.read_text())
    assert type(lock) is dict and lock, "final source lock is required"
    version = json.loads(version_path.read_text())
    if version.get("source_lock_files") is not None:
        assert version["source_lock_files"] == len(lock)
    if version.get("source_lock_sha256") is not None:
        assert version["source_lock_sha256"] == sha(lock_path)
    for value, digest in lock.items():
        path = project_path(value)
        assert type(digest) is str and len(digest) == 64
        assert path.is_file() and not path.is_symlink(), str(path)
        assert sha(path) == digest, "source lock mismatch: " + str(path)
    own_digest = sha(ROOT / PACKAGE_SCRIPT)
    lock[PACKAGE_SCRIPT] = own_digest
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    version.update(source_lock_files=len(lock), source_lock_sha256=sha(lock_path))
    version_path.write_text(json.dumps(version, indent=2), encoding="utf-8")

    files = {}

    def add(value):
        path = project_path(value)
        assert path.is_file() and not path.is_symlink(), str(path)
        assert path.suffix.lower() not in PAYLOAD_SUFFIXES, "payload prohibited: " + str(path)
        assert "models" not in path.parts and "__pycache__" not in path.parts, str(path)
        if path.is_relative_to(ROOT):
            evidence_root = ROOT
            name = path.relative_to(ROOT).as_posix()
        elif path.is_relative_to(AUX):
            evidence_root = AUX
            name = "auxiliary/" + path.relative_to(AUX).as_posix()
        else:
            raise ValueError("outside delivery evidence roots: " + str(path))
        # Prevent a parent-directory symlink from redirecting locked evidence.
        cursor = path.parent
        while cursor != evidence_root:
            assert not cursor.is_symlink(), "redirected evidence directory: " + str(cursor)
            assert cursor != cursor.parent
            cursor = cursor.parent
        assert not evidence_root.is_symlink()
        if name in files:
            assert files[name] == path, "ambiguous archive name: " + name
        files[name] = path

    def add_top(directory):
        directory = project_path(directory)
        assert directory.is_dir() and not directory.is_symlink(), str(directory)
        for path in sorted(directory.iterdir()):
            if path.is_file():
                add(path)

    for value, digest in lock.items():
        path = project_path(value)
        assert sha(path) == digest, "source changed during packaging: " + str(path)
        add(path)

    for directory, pattern in [
        ("docs/prefix_io_v1", "*.md"),
        ("docs/prefix_io_v1/templates", "*"),
        ("experiments/prefix_io_v1/locks", "*.json"),
        ("patches/prefix_io_v1", "*.patch"),
        ("src/prefix_io_control", "*.py"),
        ("experiments/prefix_io_v1/scripts", "*.py"),
    ]:
        for path in sorted((ROOT / directory).rglob(pattern)):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path)
    for directory in sorted((ROOT / "tests").glob("prefix_io_v1*")):
        if directory.is_dir():
            for path in sorted(directory.rglob("*.py")):
                if "__pycache__" not in path.parts:
                    add(path)
    for value in [
        "experiments/prefix_io_v1/execution_state.json",
        "experiments/prefix_io_v1/gpu-budget-ledger.json",
        "experiments/prefix_io_v1/configs/permissions.yaml",
        "REPRODUCE_SERVER08_P3_DISPATCH_SHADOW.md",
    ]:
        add(value)

    for plan_name in ("gpu-plan.json", "native-reference-plan.json", "shadow-gpu-plan.json"):
        entry = json.loads((OUT / plan_name).read_text())
        command = entry["command"]
        assert "--output" in command, plan_name
        folder = project_path(command[command.index("--output") + 1])
        assert folder == project_path(entry["output"]), "frozen output differs from argv"
        add_top(folder)
        run_root = PRIMARY / entry["label"]
        for name in ("process.log", "result.json"):
            add(run_root / name)

    # Original failing session and final corrected session are both preserved.
    for label in ("server08-p3-13-shadow-matrix", "server08-p3-13-shadow-matrix-02"):
        run_root = PRIMARY / label
        add_top(run_root)
        add_top(run_root / "cpu-evidence")
    add_top(AUX / "cpu-evidence/server08-p3-13-migration-matrix")
    add(AUX / "runs/server07-p3-c2-fullref-01/details/result.json")
    for name in ("published-source-clone-fullhash.json", "published-source-post-GPU-fullhash.json"):
        evidence = json.loads((OUT / name).read_text())
        add(evidence["source_manifest"])

    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in EXCLUDED_DELIVERY_NAMES:
            add(path)
    assert_idle(ledger_path)
    assert sha(ledger_path) == ledger_digest, "GPU budget changed while gathering evidence"

    manifest = dict(
        schema_version=1, created_unix=time.time(),
        project_root=str(ROOT), auxiliary_root=str(AUX),
        report="docs/prefix_io_v1/SERVER08_P3_DISPATCH_SHADOW_REPORT.md",
        destination=str(DESTINATION), reservation_bytes=RESERVATION_BYTES,
        notes=("Model weights and cache .bin payloads remain on server. Includes failed and final "
               "CPU logs, real GPU correctness/reference outputs, locks, patches and provenance. "
               "Completed labels and delivery commands are append-only; do not rerun them."),
        files=[dict(path=name, server_path=str(path), bytes=path.stat().st_size, sha256=sha(path))
               for name, path in sorted(files.items())],
    )
    manifest_path = OUT / "delivery-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    assert not DESTINATION.parent.is_symlink()
    assert_idle(ledger_path)
    assert sha(ledger_path) == ledger_digest
    # Revalidate the same reservation immediately before creating the archive.
    preflight(DESTINATION, RESERVATION_BYTES)
    with zipfile.ZipFile(DESTINATION, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.write(manifest_path, "delivery-manifest.json")
    assert DESTINATION.stat().st_size <= RESERVATION_BYTES, "ZIP exceeded frozen reservation"
    with zipfile.ZipFile(DESTINATION) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert len(names) == len(files) + 1 and len(names) == len(set(names))
        for item in manifest["files"]:
            contents = archive.read(item["path"])
            assert len(contents) == item["bytes"]
            assert hashlib.sha256(contents).hexdigest() == item["sha256"], item["path"]
        assert archive.read("delivery-manifest.json") == manifest_path.read_bytes()
    assert_idle(ledger_path)
    assert sha(ledger_path) == ledger_digest
    receipt = dict(
        path=str(DESTINATION), bytes=DESTINATION.stat().st_size, sha256=sha(DESTINATION),
        manifest_files=len(files), source_lock_files=len(lock),
        package_script_sha256=own_digest, zip_roundtrip_all_hashes_verified=True,
        gpu_budget_idle=True, storage_before=storage_before,
        storage_after=preflight(PRIMARY / "deliveries/next", 0),
    )
    (OUT / "delivery-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
