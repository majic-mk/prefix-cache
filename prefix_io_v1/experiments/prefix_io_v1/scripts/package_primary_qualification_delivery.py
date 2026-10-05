"""Package P314 qualification evidence on approved PRIMARY storage.

Run only after final source/version locks, report and patches exist. This script
never edits those locks or packages model/cache payloads or CPU temporary trees.
"""
import hashlib
import json
import time
import zipfile
from pathlib import Path

from experiment_storage import ROOT, permission, preflight

OUT = ROOT / "artifacts/prefix_io_v1/server08-p3-14"
AUX = Path("/root/prefix-io-v1-validation")
PRIMARY = ROOT / "experiments/prefix_io_v1/runs"
DESTINATION = PRIMARY / "deliveries/server08-p3-primary-qualification.zip"
RESERVATION_BYTES = 256 * 1024**2
PACKAGE_SCRIPT = "experiments/prefix_io_v1/scripts/package_primary_qualification_delivery.py"
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

    lock_path = OUT / "source-lock-final.json"
    version_path = OUT / "version-lock.json"
    lock = json.loads(lock_path.read_text())
    assert type(lock) is dict and lock, "final source lock is required"
    version = json.loads(version_path.read_text())
    assert version["source_lock_files"] == len(lock), "version/source lock count mismatch"
    assert version["source_lock_sha256"] == sha(lock_path), "version/source lock SHA mismatch"
    lock_digest, version_digest = sha(lock_path), sha(version_path)
    for value, digest in lock.items():
        path = project_path(value)
        assert type(digest) is str and len(digest) == 64
        assert path.is_file() and not path.is_symlink(), str(path)
        assert sha(path) == digest, "source lock mismatch: " + str(path)
    own_digest = sha(ROOT / PACKAGE_SCRIPT)
    assert lock.get(PACKAGE_SCRIPT) == own_digest, "packager must already be in final source lock"

    def assert_locks_unchanged():
        assert sha(lock_path) == lock_digest and sha(version_path) == version_digest, "final locks changed"

    files = {}

    def add(value):
        path = project_path(value)
        assert path.is_absolute() and ".." not in path.parts, "unsafe evidence path: " + str(path)
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
            assert not path.is_symlink(), "symlink in top-level evidence: " + str(path)
            if path.is_file() and path.suffix.lower() not in PAYLOAD_SUFFIXES:
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
        "REPRODUCE_SERVER08_P3_PRIMARY_QUALIFICATION.md",
        "docs/prefix_io_v1/SERVER08_P3_PRIMARY_QUALIFICATION_REPORT.md",
    ]:
        add(value)

    def add_gpu_run(plan_path, output_flag, expected_label=None):
        entry = json.loads(project_path(plan_path).read_text())
        command = entry["command"]
        assert type(command) is list and command.count(output_flag) == 1, str(plan_path)
        label = entry["label"]
        assert type(label) is str and label.startswith("server08-p3-14-")
        assert Path(label).name == label and label not in (".", ".."), "unsafe run label"
        if expected_label is not None:
            assert label == expected_label, "execution-plan/individual-plan label mismatch"
        run_root = PRIMARY / label
        folder = project_path(command[command.index(output_flag) + 1])
        assert folder == project_path(entry["output"]) == run_root / "details", "frozen output differs from argv"
        add(plan_path)
        add_top(folder)
        for name in ("process.log", "result.json"):
            add(run_root / name)
        return entry

    execution = json.loads((OUT / "execution-plan.json").read_text())
    jobs = execution["jobs"]
    assert type(jobs) is list and len(jobs) == 13, "P314 requires exactly 13 qualification plans"
    assert len({job["label"] for job in jobs}) == len(jobs), "duplicate qualification labels"
    assert len({job["plan"] for job in jobs}) == len(jobs), "duplicate qualification plans"
    for job in jobs:
        add_gpu_run(job["plan"], "--output-dir", job["label"])
    model = add_gpu_run(OUT / "model-off-plan.json", "--output")
    add(model["comparison_reference"])

    # Preserve initial failures and final CPU evidence, without pytest-tmp trees.
    for label in (
        "server08-p3-14-registration-matrix",
        "server08-p3-14-registration-matrix-02",
        "server08-p3-14-test-hardening",
        "server08-p3-14-test-hardening-02",
    ):
        run_root = PRIMARY / label
        add_top(run_root)
        add_top(run_root / "cpu-evidence")
    for name in ("registration.json", "source-manifest.json", "copy-proof.json"):
        add(PRIMARY / "server08-p3-14-source" / name)

    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in EXCLUDED_DELIVERY_NAMES and not path.name.startswith("delivery-"):
            add(path)
    assert_idle(ledger_path)
    assert sha(ledger_path) == ledger_digest, "GPU budget changed while gathering evidence"
    assert_locks_unchanged()

    manifest = dict(
        schema_version=1, created_unix=time.time(),
        project_root=str(ROOT), auxiliary_root=str(AUX),
        report="docs/prefix_io_v1/SERVER08_P3_PRIMARY_QUALIFICATION_REPORT.md",
        destination=str(DESTINATION), reservation_bytes=RESERVATION_BYTES,
        source_lock=str(lock_path), source_lock_sha256=lock_digest,
        version_lock_sha256=version_digest,
        notes=("Model weights and cache .bin payloads remain on server. Includes 13 cost "
               "qualification runs, one native/off model acceptance, initial/final CPU evidence, "
               "PRIMARY source metadata, locks, patches and provenance. No efficacy claim. "
               "Completed labels and delivery commands are append-only; do not rerun them."),
        files=[dict(path=name, server_path=str(path), bytes=path.stat().st_size, sha256=sha(path))
               for name, path in sorted(files.items())],
    )
    manifest_path = OUT / "delivery-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Reserve conservatively for ZIP overhead before any archive write.
    assert sum(item["bytes"] for item in manifest["files"]) + manifest_path.stat().st_size + 1024**2 <= RESERVATION_BYTES, "evidence exceeds frozen ZIP reservation"
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    assert not DESTINATION.parent.is_symlink()
    assert_idle(ledger_path)
    assert sha(ledger_path) == ledger_digest
    assert_locks_unchanged()
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
    assert_locks_unchanged()
    receipt = dict(
        path=str(DESTINATION), bytes=DESTINATION.stat().st_size, sha256=sha(DESTINATION),
        manifest_files=len(files), source_lock_files=len(lock),
        package_script_sha256=own_digest, source_lock_sha256=lock_digest,
        version_lock_sha256=version_digest, zip_roundtrip_all_hashes_verified=True,
        gpu_budget_idle=True, storage_before=storage_before,
        storage_after=preflight(PRIMARY / "deliveries/next", 0),
    )
    (OUT / "delivery-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
