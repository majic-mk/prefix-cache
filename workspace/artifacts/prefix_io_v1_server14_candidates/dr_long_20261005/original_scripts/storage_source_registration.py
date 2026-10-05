"""P314 explicit primary-disk registration of the one frozen heldout source.

Read-only experimental preparation. No copy, import fallback, native cache
configuration, I/O queue, GPU operation or resource lifecycle change occurs.
Only the existing P312 manifest may register a physically separate source tree.
"""
import errno
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

from experiment_storage import authorized_path, permission

ORIGIN_ROOT = Path("/root/prefix-io-v1-validation/storage/heldout-long")
ORIGIN_MANIFEST = Path(
    "/root/prefix-io-v1-validation/runs/server07-p3-12-mixed-order-01-off/"
    "details/storage-source-manifest.json"
)
ORIGIN_MANIFEST_SHA256 = "c03381abb29e21b54a62a4815892553b58e4ef39cc8087761cc4d173cf75d274"
KEYS = frozenset((
    "schema_version", "gpu_uuid", "source_root", "origin_root", "origin_manifest",
    "origin_manifest_sha256", "source_manifest", "source_manifest_sha256",
    "file_count", "total_bytes",
))


def _require(value, message):
    if not value:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _read_json(path):
    _require(path.is_file() and not path.is_symlink(), "metadata must be a regular file")
    _require(path.stat().st_size <= 8 * 1024**2, "bounded manifest/registration required")
    data = path.read_bytes()
    return json.loads(data, object_pairs_hook=_unique_object), data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
             "SHA-256 must be lowercase hexadecimal")
    return value


def _canonical(value, root, *, absolute_field=False):
    _require(isinstance(value, (str, Path)), "path must be text or Path")
    if type(value) is str:
        _require(value and "\x00" not in value and "\\" not in value, "unsafe path text")
    path = Path(value)
    _require(".." not in path.parts, "path traversal is prohibited")
    if absolute_field:
        _require(path.is_absolute(), "registration paths must be absolute")
    if not path.is_absolute():
        path = root / path
    # Check before resolve so even an alias into an otherwise approved root fails.
    cursor = path
    while True:
        _require(not cursor.is_symlink(), "symlink path or ancestor is prohibited")
        if cursor == cursor.parent:
            break
        cursor = cursor.parent
    _require(path.resolve() == path, "noncanonical path is prohibited")
    return path


def _primary(value, root, primary, *, absolute_field=False):
    path = _canonical(value, root, absolute_field=absolute_field)
    actual, auxiliary = authorized_path(path, project=root)
    _require(auxiliary is None and actual == path and path != primary
             and path.is_relative_to(primary), "path must be inside approved primary experiments")
    return path


def _device(path):
    """Separate seam for CPU-only filesystem-device fixtures."""
    return path.stat().st_dev


def _hash_file(path):
    # Called only for declared regular source members after exact-tree validation.
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
             "source member changed during validation")
    return digest.hexdigest()


def _members(rows):
    _require(type(rows) is list and 0 < len(rows) <= 3048, "bounded frozen manifest list required")
    result = {}
    for row in rows:
        _require(type(row) is dict and set(row) == {"path", "bytes", "sha256"},
                 "strict frozen member keys required")
        value = row["path"]
        _require(type(value) is str and value and "\\" not in value and "\x00" not in value,
                 "unsafe manifest member path")
        parts = value.split("/")
        _require(all(part not in ("", ".", "..") and ":" not in part for part in parts),
                 "unsafe manifest member path")
        relative = PurePosixPath(value)
        _require(not relative.is_absolute() and relative.as_posix() == value
                 and relative.suffix == ".bin", "members must be canonical relative .bin paths")
        _require(value not in result, "duplicate manifest member")
        _require(type(row["bytes"]) is int and row["bytes"] > 0, "positive member bytes required")
        _digest(row["sha256"])
        result[value] = row
    return result


def validate_registration(root, registration_path, storage_source, output_path, expected_gpu_uuid):
    """Validate the explicit source before model startup; never create/copy data.

    Paths and source content must remain frozen through the caller's run. This is
    a point-in-time proof, not an ownership reservation or concurrent writer lock.
    Tests may monkeypatch the three ORIGIN constants for private tmp_path fixtures.
    """
    root = _canonical(root, Path.cwd())
    _require(root.is_dir(), "project root must exist")
    permissions = permission(root)
    primary = _canonical(permissions["approved_experiment_root"], root, absolute_field=True)
    _require(primary.is_dir(), "approved primary root must exist")
    registration_path = _primary(registration_path, root, primary)
    registration, registration_bytes = _read_json(registration_path)
    _require(type(registration) is dict and set(registration) == KEYS,
             "strict source registration keys required")
    _require(type(registration["schema_version"]) is int and registration["schema_version"] == 1,
             "source registration schema must be 1")
    uuid = registration["gpu_uuid"]
    _require(type(uuid) is str and uuid and uuid == expected_gpu_uuid
             and uuid in permissions["approved_gpu_ids"], "GPU UUID is not explicitly approved")
    source = _primary(registration["source_root"], root, primary, absolute_field=True)
    _require(source == _primary(storage_source, root, primary), "storage source differs from registration")
    _require(source.is_dir(), "registered source must be an existing independent directory")
    source_manifest = _primary(registration["source_manifest"], root, primary, absolute_field=True)
    origin = _canonical(registration["origin_root"], root, absolute_field=True)
    _require(origin == Path(ORIGIN_ROOT) and origin.is_dir(), "only the existing heldout origin is allowed")
    _, origin_grant = authorized_path(origin, project=root)
    _require(origin_grant is not None, "origin must remain in approved auxiliary storage")
    origin_manifest = _canonical(registration["origin_manifest"], root, absolute_field=True)
    _require(origin_manifest == Path(ORIGIN_MANIFEST), "only the fixed P312 origin manifest is allowed")
    # Project evidence is read-only; auxiliary evidence also needs its explicit grant.
    if not origin_manifest.is_relative_to(root):
        authorized_path(origin_manifest, project=root)
    _require(not (source == origin or source.is_relative_to(origin) or origin.is_relative_to(source)),
             "registered source may not overlap origin")
    _require(not os.path.samefile(source, origin), "registered source aliases origin directory")
    _require(not registration_path.is_relative_to(source) and not source_manifest.is_relative_to(source),
             "source metadata must stay outside the exact payload tree")
    _require(registration_path != source_manifest, "registration and manifest must be independent metadata")

    origin_rows, origin_bytes = _read_json(origin_manifest)
    frozen_digest = _digest(ORIGIN_MANIFEST_SHA256)
    _require(_sha(origin_bytes) == frozen_digest, "frozen origin manifest changed")
    _require(_digest(registration["origin_manifest_sha256"]) == frozen_digest,
             "declared origin digest differs from frozen manifest")
    source_rows, manifest_bytes = _read_json(source_manifest)
    _require(_digest(registration["source_manifest_sha256"]) == frozen_digest
             and _sha(manifest_bytes) == frozen_digest and manifest_bytes == origin_bytes,
             "source manifest must be byte-identical to frozen origin")
    members = _members(origin_rows)
    _require(source_rows == origin_rows, "source namespace differs from frozen origin")
    count = registration["file_count"]
    total = registration["total_bytes"]
    _require(type(count) is int and count == len(members), "registered file count mismatch")
    _require(type(total) is int and total == sum(row["bytes"] for row in members.values()),
             "registered total bytes mismatch")

    output = _primary(output_path, root, primary)
    _require(not (output == source or output.is_relative_to(source) or source.is_relative_to(output)),
             "output may not alias, contain or be contained by source")
    if output.exists():
        _require(output.is_dir(), "existing output must be a directory")
        _require(not os.path.samefile(output, source), "output aliases source directory")
    nearest = output
    while not nearest.exists():
        nearest = nearest.parent
    _require(nearest.is_dir(), "nearest existing output parent must be a directory")
    source_device, output_device = _device(source), _device(nearest)
    if source_device != output_device:
        raise OSError(errno.EXDEV, "registered source/output hardlink path crosses devices")

    # Check names/types only first; do not read an undeclared source payload.
    observed = set()
    for directory, directories, names in os.walk(source, followlinks=False):
        base = Path(directory)
        for name in directories:
            _require(not (base / name).is_symlink(), "source directory symlink prohibited")
        for name in names:
            path = base / name
            _require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
                     "source members must be nonsymlink regular files")
            observed.add(path.relative_to(source).as_posix())
    _require(observed == set(members), "source tree has missing or extra files")
    proof = hashlib.sha256()
    for relative, row in sorted(members.items()):
        path = _canonical(source / relative, root)
        _require(path.stat().st_size == row["bytes"], "source member size changed")
        if _device(path) != source_device:
            raise OSError(errno.EXDEV, "source member crosses output device")
        digest = _hash_file(path)
        _require(digest == row["sha256"], "source member content changed")
        proof.update((relative + "\0" + str(row["bytes"]) + "\0" + digest + "\n").encode())
    _require(origin_manifest.read_bytes() == origin_bytes
             and source_manifest.read_bytes() == manifest_bytes
             and registration_path.read_bytes() == registration_bytes,
             "registration or frozen metadata changed during validation")
    return dict(
        schema_version=1, gpu_uuid=uuid, source_root=str(source), origin_root=str(origin),
        registration_path=str(registration_path), registration_sha256=_sha(registration_bytes),
        origin_manifest=str(origin_manifest), origin_manifest_sha256=frozen_digest,
        source_manifest=str(source_manifest), source_manifest_sha256=frozen_digest,
        source_content_proof_sha256=proof.hexdigest(), file_count=count, total_bytes=total,
        source_device=source_device, output_parent_device=output_device,
        nearest_existing_output_parent=str(nearest), output_path=str(output),
        source_tree_exact=True, full_source_hashes_verified=True,
        original_payloads_read=False, source_materialization_performed=False,
        native_cache_engine_modified=False, gpu_operations_performed=False,
        proof_scope="point-in-time frozen source/device validation before model startup",
    )
