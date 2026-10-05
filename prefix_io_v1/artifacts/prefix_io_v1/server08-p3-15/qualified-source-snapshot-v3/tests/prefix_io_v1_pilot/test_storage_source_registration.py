"""Private tmp_path fixtures only; no real origin/source, native engine or GPU."""
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import storage_source_registration as mod


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "project"
    primary = root / "experiments/prefix_io_v1/runs"
    auxiliary = tmp_path / "auxiliary"
    source = primary / "registered-heldout-source"
    origin = auxiliary / "storage/heldout-long"
    origin_manifest = auxiliary / "runs/server07-p3-12-mixed-order-01-off/details/storage-source-manifest.json"
    source_manifest = primary / "registration/source-manifest.json"
    registration_path = primary / "registration/source-registration.json"
    output = primary / "fresh-cohort/details"
    source.mkdir(parents=True)
    origin.mkdir(parents=True)
    namespace = ("Qwen/Qwen2.5-7B-Instruct/block_size_16_blocks_per_file_1/"
                 "tp_1_pp_size_1_pcp_size_1/rank_0/auto")
    rows = []
    for index, payload in enumerate((b"KV-first\x00", b"KV-second\x01")):
        relative = namespace + "/" + str(index).zfill(3) + "/aa/" + str(index) * 64 + ".bin"
        for directory in (source, origin):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        rows.append(dict(path=relative, bytes=len(payload), sha256=digest(payload)))
    save(origin_manifest, rows)
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_bytes(origin_manifest.read_bytes())
    manifest_sha = digest(origin_manifest.read_bytes())
    registration = dict(
        schema_version=1, gpu_uuid="GPU-test", source_root=str(source), origin_root=str(origin),
        origin_manifest=str(origin_manifest), origin_manifest_sha256=manifest_sha,
        source_manifest=str(source_manifest), source_manifest_sha256=manifest_sha,
        file_count=len(rows), total_bytes=sum(row["bytes"] for row in rows),
    )
    save(registration_path, registration)
    flags = {name: False for name in (
        "allow_project_local_edits", "allow_cpu_tests", "allow_public_source_read",
        "allow_remote_push", "allow_new_cloud_rental", "allow_payment",
        "allow_driver_or_system_changes", "allow_shared_data_deletion",
        "allow_gpu_runs", "allow_model_downloads",
    )}
    flags.update(allow_project_local_edits=True, allow_cpu_tests=True, allow_gpu_runs=True)
    authorization_path = root / "experiments/prefix_io_v1/configs/authorizations/fixture.json"
    auxiliary_grant = dict(root=str(auxiliary), max_bytes=20 * 1024**3,
                           minimum_free_bytes=8 * 1024**3,
                           authorization_record=authorization_path.relative_to(root).as_posix())
    permissions = dict(
        flags, schema_version=1, max_gpu_hours=8, max_model_download_gib=20,
        approved_gpu_ids=["GPU-test"], approved_experiment_root=str(primary),
        approved_dependency_root=str(root), approved_auxiliary_storage=auxiliary_grant,
    )
    permission_path = root / "experiments/prefix_io_v1/configs/permissions.yaml"
    save(permission_path, permissions)  # JSON is valid YAML; exercise real permission parser.
    save(authorization_path, dict(auxiliary_grant, gpu_uuid="GPU-test", max_gpu_hours_unchanged=8))
    monkeypatch.setattr(mod, "ORIGIN_ROOT", origin)
    monkeypatch.setattr(mod, "ORIGIN_MANIFEST", origin_manifest)
    monkeypatch.setattr(mod, "ORIGIN_MANIFEST_SHA256", manifest_sha)
    return SimpleNamespace(
        root=root, primary=primary, auxiliary=auxiliary, source=source, origin=origin,
        origin_manifest=origin_manifest, source_manifest=source_manifest,
        registration_path=registration_path, registration=registration, rows=rows,
        output=output, permissions=permissions, permission_path=permission_path,
        authorization_path=authorization_path,
    )


def validate(f, **changes):
    arguments = dict(root=f.root, registration_path=f.registration_path,
                     storage_source=f.source, output_path=f.output, expected_gpu_uuid="GPU-test")
    arguments.update(changes)
    return mod.validate_registration(**arguments)


def rewrite(f, **changes):
    value = dict(f.registration, **changes)
    save(f.registration_path, value)
    return value


def refreeze(f, monkeypatch, rows):
    """Fixture-only malformed-frozen-manifest seam; production constants stay fixed."""
    save(f.origin_manifest, rows)
    f.source_manifest.write_bytes(f.origin_manifest.read_bytes())
    value = digest(f.origin_manifest.read_bytes())
    monkeypatch.setattr(mod, "ORIGIN_MANIFEST_SHA256", value)
    rewrite(f, origin_manifest_sha256=value, source_manifest_sha256=value,
            file_count=len(rows), total_bytes=sum(row["bytes"] for row in rows))


def test_valid_registration_verifies_only_declared_new_payloads(fixture, monkeypatch):
    f = fixture
    visited = []
    original = mod._hash_file
    def observe(path):
        visited.append(path)
        assert path.is_relative_to(f.source)
        return original(path)
    monkeypatch.setattr(mod, "_hash_file", observe)
    report = validate(f)
    assert set(visited) == {f.source / row["path"] for row in f.rows}
    assert report["source_tree_exact"] and report["full_source_hashes_verified"]
    assert report["file_count"] == 2 and report["total_bytes"] == sum(row["bytes"] for row in f.rows)
    assert report["source_device"] == report["output_parent_device"]
    assert report["original_payloads_read"] is False
    assert report["source_materialization_performed"] is False
    assert report["gpu_operations_performed"] is False
    assert not f.output.exists()


def test_missing_registration_key_rejected(fixture):
    value = dict(fixture.registration)
    del value["origin_manifest_sha256"]
    save(fixture.registration_path, value)
    with pytest.raises(ValueError):
        validate(fixture)


def test_extra_registration_key_rejected(fixture):
    rewrite(fixture, copy_fallback=True)
    with pytest.raises(ValueError):
        validate(fixture)


def test_duplicate_json_registration_key_rejected(fixture):
    path = fixture.registration_path
    path.write_text(path.read_text().replace('"schema_version":1', '"schema_version":1,"schema_version":1', 1))
    with pytest.raises(ValueError, match="duplicate JSON"):
        validate(fixture)


def test_schema_and_count_types_do_not_accept_bool_or_float(fixture):
    for changes in (dict(schema_version=True), dict(file_count=True), dict(file_count=2.0),
                    dict(total_bytes=True), dict(total_bytes=1)):
        rewrite(fixture, **changes)
        with pytest.raises(ValueError):
            validate(fixture)


def test_storage_source_argument_must_match_registered_path(fixture):
    other = fixture.primary / "other-source"
    other.mkdir()
    with pytest.raises(ValueError):
        validate(fixture, storage_source=other)


def test_source_outside_primary_rejected(fixture):
    rewrite(fixture, source_root=str(fixture.origin))
    with pytest.raises(ValueError):
        validate(fixture, storage_source=fixture.origin)


def test_registration_outside_primary_rejected(fixture):
    other = fixture.root / "artifacts/registration.json"
    save(other, fixture.registration)
    with pytest.raises(ValueError):
        validate(fixture, registration_path=other)


def test_source_manifest_outside_primary_rejected(fixture):
    rewrite(fixture, source_manifest=str(fixture.origin_manifest))
    with pytest.raises(ValueError):
        validate(fixture)


def test_origin_root_cannot_be_substituted(fixture):
    other = fixture.auxiliary / "storage/unrelated"
    other.mkdir()
    rewrite(fixture, origin_root=str(other))
    with pytest.raises(ValueError):
        validate(fixture)


def test_equal_content_alternative_origin_manifest_rejected(fixture):
    other = fixture.auxiliary / "runs/another-manifest.json"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(fixture.origin_manifest.read_bytes())
    rewrite(fixture, origin_manifest=str(other))
    with pytest.raises(ValueError):
        validate(fixture)


def test_wrong_expected_gpu_uuid_rejected(fixture):
    with pytest.raises(ValueError):
        validate(fixture, expected_gpu_uuid="GPU-other")


def test_wrong_registered_gpu_uuid_rejected(fixture):
    rewrite(fixture, gpu_uuid="GPU-other")
    with pytest.raises(ValueError):
        validate(fixture)


def test_gpu_uuid_must_be_in_permissions_even_when_arguments_agree(fixture):
    f = fixture
    permissions = dict(f.permissions, approved_gpu_ids=["GPU-other"])
    save(f.permission_path, permissions)
    record = json.loads(f.authorization_path.read_text())
    record["gpu_uuid"] = "GPU-other"
    save(f.authorization_path, record)
    with pytest.raises(ValueError):
        validate(f)


def test_changed_old_manifest_not_accepted_by_refreshing_registration_hashes(fixture):
    f = fixture
    data = f.origin_manifest.read_bytes() + b"\n"
    f.origin_manifest.write_bytes(data)
    f.source_manifest.write_bytes(data)
    rewrite(f, origin_manifest_sha256=digest(data), source_manifest_sha256=digest(data))
    with pytest.raises(ValueError, match="frozen origin manifest changed"):
        validate(f)


def test_source_manifest_cannot_change_format_or_hash(fixture):
    fixture.source_manifest.write_bytes(fixture.source_manifest.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte-identical"):
        validate(fixture)


def test_equal_size_source_content_corruption_rejected(fixture):
    path = fixture.source / fixture.rows[0]["path"]
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="content changed"):
        validate(fixture)


def test_source_size_change_rejected(fixture):
    path = fixture.source / fixture.rows[0]["path"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="size changed"):
        validate(fixture)


def test_extra_payload_rejected_before_any_source_payload_read(fixture, monkeypatch):
    (fixture.source / "undeclared.bin").write_bytes(b"do not read")
    def must_not_read(path):
        raise AssertionError("tree rejection must precede all payload reads")
    monkeypatch.setattr(mod, "_hash_file", must_not_read)
    with pytest.raises(ValueError, match="extra files"):
        validate(fixture)


def test_missing_payload_rejected(fixture):
    (fixture.source / fixture.rows[0]["path"]).unlink()
    with pytest.raises(ValueError, match="missing or extra"):
        validate(fixture)


def test_duplicate_manifest_member_rejected(fixture, monkeypatch):
    rows = copy.deepcopy(fixture.rows)
    rows.append(dict(rows[0]))
    refreeze(fixture, monkeypatch, rows)
    with pytest.raises(ValueError, match="duplicate manifest"):
        validate(fixture)


def test_unsafe_manifest_members_never_escape_the_source_tree(fixture, monkeypatch):
    paths = ("../escape.bin", "/absolute.bin", "a/../../escape.bin", "a//b.bin",
             "a/./b.bin", "a\\b.bin", "C:/escape.bin", "a/\x00.bin")
    for unsafe in paths:
        rows = copy.deepcopy(fixture.rows)
        rows[0]["path"] = unsafe
        refreeze(fixture, monkeypatch, rows)
        with pytest.raises(ValueError):
            validate(fixture)


def test_non_bin_member_and_unknown_member_fields_rejected(fixture, monkeypatch):
    for mutation in (dict(path="README.txt"), dict(extra="unapproved")):
        rows = copy.deepcopy(fixture.rows)
        rows[0].update(mutation)
        refreeze(fixture, monkeypatch, rows)
        with pytest.raises(ValueError):
            validate(fixture)


def test_member_symlink_to_equal_bytes_is_rejected(fixture):
    f = fixture
    path = f.source / f.rows[0]["path"]
    outside = f.root / "outside.bin"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="nonsymlink"):
        validate(f)


def test_directory_symlink_does_not_get_followed(fixture):
    f = fixture
    directory = f.source / "Qwen"
    moved = f.root / "preserved-Qwen"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="directory symlink"):
        validate(f)


def test_source_root_symlink_alias_rejected(fixture):
    f = fixture
    moved = f.primary / "real-source"
    f.source.rename(moved)
    f.source.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate(f)


def test_registration_symlink_rejected(fixture):
    f = fixture
    moved = f.registration_path.with_name("backing.json")
    f.registration_path.rename(moved)
    f.registration_path.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        validate(f)


def test_output_cannot_equal_contain_or_be_contained_by_source(fixture):
    for output in (fixture.source, fixture.source / "child", fixture.primary):
        with pytest.raises(ValueError):
            validate(fixture, output_path=output)


def test_output_symlink_alias_to_source_rejected(fixture):
    alias = fixture.primary / "output-alias"
    alias.symlink_to(fixture.source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        validate(fixture, output_path=alias)


def test_cross_device_output_parent_fails_with_exdev_before_hashes(fixture, monkeypatch):
    f = fixture
    original = mod._device
    def devices(path):
        return original(path) + (1 if path == f.primary else 0)
    monkeypatch.setattr(mod, "_device", devices)
    def must_not_hash(path):
        raise AssertionError("EXDEV must be detected before source hashing/model startup")
    monkeypatch.setattr(mod, "_hash_file", must_not_hash)
    with pytest.raises(OSError) as error:
        validate(f)
    assert error.value.errno == errno.EXDEV


def test_cross_device_member_also_fails_with_exdev(fixture, monkeypatch):
    member = fixture.source / fixture.rows[0]["path"]
    original = mod._device
    monkeypatch.setattr(mod, "_device", lambda path: original(path) + (1 if path == member else 0))
    with pytest.raises(OSError) as error:
        validate(fixture)
    assert error.value.errno == errno.EXDEV


def test_source_metadata_must_stay_outside_exact_payload_tree(fixture):
    manifest = fixture.source / "source-manifest.json"
    manifest.write_bytes(fixture.source_manifest.read_bytes())
    rewrite(fixture, source_manifest=str(manifest))
    with pytest.raises(ValueError, match="metadata"):
        validate(fixture)


def test_nonregular_declared_source_file_never_blocks_on_read(fixture):
    path = fixture.source / fixture.rows[0]["path"]
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular files"):
        validate(fixture)


def test_relative_field_and_traversal_arguments_rejected(fixture):
    rewrite(fixture, source_root="relative/source")
    with pytest.raises(ValueError):
        validate(fixture)
    save(fixture.registration_path, fixture.registration)
    escaped = str(fixture.primary / ".." / "runs" / "registered-heldout-source")
    with pytest.raises(ValueError):
        validate(fixture, storage_source=escaped)
