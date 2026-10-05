"""CPU contracts for an explicit server09 -> server10 reference migration.

Reuse the immutable original scalar/path helpers and experiment definition.
Only machine identity, exclusive output names and effective permissions change.
This module never imports a backend, creates GPU authority, reserves budget,
launches a process or changes an ancestor file.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import types


OLD_DELIVERY = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
DELIVERY = "artifacts/prefix_io_v1/server10-reference-migration-v1-20261003"
SDK_DELIVERY = "artifacts/prefix_io_v1/server10-sdk-rebind-v1-20261003"
SDK_CPU_DELIVERY = "artifacts/prefix_io_v1/server10-cuda13-cpu-20261003"
_original_ref = dict(bytes=19355,
    sha256="600d1cf449c964339c4515de3e2f4a6fe9dbf21c715acf164f102c90eedce017")
_here = Path(__file__).resolve().parent
_candidates = (
    _here.parent / Path(OLD_DELIVERY).name / "g3_reference_plan.py",
    _here.parent.parent / "prefix_io_v1_server09_candidates" /
        "g3_calibration_launcher_cpu_v1/g3_reference_plan.py",
)
_available = [path for path in _candidates if path.is_file()]
if len(_available) != 1:
    raise ValueError("one existing immutable reference helper must be available")
_path = _available[0]
_raw = _path.read_bytes()
if (_path.is_symlink() or len(_raw) != _original_ref["bytes"] or
        hashlib.sha256(_raw).hexdigest() != _original_ref["sha256"]):
    raise ValueError("immutable original reference helper byte drift")
_namespace = dict(__name__="_server10_original_reference_contract", __file__=str(_path))
exec(compile(_raw, str(_path), "exec", dont_inherit=True), _namespace)
ORIGINAL = types.SimpleNamespace(**_namespace)
del _raw, _namespace, _path, _available, _candidates, _here

# These functions retain their original immutable globals. No parent attribute
# mutation is used to pretend that a function's __globals__ changed.
require = ORIGINAL.require
integer = ORIGINAL.integer
number = ORIGINAL.number
same = ORIGINAL.same
relative = ORIGINAL.relative
safe = ORIGINAL.safe
ref_shape = ORIGINAL.ref_shape
checked_ref = ORIGINAL.checked_ref
parse_json = ORIGINAL.parse_json
read_json = ORIGINAL.read_json

ROOT = PROJECT_ROOT = ORIGINAL.PROJECT_ROOT
PURPOSE = ORIGINAL.PURPOSE
SOURCE_GPU_UUID = ORIGINAL.GPU_UUID
GPU_UUID = "GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f"
MODEL_RELATIVE = ORIGINAL.MODEL_RELATIVE
MODEL_PLAN_RELATIVE = ORIGINAL.MODEL_PLAN_RELATIVE
GUARD_REF = copy.deepcopy(ORIGINAL.GUARD_REF)
ACQUIRE_REF = copy.deepcopy(ORIGINAL.ACQUIRE_REF)
# This is the historical cache publication's ancestor, not the migration lock.
BASELINE_REF = copy.deepcopy(ORIGINAL.BASELINE_REF)
BASELINE_SHA = BASELINE_REF["sha256"]
BASELINE_FILE_COUNT = ORIGINAL.BASELINE_FILE_COUNT
MIGRATION_ANCESTOR_REF = dict(path=OLD_DELIVERY + "/gpu-source-lock-reference-v1.json",
    bytes=910072, sha256="8443f45dd4dd6c1c64595199051f5e88449268af347bcbd836c927b754eabe30")
MIGRATION_ANCESTOR_FILE_COUNT = 4088
MAX_NEW_SOURCE_REFS = 32
MAX_APPENDED_REFS = MAX_NEW_SOURCE_REFS + 1
MODES = ("cold", "paired")
JOB_NAMES = dict(cold="server10-g3-reference-native-01", paired="server10-g3-reference-paired-01")
STORAGE_RELATIVE = "experiments/prefix_io_v1/runs/server10-g3-reference-native-01-unused-storage"
PAIRED_STORAGE_RELATIVE = ORIGINAL.PAIRED_STORAGE_RELATIVE
SECONDS_LIMIT = 300
GUARD_RESERVE_SECONDS = 20
TOTAL_RESERVED_SECONDS = 640
GPU_BUDGET_SECONDS = 28800
JOB_STORAGE_RESERVE_BYTES = 1024 ** 3
TOTAL_STORAGE_RESERVE_BYTES = 2 * JOB_STORAGE_RESERVE_BYTES
STORAGE_FLOOR_BYTES = 8 * 1024 ** 3
PERMISSIONS = "experiments/prefix_io_v1/configs/permissions.server10.reference.yaml"
_permission_text = """# User-designated server10 migration, 2026-10-03; original cumulative budget and restrictions preserved.
schema_version: 1
allow_project_local_edits: true
allow_cpu_tests: true
allow_public_source_read: true
allow_remote_push: false
allow_new_cloud_rental: false
allow_payment: false
allow_driver_or_system_changes: false
allow_shared_data_deletion: false
allow_gpu_runs: true
allow_model_downloads: false
max_gpu_hours: 8
max_model_download_gib: 20
approved_gpu_ids:
- GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f
approved_experiment_root: /root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs
approved_dependency_root: /root/autodl-tmp/prefix-io-v1-handoff/project
"""
PERMISSIONS_BYTES = _permission_text.encode("utf-8")
PERMISSIONS_REF = dict(path=PERMISSIONS, bytes=len(PERMISSIONS_BYTES),
    sha256=hashlib.sha256(PERMISSIONS_BYTES).hexdigest())
require(PERMISSIONS_REF["bytes"] == 685 and PERMISSIONS_REF["sha256"] ==
    "aa54897390dd18c71f4578d45c87b74a3d26a3c63daea2660bf752c41be3be50",
    "exact real server10 permission bytes")
del _permission_text
INPUT_REF = dict(path=OLD_DELIVERY + "/G3_REFERENCE_INPUT_MANIFEST.json", bytes=9730,
    sha256="d601114bd93fb99643bc0c9c5be8c5ab2fa83deec95ecc1e366ed48d00b8af24")


def machine_migration():
    return dict(source_gpu_uuid=SOURCE_GPU_UUID, execution_gpu_uuid=GPU_UUID,
        project_root=PROJECT_ROOT, execution_driver_version="580.95.05",
        execution_driver_path="/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05",
        driver_assets_require_new_real_CPU_compile_link_proof=True,
        historical_evidence_modified=False, old_machine_timings_reused_for_fit=False,
        original_experiment_definition_preserved=True, budget_reset=False)


def fixed_context():
    context = copy.deepcopy(ORIGINAL.fixed_context())
    context["gpu_uuid"] = GPU_UUID
    context["storage_by_mode"]["cold"] = STORAGE_RELATIVE
    return context


def storage_relative(mode):
    require(type(mode) is str and mode in MODES, "fixed migrated reference mode")
    return STORAGE_RELATIVE if mode == "cold" else PAIRED_STORAGE_RELATIVE


def acquisition_arguments(mode, project_root=PROJECT_ROOT):
    same(project_root, PROJECT_ROOT, "fixed existing migration project root")
    storage = storage_relative(mode)
    out = project_root + "/experiments/prefix_io_v1/runs/" + JOB_NAMES[mode] + "/details/acquisition"
    return ["--mode", mode, "--domain", "1024", "--sizes", "128", "--reps", "3",
        "--max-num-seqs", "1", "--iodepth", "4", "--model-dir", project_root + "/" + MODEL_RELATIVE,
        "--model-plan", project_root + "/" + MODEL_PLAN_RELATIVE,
        "--storage", project_root + "/" + storage, "--output-dir", out,
        *fixed_context()["diagnostic_flags"][mode]]


def _ancestor(raw):
    require(type(raw) is bytes and len(raw) == MIGRATION_ANCESTOR_REF["bytes"] and
        hashlib.sha256(raw).hexdigest() == MIGRATION_ANCESTOR_REF["sha256"],
        "exact immutable 8443 migration ancestor required")
    ancestor = parse_json(raw)
    require(type(ancestor) is dict and type(ancestor.get("files")) is list and
        len(ancestor["files"]) == MIGRATION_ANCESTOR_FILE_COUNT,
        "all 4088 original source references must remain")
    rows = {}
    for row in ancestor["files"]:
        ref_shape(row)
        require(row["path"] not in rows, "duplicate original reference")
        rows[row["path"]] = row
    for row in (GUARD_REF, ACQUIRE_REF, ORIGINAL.PERMISSIONS_REF, BASELINE_REF, INPUT_REF):
        same(rows.get(row["path"]), row, "immutable original guard/acquirer/permission/cache ancestor pin")
    return ancestor


def _new_rows(rows, old_paths=()):
    require(type(rows) is list and 1 <= len(rows) <= MAX_NEW_SOURCE_REFS,
        "bounded migration append of at most 32 files")
    seen = set(old_paths) | {MIGRATION_ANCESTOR_REF["path"]}
    for row in rows:
        ref_shape(row)
        path = row["path"]
        name = Path(path).name.lower()
        permission = path == PERMISSIONS
        allowed = permission or (any(path.startswith(prefix + "/") for prefix in (DELIVERY, SDK_DELIVERY, SDK_CPU_DELIVERY))
            and Path(path).suffix in (".py", ".json"))
        require(allowed and 0 < row["bytes"] <= 4 * 1024 ** 2 and path not in seen
            and not any(word in name for word in ("authorization", "human", "scope", "source-lock")),
            "finite separate migration/SDK source append; no replacement or authority cycles")
        if permission:
            same(row, PERMISSIONS_REF, "exact new UUID permission bytes")
        seen.add(path)
    return copy.deepcopy(rows)


def assemble_source_lock(ancestor_raw, new_source_refs):
    ancestor = _ancestor(ancestor_raw)
    added = _new_rows(new_source_refs, [row["path"] for row in ancestor["files"]])
    require(PERMISSIONS_REF in added, "new effective permission must be source locked")
    return dict(schema_version=1, status="CPU_SERVER10_MIGRATION_SOURCE_CANDIDATE_NOT_AUTHORIZED",
        allow_gpu_initialization=False, allow_gpu_runs=False, production_qualified=False,
        baseline_source_lock=copy.deepcopy(MIGRATION_ANCESTOR_REF), machine_migration=machine_migration(),
        new_source_refs=added,
        files=copy.deepcopy(ancestor["files"]) + [copy.deepcopy(MIGRATION_ANCESTOR_REF)] + added)


def validate_source_manifest(candidate, ancestor_raw):
    require(type(candidate) is dict and set(candidate) == {"schema_version", "status",
        "allow_gpu_initialization", "allow_gpu_runs", "production_qualified", "baseline_source_lock",
        "machine_migration", "new_source_refs", "files"}, "exact migration manifest fields")
    expected = assemble_source_lock(ancestor_raw, candidate["new_source_refs"])
    same(candidate, expected, "all original rows/order/types plus bounded migration append unchanged")
    return dict(status="CPU_MIGRATION_MANIFEST_ONLY", files=len(candidate["files"]),
        full_source_bytes_verified=False, authorizes_gpu=False, production_qualified=False)


def verify_source_lock(root, source_lock_ref):
    _delivery_ref(source_lock_ref, "new migration source lock")
    candidate = parse_json(checked_ref(root, source_lock_ref).read_bytes())
    ancestor_raw = checked_ref(root, MIGRATION_ANCESTOR_REF).read_bytes()
    validate_source_manifest(candidate, ancestor_raw)
    refs = {}
    for row in candidate["files"]:
        checked_ref(root, row)
        refs[row["path"]] = copy.deepcopy(row)
    return refs


def _delivery_ref(row, reason):
    ref_shape(row)
    require(row["path"].startswith(DELIVERY + "/") and Path(row["path"]).suffix == ".json", reason)
    return copy.deepcopy(row)


def _input_ref(row):
    same(row, INPUT_REF, "actual immutable server09 published cache input manifest")
    return copy.deepcopy(row)


def build_reference_plan(new_source_refs, input_manifest_ref, *, analyzer_plan_ref=None):
    rows = _new_rows(new_source_refs)
    input_ref = _input_ref(input_manifest_ref)
    analysis_ref = None if analyzer_plan_ref is None else _delivery_ref(analyzer_plan_ref, "migration analyzer plan")
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_REFERENCE_PLAN", purpose=PURPOSE,
        qualification_context=fixed_context(), baseline_source_lock=copy.deepcopy(BASELINE_REF),
        migration_ancestor_source_lock=copy.deepcopy(MIGRATION_ANCESTOR_REF), machine_migration=machine_migration(),
        new_source_refs=rows, input_manifest_ref=input_ref, analyzer_plan_ref=analysis_ref,
        source_bytes_verified=False, authorizes_gpu=False, production_qualified=False,
        jobs=[dict(mode=mode, label=JOB_NAMES[mode], original_argv=acquisition_arguments(mode),
            requests=6, output_tokens_per_request=1, diagnostic_logprobs=5,
            seconds_limit=300, guard_reserve_seconds=20, retries=0,
            requires_prior_success_original_shutdown_and_OS_drain=mode == "paired") for mode in MODES])


def scope_template(source_lock_ref=None, plan_ref=None, input_manifest_ref=None, analyzer_plan_ref=None):
    for row in (source_lock_ref, plan_ref, analyzer_plan_ref):
        if row is not None:
            _delivery_ref(row, "same migration delivery byte reference")
    if input_manifest_ref is not None:
        _input_ref(input_manifest_ref)
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_REFERENCE_TEMPLATE",
        allow_gpu_initialization=False, allow_gpu_runs=False, purpose=PURPOSE, gpu_uuid=GPU_UUID,
        allowed_modes=list(MODES), permitted_run_names=copy.deepcopy(JOB_NAMES), maximum_jobs=2,
        maximum_total_planned_reserve_seconds=640, seconds_limit_per_job=300, guard_reserve_seconds_per_job=20,
        qualification_context=fixed_context(), baseline_source_lock=copy.deepcopy(BASELINE_REF),
        migration_ancestor_source_lock=copy.deepcopy(MIGRATION_ANCESTOR_REF), machine_migration=machine_migration(),
        source_lock=copy.deepcopy(source_lock_ref), cpu_plan_ref=copy.deepcopy(plan_ref),
        input_manifest_ref=copy.deepcopy(input_manifest_ref), analyzer_plan_ref=copy.deepcopy(analyzer_plan_ref),
        base_permissions=copy.deepcopy(PERMISSIONS_REF), human_authorization_record=None)


def _scope_shape(scope):
    require(type(scope) is dict and set(scope) == set(scope_template()), "exact migrated reference scope fields")
    require(scope["status"] == "USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC" and
        scope["allow_gpu_initialization"] is True and scope["allow_gpu_runs"] is True,
        "explicit migrated human reference scope required")
    refs = [scope[key] for key in ("source_lock", "cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref")]
    require(all(row is not None for row in refs), "source/plan/input/analyzer bindings all required")
    expected = scope_template(*refs)
    for key in expected:
        if key not in ("status", "allow_gpu_initialization", "allow_gpu_runs", "human_authorization_record"):
            same(scope[key], expected[key], "exact typed migration reference context: " + key)
    human = _delivery_ref(scope["human_authorization_record"], "separate migration human record")
    require(len({row["path"] for row in refs + [human]}) == 5, "distinct bound reference files")
    return scope


def validate_scope_records(scope, human, refs):
    _scope_shape(scope)
    require(type(refs) is dict, "actual verified source references required")
    for row in (PERMISSIONS_REF, MIGRATION_ANCESTOR_REF):
        same(refs.get(row["path"]), row, "new permission and immutable migration ancestor pins")
    for key in ("cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref"):
        same(refs.get(scope[key]["path"]), scope[key], "source-locked migrated reference " + key)
    require(type(human) is dict and human.get("authorization_origin") == "direct_human_reply" and
        all(type(human.get(key)) is str and 0 < len(human[key]) <= 16384 for key in
            ("question", "verbatim_user_answer", "reply_observed_utc", "question_item_id")),
        "direct human migration request record required; CPU fixtures are not authority")
    for key in ("purpose", "gpu_uuid", "allowed_modes", "permitted_run_names", "maximum_jobs",
        "maximum_total_planned_reserve_seconds", "seconds_limit_per_job", "guard_reserve_seconds_per_job",
        "qualification_context", "source_lock", "cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref",
        "base_permissions", "baseline_source_lock", "migration_ancestor_source_lock", "machine_migration"):
        same(human.get(key), scope[key], "human record must bind migrated context: " + key)
    require(human.get("allow_gpu_initialization") is True and human.get("allow_gpu_runs") is True,
        "actual user migration record must permit the two reference jobs")
    if human.get("reply_medium") == "direct_typed_user_message":
        require(human.get("question_item_id") == "not_applicable:direct_typed_user_message" and
            human.get("form_response_id_fabricated") is False, "typed reply has no fabricated form identifier")
    return dict(status="CPU_REFERENCE_SCOPE_RECORD_CONSISTENCY_ONLY", human_record_matches=True,
        independent_human_provenance_verified=False, authorizes_gpu=False, production_qualified=False)


def validate_scope(root, scope, refs):
    _scope_shape(scope)
    human = parse_json(checked_ref(root, scope["human_authorization_record"]).read_bytes())
    result = validate_scope_records(scope, human, refs)
    for key in ("source_lock", "cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref", "base_permissions"):
        checked_ref(root, scope[key])
    plan = parse_json(checked_ref(root, scope["cpu_plan_ref"]).read_bytes())
    same(plan, build_reference_plan(plan.get("new_source_refs"), scope["input_manifest_ref"],
        analyzer_plan_ref=scope["analyzer_plan_ref"]), "exact frozen migration CPU reference plan")
    for row in plan["new_source_refs"]:
        same(refs.get(row["path"]), row, "migration plan source pin missing from lock")
    return result


def validate_budget_and_storage(ledger, storage_snapshot, *, remaining_jobs=2, now_unix=None):
    integer(remaining_jobs, 1)
    require(remaining_jobs <= 2, "finite two-job migration reservation")
    # Same ROOT, cumulative cap, reservation and PRIMARY floor as original.
    return ORIGINAL.validate_budget_and_storage(ledger, storage_snapshot,
        remaining_jobs=remaining_jobs, now_unix=now_unix)


RECEIPT_FIELDS = frozenset(ORIGINAL.RECEIPT_FIELDS)


def next_job(receipts):
    require(type(receipts) is list and len(receipts) <= 2, "two ordered migrated jobs; no retry")
    for mode, row in zip(MODES, receipts):
        require(type(row) is dict and set(row) == RECEIPT_FIELDS, "complete bound original receipt")
        same(row["mode"], mode, "native then paired migration order")
        same(row["label"], JOB_NAMES[mode], "exclusive migrated reference label")
        for key in ("guard_exit", "child_exit", "original_exit_code"):
            same(row[key], 0, "prior failure ends migrated scope without retry")
        same(row["timed_out"], False, "timeout ends migrated reference scope")
        same(row["error"], None, "original error ends migrated reference scope")
        same(row["session_drained"], True, "actual OS session drain required")
        for key in ("session_members_before_cleanup", "session_members_after_cleanup"):
            same(row[key], [], "successful original job leaves no cleanup")
        same(row["original_shutdown_completed"], True, "original shutdown must complete")
        same(row["original_acquisition_status"], "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            "actual original reference acquisition must pass")
    return MODES[len(receipts)] if len(receipts) < 2 else None


def validate_execution_guard(scope, ledger, mode, *, actual_session_id, expected_command, scope_sha256):
    _scope_shape(scope)
    require(type(mode) is str and mode in MODES, "fixed migrated reference mode")
    integer(actual_session_id, 1)
    require(type(expected_command) is list and 1 <= len(expected_command) <= 64 and
        all(type(part) is str and 0 < len(part) <= 4096 for part in expected_command), "actual bounded argv")
    require(type(scope_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", scope_sha256), "actual scope SHA")
    require(type(ledger) is dict and type(ledger.get("active_reservation")) is dict,
        "actual original guard reservation required")
    active = ledger["active_reservation"]
    require(type(ledger.get("events")) is list and len(ledger["events"]) <= 10000 and
        all(type(event) is dict for event in ledger["events"]) and
        not any(event.get("label") == JOB_NAMES[mode] for event in ledger["events"]),
        "used migrated reference label cannot retry")
    for key, expected in (("label", JOB_NAMES[mode]), ("gpu_uuid", GPU_UUID),
        ("seconds_limit", 300), ("reserved_seconds", 320), ("session_id", actual_session_id),
        ("process_group", actual_session_id), ("permissions", PERMISSIONS_REF), ("command", expected_command)):
        same(active.get(key), expected, "actual migrated guard reservation mismatch: " + key)
    require(number(ledger.get("gpu_wall_seconds")) + 320 <= GPU_BUDGET_SECONDS, "original cumulative budget")
    require(type(active.get("id")) is str and re.fullmatch(r"[0-9a-f]{32}", active["id"]), "actual reservation ID")
    return dict(schema_version=1, purpose=PURPOSE, gpu_uuid=GPU_UUID, label=JOB_NAMES[mode],
        source_lock_sha256=scope["source_lock"]["sha256"], scope_sha256=scope_sha256,
        permissions_ref=copy.deepcopy(PERMISSIONS_REF), session_id=actual_session_id,
        guard_command=list(expected_command), seconds_limit=300, reserved_seconds=320,
        authorized_scope_verified=True, active_reservation_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-template", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(scope_template(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
