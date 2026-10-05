"""Finite CPU contracts for original native/cached numerical references.

This module reads fixed source bytes and caller records. It never imports a
backend, launches a process, writes evidence, creates authority or resets budget.
The outer launcher must bind actual human replies, guard/session, inputs and
analyzer bytes. CPU fixtures/templates do not establish native/GPU provenance.
"""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import re
import types


PARENT_PATH = Path(__file__).with_name("g3_calibration_plan_metrics_v2.py")
PARENT_REF = dict(bytes=24591,
    sha256="96abdd444c187b70c81de82a63f24b990e3fabe65ecd37307909602181648543")
_parent_raw = PARENT_PATH.read_bytes()
if (PARENT_PATH.is_symlink() or len(_parent_raw) != PARENT_REF["bytes"] or
        hashlib.sha256(_parent_raw).hexdigest() != PARENT_REF["sha256"]):
    raise ValueError("exact frozen parent CPU helper required")
_parent_namespace = dict(__name__="_g3_reference_frozen_parent", __file__=str(PARENT_PATH))
exec(compile(_parent_raw, str(PARENT_PATH), "exec", dont_inherit=True), _parent_namespace)
P = types.SimpleNamespace(**_parent_namespace)
del _parent_raw, _parent_namespace

require = P.require
integer = P.integer
number = P.number
relative = P.relative
safe = P.safe
ref_shape = P.ref_shape
checked_ref = P.checked_ref
parse_json = P.parse_json
read_json = P.read_json

PURPOSE = "CURRENT_CONTEXT_CACHED_NUMERICAL_REFERENCE_ONLY"
PROJECT_ROOT = P.PROJECT_ROOT
DELIVERY = P.DELIVERY
GPU_UUID = P.GPU_UUID
MODEL_RELATIVE = P.MODEL_RELATIVE
MODEL_PLAN_RELATIVE = P.MODEL_PLAN_RELATIVE
GUARD_REF = copy.deepcopy(P.GUARD_REF)
ACQUIRE_REF = copy.deepcopy(P.ACQUIRE_REF)
PERMISSIONS_REF = copy.deepcopy(P.PERMISSIONS_REF)
BASELINE_REF = dict(path=DELIVERY + "/gpu-source-lock-metrics-v3-final.json", bytes=910009,
    sha256="b8ad831c294d978e36a0b1a6343118516af5c55a6e0a9c4380ff6644ec91092b")
BASELINE_SHA = BASELINE_REF["sha256"]
BASELINE_FILE_COUNT = 4075
MAX_APPENDED_REFS = 16  # Includes the immutable ancestor lock itself.
MAX_NEW_SOURCE_REFS = MAX_APPENDED_REFS - 1
MODES = ("cold", "paired")
JOB_NAMES = dict(cold="server09-g3-reference-native-01", paired="server09-g3-reference-paired-01")
STORAGE_RELATIVE = "experiments/prefix_io_v1/runs/server09-g3-reference-native-01-unused-storage"
PAIRED_STORAGE_RELATIVE = "experiments/prefix_io_v1/runs/server09-g3-calibration-02-private-storage"
SECONDS_LIMIT = 300
GUARD_RESERVE_SECONDS = 20
TOTAL_RESERVED_SECONDS = 640
GPU_BUDGET_SECONDS = 28800
JOB_STORAGE_RESERVE_BYTES = 1024 ** 3
TOTAL_STORAGE_RESERVE_BYTES = 2 * JOB_STORAGE_RESERVE_BYTES
STORAGE_FLOOR_BYTES = 8 * 1024 ** 3


def same(actual, expected, reason):
    """Preserve exact scalar types and the sign of an expected floating zero."""
    require(type(actual) is type(expected), reason)
    if type(expected) is dict:
        require(set(actual) == set(expected), reason)
        for key in expected:
            same(actual[key], expected[key], reason)
    elif type(expected) in (list, tuple):
        require(len(actual) == len(expected), reason)
        for left, right in zip(actual, expected):
            same(left, right, reason)
    else:
        require(actual == expected, reason)
        if type(expected) is float and expected == 0.0:
            require(math.copysign(1.0, actual) == math.copysign(1.0, expected), reason)


def fixed_context():
    context = P.fixed_context()
    context.update(purpose=PURPOSE, requests_by_mode=dict(cold=6, paired=6),
        total_requests=12, total_outputs=12, outputs_per_request=1,
        sampling=dict(context["sampling"], logprobs=5),
        storage_by_mode=dict(cold=STORAGE_RELATIVE, paired=PAIRED_STORAGE_RELATIVE),
        shared_storage_relative=PAIRED_STORAGE_RELATIVE,
        diagnostic_flags=dict(cold=["--native-hot-diagnostic", "--diagnostic-logprobs"],
                              paired=["--cached-reference-logprobs"]),
        reference_routes=dict(cold=["f", "gpu_hot"], paired=["g_ssd", "g_mem"]),
        cold_external_connector_enabled=False, paired_existing_storage_required=True,
        paired_original_writes_permitted=True, deletes_existing_input=False,
        input_contract="actual pre/post cache byte manifests; no inferred logical KV equivalence",
        analysis_contract="frozen top-5 logprob analyzer plan; no curve or latency-effect qualification",
        timing_scope="original diagnostic TTFT only; excluded from latency fitting",
        total_reserved_seconds=TOTAL_RESERVED_SECONDS,
        total_storage_reserve_bytes=TOTAL_STORAGE_RESERVE_BYTES)
    return context


def storage_relative(mode):
    require(type(mode) is str and mode in MODES, "fixed reference mode")
    return STORAGE_RELATIVE if mode == "cold" else PAIRED_STORAGE_RELATIVE


def acquisition_arguments(mode, project_root=PROJECT_ROOT):
    same(project_root, PROJECT_ROOT, "fixed existing reference project root")
    storage = storage_relative(mode)
    out = project_root + "/experiments/prefix_io_v1/runs/" + JOB_NAMES[mode] + "/details/acquisition"
    return ["--mode", mode, "--domain", "1024", "--sizes", "128", "--reps", "3",
        "--max-num-seqs", "1", "--iodepth", "4", "--model-dir", project_root + "/" + MODEL_RELATIVE,
        "--model-plan", project_root + "/" + MODEL_PLAN_RELATIVE,
        "--storage", project_root + "/" + storage, "--output-dir", out,
        *fixed_context()["diagnostic_flags"][mode]]


def _baseline(raw, g2_baseline_raw=None):
    require(type(raw) is bytes and len(raw) == BASELINE_REF["bytes"] and
        hashlib.sha256(raw).hexdigest() == BASELINE_SHA, "exact complete 4075-reference ancestor required")
    baseline = parse_json(raw)
    require(type(baseline) is dict and type(baseline.get("files")) is list and
        len(baseline["files"]) == BASELINE_FILE_COUNT, "all 4075 ancestor references required")
    seen = {}
    for row in baseline["files"]:
        ref_shape(row)
        require(row["path"] not in seen, "duplicate ancestor reference")
        seen[row["path"]] = row
    for row in (GUARD_REF, ACQUIRE_REF, PERMISSIONS_REF):
        same(seen.get(row["path"]), row, "original guard/acquirer/permission ancestor pin")
    if g2_baseline_raw is not None:
        # Parent validates its original 4050 rows and original finite append.
        # The new append is separate; it never accumulates into the parent cap.
        P.validate_source_manifest(baseline, g2_baseline_raw)
    return baseline


def _new_rows(rows, old_paths=()):
    require(type(rows) is list and 1 <= len(rows) <= MAX_NEW_SOURCE_REFS,
        "new reference append including ancestor self must fit 16 refs")
    seen = set(old_paths) | {BASELINE_REF["path"]}
    for row in rows:
        ref_shape(row)
        name = Path(row["path"]).name.lower()
        require(row["path"].startswith(DELIVERY + "/") and Path(row["path"]).suffix in (".py", ".json")
            and 0 < row["bytes"] <= 4 * 1024 ** 2 and row["path"] not in seen
            and not any(word in name for word in ("authorization", "human", "scope", "source-lock")),
            "finite source/input/analysis append; no authority/lock cycles or prior replacement")
        seen.add(row["path"])
    return copy.deepcopy(rows)


def assemble_source_lock(baseline_raw, new_source_refs, *, g2_baseline_raw=None):
    baseline = _baseline(baseline_raw, g2_baseline_raw)
    added = _new_rows(new_source_refs, [row["path"] for row in baseline["files"]])
    return dict(schema_version=1, status="CPU_G3_REFERENCE_SOURCE_CANDIDATE_NOT_AUTHORIZED",
        allow_gpu_initialization=False, allow_gpu_runs=False, production_qualified=False,
        baseline_source_lock=copy.deepcopy(BASELINE_REF), new_source_refs=added,
        files=copy.deepcopy(baseline["files"]) + [copy.deepcopy(BASELINE_REF)] + added)


def validate_source_manifest(candidate, baseline_raw, *, g2_baseline_raw=None):
    require(type(candidate) is dict and set(candidate) == {
        "schema_version", "status", "allow_gpu_initialization", "allow_gpu_runs",
        "production_qualified", "baseline_source_lock", "new_source_refs", "files"}, "exact reference manifest")
    expected = assemble_source_lock(baseline_raw, candidate["new_source_refs"], g2_baseline_raw=g2_baseline_raw)
    same(candidate, expected, "complete ancestor rows/order/types and finite append must stay unchanged")
    return dict(status="CPU_REFERENCE_MANIFEST_ONLY", files=len(candidate["files"]),
        full_source_bytes_verified=False, authorizes_gpu=False, production_qualified=False)


def verify_source_lock(root, source_lock_ref):
    ref_shape(source_lock_ref)
    require(source_lock_ref["path"].startswith(DELIVERY + "/"), "new reference source lock")
    candidate = parse_json(checked_ref(root, source_lock_ref).read_bytes())
    baseline_raw = checked_ref(root, BASELINE_REF).read_bytes()
    g2_raw = checked_ref(root, P.BASELINE_REF).read_bytes()
    validate_source_manifest(candidate, baseline_raw, g2_baseline_raw=g2_raw)
    refs = {}
    for row in candidate["files"]:
        checked_ref(root, row)
        refs[row["path"]] = copy.deepcopy(row)
    return refs


def _delivery_ref(row, reason):
    ref_shape(row)
    require(row["path"].startswith(DELIVERY + "/") and Path(row["path"]).suffix == ".json",
        reason)
    return copy.deepcopy(row)


def build_reference_plan(new_source_refs, input_manifest_ref, *, analyzer_plan_ref=None):
    rows = _new_rows(new_source_refs)
    input_ref = _delivery_ref(input_manifest_ref, "same delivery frozen input manifest")
    analysis_ref = None if analyzer_plan_ref is None else _delivery_ref(analyzer_plan_ref, "same delivery analyzer plan")
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_REFERENCE_PLAN", purpose=PURPOSE,
        qualification_context=fixed_context(), baseline_source_lock=copy.deepcopy(BASELINE_REF),
        new_source_refs=rows, input_manifest_ref=input_ref, analyzer_plan_ref=analysis_ref,
        source_bytes_verified=False, authorizes_gpu=False, production_qualified=False,
        jobs=[dict(mode=mode, label=JOB_NAMES[mode], original_argv=acquisition_arguments(mode),
            requests=6, output_tokens_per_request=1, diagnostic_logprobs=5,
            seconds_limit=300, guard_reserve_seconds=20, retries=0,
            requires_prior_success_original_shutdown_and_OS_drain=mode == "paired") for mode in MODES])


def scope_template(source_lock_ref=None, plan_ref=None, input_manifest_ref=None, analyzer_plan_ref=None):
    for row in (source_lock_ref, plan_ref, input_manifest_ref, analyzer_plan_ref):
        if row is not None:
            _delivery_ref(row, "same reference delivery byte ref")
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_REFERENCE_TEMPLATE",
        allow_gpu_initialization=False, allow_gpu_runs=False, purpose=PURPOSE, gpu_uuid=GPU_UUID,
        allowed_modes=list(MODES), permitted_run_names=copy.deepcopy(JOB_NAMES), maximum_jobs=2,
        maximum_total_planned_reserve_seconds=640, seconds_limit_per_job=300, guard_reserve_seconds_per_job=20,
        qualification_context=fixed_context(), baseline_source_lock=copy.deepcopy(BASELINE_REF),
        source_lock=copy.deepcopy(source_lock_ref), cpu_plan_ref=copy.deepcopy(plan_ref),
        input_manifest_ref=copy.deepcopy(input_manifest_ref), analyzer_plan_ref=copy.deepcopy(analyzer_plan_ref),
        base_permissions=copy.deepcopy(PERMISSIONS_REF), human_authorization_record=None)


def _scope_shape(scope):
    require(type(scope) is dict and set(scope) == set(scope_template()), "exact new reference scope fields")
    require(scope["status"] == "USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC" and
        scope["allow_gpu_initialization"] is True and scope["allow_gpu_runs"] is True,
        "no new human reference scope; templates/old G3 scopes refuse")
    refs = [scope[key] for key in ("source_lock", "cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref")]
    require(all(row is not None for row in refs), "source/CPU plan/input/analyzer bindings all required")
    expected = scope_template(*refs)
    for key in expected:
        if key not in ("status", "allow_gpu_initialization", "allow_gpu_runs", "human_authorization_record"):
            same(scope[key], expected[key], "exact typed numerical reference context: " + key)
    human = _delivery_ref(scope["human_authorization_record"], "same reference delivery human record")
    require(len({row["path"] for row in refs + [human]}) == 5, "distinct source/plan/input/analyzer/human refs")
    return scope


def validate_scope_records(scope, human, refs):
    _scope_shape(scope)
    require(type(refs) is dict, "actual verified source refs required")
    same(refs.get(PERMISSIONS_REF["path"]), PERMISSIONS_REF, "original effective permission pin")
    for key in ("cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref"):
        same(refs.get(scope[key]["path"]), scope[key], "source-locked reference " + key)
    require(type(human) is dict and human.get("authorization_origin") == "direct_human_reply" and
        all(type(human.get(key)) is str and 0 < len(human[key]) <= 16384 for key in
            ("question", "verbatim_user_answer", "reply_observed_utc", "question_item_id")),
        "actual direct-human record required; caller fixture is not authority")
    for key in ("purpose", "gpu_uuid", "allowed_modes", "permitted_run_names", "maximum_jobs",
        "maximum_total_planned_reserve_seconds", "seconds_limit_per_job", "guard_reserve_seconds_per_job",
        "qualification_context", "source_lock", "cpu_plan_ref", "input_manifest_ref", "analyzer_plan_ref",
        "base_permissions", "baseline_source_lock"):
        same(human.get(key), scope[key], "human must bind exact reference scope: " + key)
    require(human.get("allow_gpu_initialization") is True and human.get("allow_gpu_runs") is True,
        "actual new human record must allow these two jobs")
    if human.get("reply_medium") == "direct_typed_user_message":
        require(human.get("question_item_id") == "not_applicable:direct_typed_user_message" and
            human.get("form_response_id_fabricated") is False, "typed replies cannot fabricate a form identifier")
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
        analyzer_plan_ref=plan.get("analyzer_plan_ref")), "frozen exact reference CPU plan")
    if plan["analyzer_plan_ref"] is not None:
        same(plan["analyzer_plan_ref"], scope["analyzer_plan_ref"], "CPU plan analyzer binding")
    for row in plan["new_source_refs"]:
        same(refs.get(row["path"]), row, "reference CPU plan source pins differ from lock")
    return result


def validate_budget_and_storage(ledger, storage_snapshot, *, remaining_jobs=2, now_unix=None):
    integer(remaining_jobs, 1)
    require(remaining_jobs <= 2, "finite two-job reference reservation")
    return P.validate_budget_and_storage(ledger, storage_snapshot,
        remaining_jobs=remaining_jobs, now_unix=now_unix)


RECEIPT_FIELDS = frozenset(P.RECEIPT_FIELDS)


def next_job(receipts):
    require(type(receipts) is list and len(receipts) <= 2, "two ordered jobs only; no third job or retry")
    for mode, row in zip(MODES, receipts):
        require(type(row) is dict and set(row) == RECEIPT_FIELDS, "complete bound original receipt")
        same(row["mode"], mode, "strict native then cached-reference order")
        same(row["label"], JOB_NAMES[mode], "fixed new reference job label")
        for key in ("guard_exit", "child_exit", "original_exit_code"):
            same(row[key], 0, "prior failure ends scope without retry")
        same(row["timed_out"], False, "timeout ends reference scope")
        same(row["error"], None, "prior original error ends reference scope")
        same(row["session_drained"], True, "actual original OS drain required")
        for key in ("session_members_before_cleanup", "session_members_after_cleanup"):
            same(row[key], [], "successful original job leaves no OS session cleanup")
        same(row["original_shutdown_completed"], True, "original shutdown not established")
        same(row["original_acquisition_status"], "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            "actual original reference acquisition must pass")
    return MODES[len(receipts)] if len(receipts) < 2 else None


def validate_execution_guard(scope, ledger, mode, *, actual_session_id, expected_command, scope_sha256):
    _scope_shape(scope)
    require(type(mode) is str and mode in MODES, "fixed reference mode")
    integer(actual_session_id, 1)
    require(type(expected_command) is list and 1 <= len(expected_command) <= 64 and
        all(type(part) is str and 0 < len(part) <= 4096 for part in expected_command), "actual bounded original argv")
    require(type(scope_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", scope_sha256), "actual scope SHA")
    require(type(ledger) is dict and type(ledger.get("active_reservation")) is dict,
        "actual original guard reservation required")
    active = ledger["active_reservation"]
    require(type(ledger.get("events")) is list and len(ledger["events"]) <= 10000 and
        all(type(event) is dict for event in ledger["events"]) and
        not any(event.get("label") == JOB_NAMES[mode] for event in ledger["events"]), "used reference label cannot retry")
    for key, expected in (("label", JOB_NAMES[mode]), ("gpu_uuid", GPU_UUID),
        ("seconds_limit", 300), ("reserved_seconds", 320), ("session_id", actual_session_id),
        ("process_group", actual_session_id), ("permissions", PERMISSIONS_REF), ("command", expected_command)):
        same(active.get(key), expected, "original guard reservation mismatch: " + key)
    require(number(ledger.get("gpu_wall_seconds")) + 320 <= GPU_BUDGET_SECONDS, "original cumulative budget")
    require(type(active.get("id")) is str and re.fullmatch(r"[0-9a-f]{32}", active["id"]), "actual reservation identity")
    return dict(schema_version=1, purpose=PURPOSE, gpu_uuid=GPU_UUID, label=JOB_NAMES[mode],
        source_lock_sha256=scope["source_lock"]["sha256"], scope_sha256=scope_sha256,
        permissions_ref=copy.deepcopy(PERMISSIONS_REF), session_id=actual_session_id,
        guard_command=list(expected_command), seconds_limit=300, reserved_seconds=320,
        authorized_scope_verified=True, active_reservation_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-template", action="store_true", required=True)
    parser.parse_args(argv)
    import json
    print(json.dumps(scope_template(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
