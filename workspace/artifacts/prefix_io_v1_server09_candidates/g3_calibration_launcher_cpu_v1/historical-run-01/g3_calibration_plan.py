"""Finite, read-only CPU contracts for original raw native-cost acquisition.

No backend import, execution, environment change, authorization creation, file
write or budget mutation occurs here. A matching caller record is a contract
check, not independent human/run provenance; the outer launcher must bind actual
bytes and the original guard session. CPU templates never authorize a GPU.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

PURPOSE = "CURRENT_CONTEXT_NATIVE_RAW_COST_PILOT_ONLY"
PROJECT_ROOT = "/root/autodl-tmp/prefix-io-v1-handoff/project"
GPU_UUID = "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2"
MODES = ("cold", "populate", "paired")
JOB_NAMES = {mode: "server09-g3-calibration-" + mode + "-01" for mode in MODES}
DELIVERY = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
BASELINE_SHA = "0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b"
BASELINE_REF = dict(path="artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json", bytes=899396, sha256=BASELINE_SHA)
GUARD_REF = dict(path="experiments/prefix_io_v1/scripts/run_gpu_stage.py", bytes=13013, sha256="3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a")
ACQUIRE_REF = dict(path="experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py", bytes=18995, sha256="68b2c045bcc9a7de84771d556b0e01180a592280002a2d4360d1c7500c9c856e")
PERMISSIONS_REF = dict(path="experiments/prefix_io_v1/configs/permissions.server09.g1.yaml", bytes=671, sha256="eebc8ff9b3ca09c844514847e615099cf96c80d0361c7a493e0d5fdfc5a10e20")
MODEL_RELATIVE = "models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444"
MODEL_PLAN_RELATIVE = "artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json"
STORAGE_RELATIVE = "experiments/prefix_io_v1/runs/server09-g3-calibration-01-private-storage"
SECONDS_LIMIT = 300
GUARD_RESERVE_SECONDS = 20
TOTAL_RESERVED_SECONDS = 960
GPU_BUDGET_SECONDS = 28800
JOB_STORAGE_RESERVE_BYTES = 1024 ** 3
TOTAL_STORAGE_RESERVE_BYTES = 3 * JOB_STORAGE_RESERVE_BYTES
STORAGE_FLOOR_BYTES = 8 * 1024 ** 3


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def same(actual, expected, reason):
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


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "exact nonnegative integer required")
    return value


def number(value, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "finite nonnegative scalar required")
    return value


def relative(value):
    require(type(value) is str and 0 < len(value) <= 4096 and "\\" not in value and
            not PurePosixPath(value).is_absolute() and
            all(part not in ("", ".", "..") for part in value.split("/")),
            "project-relative POSIX path required")
    return value


def safe(root, rel):
    relative(rel)
    root = Path(root).resolve(strict=True)
    path = root / rel
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), "symlink path rejected")
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), "path outside project")
    return path


def ref_shape(row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact byte ref required")
    relative(row["path"])
    require(integer(row["bytes"]) <= 4 * 1024 ** 3 and type(row["sha256"]) is str and
            re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "bounded source ref required")
    return row


def checked_ref(root, row):
    ref_shape(row)
    path = safe(root, row["path"])
    require(path.is_file() and path.stat().st_size == row["bytes"], "source/evidence size drift")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    require(digest.hexdigest() == row["sha256"], "source/evidence SHA drift")
    return path


def parse_json(raw):
    require(type(raw) is bytes and len(raw) <= 4 * 1024 ** 2, "bounded raw JSON required")
    def unique(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def fixed_context():
    return dict(purpose=PURPOSE, project_root=PROJECT_ROOT, model_relative=MODEL_RELATIVE,
        model_plan_relative=MODEL_PLAN_RELATIVE, model_alias="Qwen/Qwen2.5-7B-Instruct",
        gpu_uuid=GPU_UUID, acquisition_source=ACQUIRE_REF, engine_source="experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py",
        acquisition_domain=1024, sizes=[128], reps=3, max_num_seqs=1, iodepth=4,
        prompt_tokens_per_main_request=129, reusable_prefix_tokens=128,
        outputs_per_request=1, requests_by_mode=dict(cold=3, populate=9, paired=6),
        total_requests=18, total_outputs=18, warmup_rep=0,
        measured_samples_per_point_per_route=2,
        engine_overrides=dict(async_scheduling=False, max_model_len=1040,
            max_num_batched_tokens=1040, kv_cache_memory_bytes=268435456,
            prefix_caching_hash_algo="sha256", disable_log_stats=False),
        sampling=dict(temperature=0.0, seed=0, max_tokens=1, min_tokens=1,
            ignore_eos=True, detokenize=False),
        connector=dict(io_backend="linux_aio", sync_on_store=False, staging_mem=0.125,
            enable_preload=True, preload_share_staging=True, preload_lookahead_requests=1,
            staging_cache="lru", load_planner="off", prefix_cache_break_even_path=None),
        shared_storage_relative=STORAGE_RELATIVE, strategies="off",
        asynchronous_author_io_preserved=True, new_executor=False,
        reset_scope="original calibration GPU-only resets; reset_connector=False",
        timing_scope="original metrics.first_token_latency wall-clock TTFT; not GPU event cost",
        curve_output=False, cost_qualified=False, production_qualified=False,
        performance_claim=False, budget_reset=False, cumulative_gpu_budget_seconds=GPU_BUDGET_SECONDS,
        seconds_limit_per_job=SECONDS_LIMIT, guard_reserve_seconds_per_job=GUARD_RESERVE_SECONDS,
        total_reserved_seconds=TOTAL_RESERVED_SECONDS,
        storage_reserve_bytes_per_job=JOB_STORAGE_RESERVE_BYTES,
        total_storage_reserve_bytes=TOTAL_STORAGE_RESERVE_BYTES, storage_floor_bytes=STORAGE_FLOOR_BYTES,
        existing_assets_only=True, no_download=True, no_system_or_author_changes=True, no_data_deletion=True)


def _baseline(raw):
    require(type(raw) is bytes and len(raw) == BASELINE_REF["bytes"] and
            hashlib.sha256(raw).hexdigest() == BASELINE_SHA, "exact complete G2 baseline required")
    baseline = parse_json(raw)
    require(type(baseline) is dict and type(baseline.get("files")) is list and len(baseline["files"]) == 4050,
            "all 4050 prior refs required")
    refs = {}
    for row in baseline["files"]:
        ref_shape(row)
        require(row["path"] not in refs, "duplicate baseline ref")
        refs[row["path"]] = row
    for row in (GUARD_REF, ACQUIRE_REF, PERMISSIONS_REF):
        same(refs.get(row["path"]), row, "current guard/acquirer/effective permission pin differs")
    return baseline


def _new_rows(rows, old_paths):
    require(type(rows) is list and 1 <= len(rows) <= 32, "finite new source refs required")
    seen = set(old_paths)
    for row in rows:
        ref_shape(row)
        require(row["path"].startswith("artifacts/prefix_io_v1/server09-g3-") and
                Path(row["path"]).suffix in (".py", ".json") and
                row["bytes"] <= 4 * 1024 ** 2 and row["path"] not in seen and
                not any(word in Path(row["path"]).name.lower() for word in ("authorization", "human", "scope", "source-lock")),
                "new source-only append; no authority/lock cycles or prior replacement")
        seen.add(row["path"])
    return copy.deepcopy(rows)


def assemble_source_lock(baseline_raw, new_source_refs):
    baseline = _baseline(baseline_raw)
    appended = _new_rows(new_source_refs, [row["path"] for row in baseline["files"]] + [BASELINE_REF["path"]])
    return dict(schema_version=1, status="CPU_G3_RAW_CALIBRATION_SOURCE_CANDIDATE_NOT_AUTHORIZED",
        allow_gpu_initialization=False, allow_gpu_runs=False, production_qualified=False,
        baseline_source_lock=copy.deepcopy(BASELINE_REF), new_source_refs=appended,
        files=copy.deepcopy(baseline["files"]) + [copy.deepcopy(BASELINE_REF)] + appended)


def validate_source_manifest(candidate, baseline_raw):
    require(type(candidate) is dict and set(candidate) == {
        "schema_version", "status", "allow_gpu_initialization", "allow_gpu_runs",
        "production_qualified", "baseline_source_lock", "new_source_refs", "files"}, "exact candidate manifest")
    expected = assemble_source_lock(baseline_raw, candidate["new_source_refs"])
    same(candidate, expected, "complete prior rows/order/types must stay unchanged")
    return dict(status="CPU_MANIFEST_ONLY", files=len(candidate["files"]),
                full_source_bytes_verified=False, authorizes_gpu=False)


def verify_source_lock(root, source_lock_ref):
    require(source_lock_ref["path"].startswith(DELIVERY + "/"), "new calibration lock required")
    candidate = parse_json(checked_ref(root, source_lock_ref).read_bytes())
    baseline_raw = checked_ref(root, BASELINE_REF).read_bytes()
    validate_source_manifest(candidate, baseline_raw)
    refs = {}
    for row in candidate["files"]:
        checked_ref(root, row)
        refs[row["path"]] = copy.deepcopy(row)
    return refs


def acquisition_arguments(mode, project_root=PROJECT_ROOT):
    require(mode in MODES and type(mode) is str, "fixed raw acquisition mode only")
    same(project_root, PROJECT_ROOT, "fixed existing server project root")
    out = project_root + "/experiments/prefix_io_v1/runs/" + JOB_NAMES[mode] + "/details"
    return ["--mode", mode, "--domain", "1024", "--sizes", "128", "--reps", "3",
        "--max-num-seqs", "1", "--iodepth", "4", "--model-dir", project_root + "/" + MODEL_RELATIVE,
        "--model-plan", project_root + "/" + MODEL_PLAN_RELATIVE,
        "--storage", project_root + "/" + STORAGE_RELATIVE, "--output-dir", out]


def build_calibration_plan(project_root=PROJECT_ROOT, *, new_source_refs):
    same(project_root, PROJECT_ROOT, "fixed existing project root")
    rows = _new_rows(new_source_refs, [BASELINE_REF["path"]])
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_PLAN", purpose=PURPOSE,
        qualification_context=fixed_context(), baseline_source_lock=copy.deepcopy(BASELINE_REF),
        new_source_refs=rows, source_bytes_verified=False, authorizes_gpu=False,
        jobs=[dict(mode=mode, label=JOB_NAMES[mode], original_argv=acquisition_arguments(mode),
            requests=fixed_context()["requests_by_mode"][mode], output_tokens_per_request=1,
            seconds_limit=300, guard_reserve_seconds=20, retries=0,
            requires_prior_success_original_shutdown_and_OS_drain=mode != "cold") for mode in MODES])


def scope_template(source_lock_ref=None, *, plan_ref=None):
    for row in (source_lock_ref, plan_ref):
        if row is not None:
            ref_shape(row)
            require(row["path"].startswith(DELIVERY + "/"), "same new calibration delivery ref")
    return dict(schema_version=1, status="NOT_AUTHORIZED_CPU_TEMPLATE",
        allow_gpu_initialization=False, allow_gpu_runs=False, purpose=PURPOSE, gpu_uuid=GPU_UUID,
        allowed_modes=list(MODES), permitted_run_names=copy.deepcopy(JOB_NAMES), maximum_jobs=3,
        maximum_total_planned_reserve_seconds=960, seconds_limit_per_job=300, guard_reserve_seconds_per_job=20,
        qualification_context=fixed_context(), baseline_source_lock=copy.deepcopy(BASELINE_REF),
        source_lock=copy.deepcopy(source_lock_ref), cpu_plan_ref=copy.deepcopy(plan_ref),
        base_permissions=copy.deepcopy(PERMISSIONS_REF), human_authorization_record=None)


def validate_scope_records(scope, human, refs):
    """Typed record consistency only; caller dictionaries are not human authority."""
    require(type(scope) is dict and set(scope) == set(scope_template()), "exact new calibration scope fields")
    require(scope["status"] == "USER_AUTHORIZED_G3_RAW_COST_PILOT" and
        scope["allow_gpu_initialization"] is True and scope["allow_gpu_runs"] is True,
        "no new human GPU scope; templates/G2 scopes refuse")
    expected = scope_template(scope["source_lock"], plan_ref=scope["cpu_plan_ref"])
    for key in expected:
        if key not in ("status", "allow_gpu_initialization", "allow_gpu_runs", "human_authorization_record"):
            same(scope[key], expected[key], "exact typed raw calibration scope/context: " + key)
    require(scope["source_lock"] is not None and scope["cpu_plan_ref"] is not None, "frozen new source/plan refs required")
    ref_shape(scope["human_authorization_record"])
    require(type(refs) is dict and refs.get(PERMISSIONS_REF["path"]) == PERMISSIONS_REF and
        refs.get(scope["cpu_plan_ref"]["path"]) == scope["cpu_plan_ref"], "bound effective permission and CPU plan refs")
    require(type(human) is dict and human.get("authorization_origin") == "direct_human_reply" and
        all(type(human.get(key)) is str and 0 < len(human[key]) <= 16384 for key in
            ("question", "verbatim_user_answer", "reply_observed_utc", "question_item_id")),
        "frozen actual direct-human reply record required")
    for key in ("purpose", "gpu_uuid", "allowed_modes", "permitted_run_names", "maximum_jobs",
                "maximum_total_planned_reserve_seconds", "seconds_limit_per_job", "guard_reserve_seconds_per_job",
                "qualification_context", "source_lock", "cpu_plan_ref", "base_permissions", "baseline_source_lock"):
        same(human.get(key), scope[key], "human record must explicitly bind new calibration scope: " + key)
    require(human.get("allow_gpu_initialization") is True and human.get("allow_gpu_runs") is True,
        "actual human record must explicitly allow these GPU jobs")
    return dict(status="CPU_SCOPE_RECORD_CONSISTENCY_ONLY", human_record_matches=True,
        independent_human_provenance_verified=False, authorizes_gpu=False, production_qualified=False)


def validate_scope(root, scope, refs):
    require(type(scope) is dict and type(scope.get("human_authorization_record")) is dict,
            "no frozen actual human record")
    human = parse_json(checked_ref(root, scope["human_authorization_record"]).read_bytes())
    result = validate_scope_records(scope, human, refs)
    checked_ref(root, scope["base_permissions"])
    plan = parse_json(checked_ref(root, scope["cpu_plan_ref"]).read_bytes())
    same(plan, build_calibration_plan(PROJECT_ROOT, new_source_refs=plan.get("new_source_refs")), "frozen exact CPU plan")
    return result


def validate_budget_and_storage(ledger, storage_snapshot, *, remaining_jobs=3, now_unix=None):
    integer(remaining_jobs, 1)
    require(remaining_jobs <= 3 and type(ledger) is dict and ledger.get("active_reservation") is None,
            "finite jobs and resolved original reservation required")
    used = number(ledger.get("gpu_wall_seconds"))
    require(type(ledger.get("events")) is list and len(ledger["events"]) <= 10000 and
            all(type(event) is dict for event in ledger["events"]), "actual original cumulative ledger required")
    require(used + remaining_jobs * 320 <= GPU_BUDGET_SECONDS, "insufficient unchanged cumulative GPU budget")
    require(type(storage_snapshot) is dict and set(storage_snapshot) == {
        "origin", "captured_unix", "primary_root", "free_bytes"}, "explicit actual PRIMARY snapshot")
    same(storage_snapshot["origin"], "actual_statvfs", "no estimated/frozen free-space claim")
    same(storage_snapshot["primary_root"], PROJECT_ROOT, "PRIMARY-only existing project")
    captured = number(storage_snapshot["captured_unix"], 1)
    now = time.time() if now_unix is None else number(now_unix, 1)
    require(-5 <= now - captured <= 120, "fresh actual storage snapshot required")
    free = integer(storage_snapshot["free_bytes"])
    reserve = remaining_jobs * JOB_STORAGE_RESERVE_BYTES
    require(free - reserve >= STORAGE_FLOOR_BYTES, "insufficient total persistent reservation plus 8 GiB floor")
    return dict(status="CPU_LEDGER_STORAGE_CONSISTENCY_ONLY", remaining_gpu_seconds=GPU_BUDGET_SECONDS-used,
        reserved_gpu_seconds=remaining_jobs*320, PRIMARY_free_bytes=free,
        remaining_storage_reserve_bytes=reserve, storage_floor_bytes=STORAGE_FLOOR_BYTES,
        actual_free_space_independently_verified=False, authorizes_gpu=False)


RECEIPT_FIELDS = {"mode", "label", "guard_exit", "child_exit", "timed_out", "error",
    "session_drained", "session_members_before_cleanup", "session_members_after_cleanup",
    "original_exit_code", "original_acquisition_status", "original_shutdown_completed"}


def next_job(receipts):
    """Consume externally byte-bound summaries; never interpret flags as live proof."""
    require(type(receipts) is list and len(receipts) <= 3, "finite ordered receipts; no fourth job/retry")
    for mode, row in zip(MODES, receipts):
        require(type(row) is dict and set(row) == RECEIPT_FIELDS, "complete original guard/runtime receipt required")
        same(row["mode"], mode, "ordered cold/populate/paired; no retry")
        same(row["label"], JOB_NAMES[mode], "same preregistered job identity")
        for key in ("guard_exit", "child_exit", "original_exit_code"):
            same(row[key], 0, "prior job failed; scope stops without retry")
        same(row["timed_out"], False, "timeout stops scope")
        same(row["error"], None, "error stops scope")
        same(row["session_drained"], True, "original OS session unresolved")
        for key in ("session_members_before_cleanup", "session_members_after_cleanup"):
            same(row[key], [], "successful original job must leave no child cleanup")
        same(row["original_shutdown_completed"], True, "original engine shutdown not proven")
        same(row["original_acquisition_status"], "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
             "actual original acquisition must succeed")
    return MODES[len(receipts)] if len(receipts) < 3 else None


def validate_execution_guard(scope, ledger, mode, *, actual_session_id, expected_command, scope_sha256):
    require(type(mode) is str and mode in MODES, "fixed calibration mode")
    require(type(scope) is dict and scope.get("status") == "USER_AUTHORIZED_G3_RAW_COST_PILOT" and
        scope.get("allow_gpu_runs") is True and scope.get("allow_gpu_initialization") is True and
        scope.get("purpose") == PURPOSE, "actual already-validated calibration scope required")
    integer(actual_session_id, 1)
    require(type(expected_command) is list and 1 <= len(expected_command) <= 64 and
        all(type(part) is str and 0 < len(part) <= 4096 for part in expected_command), "exact bounded guard command")
    require(type(scope_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", scope_sha256), "actual frozen scope SHA")
    require(type(ledger) is dict and type(ledger.get("active_reservation")) is dict,
        "must execute in actual original guard reservation")
    active = ledger["active_reservation"]
    for key, expected in (("label", JOB_NAMES[mode]), ("gpu_uuid", GPU_UUID),
        ("seconds_limit", 300), ("reserved_seconds", 320), ("session_id", actual_session_id),
        ("process_group", actual_session_id), ("permissions", PERMISSIONS_REF), ("command", expected_command)):
        same(active.get(key), expected, "original guard reservation mismatch: " + key)
    number(ledger.get("gpu_wall_seconds"))
    require(ledger["gpu_wall_seconds"] + 320 <= GPU_BUDGET_SECONDS, "original cumulative execution budget")
    require(type(active.get("id")) is str and re.fullmatch(r"[0-9a-f]{32}", active["id"]), "durable original reservation identity")
    source_lock = ref_shape(scope["source_lock"])
    return dict(schema_version=1, purpose=PURPOSE, gpu_uuid=GPU_UUID, label=JOB_NAMES[mode],
        source_lock_sha256=source_lock["sha256"], scope_sha256=scope_sha256,
        permissions_ref=copy.deepcopy(PERMISSIONS_REF), session_id=actual_session_id,
        guard_command=list(expected_command), seconds_limit=300, reserved_seconds=320,
        authorized_scope_verified=True, active_reservation_verified=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--scope-template", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    parser.add_argument("--project", default=PROJECT_ROOT)
    parser.add_argument("--new-refs")
    parser.add_argument("--source-lock-ref")
    parser.add_argument("--plan-ref")
    parser.add_argument("--scope-record")
    parser.add_argument("--storage-snapshot")
    args = parser.parse_args(argv)
    same(args.project, PROJECT_ROOT, "fixed designated server")
    def supplied(path):
        require(path is not None, "explicit input JSON file required")
        return parse_json(Path(path).read_bytes())
    if args.plan:
        result = build_calibration_plan(args.project, new_source_refs=supplied(args.new_refs))
    elif args.scope_template:
        result = scope_template(supplied(args.source_lock_ref) if args.source_lock_ref else None,
                                plan_ref=supplied(args.plan_ref) if args.plan_ref else None)
    else:
        root = Path(args.project)
        lock_ref = supplied(args.source_lock_ref)
        refs = verify_source_lock(root, lock_ref)
        require(args.scope_record is not None, "NO_NEW_G3_HUMAN_SCOPE")
        scope = parse_json(safe(root, args.scope_record).read_bytes())
        same(scope.get("source_lock"), lock_ref, "same actual candidate source lock")
        checked = validate_scope(root, scope, refs)
        ledger = parse_json(safe(root, "experiments/prefix_io_v1/gpu-budget-ledger.json").read_bytes())
        budget = validate_budget_and_storage(ledger, supplied(args.storage_snapshot))
        result = dict(status="CPU_PREFLIGHT_ONLY_NO_GPU_LAUNCH", source_refs_verified=len(refs),
                      scope=checked, budget_storage=budget, authorizes_gpu=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
