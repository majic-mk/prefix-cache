"""Post-guard native reserve verification. CPU labels cannot issue authority.

The caller pins a prospective plan before launch and receives the expected guard
reference from the original guard after exit. No GPU operation occurs here.
No deadline is inferred from A_max, costs, a wall-minus-CUDA delta, or results.
"""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

from host_control_observer import (CATEGORIES, canonical_sha, checked_intervals,
                                  closed_ref, integer, json_ref, read_ref,
                                  require, reserve_candidate)

PROTOCOL_SHA = "7fbc544597c9abe334122d354e7494b590e810ab0682f1d2d155408cc4f618eb"
NATIVE_VALIDATOR_SHA = "675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"
GUARD_SHA = "3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a"
SDK_ADAPTER_SHA = "35c3d68ff2c53e0213806331d98e94ac0dba7c3c3d97b20e6e9f7fdfc2a274e8"
ISSUER_SHA = "455abbd6391616fa49be323ba13daa4cdc3eda1b3642152b6f597c6ad681dcfe"


def _load_pinned(ref, expected_sha, label):
    raw = read_ref(ref)
    require(ref["sha256"] == expected_sha, label + " exact original source pin")
    name = "_reserve_" + label + "_" + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, ref["path"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, ref["path"], "exec", dont_inherit=True), module.__dict__)
        require(read_ref(ref) == raw, label + " source drift during validation")
        return module
    finally:
        sys.modules.pop(name, None)


def _absolute(root, ref):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "exact project source reference")
    relative = ref["path"]
    require(type(relative) is str and relative and ":" not in relative and "\\" not in relative and
            not relative.startswith("/") and all(p not in ("", ".", "..") for p in relative.split("/")), "safe relative source path")
    return dict(ref, path=(root / relative).as_posix())


def _capture(capture, frontend, *, run_id, expected_ordinals, source_ref):
    require(capture.get("run_id") == run_id and capture.get("origin") == "native_gpu_recording" and
            capture.get("valid") is True and capture.get("scope") in
            ("server11_full_step_native_capture_v1", "bounded_original_full_step_stream_v1"), "actual native full-step capture")
    require(capture.get("failures") == [] and type(capture.get("pending_event_pairs")) is int and
            capture["pending_event_pairs"] == 0 and capture.get("open_event_pair") is False and
            capture.get("no_added_synchronization") is True and capture.get("cross_clock_absolute_mapping") is False,
            "original complete CUDA query-only tail")
    frames, witnesses = capture.get("frames"), capture.get("event_witnesses")
    require(type(frames) is list and type(witnesses) is list and 128 <= len(frames) <= 4096 and
            len(frames) == len(witnesses) == len(expected_ordinals), "all actual native steps and at least128 CUDA frames")
    require(capture.get("selected_offsets") == [] and capture.get("actions") == [], "reserve observation adds no native action")
    output_by_native = {}
    names = ("start_record_before_ns", "start_record_after_ns", "start_completed_query_ns",
             "end_record_before_ns", "end_record_after_ns", "end_completed_query_ns")
    for frame, witness, ordinal in zip(frames, witnesses, expected_ordinals):
        require(type(frame) is dict and type(witness) is dict and
                frame.get("native_step_ordinal") == witness.get("native_step_ordinal") == ordinal,
                "host control intervals match every actual native ordinal")
        begin = integer(frame.get("start_ns"), "native original host frame start", 1)
        end = integer(frame.get("end_ns"), "native original host frame end", 1)
        require(end > begin and frame.get("gpu_elapsed_ns") is None and frame.get("existing_io") is None and
                frame.get("new_io") is None and frame.get("intended_timing_scope") == "full_decode_step",
                "unpromoted original scalar frame")
        values = {name: integer(witness.get(name), "actual CUDA host enclosure " + name, 1) for name in names}
        require(values["start_record_before_ns"] <= values["start_record_after_ns"] <= values["start_completed_query_ns"]
                and values["start_record_after_ns"] <= begin < end <= values["end_record_before_ns"] <=
                values["end_record_after_ns"] <= values["end_completed_query_ns"], "actual original CUDA record/query causal bounds")
        require(witness.get("event_elapsed_source") == "torch.cuda.Event.elapsed_time" and
                integer(witness.get("gpu_elapsed_ns"), "actual CUDA duration", 1) <=
                values["end_completed_query_ns"] - values["start_record_before_ns"], "actual CUDA duration inside host enclosure")
        prepared = frame.get("prepared")
        require(type(prepared) is dict and prepared.get("native_step_ordinal") == ordinal and
                all(type(prepared.get(k)) is int and prepared[k] >= 0 for k in
                    ("batch", "active_decode", "prefill_tokens", "context_length")), "actual prepared metadata scalars")
        require(type(frame.get("outputs")) is list and frame["outputs"], "actual original sampled outputs")
        for item in frame["outputs"]:
            require(type(item) is list and len(item) == 2 and type(item[0]) is str and item[0] and
                    type(item[1]) is list and len(item[1]) == 1 and type(item[1][0]) is int and item[1][0] >= 0,
                    "each actual original sampled increment exactly one token")
            output_by_native.setdefault(item[0], []).extend(item[1])
    require(frontend.get("status") == "PASS_COMPLETE_ORIGINAL_REQUEST_OUTPUTS" and
            frontend.get("actual_request_stream_completed") is True and
            type(frontend.get("failure_or_unsubmitted_requests")) is int and frontend["failure_or_unsubmitted_requests"] == 0,
            "actual complete original requests")
    rows = frontend.get("rows")
    require(type(rows) is list and 1 <= len(rows) <= 32 and
            type(frontend.get("successful_requests")) is int and frontend["successful_requests"] == len(rows) ==
            frontend.get("planned_requests") and len(output_by_native) == len(rows), "all accepted and planned requests accounted")
    seen = set()
    for row in rows:
        native_id = row.get("native_request_id")
        require(type(native_id) is str and native_id not in seen and row.get("state") == "COMPLETED", "unique complete original request")
        seen.add(native_id)
        ids, times, itl = row.get("output_token_ids"), row.get("token_return_ns"), row.get("itl_ns")
        require(type(ids) is list and len(ids) == 128 and output_by_native.get(native_id) == ids,
                "complete128 actual outputs agree with every CUDA frame")
        require(type(times) is list and len(times) == 128 and all(type(t) is int and t > 0 for t in times)
                and times == sorted(times) and type(itl) is list and len(itl) == 127 and
                itl == [b-a for a,b in zip(times,times[1:])], "actual complete typed token return timeline")
    return dict(full_native_steps=len(frames), complete128_requests=len(rows), collector_source_ref=source_ref)


def match_preview_interval_shape(observation, windows, *, clock_scope):
    """Pure interval association, also tested with explicitly labelled CPU fixtures.

    It grants no native, cost, controller, or effect qualification. Sparse steps
    without an original preview keep their full host records and need no fake
    preview. Every real preview interval and every supplied attempt must match.
    """
    preview_id = "qualified_finite_controller_preview"
    require(observation["boundaries"].get(preview_id, {}).get("category") == "controller", "actual qualified controller method source")
    intervals = [(step["native_step_ordinal"], row) for step in observation["per_step"] for row in step["intervals"] if row["boundary_id"] == preview_id]
    require(type(windows) is list and len(intervals) == len(windows), "all actual preview method intervals and windows must agree; no post-selection")
    matched = set()
    for number, window in enumerate(windows, 1):
        require(type(window) is dict and window.get("attempt_id") == number and type(window["attempt_id"]) is int,
                "contiguous bounded actual preview attempts")
        ordinal = integer(window.get("native_step_ordinal"), "actual preview ordinal")
        begin, end = integer(window.get("begin_ns"), "actual preview host begin", 1), integer(window.get("end_ns"), "actual preview host end", 1)
        require(end > begin and window.get("clock_scope") == clock_scope, "actual preview same host boot clock")
        matches = [i for i,(step,row) in enumerate(intervals) if i not in matched and step == ordinal and row["start_ns"] <= begin < end <= row["end_ns"]]
        require(len(matches) == 1, "preview belongs to exactly one actual source-bound host interval")
        matched.add(matches[0])
    require(len(matched) == len(intervals), "unmatched actual preview interval")
    return dict(schema="CPU_interval_shape_check_v1", associated_attempts=len(matched),
                full_native_step_ordinals=[step["native_step_ordinal"] for step in observation["per_step"]],
                actual_native_gpu_run=False, formal_goodput_allowed=False)


def _eligible_coverage(plan, native, observation, candidate, identity, table):
    """All natural attempts remain recorded; select only preregistered eligible states.

    No attempt/window is injected. Unknown finite states retain original U.
    Every actual qualified-preview method interval must have one raw attempt.
    A completely empty eligible subset cannot issue an I controller reserve.
    """
    contract = plan.get("coverage_contract")
    require(type(contract) is dict and contract.get("schema") == "finite_natural_eligible_host_reserve_v1" and
            contract.get("selection") == "all_actual_original_ordinary_preload_preview_windows_in_exact_issued_cells" and
            contract.get("required_signatures") == [list(cell) for cell in identity.cells] and
            contract.get("unknown_domain_fallback") == "U" and contract.get("synthetic_candidates_allowed") is False,
            "prospective finite natural eligibility contract")
    coverage = native.get("host_controller_coverage")
    require(type(coverage) is dict and coverage.get("kind") == "same_qualified_I_lookup_binding_path_observed_only_no_issue"
            and coverage.get("native_step_ordinals") == candidate["native_step_ordinals"] and
            coverage.get("ordinary_IO_issued_by_observer") is False and
            coverage.get("qualified_identity_source_lock_sha256") == identity.source_lock_sha256 and
            coverage.get("qualified_cells") == [list(cell) for cell in identity.cells] and
            coverage.get("overflow") is False and coverage.get("unknown_coverage") is False,
            "actual qualified I path; missing/unknown/overflow coverage is closed")
    windows = coverage.get("preview_windows")
    require(type(windows) is list and windows and integer(coverage.get("attempt_count"), "actual attempt count", 1, 4096*32) == len(windows),
            "complete actual ordinary preview attempt list; no eligible window cannot qualify")
    association = match_preview_interval_shape(observation, windows, clock_scope=candidate["clock_scope"])
    witness_by_step = {w["native_step_ordinal"]: w for w in native["full_original_step_capture"]["event_witnesses"]}
    frame_by_step = {f["native_step_ordinal"]: f for f in native["full_original_step_capture"]["frames"]}
    max_age = integer(contract.get("condition_max_age_ns"), "original condition freshness bound", 1)
    require(coverage.get("condition_max_age_ns") == max_age, "actual original condition freshness configuration")
    eligible_ordinals, signatures = set(), set()
    Amount = sys.modules[type(table).__module__].Amount
    for number, window in enumerate(windows, 1):
        ordinal, begin, end = window["native_step_ordinal"], window["begin_ns"], window["end_ns"]
        verdict = window.get("verdict")
        require(verdict in ("eligible", "native_fallback"), "explicit actual eligibility verdict")
        if verdict == "native_fallback":
            require(type(window.get("reason")) is str and window["reason"], "native fallback reason cannot hide an attempt")
            continue
        signature = window.get("signature")
        require(type(signature) is list and tuple(signature) in identity.cells, "same exact9 qualified signature")
        before, after = window.get("state_before"), window.get("state_after")
        require(type(before) is list and len(before) == 7 and before == after and
                all(type(v) is int for v in before) and before[0] == ordinal and
                0 < before[1] <= before[2] <= begin and 0 <= begin-before[2] <= max_age and
                before[3:] == signature[4:8], "actual source-bound prepared tuple two rechecks and freshness")
        witness, prepared = witness_by_step.get(ordinal), frame_by_step.get(ordinal, {}).get("prepared")
        require(witness and prepared and witness["start_record_before_ns"] == before[1] and
                witness["start_completed_query_ns"] == before[2] and end <= witness["end_record_before_ns"] and
                [prepared[k] for k in ("batch", "active_decode", "prefill_tokens", "context_length")] == before[3:],
                "eligible condition joins original real CUDA causal/prepared scalars")
        require(window.get("start_query_complete") is True and window.get("end_record_started") is False and
                window.get("mandatory_or_max_wait_override") is False, "actual original ordinary eligible window")
        captured = integer(window.get("snapshot_captured_ns"), "actual owner StageAccounting capture", 1)
        require(0 <= begin-captured <= max_age, "actual owner I/O vector freshness")
        vector = window.get("existing_io")
        require(type(vector) is list and len(vector) == 4 and all(type(v) is dict and set(v) == {"ops", "nbytes"} for v in vector)
                and window.get("existing_io_origin") == "actual_owner_StageAccounting_snapshot", "real original physical existing-I/O vector")
        amounts = tuple(Amount(integer(v["ops"], "existing physical ops"), integer(v["nbytes"], "existing physical bytes")) for v in vector)
        physical = integer(window.get("physical_bytes"), "actual candidate physical bytes", 1)
        require(physical == signature[-1] and window.get("stage") == "ssd_read" and
                table.lookup(tuple(signature), amounts, "ssd_read", physical, execution="production") is not None,
                "real exact stage/vector/cell; unknown remains U")
        eligible_ordinals.add(ordinal)
        signatures.add(tuple(signature))
    require(association["associated_attempts"] == len(windows) and eligible_ordinals, "no complete real eligible preview subset; reserve remains blocked")
    return dict(eligible_ordinals=sorted(eligible_ordinals), covered_signatures=[list(s) for s in sorted(signatures)],
                all_actual_attempts=len(windows), per_step_control_intervals=candidate["per_step_control_intervals"],
                all_original_step_host_intervals_retained=True,
                scope="all_observed_natural_eligible_ordinary_preview_windows_of_exact_issued_cells")


def verify_development_inputs(project, *, plan_ref, expected_plan_ref, guard_ref, expected_guard_ref,
                              observation_ref, native_result_ref,
                              qualified_controller_table=None, issuer=None):
    """Issue only after independent deadline + original guard + genuine raw closure.

    Expected plan/guard refs come from the parent guarded workflow, not fields
    copied out of the candidate. The returned receipt qualifies only measured
    development reserve, never a cost cell, strategy effect, or formal goodput.
    CPU tests intentionally contain no successful native issuance fixture.
    """
    require(plan_ref == expected_plan_ref and guard_ref == expected_guard_ref,
            "independent prelaunch plan and completed original guard anchors")
    plan, guard = json_ref(plan_ref), json_ref(guard_ref)
    require(plan.get("schema") == "guarded_host_control_development_plan_v1" and plan.get("phase") == "development"
            and plan.get("first_on_has_run") is False and plan.get("evaluation_used") is False,
            "prospective development-only reserve plan")
    root = Path(project).resolve(strict=True)
    require(plan.get("project_root") == root.as_posix(), "actual project root binding")
    require(plan.get("mode") == "shadow" and plan.get("arm") == "I" and plan.get("native_phase") == "development",
            "qualified I observe-only development; off bookkeeping cannot issue its reserve")
    require(plan.get("guard_source_ref", {}).get("sha256") == GUARD_SHA, "original guard source immutable")
    read_ref(plan["guard_source_ref"])
    protocol = _load_pinned(plan["protocol_source_ref"], PROTOCOL_SHA, "sealed_protocol")
    original = _load_pinned(plan["native_validator_ref"], NATIVE_VALIDATOR_SHA, "original_native_validator")
    declaration, observation = json_ref(plan["declaration_ref"]), json_ref(observation_ref)
    candidate = reserve_candidate(observation, declaration=declaration, protocol=protocol)
    require(plan.get("deadline_declaration_sha256") == candidate["deadline_declaration_sha256"] and
            plan.get("clock_scope") == candidate["clock_scope"], "prelaunch independent clock/deadline binding")
    require(integer(plan.get("planned_monotonic_ns"), "prelaunch plan time", 1) <
            candidate["first_development_monotonic_ns"], "plan did not precede measured development")
    expected_sources = plan.get("source_refs")
    require(type(expected_sources) is list and expected_sources and
            len({r["path"] for r in expected_sources}) == len(expected_sources), "closed prelaunch source list")
    for source in expected_sources:
        read_ref(source)
    require(all(row in expected_sources for row in candidate["source_refs"]), "observer and every method were frozen before launch")
    for field in ("guard_source_ref", "protocol_source_ref", "native_validator_ref", "native_runtime_ref",
                  "sdk_adapter_ref", "collector_source_ref", "wrapper_ref"):
        require(plan.get(field) in expected_sources, "verifier/runtime/code not frozen in launch closure: " + field)
    require(closed_ref(__file__) in expected_sources, "native reserve verifier itself must be frozen before launch")
    # A CPU-created mock or a receipt's boolean cannot provide this capability.
    # The original sealed issuer registry is populated only by full raw-GPU
    # calibration + independently held-out verification. Reusing this live
    # identity is not production I activation and never starts an I/O operation.
    require(issuer is not None and qualified_controller_table is not None,
            "same I controller reserve needs independently issued real exact-cell table")
    issuer_ref = closed_ref(issuer.__file__)
    require(issuer_ref in expected_sources and issuer_ref == plan.get("controller_issuer_source_ref")
            and issuer_ref["sha256"] == ISSUER_SHA and
            Path(issuer.qualified_identity.__code__.co_filename).resolve() == Path(issuer.__file__).resolve(),
            "actual sole source-bound private issuer")
    identity = issuer.qualified_identity(qualified_controller_table)
    require(identity is not None and qualified_controller_table.production_qualified is True,
            "mock/conditional table cannot qualify controller reserve")
    require(identity.common_runtime_domain_sha256 == plan.get("common_runtime_domain_sha256") and
            identity.gpu_uuid == plan.get("gpu_uuid"), "same strong hardware/domain controller table")
    require(closed_ref(inspect.getfile(type(qualified_controller_table))) in expected_sources,
            "actual issued CostTable class source frozen in launch closure")
    lock = json_ref(plan["source_lock_ref"])
    require(type(lock.get("files")) is list and 2000 <= len(lock["files"]) <= 8192, "full inherited source/model/SDK closure")
    refs = {row["path"]: row for row in lock["files"]}
    require(len(refs) == len(lock["files"]), "duplicate immutable source rows")
    for row in expected_sources:
        relative = Path(row["path"]).relative_to(root).as_posix()
        require(refs.get(relative) == dict(row, path=relative), "observed source outside original launch lock")
    source_proof = json_ref(plan["source_proof_ref"])
    require(source_proof.get("schema") == "strong_trace_source_proof_v1" and
            source_proof.get("status") == "PASS_FULL_CPU_SOURCE_BYTES" and source_proof.get("source_count") == len(refs)
            and source_proof.get("source_lock_ref") == dict(plan["source_lock_ref"], path=Path(plan["source_lock_ref"]["path"]).relative_to(root).as_posix()),
            "actual full prelaunch CPU source verification")
    history = json_ref(plan["pre_on_history_ref"])
    require(history.get("schema") == "guarded_new_domain_pre_on_history_v1" and
            history.get("common_runtime_domain_sha256") == plan.get("common_runtime_domain_sha256") and
            history.get("first_on_has_run") is False and history.get("evaluation_has_run") is False,
            "actual guarded new-domain pre-on history")
    require(type(history.get("prior_runs")) is list, "explicit prior guarded development history")
    for prior in history["prior_runs"]:
        previous = json_ref(prior["native_result_ref"])
        prior_guard = json_ref(prior["guard_ref"])
        require(((previous.get("arm") == "U" and previous.get("mode") in ("off", "shadow")) or
                 (previous.get("arm") == "I" and previous.get("mode") == "shadow" and previous.get("phase") == "development")) and
                previous.get("common_runtime_domain_sha256") == plan["common_runtime_domain_sha256"], "on/evaluation history cannot fit reserve")
        original.validate_guard(prior_guard, gpu_uuid=previous["gpu_uuid"], job_id=previous["run_id"], wrapper_path=prior["wrapper_path"])
    original.validate_guard(guard, gpu_uuid=plan["gpu_uuid"], job_id=plan["run_id"], wrapper_path=plan["wrapper_ref"]["path"])
    native = json_ref(native_result_ref)
    expected_status = "PASS_FINITE_DEVELOPMENT_SHADOW_REQUIRES_RESERVE_GUARD_JOIN"
    require(native.get("schema") == "strong_native_original_workload_result_v1" and native.get("origin") == "actual_guarded_original_runtime"
            and native.get("status") == expected_status and native.get("run_id") == plan["run_id"] == candidate["run_id"] and
            native.get("gpu_uuid") == plan["gpu_uuid"] and native.get("guard_reservation_id") == guard["reservation_id"]
            and native.get("arm") == "I" and native.get("mode") == "shadow" and native.get("phase") == "development"
            and type(native.get("actual_gpu_runs")) is int and native["actual_gpu_runs"] == 1, "actual completed native guarded run identity")
    require(native.get("host_control_observation_ref") == observation_ref and native.get("host_clock_scope") == candidate["clock_scope"],
            "host intervals joined to actual native run and same boot/clock")
    for key in ("framework_import_attempted", "original_engine_initialization_attempted", "original_engine_shutdown_returned",
                "native_tail_drained", "immutable_runner_config_workload_unchanged", "optional_probe_restored"):
        require(native.get(key) is True, "actual original runtime lifecycle: " + key)
    for key in ("synthetic_io_or_stall_injected", "midstream_cache_reset", "selected_natural_request_prefix_injected",
                "formal_goodput_allowed", "strategy_improvement_proved"):
        require(native.get(key) is False, "development source must remain unmodified and effect-unqualified: " + key)
    for key in ("source_lock_ref", "workload_ref", "pair_config_ref"):
        require(native.get(key) == dict(plan[key], path=Path(plan[key]["path"]).relative_to(root).as_posix()), "native frozen asset identity: " + key)
    planner = native.get("original_planner_identity")
    require(type(planner) is dict and planner.get("actual_planner_present") is True and planner.get("same_manager_planner") is True
            and planner.get("load_planner") == "on" and planner.get("enable_preload") is True and planner.get("preload_share_staging") is True,
            "real strong planner/preload/shared staging retained")
    runtime = _load_pinned(plan["native_runtime_ref"], plan["native_runtime_ref"]["sha256"], "prelaunch_native_runtime")
    runtime.verify_original_tail(native["post_original_shutdown"])
    for field in ("before_workload_native_drain", "after_workload_native_drain", "final_before_shutdown_drain"):
        runtime.validate_drained_snapshot(native[field]["owner_snapshot"])
    sdk = _load_pinned(plan["sdk_adapter_ref"], SDK_ADAPTER_SHA, "original_sdk_adapter")
    _, pin, _ = sdk.load_site_assets(root, refs)  # read-only, hashes real SDK/driver; no compiler/library load
    require(native.get("sdk_environment", {}).get("current_driver_ref") == pin["driver"] and
            native["sdk_environment"].get("CPU_assets_verified") is True and
            native["sdk_environment"].get("stubs_on_runtime_library_path") is False, "actual same SDK and nonstub real driver")
    require(native.get("complete_observation_config", {}).get("source_ref") ==
            dict(plan["collector_source_ref"], path=Path(plan["collector_source_ref"]["path"]).relative_to(root).as_posix()),
            "actual frozen full CUDA collector source")
    capture_proof = _capture(native["full_original_step_capture"], native["frontend"], run_id=plan["run_id"],
                             expected_ordinals=candidate["native_step_ordinals"], source_ref=plan["collector_source_ref"])
    eligible = _eligible_coverage(plan, native, observation, candidate, identity, qualified_controller_table)
    integer(plan.get("reserve_uncertainty_ns"), "prospective explicit reserve uncertainty", 0)
    source = dict(schema="actual_development_reserve_source_v1", actual_native_gpu_run=True, synthetic_fixture=False,
        phase="development", run_id=plan["run_id"], clock_scope=candidate["clock_scope"],
        control_intervals_sha256=canonical_sha(eligible["per_step_control_intervals"]),
        observation_ref=observation_ref, native_result_ref=native_result_ref, completed_original_guard_ref=guard_ref,
        independently_expected_plan_ref=expected_plan_ref, source_refs=expected_sources,
        native_qualification_verifier_source_ref=closed_ref(__file__), capture_proof=capture_proof,
        reserve_scope=eligible["scope"], covered_cell_signatures=eligible["covered_signatures"],
        all_actual_attempts=eligible["all_actual_attempts"], eligible_native_step_ordinals=eligible["eligible_ordinals"],
        all_original_step_host_intervals_retained=True,
        real_SDK_and_driver_revalidated=True, production_cost_qualified=False, formal_goodput_allowed=False,
        strategy_effect_qualified=False, no_GPU_delta_subtracted=True)
    receipt = dict(schema="actual_development_control_reserve_v1", actual_native_gpu_run=True, synthetic_fixture=False,
        deadline_declaration_sha256=candidate["deadline_declaration_sha256"], clock_scope=candidate["clock_scope"],
        first_development_monotonic_ns=candidate["first_development_monotonic_ns"], phase="development", first_on_has_run=False,
        clock="monotonic_ns_host_control_intervals", all_control_categories=list(CATEGORIES),
        per_step_control_intervals=eligible["per_step_control_intervals"], reserve_uncertainty_ns=plan["reserve_uncertainty_ns"])
    observed = max(protocol.union_ns(step) for step in receipt["per_step_control_intervals"])
    require(observed + receipt["reserve_uncertainty_ns"] < integer(declaration.get("full_control_window_deadline_ns"), "independent deadline", 1),
            "ordinary independent budget is not positive")
    return dict(source=source, receipt=receipt, declaration=declaration, protocol=protocol,
                independently_verified_source_refs=expected_sources)


def _verification(source_ref, receipt_ref, *, budget, source_refs):
    source = json_ref(source_ref)
    return dict(schema="actual_development_native_reserve_verification_v1", status="PASS_ACTUAL_DEVELOPMENT_RESERVE_ONLY",
        reserve_receipt_ref=receipt_ref, reserve_source_ref=source_ref,
        verifier_source_ref=closed_ref(__file__), independent_source_refs=source_refs,
        actual_native_gpu_run=True, synthetic_fixture=False, independent_deadline_prospective=True,
        actual_guard_native_SDK_CUDA_outputs_and_all4host_categories_verified=True,
        covered_cell_signatures=source["covered_cell_signatures"], reserve_scope=source["reserve_scope"],
        formal_goodput_allowed=False, cost_qualified=False, I_activation_allowed=False, frozen_budget=budget)


def join_completed_development(project, *, output_directory, **evidence):
    """Append-only post-guard issuer. Effect gates must use the read API below."""
    verified = verify_development_inputs(project, **evidence)
    source, receipt = verified["source"], verified["receipt"]
    declaration, protocol = verified["declaration"], verified["protocol"]
    root, out = Path(project).resolve(strict=True), Path(output_directory).absolute()
    require(out.is_dir() and not out.is_symlink() and root in out.resolve().parents, "existing new project output directory")
    source_path, receipt_path, proof_path = (out / name for name in
        ("actual-development-reserve-source.json", "actual-development-control-reserve.json", "actual-development-native-reserve-verification.json"))
    require(not any(p.exists() for p in (source_path, receipt_path, proof_path)), "append-only native reserve issuance")
    with source_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(source, indent=2, sort_keys=True, allow_nan=False) + "\n")
    receipt["native_result_ref"] = closed_ref(source_path)
    budget = protocol.freeze_development_budget(declaration, receipt)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    proof = _verification(closed_ref(source_path), closed_ref(receipt_path), budget=budget,
                          source_refs=verified["independently_verified_source_refs"])
    with proof_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(proof, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return proof


def verify_existing_native_reserve(project, *, reserve_source_ref, reserve_receipt_ref,
                                  native_verification_ref, **evidence):
    """Strict read-only raw replay. No temp file, issuance, or new authority."""
    verified = verify_development_inputs(project, **evidence)
    source, receipt = verified["source"], verified["receipt"]
    require(json_ref(reserve_source_ref) == source, "frozen actual reserve source differs from native raw replay")
    receipt["native_result_ref"] = reserve_source_ref
    require(json_ref(reserve_receipt_ref) == receipt, "frozen actual reserve receipt differs from actual host intervals")
    budget = verified["protocol"].freeze_development_budget(verified["declaration"], receipt)
    expected = _verification(reserve_source_ref, reserve_receipt_ref, budget=budget,
                             source_refs=verified["independently_verified_source_refs"])
    require(json_ref(native_verification_ref) == expected, "frozen actual native reserve qualification proof differs from raw replay")
    return dict(schema="existing_actual_native_development_reserve_verified_v1", status="PASS_READ_ONLY_NATIVE_RESERVE_REPLAY",
        actual_native_gpu_run=True, synthetic_fixture=False, receipt_ref=reserve_receipt_ref,
        source_ref=reserve_source_ref, proof_ref=native_verification_ref,
        frozen_budget=budget, formal_goodput_allowed=False, cost_qualified=False, I_activation_allowed=False,
        covered_cell_signatures=source["covered_cell_signatures"], reserve_scope=source["reserve_scope"],
        native_qualification_verifier_required_separately=False, new_qualified_receipts_written=0)
