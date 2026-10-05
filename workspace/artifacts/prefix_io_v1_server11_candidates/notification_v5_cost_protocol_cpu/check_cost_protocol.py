"""Pure CPU protocol consistency checks. Cannot issue costs/receipts or GPU jobs."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

SCHEMA = "c5_common_cost_then_notification_qualification_cpu_protocol_v1"
FORMULA = "ceil_calibration_means_plus_max_positive_residual_v1"
CONDITION = dict(prompt_tokens=129, output_tokens=128, selected_offset=16,
    physical_bytes=917504, operations=1, max_accepted_parents=8,
    sample_max_age_ns=100000000, max_wait_ns=100000000,
    batch=1, active_decode=1, prefill_tokens=0, context_length=144,
    stage="ssd_read", existing_io=[[0, 0]] * 4,
    common_prompt_first=[18100, 19100, 20100], common_seeds=[1829, 1830, 1831],
    runtime_prompt_first=28100, runtime_seed=2829)
WINDOWS = [dict(pair=i, arm=arm, split="calibration" if i < 2 else "holdout")
           for i, arms in enumerate(("AB", "BA", "AB")) for arm in arms]
DEPENDENCIES = {
    "cpu_environment_qualification": [],
    "final_c5_sources_and_strict_factory": [],
    "new_common_calibration_permission": [],
    "six_common_native_windows": ["cpu_environment_qualification", "final_c5_sources_and_strict_factory", "new_common_calibration_permission"],
    "strict_common_cost_receipt": ["six_common_native_windows"],
    "new_runtime_permission": [],
    "off_native": ["strict_common_cost_receipt", "new_runtime_permission"],
    "shadow_native": ["off_native"],
    "on_notification_native": ["shadow_native"],
    "bounded_qualification_report": ["on_notification_native"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(actual, expected, description):
    # JSON encoding keeps bool/int distinct and requires the full exact schema.
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), description)


def topological_order(dependencies):
    remaining = {k: set(v) for k, v in dependencies.items()}
    require(all(x in remaining for values in remaining.values() for x in values), "unknown dependency")
    result = []
    while remaining:
        ready = sorted(k for k, values in remaining.items() if not values)
        require(bool(ready), "receipt/qualification dependency cycle")
        for name in ready:
            result.append(name); del remaining[name]
        for values in remaining.values(): values.difference_update(ready)
    return result


def check_protocol(document):
    require(type(document) is dict and set(document) == {
        "schema", "execution", "condition", "common", "runtime", "dependencies", "budget_reference", "current_state"},
        "exact CPU protocol fields")
    require(document["schema"] == SCHEMA, "protocol schema")
    exact(document["execution"], dict(kind="cpu_protocol_only", gpu_launch_allowed=False,
        native_execution_verified=False, create_jobs=False, create_scope=False, allocate_budget=False,
        receipt_issued=False, performance_claim=False), "protocol cannot create native qualification or GPU work")
    exact(document["condition"], CONDITION, "original fixed condition/thresholds/workload")
    common = document["common"]
    exact(common, dict(windows=WINDOWS, fresh_process_per_window=True, bridge=None,
        notification_installed=False, queue_observation_installed=False,
        capture_run_namespace="frontend_RID", frontend_request_namespace="frontend_RID",
        native_request_id="original_native_request_id",
        formula=FORMULA, calibration_pairs=2, holdout_pairs=1, holdout_refit=False,
        budget_formula="ceil_mean_calibration_A_plus_max_positive_A_residual",
        budget_uses=["calibration_A_only"], cost_upper_ns=None, step_budget_ns=None,
        qualification_rule="zero_observed_holdout_underprediction_no_refit_v1",
        receipt_scope="finite_common_io_condition_only",
        measured_paths=["common_original_io", "original_full_step_capture", "C5_uninstalled_wait_branch"],
        unmeasured_paths=["active_policy_preview", "collector_notification_attachment", "armed_queue_get", "notification_queue_observer"],
        raw_required=["new_parent_and_six_fresh_children", "actual_C5_loaded_modules_and_run_code", "same_actual_collector_128_frames",
            "original_cuda_event_and_model_runner", "physical_original_AIO_journal", "source_before_after", "original_budget_guard",
            "full_output_equality", "holdout_no_refit"]), "common six-window scope/formula cannot claim active notification cost")
    runtime = document["runtime"]
    exact(runtime, dict(order=["off", "shadow", "on"],
        arms=[dict(mode="off", bridge=None, notification=False, queue_observer=False),
              dict(mode="shadow", bridge="interference_shadow", notification=False, queue_observer=False),
              dict(mode="on", bridge="interference_active", notification=True, queue_observer=True)],
        capture_run_namespace="owner_LABEL", frontend_request_namespace="frontend_RID",
        native_request_id="original_native_request_id", max_captures_per_fresh_owner=1,
        receipt_dependency="strict_new_common_cost_receipt",
        source_bindings=["same_final_C5_reactor", "same_actual_collector_overlay_ref_equals_calibration", "same_model_runner_Event_geometry_kernel",
                         "final_runtime_adapter_launcher_verifier_refs"],
        controls_migration_gate="selected_gpu_ns_le_common_upper",
        on_migration_gate="record_only_not_cost_coverage",
        issue_relation="common_upper_le_A_only_budget",
        issue_outcome="NOT_EXERCISED_NO_FORCED_DEFER",
        defer_relation="common_upper_gt_A_only_budget_and_all_original_live_checks",
        invalidation="matching_original_queue_wake_or_original_earliest_deadline",
        latency_fields=["selected_gpu_ns", "all_128_gpu_ns", "whole_request_host_ns", "drain_host_ns", "original_queue_witness"],
        success_requirements=["full_128_output_equality", "same_128_original_native_request_frames", "real_install_identity",
            "real_original_queue_witness", "registration_clear_after_original_drain", "no_notification_fault_or_overflow", "original_shutdown_and_session_drain"],
        on_timing_interpretation="combined_observed_runtime_not_isolated_instrumentation_cost",
        upper_refit_after_runtime=False, add_cpu_cost_to_upper=False, change_budget_to_force_defer=False,
        performance_claim=False, instrumentation_cost_qualified=False,
        future_runner_implemented=False, future_gpu_authorized=False), "separate runtime notification lifecycle/cost limits")
    topological_order(document["dependencies"])
    exact(document["dependencies"], DEPENDENCIES, "finite acyclic qualification dependencies")
    exact(document["budget_reference"], dict(cumulative_cap_seconds=28800,
        historical_common_limit_seconds=1200, historical_common_reserved_seconds=1220,
        historical_runtime_limit_seconds=300, historical_runtime_reserved_seconds=320,
        remaining_seconds=None, new_allocation=None, new_authorization=None,
        meaning="historical_ceiling_reference_not_new_permission"), "no new budget, permission, or invented remaining time")
    exact(document["current_state"], dict(cpu_environment="NOT_QUALIFIED", new_native_receipt="UNAVAILABLE",
        common_calibration="NOT_RUN", runtime="NOT_RUN", overall="GPU_BLOCKED",
        gpu_jobs_created=0, gpu_runs_performed=0), "actual current state must remain blocked")
    return dict(protocol_valid=True, dependency_order=topological_order(DEPENDENCIES),
        native_execution_verified=False, receipt_issued=False, gpu_launch_allowed=False,
        cost_values_available=False, performance_claim=False)


def simulated_next_obligation(state):
    """Test-only planning enum; no numeric costs, permission, or real qualification."""
    require(set(state) == {"origin", "completed", "upper_relation", "notification_exercised"} and
            state["origin"] == "synthetic_protocol_test", "only explicit symbolic CPU simulation")
    completed = state["completed"]
    require(type(completed) is list and len(set(completed)) == len(completed) and
            all(x in DEPENDENCIES for x in completed), "bounded known stages")
    for stage in completed:
        require(set(DEPENDENCIES[stage]).issubset(completed), "stage without prior evidence")
    relation = state["upper_relation"]
    require(relation in ("unavailable", "le", "gt"), "symbolic policy relation only")
    require(type(state["notification_exercised"]) is bool, "exact exercised boolean")
    if relation != "unavailable":
        require("strict_common_cost_receipt" in completed, "no cost relation before strict common receipt")
    if state["notification_exercised"]:
        require("on_notification_native" in completed and relation == "gt", "notification cannot be invented from common-only/issue path")
    if relation == "le":
        require("on_notification_native" not in completed, "do not run or fake on notification qualification after issue-only result")
        status = "NOT_EXERCISED_NO_FORCED_DEFER"
    elif "on_notification_native" in completed and not state["notification_exercised"]:
        status = "ON_NOTIFICATION_NOT_DEMONSTRATED"
    else:
        available = [x for x in topological_order(DEPENDENCIES) if x not in completed and set(DEPENDENCIES[x]).issubset(completed)]
        status = "NEXT_OBLIGATION:" + (available[0] if available else "bounded_qualification_report")
    return dict(symbolic_status=status, origin="synthetic_protocol_test", gpu_launch_allowed=False,
        native_execution_verified=False, receipt_issued=False, performance_claim=False)


def audit_sources(pins, roots):
    sources = {}
    for row in pins["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "relative source pin")
        raw = (roots[row["scope"]] / relative).read_bytes()
        require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "source hash: " + row["name"])
        sources[row["name"]] = ast.parse(raw.decode("utf-8-sig"))
    common = sources["common_runner"]
    bridge_args = [kw.value for node in ast.walk(common) if isinstance(node, ast.Call)
                   for kw in node.keywords if kw.arg == "p4_bridge"]
    require(len(bridge_args) == 1 and isinstance(bridge_args[0], ast.Constant) and bridge_args[0].value is None,
            "actual original six-window coordinator bridge is None")
    require(not any(isinstance(node, ast.Attribute) and node.attr in
                    ("attach_single_file_wait", "prepare_notification_capture") for node in ast.walk(common)),
            "common six-window source does not activate notification")
    adapter = sources["prepared_adapter"]
    require(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                node.func.attr == "attach_single_file_wait" for node in ast.walk(adapter)), "explicit prepared active attachment exists")
    return dict(source_pins_verified=len(sources), common_bridge_none_proved=True,
        common_does_not_execute_notification_proved=True, prepared_active_attachment_exists=True,
        gpu_launch_allowed=False, native_execution_verified=False, timing_measured=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("PROTOCOL.json"))
    args = parser.parse_args()
    print(json.dumps(check_protocol(json.loads(args.protocol.read_text(encoding="utf-8"))), sort_keys=True))


if __name__ == "__main__":
    main()
