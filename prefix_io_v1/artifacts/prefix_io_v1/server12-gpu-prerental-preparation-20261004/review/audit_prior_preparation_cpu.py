"""Bounded read-only audit of retained preparation; no engine/GPU/network imports."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def read(root, relative, refs):
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("missing, nonregular or oversized input: " + relative)
    data = path.read_bytes()
    refs.append({"path": relative, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return data.decode("utf-8")


def audit(root):
    old = "artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/"
    original = "artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/"
    refs, checks = [], []
    def check(name, condition):
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise ValueError("retained preparation audit rejected: " + name)

    execution = read(root, "artifacts/prefix_io_v1_server12_candidates/migration_audit/04_CODEX_EXECUTION.md", refs)
    check("original_LoadPlanner_remains_unique_admission_contract", "保留原LoadPlanner唯一准入入口" in execution)
    builder = read(root, original + "vllm.py", refs)
    check("original_builder_SHA_matches_retained_CPU_replay_pin", refs[-1]["sha256"] == "901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1")
    tree = ast.parse(builder)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PyKvCacheOffloadingSpec")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_build_planner")
    fn_text = ast.get_source_segment(builder, fn)
    check("original_recompute_overpriced_warning_is_retained", "recompute is overpriced" in fn_text)
    check("original_builder_enforces_v2_curves_and_preload_shared_staging", all(word in fn_text for word in ("curves is None", "enable_preload", "preload_share_staging", "PLAN_API_AVAILABLE")))

    cost = read(root, original + "cost_model.py", refs)
    check("original_cost_model_SHA_matches_retained_CPU_replay_pin", refs[-1]["sha256"] == "e85922054fe96dce5c0221f1cc17cc3f1b2d7addd7eae452cabad305fc412397")
    check("author_cost_domain_uses_recompute_SSD_and_memory_curves", all(word in cost for word in ("f_s", "g_ssd_s", "g_mem_s", "cost model extrapolates")))
    stage = json.loads(read(root, old + "FINAL_CPU_STAGE_DECISION.json", refs))
    check("retained_CPU_test_total_has_no_GPU_effect_promotion", stage["distinct_final_tests_passed"] == 209 and stage["performance_benefit_proved"] is False and stage["actual_GPU_runs_this_turn"] == 0)
    check("planner_off_receipt_explicitly_does_not_cover_strong_domain", stage["diagnostic_receipt_covers_strong_planner_on_effect"] is False and stage["additional_strong_planner_on_calibration_required"] is True)
    check("old_budget_has_not_been_widened", stage["original_cumulative_GPU_remaining_seconds"] == 4121.104761094321)
    protocol = json.loads(read(root, old + "protocol/I_PILOT_PROTOCOL.json", refs))
    check("old_null_deadline_cannot_authorize_on", protocol["development"]["declared_full_control_window_deadline_ns"] is None and protocol["development"]["internal_step_budget_ns"] is None)
    check("old_singlefile_protocol_does_not_claim_natural_service_effect", protocol["evaluation"]["natural_service_stream_effect_claim_allowed"] is False and protocol["calibration"]["normal_workload_generalization_claim"] is False)
    check("all_old_max_reservations_sum_to_frozen_plan", sum(row["reserved_seconds"] for row in protocol["jobs"]) == 3780)
    prepare = read(root, old + "source_inputs/prepare_p3_pilot.py", refs)
    check("retained_P3_generated_tokens_are_explicitly_synthetic", "controlled synthetic-token development replay" in prepare)
    author = read(root, old + "source_inputs/shared_storage_trace_replay.py", refs)
    check("original_author_replay_has_optional_injection_and_wipe_that_must_stay_disabled", "maybe_inject_repeated_prefix" in author and "wipe_shared_storage" in author)
    strong = read(root, old + "strong_baseline/strong_baseline_config.py", refs)
    check("strong_configuration_rejects_planner_off_and_keeps_prefix_cache", 'extra.get("load_planner") == "on"' in strong and '"enable_prefix_caching"' in strong)

    # Re-read actual source bytes after all parsing, without engine execution.
    unchanged = all((root / ref["path"]).stat().st_size == ref["bytes"] and hashlib.sha256((root / ref["path"]).read_bytes()).hexdigest() == ref["sha256"] for ref in refs)
    check("all_retained_inputs_unchanged_after_read", unchanged)
    return {"schema": "gpu_prerental_independent_prior_review_v1", "status": "PASS_RETAINED_BOUNDARIES_CURRENT_EFFECT_GATES_STILL_REQUIRED", "checks": checks, "source_refs": refs,
            "GPU_operations": 0, "network_operations": 0, "engine_imports": 0, "current_GPU_available_claim": False,
            "original_remaining_snapshot_seconds": 4121.104761094321,
            "old_plan_max_reserved_seconds": 3780, "old_mechanism_calibration_reservation_seconds": 1220,
            "old_plan_includes_strong_domain_qualification": False, "new_runner_protocol_review_completed": False,
            "strong_U_I_effect_ready_claim": False, "method_effectiveness_claim": False,
            "risk_actions": ["replace the old mechanism-only first-job plan with an explicit new strong-domain investment plan; do not mutate old records", "freeze trace provenance and an independent deadline before development; CPU fixtures cannot supply measured reserve", "bound exact cost cells and unknown fallback before spending GPU time", "retain author LoadPlanner warning and require actual domain coverage; do not disable features or change admission to hide it", "private JIT CPU build is not a loaded CUDA/native execution capability proof"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.project_root.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "checks_passed": len(result["checks"]), "source_inputs": len(result["source_refs"]), "GPU_operations": 0}))


if __name__ == "__main__":
    main()
