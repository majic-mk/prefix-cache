"""Append-only seal of local CPU review evidence; runs no tests or GPU code."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MANIFEST = HERE / "INDEPENDENT_REVIEW_SOURCE_FREEZE.json"
FINAL = HERE / "INDEPENDENT_REVIEW_FINAL.json"
TESTS = (
    ("PROTOCOL_INDEPENDENT_REJECTIONS_FINAL.json", 7),
    ("ORIGINAL_REQUEST_ID_CPU_REPLAY.json", 6),
    ("FINITE_CAPABILITY_INDEPENDENT_REFUSALS_FINAL.json", 9),
    ("OBSERVER_CAPACITY_INDEPENDENT_FINAL.json", 3),
    ("RAW_REQUEST_ID_MAPPING_INDEPENDENT_FINAL.json", 3),
    ("HOST_CONTROL_REFUSALS_INDEPENDENT_FINAL.json", 6),
    ("HOST_BOUNDARY_FALLBACK_INDEPENDENT_FINAL.json", 2),
)
SUBJECTS = (
    "activation/control_observation/host_control_observer.py",
    "activation/control_observation/reserve_join.py",
    "activation/control_observation/CONTROL_OBSERVATION_SOURCE_FREEZE.json",
    "activation/source/prefix_io_control/gpu_cell_issuer.py",
    "activation/source/prefix_io_control/p4_cost_table.py",
    "activation/native_conditional_cost.py",
    "protocol/prerental_protocol.py",
    "runner/strong_native_cost_runner.py",
    "runner/bounded_native_full_step_collector.py",
    "runner/host_boundary_binding.py",
    "runner/finite_current_binding.py",
    "runner/finite_startup.py",
    "runner/activation_request.py",
    "runner/native_runtime.py",
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def ref(path, relative):
    require(path.is_file() and not path.is_symlink(), "regular reviewed file: " + relative)
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def append(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def main():
    require(not MANIFEST.exists() and not FINAL.exists(), "seal files are append-only")
    records = []
    for name, expected in TESTS:
        proof = json.loads((HERE / name).read_bytes())
        require(str(proof.get("status", "")).startswith("PASS") and proof.get("tests_run") == expected,
                "actual completed local test result: " + name)
        require(proof.get("failures") in (0, []) and proof.get("errors") in (0, []), "test failure: " + name)
        require(proof.get("skipped", proof.get("skips")) == 0, "skipped independent test: " + name)
        records.append(dict(result_ref=ref(HERE / name, name), tests=expected, execution_environment="local_CPU_only"))
    audit_name = "PRIOR_PREPARATION_REVIEW_CPU_RESULT_V2.json"
    audit = json.loads((HERE / audit_name).read_bytes())
    require(str(audit.get("status", "")).startswith("PASS") and len(audit["checks"]) == 16 and
            all(row["passed"] is True for row in audit["checks"]), "actual completed16 local audit checks")
    subjects = [ref(BASE / name, name) for name in SUBJECTS]
    files = [ref(path, path.name) for path in sorted(HERE.iterdir())
             if path.is_file() and path.suffix in (".py", ".json", ".md") and path not in (MANIFEST, FINAL)]
    manifest = dict(schema="independent_local_CPU_review_source_freeze_v1", files=files,
        source_count=len(files), reviewed_subject_source_refs=subjects, subject_paths_relative_to="gpu_prerental_preparation_20261004",
        local_unit_tests_passed=sum(row["tests"] for row in records), local_audit_checks_passed=16,
        repeated_reverification_added_to_test_count=False, server_test_success_claimed=False,
        original_inputs_modified_by_review=False, failed_and_superseded_evidence_preserved=True,
        actual_GPU_runs=0, actual_RPC_calls=0, genuine_GPU_capabilities_issued=0,
        genuine_native_reserve_receipts_issued=0, full_SDK_runtime_PASS_claimed=False,
        future_GPU_positive_acceptance_tested=False, method_improvement_proved=False)
    append(MANIFEST, manifest)
    require(all(ref(HERE / row["path"], row["path"]) == row for row in files), "review source drift during seal")
    require(all(ref(BASE / row["path"], row["path"]) == row for row in subjects), "review subject drift during seal")
    final = dict(schema="independent_pre_rental_local_CPU_review_final_v1", status="PASS_CPU_INTERFACES_REFUSALS_AND_READ_ONLY_REVIEW",
        manifest_ref=ref(MANIFEST, MANIFEST.name), report_ref=ref(HERE / "INDEPENDENT_PRE_RENT_REVIEW.md", "INDEPENDENT_PRE_RENT_REVIEW.md"),
        unit_test_records=records, audit_record=dict(result_ref=ref(HERE / audit_name, audit_name), checks=16),
        unique_local_unit_tests=36, unique_local_audit_checks=16, repeated_tests_double_counted=False,
        actual_GPU_runs=0, actual_RPC_calls=0, genuine_GPU_capabilities_issued=0, genuine_native_reserve_receipts_issued=0,
        future_GPU_positive_acceptance_tested=False, real_method_effect_tested=False, formal_goodput_allowed=False,
        original_inputs_modified_by_review=False, GPU_stage="UNTESTED_IN_THIS_CPU_PREPARATION",
        next_allowed_GPU_phase="one original-guard strong U/off qualification only after live device/driver/UUID/source/budget gates",
        first_job_maximum_reserved_seconds=320, remaining_original_GPU_budget_snapshot_seconds=4121.104761094321,
        live_original_ledger_recheck_required=True, complete_effect_or_paper_result_promised=False,
        parent_remaining_work=["server_CPU_verification", "full_source_asset_freeze", "CPU_template_preflight", "original_input_AFTER_protection"],
        limitations=["controlled_P3_input_is_not_natural_production_holdout", "zero_scheduled_or_empty_prefill_frames_remain_UNKNOWN",
                     "real_exact_cell_cost_and_controller_reserve_require_GPU", "no_claim_current_full_SDK_runtime_PASS"])
    append(FINAL, final)
    print(json.dumps(dict(status=final["status"], unique_local_tests=36, local_audit_checks=16,
                          review_files=len(files), reviewed_subject_sources=len(subjects), GPU_runs=0, RPC_calls=0,
                          manifest_ref=final["manifest_ref"], final_ref=ref(FINAL, FINAL.name))))


if __name__ == "__main__":
    main()
