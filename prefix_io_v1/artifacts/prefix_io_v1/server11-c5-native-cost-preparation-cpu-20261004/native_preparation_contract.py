"""CPU draft and real raw reconstruction audit, never an issuance interface."""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
DELIVERY = "artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004"
CANDIDATE = DELIVERY + "/common_candidate"
LABEL = "server11-c5-native-common-cost-preparation01"
BLOCKED = "GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY"
C5_REACTOR_SHA256 = "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47"
C5_COLLECTOR_SHA256 = "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"
ORIGINAL_ESTIMATOR_SHA256 = "3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac"
ORIGINAL_NATIVE_VERIFIER_SHA256 = "e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291"


def require(condition, message):
    if not condition: raise ValueError(message)


def blocked_document():
    return dict(status=BLOCKED, actual_gpu_runs=0, gpu_launch_allowed=False,
        native_execution_verified=False, native_cost_qualified=False, valid_native_receipt=None,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        effective_cost_upper_ns=None, effective_step_budget_ns=None)


def block_gpu_start():
    raise RuntimeError(BLOCKED)


def load_serializer():
    name = "_c5_native_preparation_real_serializer"
    path = HERE / "prepare_and_verify_native_cost.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def draft_common_calibration():
    """An inert source draft; deliberately not a native conditional plan."""
    return dict(scope="c5_native_common_source_draft_v1", status="CPU_DRAFT_GPU_UUID_UNRESOLVED",
        gpu_uuid=None, job_namespace=LABEL, actual_gpu_runs=0, gpu_launch_allowed=False,
        fixed_order=["A0", "B0", "B1", "A1", "A2", "B2"], calibration_pairs=2, validation_pairs=1,
        max_accepted_parents=8, bridge_is_none=True, common_candidate=CANDIDATE,
        required_output_tokens_each=128, prompt_tokens=129, cached_prompt_tokens=128,
        measured_offset=16, warmup_offsets=[1], action_operations=1, action_physical_bytes=917504,
        six_fresh_original_processes=True, native_execution_verified=False, native_cost_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        valid_native_receipt=None, effective_cost_upper_ns=None, effective_step_budget_ns=None,
        unresolved=["physical_gpu_uuid", "same_uuid_permission", "new_authorized_source_lock",
            "new_guarded_native_raw_receipts", "unthrottled_complete_entry_CPU_qualification"],
        native_plan_created=False, gpu_scope_created=False, gpu_job_created=False)


def assert_original_algorithms(original_verifier_path):
    """All numerical/capture/causality functions are the original exact AST."""
    path = Path(original_verifier_path)
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == ORIGINAL_NATIVE_VERIFIER_SHA256, "original verifier source drift")
    parent = {node.name: node for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef)}
    current = {node.name: node for node in ast.parse((HERE / "native_conditional_cost.py").read_bytes()).body
        if isinstance(node, ast.FunctionDef)}
    names = ("original_estimator", "validate_capture", "validate_io", "original_post_shutdown_drain", "analyze_paired")
    for name in names:
        require(ast.dump(parent[name], include_attributes=False) == ast.dump(current[name], include_attributes=False),
            "original algorithm AST drift: " + name)
    return dict(status="PASS_ORIGINAL_ALGORITHM_AST", functions=list(names), numerical_ast_changed=False,
        original_source_sha256=ORIGINAL_NATIVE_VERIFIER_SHA256, native_execution_verified=False,
        native_cost_qualified=False, gpu_launch_allowed=False)


def audit_synthetic_raw(root, *, envelope, original_estimator_path):
    """Read six actual files and reconstruct every raw field with new serializer.

    'actual' here refers to reading the synthetic files, not actual GPU/native
    execution. Inner legacy raw origin vocabulary is required by the unchanged
    shape checker; the enclosing origin is always explicitly synthetic.
    """
    require(type(envelope) is dict and set(envelope) == {"origin", "runtime_relative", "plan_ref",
        "source_verification_refs"}, "exact synthetic raw envelope")
    require(envelope["origin"] == "synthetic_cpu_contract", "explicit synthetic CPU raw input required")
    serializer = load_serializer()
    plan = serializer.C.EvidenceRef.from_mapping(envelope["plan_ref"]).json(root)
    expected = dict(stage="ssd_read", units=1, operations=1, transfer_quantum_bytes=917504,
        cached_prompt_tokens=128, prompt_tokens=129, output_tokens=128, measured_offset=16)
    require(all(type(plan.get(name)) is type(value) and plan[name] == value for name, value in expected.items()) and
        plan.get("warmup_offsets") == [1] and all(type(offset) is int for offset in plan["warmup_offsets"]),
        "unchanged C5 finite formula workload")
    record = serializer.serialize_runtime_record(root, runtime_relative=envelope["runtime_relative"],
        plan_ref=envelope["plan_ref"], source_verification_refs=envelope["source_verification_refs"], synthetic_cpu=True)
    require(record["origin"] == "synthetic_cpu_contract" and record["native_execution_verified"] is False,
        "synthetic raw record cannot be native evidence")
    original = Path(original_estimator_path)
    require(original.is_file() and not original.is_symlink() and
        sha256(original.read_bytes()).hexdigest() == ORIGINAL_ESTIMATOR_SHA256, "original math source drift")
    result = serializer.C.analyze_paired(plan, record["windows"], original_source_path=original)
    require(result["native_execution_verified"] is False and result["conditional_cost_cell_qualified"] is False and
        result["production_qualified"] is False and result["holdout_used_to_refit"] is False,
        "CPU raw path cannot issue native qualification")
    return dict(status="PASS_C5_SYNTHETIC_SIX_RAW_RECONSTRUCTION_AND_ORIGINAL_FORMULA",
        origin="synthetic_cpu_contract", raw_runtime_ref=record["raw_runtime_ref"], plan_ref=envelope["plan_ref"],
        raw_child_refs=[row["raw_child_ref"] for row in record["windows"]],
        synthetic_process_receipt_count=record["actual_subprocess_count"], actual_native_processes_started=0,
        full_output_tokens=result["full_output_tokens"], full_frame_count_per_window=128,
        original_estimator_proof=result["original_estimator_proof"], synthetic_holdout_covered=result["heldout_covered"],
        holdout_used_to_refit=False, actual_gpu_runs=0, native_execution_verified=False,
        conditional_cost_cell_qualified=False, native_cost_qualified=False, gpu_launch_allowed=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        effective_cost_upper_ns=None, effective_step_budget_ns=None, valid_native_receipt=None,
        production_qualified=False, performance_claim=False)


def main(argv=None):
    print(json.dumps(blocked_document(), sort_keys=True)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
