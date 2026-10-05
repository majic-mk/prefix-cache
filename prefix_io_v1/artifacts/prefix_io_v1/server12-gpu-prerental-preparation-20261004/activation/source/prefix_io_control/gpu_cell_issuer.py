"""Finite exact-cell GPU evidence issuer, with no device or execution operations.

An origin string, CPU semantic candidate, table flag, or old single-file receipt
is insufficient. Independent parent-frozen references and every raw native
capture are replayed before a private table capability can be minted.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
from .dispatch_budget import Amount, ZERO


VALIDATION_SHA256 = "675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"
NATIVE_SHA256 = "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47"
COLLECTOR_SHA256 = "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"
STRONG_PAIR_VALIDATOR_SHA256 = "d2d1d8e35c582df3a6f4857f91b21c224d524ed2cb0ca9a35948f19195396478"
ORIGINAL_GUARD_SHA256 = "3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a"
_ISSUED = {}
PAIR_WRAPPER_KEYS = frozenset(("schema", "configurations", "contract", "gpu_uuid", "GPU_operations",
    "native_execution_verified", "gpu_effect_qualified", "formal_goodput_allowed", "independent_internal_step_budget_ns",
    "development_deadline_ns", "SLO", "cost_receipt", "source_inputs", "original_planner_curve_ref",
    "original_planner_curve_status", "original_recompute_overpriced_warning_preserved", "qualification_workload_kind",
    "natural_trace_bound", "future_effect_activation"))


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, message, minimum=0):
    require(type(value) is int and value >= minimum, message)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return sha256(canonical(value)).hexdigest()


def sha(value, name):
    require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value), name)
    return value


def read_ref(project, ref):
    require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "strict actual byte reference")
    sha(ref["sha256"], "actual SHA-256 hex")
    integer(ref["bytes"], "actual bounded source bytes", 1)
    require(ref["bytes"] <= 32 * 1024**2, "bounded actual evidence file")
    text = ref["path"]
    require(type(text) is str and text and "\\" not in text and ":" not in text and "\0" not in text, "POSIX relative evidence path")
    pure = PurePosixPath(text)
    require(not pure.is_absolute() and all(part not in ("", ".", "..") for part in text.split("/")), "safe evidence reference")
    root = Path(project).resolve(strict=True)
    path = root
    for part in pure.parts:
        path /= part
        require(not path.is_symlink(), "symlink source rejected")
    require(path.is_file() and root in path.resolve().parents and path.stat().st_size == ref["bytes"], "existing actual evidence file size")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == ref["sha256"], "actual evidence byte drift")
    return raw


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate raw JSON key")
        result[key] = value
    return result


def json_ref(project, ref):
    return json.loads(read_ref(project, ref), object_pairs_hook=_pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(ValueError("nonfinite raw evidence")))


def load_pinned_module(project, ref, name, expected_sha):
    require(ref["sha256"] == expected_sha, "independent validation implementation source drift")
    read_ref(project, ref)
    path = Path(project) / ref["path"]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@dataclass(frozen=True)
class ExactGpuIdentity:
    gpu_uuid: str
    common_runtime_domain_sha256: str
    source_lock_sha256: str
    model_sha256: str
    kv_layout_sha256: str
    kernel_mode: str
    native_source_sha256: str
    collector_source_sha256: str
    cuda_event_source_sha256: str
    runtime_refs: tuple
    cells: tuple

    def signature(self, batch, active_decode, prefill_tokens, context_length, physical_bytes):
        for value in (batch, active_decode, context_length, physical_bytes):
            integer(value, "actual positive exact-cell load", 1)
        integer(prefill_tokens, "actual exact prefill")
        return (self.gpu_uuid, self.common_runtime_domain_sha256, self.model_sha256, self.kv_layout_sha256,
                batch, active_decode, prefill_tokens, context_length, physical_bytes)


def _table_material(proof):
    require(proof in _ISSUED, "private capability requires verified raw GPU issuer")
    return _ISSUED[proof][:3]


def _table_proof_valid(proof, cells, source_sha256):
    record = _ISSUED.get(proof)
    return record is not None and record[0] is cells and record[1] == source_sha256


def qualified_identity(table):
    from .p4_cost_table import CostTable
    if type(table) is not CostTable or not table.production_qualified:
        return None
    return _ISSUED[table._gpu_proof][2]


def _unwrap_pair_config(document):
    require(type(document) is dict and set(document) == PAIR_WRAPPER_KEYS
            and document.get("schema") == "strong_native_u_i_cpu_configuration_v1", "strict actual private U/I configuration wrapper")
    pair = document["configurations"]
    require(type(pair) is dict and set(pair) == {"U", "I"}, "exact wrapped U/I configurations")
    require(document["GPU_operations"] == 0 and type(document["GPU_operations"]) is int
            and document["native_execution_verified"] is False and document["gpu_effect_qualified"] is False
            and document["formal_goodput_allowed"] is False, "CPU wrapper cannot grant GPU/effect qualifications")
    return pair


def _strong_domain(project, plan):
    source = plan["strong_pair_validator_source_ref"]
    validator = load_pinned_module(project, source, "_strong_gpu_exact_pair_validator", STRONG_PAIR_VALIDATOR_SHA256)
    pair = _unwrap_pair_config(json_ref(project, plan["runtime_pair_ref"]))
    validator.validate_runtime_pair(pair)
    import copy
    normalized = copy.deepcopy(pair["U"])
    require(normalized["engine"].get("enforce_eager") is True, "first exact-cell issuer only verifies eager kernel mode")
    extra = normalized["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
    require(extra["load_planner"] == "on" and extra["enable_preload"] is True and extra["preload_share_staging"] is True,
            "new strong planner/preload/shared staging domain required")
    extra.pop("shared_storage_path")
    extra.pop("prefix_io_p4_policy")
    extra["prefix_io_parent_admission"].pop("run_id")
    extra.pop("prefix_io_observation_run_id", None)
    return digest(normalized)


def _canonical_geometry(project, plan):
    raw = read_ref(project, plan["kv_layout_ref"])
    geometry = json.loads(raw, object_pairs_hook=_pairs)
    fields = {"model_config_sha256", "num_hidden_layers", "num_key_value_heads", "head_dim", "dtype", "dtype_bytes",
              "tokens_per_block", "tensor_parallel_size", "physical_block_bytes"}
    require(type(geometry) is dict and set(geometry) == fields and raw == canonical(geometry),
            "layout reference must be actual canonical compact tensor geometry bytes without newline")
    config_ref = plan["model_config_source_ref"]
    model = json_ref(project, config_ref)
    require(geometry["model_config_sha256"] == config_ref["sha256"] and model.get("num_hidden_layers") == 28
            and model.get("num_key_value_heads") == 4 and model.get("hidden_size") == 3584 and model.get("num_attention_heads") == 28,
            "actual Qwen config/layout source closure")
    expected = dict(model_config_sha256=config_ref["sha256"], num_hidden_layers=28, num_key_value_heads=4,
                    head_dim=128, dtype="bfloat16", dtype_bytes=2, tokens_per_block=16, tensor_parallel_size=1,
                    physical_block_bytes=28 * 2 * 4 * 128 * 2 * 16)
    require(geometry == expected and plan["kv_layout_ref"]["sha256"] == digest(expected), "actual model-derived physical geometry SHA")
    return geometry


def _fit_calibration_only(validation, original_estimator_path, base, action, pair_ids, timing_contract):
    """Original numerical AST receives fit pairs only; never fits the holdout."""
    calculate, math_proof = validation.original_estimator(original_estimator_path)
    runs = {key: {"pair_id": key} for key in pair_ids}
    groups = {"calibration": list(runs.values()), "validation": []}
    frozen_timing = dict(timing_contract)
    frozen_timing["reference_source_ref"] = validation.EvidenceRef.from_mapping(timing_contract["reference_source_ref"])
    plan = SimpleNamespace(schema_version=2, timing_contract=tuple(frozen_timing.items()))
    fitted = calculate({key: value for key, value in base.items() if key[0] in runs},
                       {key: value for key, value in action.items() if key[0] in runs}, runs, groups, plan)
    return fitted, math_proof


def replay_math_audit(validation, original_estimator_path, *, calibration_A_ns, calibration_B_ns, heldout_B_ns, reference_source_ref):
    """Pure mathematics test interface: it can never mint a GPU capability."""
    require(len(calibration_A_ns) == len(calibration_B_ns) == 2, "two fit pairs")
    timing = dict(timing_scope="full_decode_step", clock_domain="cuda_event_elapsed", reference_source_ref=reference_source_ref,
                  reference_valid=True, fallback_used=False, clock_domain_valid=True)
    base, action = {}, {}
    for index, (a, b) in enumerate(zip(calibration_A_ns, calibration_B_ns)):
        integer(a, "positive A fixture duration", 1)
        integer(b, "positive B fixture duration", 1)
        key = ("fit" + str(index), "step-16", "measured")
        base[key] = dict(output_tokens=1, timing=dict(timing, gpu_elapsed_ns=a))
        action[key] = dict(output_tokens=1, timing=dict(timing, gpu_elapsed_ns=b))
    fitted, proof = _fit_calibration_only(validation, original_estimator_path, base, action, ("fit0", "fit1"), timing)
    upper = sum(fitted[name] for name in ("baseline_ns", "incremental_or_joint_ns", "uncertainty_ns"))
    integer(heldout_B_ns, "positive heldout fixture duration", 1)
    return dict(status="CPU_MATH_AUDIT_ONLY", calibration=fitted, original_math_proof=proof, upper_ns=upper,
                heldout_B_ns=heldout_B_ns, heldout_covered=heldout_B_ns <= upper, holdout_used_to_fit=False,
                production_qualified=False, actual_gpu_execution_proved=False)


def _cell(project, validation, plan, descriptor, raw_cell):
    from .p4_cost_table import CostCell
    require(type(descriptor) is dict and descriptor["id"] == raw_cell.get("id"), "exact preregistered cell identity")
    prompt_tokens = integer(descriptor["prompt_tokens"], "actual frozen prompt length", 1)
    cached = integer(descriptor["cached_prompt_tokens"], "actual frozen cached prompt length")
    offset = integer(descriptor["measured_offset"], "actual frozen decode selection", 2)
    require(cached < prompt_tokens and offset < 128 and descriptor["warmup_offsets"] == [1], "finite full-output selection")
    operations = integer(descriptor["operations"], "finite original SSD operations", 1)
    require(operations in (1, 2, 4, 8) and descriptor["stage"] == "ssd_read", "finite SSD-only current issuer scope")
    physical = integer(descriptor["transfer_quantum_bytes"], "actual unit geometry", 1) * operations
    require(descriptor["existing_io"] == [{"ops": 0, "bytes": 0}] * 4, "issuer zero-existing-I/O only, owner journal must prove it")
    entries = descriptor["entries"]
    require(type(entries) is list and len(entries) == 3 and [entry["split"] for entry in entries] == ["calibration", "calibration", "validation"]
            and [entry["arm_order"] for entry in entries] == ["AB", "BA", "AB"] and len({entry["pair_id"] for entry in entries}) == 3,
            "two fit pairs and independent never-fitted holdout")
    require(type(raw_cell.get("windows")) is list and len(raw_cell["windows"]) == 6, "all six actual native windows")
    for entry in entries:
        tokens = entry["prompt_token_ids"]
        require(type(tokens) is list and len(tokens) == prompt_tokens and all(type(token) is int and token >= 0 for token in tokens), "complete actual prompt token pins")
        integer(entry["seed"], "frozen pair seed")
        require(entry["prompt_sha256"] == digest(tokens) and entry["prefix_family_sha256"] == digest(tokens[:16])
                and entry["trace_sha256"] == digest(dict(prompt_token_ids=tokens, seed=entry["seed"], output_tokens=128)), "prompt/prefix/trace independent pins")
    for field in ("prompt_sha256", "prefix_family_sha256", "trace_sha256"):
        require(entries[-1][field] not in {entry[field] for entry in entries[:2]}, "holdout family leakage")
    order = [(entry, arm) for entry in entries for arm in (("baseline", "action") if entry["arm_order"] == "AB" else ("action", "baseline"))]
    timing = dict(timing_scope="full_decode_step", clock_domain="cuda_event_elapsed", reference_source_ref=plan["collector_source_ref"],
                  reference_valid=True, fallback_used=False, clock_domain_valid=True)
    base, action, outputs, heldout, request_ids = {}, {}, {}, [], set()
    interval_end = 0
    for window, (entry, arm) in zip(raw_cell["windows"], order):
        require(window.get("pair_id") == entry["pair_id"] and window.get("arm") == arm and window.get("seed") == entry["seed"]
                and window.get("prompt_token_ids") == entry["prompt_token_ids"], "actual frozen window order/load/seed")
        require(window.get("gpu_uuid") == plan["gpu_uuid"] and window.get("source_lock_sha256") == plan["source_lock_ref"]["sha256"]
                and window.get("model_sha256") == plan["model_manifest_ref"]["sha256"] and window.get("kv_layout_sha256") == plan["kv_layout_ref"]["sha256"], "actual window source/model/layout/GPU")
        require(type(window.get("request_id")) is str and window["request_id"] and window["request_id"] not in request_ids, "unique actual native request")
        require(type(window.get("run_id")) is str and window["run_id"], "actual capture run identity")
        request_ids.add(window["request_id"])
        capture = window["capture"]
        event = capture.get("event_source", {})
        cuda_ref = plan["cuda_event_source_ref"]
        require(event.get("origin") == "native_gpu_recording" and event.get("module") == "torch.cuda.streams" and event.get("class_name") == "Event"
                and event.get("sha256") == cuda_ref["sha256"] and event.get("bytes") == cuda_ref["bytes"]
                and event.get("path") == str(Path(project).resolve()).replace("\\", "/") + "/" + cuda_ref["path"], "source-pinned actual CUDA Event")
        frames = validation.validate_capture(capture, run_id=window["run_id"], request_id=window["request_id"],
                    output_ids=window["output_token_ids"], prompt_tokens=prompt_tokens, measured_offset=offset, warmup_offsets=[1], cached_tokens=cached)
        require(frames[0]["witness"]["start_record_before_ns"] > interval_end, "independent windows overlap")
        interval_end = frames[-1]["witness"]["end_record_after_ns"]
        drain = validation.original_post_shutdown_drain(window["native_post_shutdown"], window["native_tail_assertions"],
                    run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_ref"]["sha256"])
        validation.validate_io(window["native_journal"], drain, capture=capture, frames=frames,
                    run_id=plan["journal_run_id"], native_source_sha256=plan["native_source_ref"]["sha256"], arm=arm,
                    measured_offset=offset, physical_bytes=physical, operations=operations, independent_payload=window["independent_payload"])
        for selected, phase in ((1, "warmup"), (offset, "measured")):
            row = dict(output_tokens=1, timing=dict(timing, gpu_elapsed_ns=frames[selected]["gpu_elapsed_ns"]))
            (base if arm == "baseline" else action)[entry["pair_id"], "step-" + str(selected), phase] = row
        outputs.setdefault(entry["pair_id"], {})[arm] = window["output_token_ids"]
        if entry["split"] == "validation" and arm == "action":
            heldout.append(frames[offset]["gpu_elapsed_ns"])
    require(all(pair["baseline"] == pair["action"] for pair in outputs.values()), "full paired output mismatch")
    original_source = Path(project) / plan["original_estimator_source_ref"]["path"]
    fitted, math_proof = _fit_calibration_only(validation, original_source, base, action, [entry["pair_id"] for entry in entries[:2]], timing)
    upper = sum(fitted[name] for name in ("baseline_ns", "incremental_or_joint_ns", "uncertainty_ns"))
    require(len(heldout) == 1 and heldout[0] <= upper, "heldout underprediction, no refit permitted")
    signature = (plan["gpu_uuid"], plan["common_runtime_domain_sha256"], plan["model_manifest_ref"]["sha256"], plan["kv_layout_ref"]["sha256"],
                 1, 1, 0, prompt_tokens + offset - 1, physical)
    cell = CostCell(signature, ZERO, "ssd_read", physical, "existing_io_plus_delta",
                    fitted["baseline_ns"], fitted["incremental_or_joint_ns"], fitted["uncertainty_ns"])
    return cell, dict(cell_id=descriptor["id"], upper_ns=upper, heldout_B_ns=heldout[0], holdout_used_to_fit=False,
                     source_math=math_proof, resource_release_credit=False)


def issue_verified_gpu_table(project, *, plan_ref, measurements_ref, guard_ref, intent_ref,
                             expected_plan_ref, expected_guard_ref, expected_intent_ref,
                             expected_runtime_domain_sha256, expected_gpu_uuid, expected_source_lock_sha256,
                             expected_collector_source_ref=None):
    require(plan_ref == expected_plan_ref and guard_ref == expected_guard_ref and intent_ref == expected_intent_ref,
            "independent parent prelaunch plan/intent and completed original guard refs")
    for value in (expected_runtime_domain_sha256, expected_source_lock_sha256):
        sha(value, "independent expected domain/source")
    require(type(expected_gpu_uuid) is str and expected_gpu_uuid.startswith("GPU-"), "actual expected GPU UUID")
    plan = json_ref(project, plan_ref)
    require(plan.get("schema") == "strong_gpu_exact_cell_prelaunch_plan_v1" and plan.get("evidence_origin") == "native_runtime_preregistered"
            and plan.get("cpu_preparation_only") is False and plan.get("synthetic_fixture") is False, "new actual native strong-domain plan, no CPU fixture or old singleton")
    require(plan.get("project_root") == str(Path(project).resolve()).replace("\\", "/") and plan.get("gpu_uuid") == expected_gpu_uuid
            and plan.get("source_lock_ref", {}).get("sha256") == expected_source_lock_sha256
            and plan.get("common_runtime_domain_sha256") == expected_runtime_domain_sha256, "independent actual project/device/source/domain")
    require(type(plan.get("job_id")) is str and 0 < len(plan["job_id"]) <= 128 and plan.get("journal_run_id") == plan["job_id"],
            "same guarded native owner job identity")
    intent = json_ref(project, intent_ref)
    require(intent.get("schema") == "strong_gpu_exact_cell_prelaunch_intent_v1" and intent.get("plan_ref") == plan_ref
            and intent.get("job_id") == plan["job_id"] and intent.get("source_lock_ref") == plan["source_lock_ref"], "independent before-launch intent byte closure")
    lock = json_ref(project, plan["source_lock_ref"])
    rows = lock.get("files")
    require(type(rows) is list and 1 <= len(rows) <= 5000 and len({row["path"] for row in rows}) == len(rows), "complete unique bounded source lock")
    locked = {row["path"]: row for row in rows}
    required_keys = ("collector_source_ref", "cuda_event_source_ref", "native_source_ref", "original_guard_source_ref",
                     "validation_source_ref", "strong_pair_validator_source_ref", "runtime_pair_ref", "model_manifest_ref", "kv_layout_ref",
                     "original_estimator_source_ref", "wrapper_source_ref", "issuer_source_ref", "cost_table_source_ref", "model_config_source_ref")
    for key in required_keys:
        require(locked.get(plan[key]["path"]) == plan[key], "actual runtime leaf absent from source lock: " + key)
        read_ref(project, plan[key])
    require(plan["validation_source_ref"]["sha256"] == VALIDATION_SHA256 and plan["native_source_ref"]["sha256"] == NATIVE_SHA256,
            "original owner/validation source pins")
    require(plan["original_guard_source_ref"]["path"] == "experiments/prefix_io_v1/scripts/run_gpu_stage.py"
            and plan["original_guard_source_ref"]["sha256"] == ORIGINAL_GUARD_SHA256, "original GPU budget guard source pin")
    if expected_collector_source_ref is None:
        require(plan["collector_source_ref"]["sha256"] == COLLECTOR_SHA256, "original collector source pin")
    else:
        require(plan["collector_source_ref"] == expected_collector_source_ref, "independent patched collector source pin")
        read_ref(project, expected_collector_source_ref)
    from . import p4_cost_table as cost_module
    require((Path(project) / plan["issuer_source_ref"]["path"]).resolve() == Path(__file__).resolve()
            and (Path(project) / plan["cost_table_source_ref"]["path"]).resolve() == Path(cost_module.__file__).resolve(),
            "actual active issuer/CostTable overlay source closure")
    require(_strong_domain(project, plan) == expected_runtime_domain_sha256 and plan.get("kernel_mode") == "eager", "actual strong common runtime domain")
    geometry = _canonical_geometry(project, plan)
    validation = load_pinned_module(project, plan["validation_source_ref"], "_strong_raw_native_validation", VALIDATION_SHA256)
    guard = json_ref(project, guard_ref)
    validation.validate_guard(guard, gpu_uuid=expected_gpu_uuid, job_id=plan["job_id"], wrapper_path=plan["wrapper_source_ref"]["path"])
    record = json_ref(project, measurements_ref)
    require(record.get("schema") == "strong_gpu_exact_cell_raw_measurements_v1" and record.get("origin") == "native_gpu_recording"
            and record.get("synthetic_fixture") is False and record.get("plan_ref") == plan_ref, "actual raw measurements origin/plan")
    require(record.get("runtime_identity") == dict(gpu_uuid=expected_gpu_uuid,
            common_runtime_domain_sha256=expected_runtime_domain_sha256, source_lock_sha256=expected_source_lock_sha256,
            model_sha256=plan["model_manifest_ref"]["sha256"], kv_layout_sha256=plan["kv_layout_ref"]["sha256"],
            kernel_mode="eager"), "actual raw runtime identity closure")
    for phase in ("before", "after"):
        receipt = json_ref(project, record["source_verification_refs"][phase])
        require(receipt.get("phase") == phase and receipt.get("source_lock_ref") == plan["source_lock_ref"]
                and receipt.get("failed") == [] and type(receipt.get("files_verified")) is int and receipt["files_verified"] == len(rows), "actual full source verification phase")
    descriptors = plan.get("cells")
    raw_cells = record.get("cells")
    require(type(descriptors) is list and type(raw_cells) is list and 1 <= len(descriptors) == len(raw_cells) <= 8
            and len({cell["id"] for cell in descriptors}) == len(descriptors), "one to eight exact preregistered cells")
    require(all(type(cell.get("transfer_quantum_bytes")) is int and cell["transfer_quantum_bytes"] == geometry["physical_block_bytes"]
            for cell in descriptors), "canonical actual cell storage-unit quantum")
    cells, audits = zip(*(_cell(project, validation, plan, descriptor, raw) for descriptor, raw in zip(descriptors, raw_cells)))
    require(len({cell.key for cell in cells}) == len(cells), "ambiguous exact verified cells")
    # Leaf closure is reread after all arithmetic; complete runtime/model closure
    # remains in the independently pinned before/after full-source receipts.
    for ref in (plan_ref, measurements_ref, guard_ref, intent_ref, plan["source_lock_ref"]):
        read_ref(project, ref)
    for key in required_keys:
        read_ref(project, plan[key])
    identity = ExactGpuIdentity(expected_gpu_uuid, expected_runtime_domain_sha256, expected_source_lock_sha256,
                    plan["model_manifest_ref"]["sha256"], plan["kv_layout_ref"]["sha256"], plan["kernel_mode"],
                    NATIVE_SHA256, plan["collector_source_ref"]["sha256"], plan["cuda_event_source_ref"]["sha256"],
                    tuple(sorted((plan[key]["path"], plan[key]["bytes"], plan[key]["sha256"]) for key in required_keys
                                 if key != "runtime_pair_ref")),
                    tuple(cell.load_signature for cell in cells))
    proof = object()
    require(len(_ISSUED) < 32, "bounded process lifetime of exact-cell capabilities")
    _ISSUED[proof] = (tuple(cells), measurements_ref["sha256"], identity, tuple(audits))
    from .p4_cost_table import _from_verified_gpu_capability
    try:
        return _from_verified_gpu_capability(proof)
    except BaseException:
        _ISSUED.pop(proof, None)
        raise
