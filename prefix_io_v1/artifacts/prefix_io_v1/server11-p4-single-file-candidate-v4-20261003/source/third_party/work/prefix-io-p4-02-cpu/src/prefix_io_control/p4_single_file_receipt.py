"""Source-bound receipt for one finite native SSD cost cell, never production.

The caller freezes the binding in the new experiment's source closure. Hashes
detect drift; they are not signatures or an authorization mechanism. Issuance
replays the frozen v6 serializer and verifier over the real raw receipts. The
private constructor prevents accidental dict/PASS promotion, not hostile Python
code in this process. No model, torch, CUDA, or storage engine is imported here.
"""
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from typing import ClassVar


_D6 = "artifacts/prefix_io_v1/server11-native-cost-v6-20261003/"
_COMMON_OVERLAY = "artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003/source"
_NATIVE = _COMMON_OVERLAY + "/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
_ISSUER = object()
_KERNEL_MODE = "original_execute_sample_eager_triton_attention"
_ORIGINAL = "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py"
_RUNNER = "third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py"


def _require(value, message):
    if not value:
        raise ValueError(message)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, name)
    return value


def _relative(project, name):
    _require(type(name) is str and name and "\\" not in name and ":" not in name
             and all(part not in ("", ".", "..") for part in name.split("/")),
             "bounded project-relative file required")
    result = project
    for part in name.split("/"):
        result /= part
        _require(not result.is_symlink(), "symlink evidence refused")
    _require(project in result.resolve().parents, "file outside project")
    return result


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class FileRef:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_mapping(cls, value):
        _require(type(value) is dict and set(value) == {"path", "bytes", "sha256"},
                 "exact file reference required")
        _integer(value["bytes"], "positive bounded file size", 1)
        _require(value["bytes"] <= 32 * 1024**2 and type(value["sha256"]) is str
                 and len(value["sha256"]) == 64
                 and all(c in "0123456789abcdef" for c in value["sha256"]), "file SHA-256")
        return cls(**value)

    def mapping(self):
        return {"path": self.path, "bytes": self.bytes, "sha256": self.sha256}

    def read(self, project):
        path = _relative(project, self.path)
        _require(path.is_file() and path.stat().st_size == self.bytes, "file size changed: " + self.path)
        raw = path.read_bytes()
        _require(len(raw) == self.bytes and sha256(raw).hexdigest() == self.sha256,
                 "file contents changed: " + self.path)
        return raw

    def json(self, project):
        return json.loads(self.read(project), object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


_VERIFIER = FileRef(_D6 + "native_conditional_cost.py", 36869,
    "e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291")
_SERIALIZER = FileRef(_D6 + "prepare_and_verify_native_cost.py", 22006,
    "6cb02c8d2ff148a1c25f221a2350f71ae71daff39c1cdcd191954ae8e6c9570f")
_MATH = FileRef(_ORIGINAL, 48136,
    "3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac")


@dataclass(frozen=True, init=False)
class ExactSingleFileReceipt:
    signature: tuple
    cost_upper_ns: int
    step_budget_ns: int
    calibration_source_lock_sha256: str
    common_source_refs: tuple
    runtime_common_refs: tuple
    runtime_overlay_refs: tuple
    cuda_event_source_sha256: str
    calibration_native_source_sha256: str
    binding_ref: FileRef
    condition_only: ClassVar[bool] = True
    production_qualified: ClassVar[bool] = False

    def __init__(self, issuer=None, **fields):
        _require(issuer is _ISSUER, "use source-bound load_verified_single_file")
        _require(set(fields) == set(self.__dataclass_fields__) - {"condition_only", "production_qualified"},
                 "complete private receipt fields")
        for name, value in fields.items():
            object.__setattr__(self, name, value)


def _calibration_a_budget(plan, verified):
    """A-only engineering envelope; no SLO or statistical coverage claim.

    Reuse the verified original estimator's calibration baseline, then add the
    largest positive A residual. Neither action nor heldout samples enter this
    budget. Cost fitting itself remains entirely in the frozen estimator.
    """
    ids = [row["pair_id"] for row in plan["entries"] if row["split"] == "calibration"]
    _require(len(ids) == 2 and len(set(ids)) == 2, "two independent calibration A samples")
    rows = [row for row in verified["windows"] if row["arm"] == "baseline" and row["pair_id"] in ids]
    _require(len(rows) == 2 and {row["pair_id"] for row in rows} == set(ids),
             "one measured A sample per calibration pair")
    baseline = _integer(verified["calibration_only_cost"]["baseline_ns"], "verified A baseline", 1)
    values = [_integer(row["selected_gpu_elapsed_ns"], "calibration A duration", 1) for row in rows]
    _require(baseline == (sum(values) + len(values) - 1) // len(values),
             "original verified baseline disagrees with actual calibration A samples")
    return baseline + max(0, max(value - baseline for value in values))


def _kernel_mode(record, project):
    """Derive the descriptor from six source-bound child engine configurations."""
    _require(len(record["windows"]) == 6, "six native windows required")
    expected = dict(dtype="bfloat16", kv_cache_dtype="auto", quantization=None,
        enforce_eager=True, compilation_config=0, attention_config={"backend": "TRITON_ATTN"},
        tensor_parallel_size=1, distributed_executor_backend="uni", async_scheduling=False,
        block_size=16, max_num_seqs=1, kv_cache_memory_bytes=268435456,
        max_model_len=1040, max_num_batched_tokens=1040, disable_log_stats=True)
    expected_extra = dict(spec_name="PyKvCacheOffloadingSpec", spec_module_path="py_kvcache.vllm",
        block_size=16, sync_on_store=False, staging_mem=0.125, iodepth=4, enable_preload=True,
        preload_lookahead_requests=1, preload_share_staging=True, staging_cache="lru",
        load_planner="off", io_backend="linux_aio")
    configs = []
    for window in record["windows"]:
        child = FileRef.from_mapping(window["raw_child_ref"]).json(project)
        config = child["engine_config"]
        _require(all(type(config.get(k)) is type(v) and config.get(k) == v for k, v in expected.items()),
                 "actual original eager kernel configuration changed")
        transfer = config["kv_transfer_config"]
        _require(transfer.get("kv_connector") == "OffloadingConnector" and transfer.get("kv_role") == "kv_both",
                 "actual original connector configuration changed")
        extra = transfer["kv_connector_extra_config"]
        _require(all(type(extra.get(k)) is type(v) and extra.get(k) == v for k, v in expected_extra.items()),
                 "actual native SSD/staging configuration changed")
        _require(extra.get("shared_storage_path") == str(_relative(project, child["private_storage"])),
                 "actual private storage pin")
        # Every non-storage engine option must agree; paths are per-process by design.
        normalized = json.loads(json.dumps(config))
        normalized["kv_transfer_config"]["kv_connector_extra_config"].pop("shared_storage_path")
        configs.append(normalized)
    _require(all(config == configs[0] for config in configs), "calibration engine options differ")
    return _KERNEL_MODE


def _runtime_refs(project, binding, plan):
    lock = FileRef.from_mapping(plan["source_lock_ref"]).json(project)
    locked = {row["path"]: row for row in lock["files"]}
    result = []
    seen = set()
    for key in ("runtime_common_refs", "runtime_overlay_refs"):
        values = binding[key]
        _require(type(values) is list and 1 <= len(values) <= 256, "bounded runtime source references")
        refs = tuple(FileRef.from_mapping(row) for row in values)
        for ref in refs:
            _require(ref.path not in seen, "duplicate/common-overlay source path")
            seen.add(ref.path)
            ref.read(project)
            if key == "runtime_common_refs":
                _require(locked.get(ref.path) == ref.mapping(), "common source differs from real calibration closure")
            if ref.path.startswith(_COMMON_OVERLAY + "/"):
                _require(locked.get(ref.path) == ref.mapping(), "C4 runtime source differs from actual C4 calibration")
        result.append(refs)
    common, overlay = result
    required = {plan["model_plan_ref"]["path"], plan["model_config_ref"]["path"],
                plan["cuda_event_source_ref"]["path"], _RUNNER, _NATIVE}
    _require(required.issubset({ref.path for ref in common}), "model/layout/Event/model runner common pins required")
    return common, overlay


def _load_frozen_serializer(project):
    # Both imported source files are pinned before import and reread afterwards.
    _VERIFIER.read(project)
    raw = _SERIALIZER.read(project)
    _MATH.read(project)
    name = "_p4_single_file_v6_serializer_" + str(id(raw))
    spec = importlib.util.spec_from_file_location(name, _relative(project, _SERIALIZER.path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(spec.origin), "exec", dont_inherit=True), module.__dict__)
        _VERIFIER.read(project)
        return module
    finally:
        sys.modules.pop(name, None)


def load_verified_single_file(project, binding_path):
    """Issue only after v6 raw reserialization, native verification, and pins.

    binding_path is a relative path explicitly frozen by the parent workflow.
    The consumer must require binding_ref in its new runtime source closure.
    This function does not grant GPU permission or production qualification.
    """
    project = Path(project).resolve(strict=True)
    path = _relative(project, binding_path)
    _require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2, "bounded binding file")
    raw = path.read_bytes()
    binding_ref = FileRef(binding_path, len(raw), sha256(raw).hexdigest())
    binding = binding_ref.json(project)
    _require(type(binding) is dict and set(binding) == {"schema_version", "scope", "calibration",
             "runtime_common_refs", "runtime_overlay_refs", "kernel_mode"}
             and type(binding["schema_version"]) is int and binding["schema_version"] == 1
             and binding["scope"] == "p4_single_file_native_binding_v1", "exact source-bound native binding")
    calibration = binding["calibration"]
    _require(type(calibration) is dict and set(calibration) == {"plan_ref", "measurements_ref", "guard_ref"},
             "independently frozen native plan/measurements/guard refs")
    refs = {key: FileRef.from_mapping(value) for key, value in calibration.items()}
    plan = refs["plan_ref"].json(project)
    record = refs["measurements_ref"].json(project)
    _require(plan.get("job_id") == "server11-native-cost-six-window-06" and plan.get("units") == 1
             and plan.get("operations") == 1 and plan.get("transfer_quantum_bytes") == 917504,
             "only the actual v6 single-file calibration is supported")
    native = FileRef.from_mapping(plan["native_source_ref"])
    _require(native.path == _NATIVE and native.sha256 == plan["native_source_sha256"]
             and plan.get("common_overlay_relative") == _COMMON_OVERLAY
             and plan.get("common_owner_parameters") == {"max_accepted_parents":8, "bridge_is_none":True},
             "actual same C4 native source and unchanged common owner parameters required")
    native.read(project)
    common, overlay = _runtime_refs(project, binding, plan)
    serializer = _load_frozen_serializer(project)
    reconstructed = serializer.serialize_runtime_record(project,
        runtime_relative=record["raw_runtime_ref"]["path"], plan_ref=calibration["plan_ref"],
        source_verification_refs=record["source_verification_refs"])
    _require(reconstructed == record, "serialized measurement differs from actual parent/child native receipts")
    verified = serializer.C.verify_native_cell(project,
        plan_ref=calibration["plan_ref"], measurements_ref=calibration["measurements_ref"],
        guard_ref=calibration["guard_ref"], expected_plan_ref=calibration["plan_ref"],
        expected_guard_ref=calibration["guard_ref"], original_source_path=_relative(project, _ORIGINAL))
    _require(verified.get("native_execution_verified") is True
             and verified.get("conditional_cost_cell_qualified") is True
             and verified.get("heldout_covered") is True
             and verified.get("production_qualified") is False
             and verified.get("holdout_used_to_refit") is False, "v6 finite native cell did not qualify")
    condition = verified["condition"]
    _require(condition["stage"] == "ssd_read" and condition["physical_bytes"] == 917504
             and condition["operations"] == 1
             and condition["existing_io"] == [{"ops": 0, "bytes": 0}] * 4
             and condition["load"] == {"active_decode": 1, "batch": 1, "prefill_tokens": 0, "context_length": 144},
             "unsupported native condition")
    mode = _kernel_mode(record, project)
    _require(binding["kernel_mode"] == mode, "binding kernel descriptor differs from actual native configuration")
    upper = _integer(verified["calibration_predicted_upper_ns"], "calibration-only cost upper", 1)
    budget = _calibration_a_budget(plan, verified)
    for ref in (binding_ref, *refs.values(), _VERIFIER, _SERIALIZER, _MATH, *common, *overlay):
        ref.read(project)
    return ExactSingleFileReceipt(_ISSUER,
        signature=(condition["model_sha256"], condition["gpu_uuid"], condition["kv_layout_sha256"],
                   mode, 1, 1, 0, 144, 917504), cost_upper_ns=upper, step_budget_ns=budget,
        calibration_source_lock_sha256=condition["source_lock_sha256"], common_source_refs=common,
        runtime_common_refs=common, runtime_overlay_refs=overlay,
        cuda_event_source_sha256=plan["cuda_event_source_ref"]["sha256"],
        calibration_native_source_sha256=plan["native_source_sha256"], binding_ref=binding_ref)
