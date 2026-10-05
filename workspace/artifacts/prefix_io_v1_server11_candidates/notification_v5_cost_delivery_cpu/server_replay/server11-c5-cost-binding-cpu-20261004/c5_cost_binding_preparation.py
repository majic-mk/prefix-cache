"""Read-only C5 source/receipt-contract preparation; never a native issuer.

The frozen six-window analyzer supplies the numerical AST and 128-frame
contract. This module does not import a launcher, native verifier entry, model,
torch, CUDA, or the old private receipt constructor. Every input timing is a
synthetic CPU contract example. Source hashes detect drift, not authorization.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys


PREPARATION_SCOPE = "p4_c5_single_file_cpu_binding_preparation_v1"
PLAN_SCOPE = "server11_c5_cost_binding_preparation_plan_v1"
BLOCKED = "GPU_BLOCKED_C5_COST_BINDING_PREPARATION_ONLY"
C5_REACTOR_SHA256 = "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47"
C5_COLLECTOR_SHA256 = "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"
VERIFIER_SHA256 = "e20944ea470a4fbdbd10cb67e516cfc2b9f1de3f53d61d0249e0fa6d0e59a291"
SERIALIZER_SHA256 = "6cb02c8d2ff148a1c25f221a2350f71ae71daff39c1cdcd191954ae8e6c9570f"
ORIGINAL_SHA256 = "3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac"
FORMULA = "ceil_calibration_means_plus_max_positive_residual_v1"
ORIGINAL_RELATIVE = "third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py"
NATIVE_SUFFIX = "source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
RUNNER_SUFFIX = "third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py"
PREPARATION_REQUIRED = ("notification_runtime_adapter.py", "run_p4_single_file_experiment.py",
    "control_p4_single_file.py", "verify_p4_single_file.py", "run_cpu_preparation.py",
    "freeze_cpu_preparation.py")
_PREPARATION_TOKEN = object()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def keys(value, names, label):
    require(type(value) is dict and set(value) == set(names), label + " exact keys")


def _pairs(items):
    result = {}
    for name, value in items:
        require(name not in result, "duplicate JSON key")
        result[name] = value
    return result


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def relative(root, value):
    require(type(value) is str and value and not value.startswith("/") and
        not any(character in value for character in ("\\", ":", "\0")) and
        all(part not in ("", ".", "..") for part in value.split("/")), "bounded project path")
    path = Path(root).resolve(strict=True)
    for part in value.split("/"):
        path /= part
        require(not path.is_symlink(), "symlink evidence refused")
    require(Path(root).resolve(strict=True) in path.resolve().parents, "project confinement")
    return path


@dataclass(frozen=True)
class FileRef:
    path: str
    bytes: int
    sha256: str

    @classmethod
    def from_mapping(cls, value):
        keys(value, ("path", "bytes", "sha256"), "file reference")
        require(type(value["bytes"]) is int and 0 < value["bytes"] <= 32 * 1024**2,
            "bounded positive evidence bytes")
        digest = value["sha256"]
        require(type(digest) is str and len(digest) == 64 and
            all(c in "0123456789abcdef" for c in digest), "file SHA-256")
        return cls(**value)

    def mapping(self):
        return dict(path=self.path, bytes=self.bytes, sha256=self.sha256)

    def read(self, root):
        path = relative(root, self.path)
        require(path.is_file() and path.stat().st_size == self.bytes, "evidence size drift: " + self.path)
        data = path.read_bytes()
        require(len(data) == self.bytes and sha256(data).hexdigest() == self.sha256,
            "evidence SHA drift: " + self.path)
        return data

    def json(self, root):
        return json.loads(self.read(root), object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def file_ref(root, value):
    path = relative(root, value)
    require(path.is_file() and 0 < path.stat().st_size <= 32 * 1024**2, "bounded evidence file")
    raw = path.read_bytes()
    return FileRef(value, len(raw), sha256(raw).hexdigest()).mapping()


def _rows(root, rows, label):
    require(type(rows) is list and 1 <= len(rows) <= 8192, label + " bounded rows")
    mapped = {}
    for value in rows:
        ref = FileRef.from_mapping(value)
        require(ref.path not in mapped, label + " duplicate source path")
        ref.read(root)
        mapped[ref.path] = ref.mapping()
    return mapped


def _fixed_protocol(plan):
    keys(plan, ("scope", "schema_version", "origin", "preparation_source_lock_ref", "source_lock_ref",
        "source_lock_files", "candidate_relative", "preparation_relative", "reactor_source_ref",
        "collector_source_ref", "wrapper_source_ref", "original_estimator_ref", "verifier_source_ref",
        "serializer_source_ref", "model_plan_ref", "model_config_ref", "cuda_event_source_ref",
        "model_runner_ref", "common_owner_parameters", "formula_contract"), "C5 preparation plan")
    require(plan["scope"] == PLAN_SCOPE and type(plan["schema_version"]) is int and
        plan["schema_version"] == 1 and plan["origin"] == "synthetic_cpu_contract",
        "new C5 synthetic preparation plan required; old/native plan refused")
    owner = plan["common_owner_parameters"]
    keys(owner, ("max_accepted_parents", "bridge_is_none"), "original calibration owner")
    require(type(owner["max_accepted_parents"]) is int and owner["max_accepted_parents"] == 8 and
        owner["bridge_is_none"] is True,
        "unchanged eight-parent bridge=None calibration owner")
    contract = plan["formula_contract"]
    keys(contract, ("scope", "qualification_rule", "stage", "units", "operations", "transfer_quantum_bytes",
        "cached_prompt_tokens", "prompt_tokens", "output_tokens", "measured_offset", "warmup_offsets",
        "external_warmup_output_tokens", "external_flush_output_tokens", "process_design", "entries"),
        "fixed six-window formula contract")
    expected = dict(scope="server11_preregistered_native_conditional_cell_v1",
        qualification_rule="zero_observed_holdout_underprediction_no_refit_v1", stage="ssd_read",
        units=1, operations=1, transfer_quantum_bytes=917504, cached_prompt_tokens=128,
        prompt_tokens=129, output_tokens=128, measured_offset=16, warmup_offsets=[1],
        external_warmup_output_tokens=128, external_flush_output_tokens=1,
        process_design="six_fresh_original_processes_and_handlers_one_guard")
    require(all(type(contract.get(name)) is type(value) and contract[name] == value
        for name, value in expected.items()), "original finite workload/selection/geometry changed")
    require(all(type(offset) is int for offset in contract["warmup_offsets"]), "exact integer warmup offsets")
    entries = contract["entries"]
    require(type(entries) is list and len(entries) == 3, "three independent frozen pairs")
    for index, entry in enumerate(entries):
        keys(entry, ("pair_id", "split", "arm_order", "seed", "prompt_token_ids", "prompt_sha256",
            "prefix_family_sha256", "trace_sha256", "workload_sha256"), "pair")
        ids = list(range(1000, 1129)); ids[0] = (18100, 19100, 20100)[index]
        seed = (1829, 1830, 1831)[index]
        require(entry["pair_id"] == "pair-" + str(index) and
            entry["split"] == ("calibration" if index < 2 else "validation") and
            entry["arm_order"] == ("AB", "BA", "AB")[index] and
            type(entry["seed"]) is int and entry["seed"] == seed and entry["prompt_token_ids"] == ids and
            all(type(token) is int for token in entry["prompt_token_ids"]), "fixed prompt/seed/order/split")
        hashes = dict(prompt_sha256=canonical_hash(ids), prefix_family_sha256=canonical_hash(ids[:16]),
            trace_sha256=canonical_hash(dict(prompt_token_ids=ids, seed=seed, output_tokens=128)),
            workload_sha256=canonical_hash(dict(prompt_token_ids=ids, output_tokens=128,
                temperature=0, ignore_eos=True)))
        require(all(entry[name] == value for name, value in hashes.items()), "recomputed workload pins")
    require(all(entries[2][name] not in {entry[name] for entry in entries[:2]}
        for name in ("prefix_family_sha256", "trace_sha256", "workload_sha256")), "heldout leakage")


def _preparation_closure(root, plan, locked):
    prep_ref = FileRef.from_mapping(plan["preparation_source_lock_ref"])
    require(locked.get(prep_ref.path) == prep_ref.mapping(), "preparation lock absent from new closure")
    lock = prep_ref.json(root)
    require(lock.get("schema") == "cpu_notification_preparation_source_lock_v1" and
        lock.get("gpu_qualified") is False and lock.get("gpu_launch_allowed") is False and
        lock.get("native_receipt") is None and lock.get("immutable_parent_sources") is True,
        "actual CPU preparation lock flags")
    prefixes = dict(candidate=plan["candidate_relative"], preparation=plan["preparation_relative"])
    require(prep_ref.path == prefixes["preparation"] + "/PREPARATION_SOURCE_LOCK.json",
        "preparation lock exact location")
    rows, seen, mapped = lock.get("files"), set(), {}
    require(type(rows) is list and 1 <= len(rows) <= 8192, "bounded preparation closure")
    for row in rows:
        keys(row, ("scope", "path", "bytes", "sha256"), "scoped preparation ref")
        require(row["scope"] in prefixes, "unknown preparation source scope")
        key = (row["scope"], row["path"])
        require(key not in seen, "duplicate scoped preparation ref"); seen.add(key)
        scoped = dict(path=prefixes[row["scope"]] + "/" + row["path"],
            bytes=row["bytes"], sha256=row["sha256"])
        ref = FileRef.from_mapping(scoped); ref.read(root)
        require(locked.get(ref.path) == scoped, "complete preparation source missing/drifted")
        mapped[ref.path] = scoped
    candidate = relative(root, prefixes["candidate"])
    actual = {path.relative_to(root).as_posix() for path in candidate.rglob("*.py")
        if "__pycache__" not in path.parts}
    declared = {name for name in mapped if name.startswith(prefixes["candidate"] + "/") and name.endswith(".py")}
    require(actual == declared, "complete actual C5 Python closure differs from preparation lock")
    require(all(prefixes["preparation"] + "/" + name in mapped for name in PREPARATION_REQUIRED),
        "full launcher/control/verifier/observer preparation required")
    for path in relative(root, prefixes["preparation"]).iterdir():
        if path.is_file() and path.suffix in (".py", ".md"):
            require(path.relative_to(root).as_posix() in mapped, "preparation top-level source omitted")
    return mapped


@dataclass(frozen=True, init=False)
class C5CostBindingPreparation:
    _document: str

    def __init__(self, *, token=None, document=None):
        require(token is _PREPARATION_TOKEN, "use prepare_cost_binding; this is not a native receipt")
        object.__setattr__(self, "_document", json.dumps(document, sort_keys=True, allow_nan=False))

    def document(self):
        return json.loads(self._document)


def prepare_cost_binding(root, *, binding_ref, expected_binding_ref, expected_plan_ref,
                         expected_source_lock_ref, expected_factory_ref):
    """Validate independently supplied refs; return no costs or native identity.

    expected_* are frozen by the caller before this read, not accepted from an
    untrusted binding's own claims. This is drift detection, not a signature.
    Every runtime ref, including collector and all preparation code, must be in
    the new closure. The old C4/v6 factory and private issuer are never invoked.
    """
    root = Path(root).resolve(strict=True)
    require(binding_ref == expected_binding_ref, "independently expected binding ref")
    binding_file = FileRef.from_mapping(binding_ref)
    binding = binding_file.json(root)
    keys(binding, ("scope", "schema_version", "origin", "plan_ref", "factory_source_ref",
        "runtime_common_refs", "runtime_overlay_refs"), "CPU-only C5 binding")
    require(binding["scope"] == PREPARATION_SCOPE and type(binding["schema_version"]) is int and
        binding["schema_version"] == 1 and binding["origin"] == "synthetic_cpu_contract",
        "old C4/v6 or native receipt binding refused")
    require(binding["plan_ref"] == expected_plan_ref and binding["factory_source_ref"] == expected_factory_ref,
        "independently expected plan/factory refs")
    plan_file = FileRef.from_mapping(expected_plan_ref); plan = plan_file.json(root)
    _fixed_protocol(plan)
    require(plan["source_lock_ref"] == expected_source_lock_ref, "independently expected new source lock")
    lock_file = FileRef.from_mapping(expected_source_lock_ref); lock = lock_file.json(root)
    require(lock.get("schema") == "c5_cost_binding_cpu_source_lock_v1" and
        lock.get("gpu_launch_allowed") is False and lock.get("native_execution_verified") is False,
        "new CPU-only source lock")
    locked = _rows(root, lock.get("files"), "new complete source lock")
    require(type(plan["source_lock_files"]) is int and len(locked) == plan["source_lock_files"],
        "actual source lock count")
    factory = FileRef.from_mapping(expected_factory_ref)
    require(factory.read(root) == Path(__file__).read_bytes(), "actual factory source differs from pinned source")
    require(locked.get(factory.path) == factory.mapping(), "factory absent from independent source closure")
    mapped = _preparation_closure(root, plan, locked)
    common = _rows(root, binding["runtime_common_refs"], "common runtime refs")
    overlay = _rows(root, binding["runtime_overlay_refs"], "overlay runtime refs")
    require(not set(common).intersection(overlay), "duplicate/common-overlay source path")
    require(all(locked.get(name) == value for name, value in (common | overlay).items()),
        "runtime source outside independent new closure")
    candidate_prefix = plan["candidate_relative"] + "/"
    prep_prefix = plan["preparation_relative"] + "/"
    native = plan["reactor_source_ref"]
    require(all(name not in common for name in mapped if name != native["path"] and
        name.startswith((candidate_prefix, prep_prefix))),
        "collector/other C5/preparation sources must remain runtime overlay")
    require(all((common if name == native["path"] else overlay).get(name) == value
        for name, value in mapped.items()) and
        overlay.get(factory.path) == factory.mapping(), "complete C5/preparation/factory overlay required")
    collector = plan["collector_source_ref"]
    require(collector["path"] == candidate_prefix + "native_full_step_collector.py" and
        collector["sha256"] == C5_COLLECTOR_SHA256 and overlay.get(collector["path"]) == collector,
        "same C5 plan collector must appear uniquely in overlay")
    require(native["path"] == candidate_prefix + NATIVE_SUFFIX and
        native["sha256"] == C5_REACTOR_SHA256 and common.get(native["path"]) == native,
        "actual fixed C5 native source; C4 calibration refused")
    required_common = ("original_estimator_ref", "verifier_source_ref", "serializer_source_ref",
        "model_plan_ref", "model_config_ref", "cuda_event_source_ref", "model_runner_ref")
    for name in required_common:
        value = FileRef.from_mapping(plan[name]).mapping()
        require(common.get(value["path"]) == value, "required original/model/Event/runner common ref: " + name)
    require(plan["original_estimator_ref"]["path"] == ORIGINAL_RELATIVE and
        plan["original_estimator_ref"]["sha256"] == ORIGINAL_SHA256 and
        plan["verifier_source_ref"]["sha256"] == VERIFIER_SHA256 and
        plan["serializer_source_ref"]["sha256"] == SERIALIZER_SHA256,
        "unchanged original numerical/serialization/verifier dependencies")
    require(plan["model_runner_ref"]["path"] == RUNNER_SUFFIX and
        plan["cuda_event_source_ref"]["path"].endswith("/torch/cuda/streams.py"), "original runner/Event path")
    wrapper = FileRef.from_mapping(plan["wrapper_source_ref"])
    require(wrapper.path == prep_prefix + "run_p4_single_file_experiment.py" and
        overlay.get(wrapper.path) == wrapper.mapping(), "complete blocked C5 launcher source binding")
    for ref in (binding_file, plan_file, lock_file, factory):
        ref.read(root)
    # Recheck every dependency after the full validation, including metadata
    # and common refs, rather than rereading just the four outer documents.
    for value in locked.values():
        FileRef.from_mapping(value).read(root)
    return C5CostBindingPreparation(token=_PREPARATION_TOKEN, document=dict(
        status="PASS_C5_COST_BINDING_CPU_PREPARATION_ONLY", origin="synthetic_cpu_contract",
        scope=PREPARATION_SCOPE, binding_ref=binding_ref, plan_ref=expected_plan_ref,
        source_lock_ref=expected_source_lock_ref, source_files_verified=len(locked),
        collector_overlay_bound=True, complete_preparation_source_bound=True,
        source_binding_prepared=True, native_cost_qualified=False,
        six_window_formula=FORMULA, six_fresh_processes_required=True, full_frame_count=128,
        actual_gpu_runs=0, native_execution_verified=False, conditional_cost_cell_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        cpu_timing_qualification=False, old_c4_v6_receipt_reusable=False,
        gpu_launch_allowed=False, permission_scope_created=False, valid_native_receipt=None,
        effective_cost_upper_ns=None, effective_step_budget_ns=None,
        production_qualified=False, strategy_effect_verified=False, performance_claim=False,
        next_stage="complete-entry CPU qualification; separately authorized new native calibration"))


def _load_pinned_analyzer(root, ref):
    source = FileRef.from_mapping(ref)
    require(source.sha256 == VERIFIER_SHA256, "frozen analyzer SHA")
    raw = source.read(root)
    name = "_c5_cpu_original_math_" + source.sha256
    module_spec = importlib.util.spec_from_file_location(name, relative(root, source.path))
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(module_spec.origin), "exec", dont_inherit=True), module.__dict__)
        source.read(root)
        return module
    finally:
        sys.modules.pop(name, None)


def audit_synthetic_formula(root, *, envelope, analyzer_ref, original_estimator_ref):
    """Exercise the existing 6x128-frame algorithm, never its native issuer.

    Legacy inner metadata says native_gpu_recording because the frozen shape
    validator requires that vocabulary. The explicit enclosing origin overrides
    no trust: these inputs are synthetic, and no native receipt can be emitted.
    Synthetic numbers are not exposed as effective cost/budget values.
    """
    keys(envelope, ("origin", "plan", "windows", "journal", "drain"), "synthetic formula envelope")
    require(envelope["origin"] == "synthetic_cpu_contract", "only explicit synthetic CPU inputs allowed")
    synthetic_plan = envelope["plan"]
    expected = dict(stage="ssd_read", units=1, operations=1, transfer_quantum_bytes=917504,
        cached_prompt_tokens=128, prompt_tokens=129, output_tokens=128, measured_offset=16)
    require(type(synthetic_plan) is dict and all(type(synthetic_plan.get(name)) is type(value) and
        synthetic_plan[name] == value for name, value in expected.items()) and
        synthetic_plan.get("warmup_offsets") == [1] and
        all(type(offset) is int for offset in synthetic_plan["warmup_offsets"]),
        "fixed C5 synthetic formula geometry/workload")
    root = Path(root).resolve(strict=True)
    original = FileRef.from_mapping(original_estimator_ref)
    require(original.sha256 == ORIGINAL_SHA256 and original.bytes == 48136,
        "frozen original numerical AST")
    original.read(root)
    analyzer = _load_pinned_analyzer(root, analyzer_ref)
    result = analyzer.analyze_paired(deepcopy(envelope["plan"]), deepcopy(envelope["windows"]),
        journal=deepcopy(envelope["journal"]), drain=deepcopy(envelope["drain"]),
        original_source_path=relative(root, original.path))
    require(result["formula"] == FORMULA and result["native_execution_verified"] is False and
        result["conditional_cost_cell_qualified"] is False and result["production_qualified"] is False and
        result["holdout_used_to_refit"] is False, "synthetic analyzer cannot issue qualification")
    original.read(root); FileRef.from_mapping(analyzer_ref).read(root)
    return dict(status="PASS_SYNTHETIC_ORIGINAL_FORMULA_CONTRACT", origin="synthetic_cpu_contract",
        original_estimator_proof=result["original_estimator_proof"], formula=FORMULA,
        calibration_pairs=2, validation_pairs=1, full_output_tokens=768, full_frame_count_per_window=128,
        synthetic_holdout_covered=result["heldout_covered"], holdout_used_to_refit=False,
        native_execution_verified=False, conditional_cost_cell_qualified=False,
        production_qualified=False, actual_gpu_runs=0, gpu_launch_allowed=False,
        native_cost_qualified=False, source_binding_prepared=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        effective_cost_upper_ns=None, effective_step_budget_ns=None, valid_native_receipt=None,
        resource_release_credit=False, performance_claim=False)


def load_verified_single_file(*args, **kwargs):
    raise RuntimeError(BLOCKED + ": no native receipt can be issued by this CPU preparation factory")


def launch_gpu(*args, **kwargs):
    raise RuntimeError(BLOCKED)


def main(argv=None):
    # Block before argument parsing, file reads, budget reservation or import.
    print(json.dumps(dict(status=BLOCKED, gpu_started=False, gpu_launch_allowed=False,
        native_execution_verified=False, valid_native_receipt=None), sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
