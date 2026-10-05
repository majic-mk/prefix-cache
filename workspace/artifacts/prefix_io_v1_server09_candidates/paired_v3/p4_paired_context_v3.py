"""Independent CPU-only schema 3 wrapper / trace paired semantics.

Only two locked function ASTs are compiled into independent globals. The
original source, module globals, selected-window rules and table loader stay
unchanged. GPU-looking event durations here are fixture metadata, not evidence
that any GPU event or model ran.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

CONTEXT_REFS = {
    "p4_complete_trace_context_v3.py": (12962, "07dd3af5932395dcc07b325b52be53ed624f4b4af8cfc4bf47dd0b7c58bbb477"),
    "test_p4_complete_trace_context_v3.py": (21769, "45cb847949b7ece2f9ac3b3ad011e0db5924f98da97853384a0e83427b153d36"),
    "P4_COMPLETE_TRACE_CONTEXT_V3_PLAN.json": (14930, "1a2571b6a823c30bae553b66bbbf294993403df7ab02ae025e88b05c00fff8e9"),
}
LOADER_REF = (2686, "23fddf7c159ff2bdcb98c000ad18f90c804183f0a5af9bf08be24ac3b54d368d")
WRAPPER_ROLES = ("baseline_wrapper", "action_wrapper")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same_expression(node, source):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(source, mode="eval").body, include_attributes=False)


@dataclass(frozen=True)
class PairedContextV3CPUSemantics:
    cell_id: str
    baseline_ns: int
    incremental_or_joint_ns: int
    empirical_residual_ns: int
    calibration_pairs: int
    validation_pairs: int
    paired_runs: int
    measured_windows: int
    warmup_windows: int
    full_steps: int
    full_output_tokens: int
    cold_prefill_rows: tuple
    evidence_refs: tuple
    source_refs: tuple
    origin: str = "cpu_fixture"
    schema_version: int = 3
    context_basis: str = "pre_computed_tokens"
    status: str = "CPU_PAIRED_CONTEXT_V3_SEMANTICS_VERIFIED_GPU_AUTHENTICITY_UNPROVEN"
    paired_cost_verification_performed: bool = True

    @property
    def gpu_verified(self):
        return False

    @property
    def production_qualified(self):
        return False

    @property
    def effect_verified(self):
        return False

    @property
    def P4_complete(self):
        return False


class PairedContextV3CPUVerifier:
    """Four raw roles plus independent schema 2 analysis; no table publishing."""
    def __init__(self, control_source_root, context_source_root=None):
        self.control_source_root = Path(control_source_root).resolve(strict=True)
        self.context_source_root = Path(context_source_root or
            Path(__file__).resolve().parent.parent / "context_v3").resolve(strict=True)
        self._check_dependency_bytes()
        source = self.context_source_root / "p4_complete_trace_context_v3.py"
        name = "_server09_paired_context_v3_dependency_" + sha256(str(source).encode()).hexdigest()[:16]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.context_module = sys.modules[name]
        require(Path(self.context_module.__file__).resolve() == source,
                "context dependency module path differs")
        self.complete = self.context_module.CompleteTraceContextV3Verifier(self.control_source_root)
        self.base = self.complete.base
        self._check_dependency_bytes()
        self._cell, self.clone_proof = self._clone_pair_chain()

    def _check_dependency_bytes(self):
        for name, expected in CONTEXT_REFS.items():
            p = self.context_source_root / name
            require(p.is_file() and not p.is_symlink(), "frozen context dependency absent/symlink: " + name)
            raw = p.read_bytes()
            require((len(raw), sha256(raw).hexdigest()) == expected,
                    "frozen context dependency bytes/SHA drift: " + name)
        loader = self.control_source_root / "prefix_io_control/p4_verified_cost_loader.py"
        require(loader.is_file() and not loader.is_symlink(), "original loader absent/symlink")
        raw = loader.read_bytes()
        require((len(raw), sha256(raw).hexdigest()) == LOADER_REF, "original loader bytes/SHA drift")

    def _check_sources(self):
        self._check_dependency_bytes()
        self.complete._check_sources()

    def _clone_pair_chain(self):
        source = self.complete.package_root / "p4_paired_measurement_verifier.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        names = ("_verify_cell", "_v2_windows")
        nodes = {n.name: deepcopy(n) for n in tree.body
                 if isinstance(n, ast.FunctionDef) and n.name in names}
        require(set(nodes) == set(names), "two exact original paired functions required")
        before = {n: sha256(ast.dump(nodes[n], include_attributes=False).encode()).hexdigest() for n in names}
        counts = dict(role_wrapper_version=0, complete_trace_call=0)

        class ExactPairPatch(ast.NodeTransformer):
            def visit_Call(self, value):
                value = self.generic_visit(value)
                if (isinstance(value.func, ast.Name) and value.func.id == "_measurement"
                        and len(value.args) == 2 and same_expression(value.args[0], "root")
                        and same_expression(value.args[1], "refs[role]")):
                    require([k.arg for k in value.keywords] == ["scope", "arm", "cell_id", "context", "version"],
                            "original measurement role call shape changed")
                    keyword = value.keywords[-1]
                    require(same_expression(keyword.value, "plan.schema_version"),
                            "original role version expression changed")
                    keyword.value = ast.copy_location(ast.parse(
                        "3 if role in ('baseline_wrapper', 'action_wrapper') else plan.schema_version",
                        mode="eval").body, keyword.value)
                    counts["role_wrapper_version"] += 1
                if (isinstance(value.func, ast.Name) and value.func.id == "_v2_complete_trace"
                        and not value.keywords and len(value.args) == 4
                        and all(same_expression(arg, wanted) for arg, wanted in
                                zip(value.args, ("root", "run", "data", "plan")))):
                    value.func = ast.copy_location(ast.Name("_complete_trace_context_v3", ast.Load()), value.func)
                    counts["complete_trace_call"] += 1
                return value

        nodes = {name: ExactPairPatch().visit(node) for name, node in nodes.items()}
        require(counts == dict(role_wrapper_version=1, complete_trace_call=1),
                "unexpected original AST; refuse broad version or paired-rule substitution")
        after = {n: sha256(ast.dump(nodes[n], include_attributes=False).encode()).hexdigest() for n in names}
        namespace = dict(self.base.__dict__)
        # Wrappers have already been strictly read as version 3. Their run ABI
        # is exactly the original v2 ABI; there is no legacy or loose dispatch.
        namespace["_runs"] = self.base._v2_runs
        namespace["_complete_trace_context_v3"] = self.complete._complete
        module = ast.fix_missing_locations(ast.Module(body=[nodes[n] for n in names], type_ignores=[]))
        exec(compile(module, str(source) + "::paired_v3_exact_patch", "exec"), namespace)
        require(namespace["_verify_cell"].__globals__ is namespace
                and namespace["_v2_windows"].__globals__ is namespace
                and namespace is not self.base.__dict__, "independent paired globals required")
        return namespace["_verify_cell"], dict(counts,
            original_function_ast_sha256=before, cloned_function_ast_sha256=after,
            original_run_checker_reused="_v2_runs", original_module_globals_changed=False)

    def verify_cell(self, root, *, cell_geometry, raw_refs, selection_plan_ref):
        self._check_sources()
        b = self.base
        root = Path(root).resolve(strict=True)
        pref = b.EvidenceRef.from_mapping(selection_plan_ref)
        plan = b.load_verification_plan(root, pref.path, expected_plan_ref=pref)
        require(plan.schema_version == 2 and dict(plan.timing_contract)["timing_scope"] == "full_decode_step",
                "unchanged schema 2 plan with full_decode_step timer required")
        require(type(raw_refs) is dict and set(raw_refs) == set(b.MEASUREMENT_ROLES) | {"pair_analysis"},
                "exact four original roles and schema 2 pair_analysis required")
        refs = {role: b.EvidenceRef.from_mapping(ref) for role, ref in raw_refs.items()}
        # This candidate never attests native/GPU origin, including native-looking
        # fixture records whose other semantic metadata happen to be consistent.
        for role, ref in refs.items():
            obj, _, _ = b._read_json(ref.verify(root), ref)
            require(type(obj) is dict and obj.get("origin") == "cpu_fixture",
                    "CPU-only paired verifier rejects native/GPU origin without actual authorization")
        b._keys(cell_geometry, ("cell_id", "load", "existing_io", "stage", "physical_bytes"),
                "unchanged original exact cell geometry")
        cell_id = cell_geometry["cell_id"]
        b._text(cell_id, "exact original cell")
        require(cell_id in dict(plan.action_operations), "cell missing frozen operation geometry")
        load = cell_geometry["load"]
        b._valid_load(load)
        require(load["active_decode"] == load["batch"] > 0 and load["prefill_tokens"] == 0,
                "selected cost load must remain positive-context ordinary pure decode")
        stage = cell_geometry["stage"]
        require(type(stage) is str and stage in b.STAGES, "one exact original physical stage required")
        size = cell_geometry["physical_bytes"]
        b._integer(size, "exact original physical bytes", 1)
        require(size % plan.context.transfer_quantum_bytes == 0
                and size // plan.context.transfer_quantum_bytes in (1, 2, 4, 8),
                "unchanged finite storage quantum geometry required")
        existing = b._mapping_vector(cell_geometry["existing_io"])
        signature = (plan.context.model_sha256, plan.context.gpu_uuid, plan.context.kv_layout_sha256,
            plan.context.kernel_mode, load["active_decode"], load["batch"], load["prefill_tokens"],
            load["context_length"], plan.context.transfer_quantum_bytes)
        geometry = b.CostCell(signature, existing, stage, size, plan.context.cost_basis, 1, 0, 0)
        bound = b.BoundCell(cell_id, geometry, tuple(sorted(refs.items())), 0, 0)
        complete_results = [self.complete.verify(root, wrapper_ref=raw_refs[role],
            selection_plan_ref=selection_plan_ref, arm=role.split("_")[0], cell_id=cell_id)
            for role in WRAPPER_ROLES]
        calculated = self._cell(root, bound, plan, check_declared=False)
        declared_bound = b.BoundCell(cell_id, calculated.cost, tuple(sorted(refs.items())),
                                   calculated.paired_runs, calculated.measured_windows)
        checked = self._cell(root, declared_bound, plan, check_declared=True)
        require(checked == calculated, "independent schema 2 analysis differs from raw recomputation")
        evidence = set(refs.values()) | {pref, plan.workload_split_ref,
            dict(plan.timing_contract)["reference_source_ref"]} | set(checked.complete_trace_refs)
        for ref in evidence:
            ref.verify(root)
        self._check_sources()
        sources = tuple((name, size, digest) for name, (size, digest) in
                        self.context_module.BASE_REFS.items())
        sources += tuple(("context_v3/" + name, size, digest)
                         for name, (size, digest) in CONTEXT_REFS.items())
        sources += (("p4_verified_cost_loader.py", *LOADER_REF),)
        return PairedContextV3CPUSemantics(cell_id, checked.cost.baseline_ns,
            checked.cost.incremental_or_joint_ns, checked.cost.uncertainty_ns,
            checked.calibration_pairs, checked.validation_pairs, checked.paired_runs,
            checked.measured_windows, checked.warmup_windows,
            sum(r.steps for r in complete_results), sum(r.full_output_tokens for r in complete_results),
            tuple((role, *row) for role, r in zip(WRAPPER_ROLES, complete_results)
                  for row in r.cold_prefill_rows),
            tuple((r.path, r.bytes, r.sha256) for r in sorted(evidence, key=lambda r: r.path)), sources)

    def require_gpu_launch(self, authorization=None):
        raise ValueError("CPU-only paired semantics has no GPU launch, native attestation or production authority")
