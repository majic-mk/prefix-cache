"""Source-bound CPU-only schema 3 complete-trace context validation.

The locked version 2 verifier remains unchanged. Only its complete-trace
function is cloned into independent globals with two exact AST changes:
trace schema 2 -> 3; row load validation -> an explicit prefill-aware helper.
Selections, selected windows, costs, loaders and GPU authority remain version 2.
"""
import ast
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys

BASE_REFS = {
    "__init__.py": [
        149,
        "0cb5d4febaf3ecb7f77d7dee56ae35f41d01300b33c108ef7f7fcb018020664c"
    ],
    "dispatch_budget.py": [
        11021,
        "412c43761cb2f59636240e15c884b25d2a1d0ee708684d496259baf2f34d667c"
    ],
    "p4_cost_table.py": [
        5071,
        "4caae858778182f5575ce2d5807bdc72520ef4c5f32a2c18f02ff2628155d0f8"
    ],
    "p4_production_table_contract.py": [
        14365,
        "ae244d09c2fc1eb077e6af8df3ae378b6b3504b4f71ff10eb26f0ada7d2c71fd"
    ],
    "p4_paired_measurement_verifier.py": [
        48136,
        "3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac"
    ]
}


def require(value, message):
    if not value:
        raise ValueError(message)


def _same_expression(node, source):
    return ast.dump(node, include_attributes=False) == ast.dump(
        ast.parse(source, mode="eval").body, include_attributes=False)


@dataclass(frozen=True)
class CompleteTraceContextV3Result:
    wrapper_ref: object
    selection_plan_ref: object
    trace_refs: tuple
    original_source_refs: tuple
    origin: str
    runs: int
    steps: int
    full_output_tokens: int
    cold_prefill_rows: tuple
    schema_version: int = 3
    context_basis: str = "pre_computed_tokens"
    status: str = "CPU_COMPLETE_TRACE_CONTEXT_V3_VERIFIED_GPU_AUTHENTICITY_UNPROVEN"
    paired_cost_verification_performed: bool = False
    runtime_hook_status: str = "not_installed"

    @property
    def gpu_verified(self):
        return False

    @property
    def production_qualified(self):
        return False


class CompleteTraceContextV3Verifier:
    """Standalone complete-trace contract; never a cost estimator or loader."""
    def __init__(self, control_source_root):
        self.control_source_root = Path(control_source_root).resolve(strict=True)
        self.package_root = self.control_source_root / "prefix_io_control"
        self._check_sources()
        tag = sha256(str(self.package_root).encode()).hexdigest()[:16]
        package_name = "_prefix_io_context_v3_locked_" + tag
        package_path = self.package_root / "__init__.py"
        if package_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(package_name, package_path,
                submodule_search_locations=[str(self.package_root)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)
        else:
            require(Path(sys.modules[package_name].__file__).resolve() == package_path,
                    "private locked package root differs")
        name = package_name + ".p4_paired_measurement_verifier"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name,
                self.package_root / "p4_paired_measurement_verifier.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        self.base = sys.modules[name]
        require(Path(self.base.__file__).resolve() ==
                self.package_root / "p4_paired_measurement_verifier.py",
                "private locked verifier source differs")
        self._complete, self.ast_patch = self._clone_complete_trace()
        self._check_sources()

    def _check_sources(self):
        for name, (size, digest) in BASE_REFS.items():
            path = self.package_root / name
            require(path.is_file() and not path.is_symlink(), "locked source absent or symlink: " + name)
            raw = path.read_bytes()
            require((len(raw), sha256(raw).hexdigest()) == (size, digest),
                    "locked original source bytes/SHA drift: " + name)

    def _valid_complete_trace_load_v3(self, row):
        """Zero is genuine only on an explicit cold-prefill complete-trace row."""
        b = self.base
        load = row["load"]
        b._keys(load, ("active_decode", "batch", "prefill_tokens", "context_length"),
                "actual schema 3 complete-step load")
        for name, value in load.items():
            b._integer(value, "actual " + name)
        if load["context_length"] >= 1:
            return b._valid_load(load)
        b._require(type(load["context_length"]) is int and load["context_length"] == 0 and
                   row.get("step_kind") == "prefill" and load["active_decode"] == 0 and
                   load["prefill_tokens"] > 0 and load["batch"] >= 1,
                   "zero prior context requires explicit actual cold prefill")
        return None

    def _clone_complete_trace(self):
        source = self.package_root / "p4_paired_measurement_verifier.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "_v2_complete_trace"]
        require(len(matches) == 1, "one exact original complete-trace function required")
        node = deepcopy(matches[0])
        before = sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        counts = dict(schema_comparison=0, complete_row_load_call=0)
        class ExactContextPatch(ast.NodeTransformer):
            def visit_Compare(self, value):
                value = self.generic_visit(value)
                if (_same_expression(value.left, "trace['schema_version']") and
                    len(value.ops) == 1 and isinstance(value.ops[0], ast.Eq) and
                    len(value.comparators) == 1 and isinstance(value.comparators[0], ast.Constant) and
                    type(value.comparators[0].value) is int and value.comparators[0].value == 2):
                    value.comparators[0] = ast.copy_location(ast.Constant(3), value.comparators[0])
                    counts["schema_comparison"] += 1
                return value
            def visit_Call(self, value):
                value = self.generic_visit(value)
                if (isinstance(value.func, ast.Name) and value.func.id == "_valid_load" and
                    len(value.args) == 1 and not value.keywords and
                    _same_expression(value.args[0], "row['load']")):
                    value.func = ast.copy_location(ast.Name("_valid_complete_trace_load_v3", ast.Load()),
                                                   value.func)
                    value.args = [ast.copy_location(ast.Name("row", ast.Load()), value.args[0])]
                    counts["complete_row_load_call"] += 1
                return value
        node = ExactContextPatch().visit(node)
        require(counts == dict(schema_comparison=1, complete_row_load_call=1),
                "unexpected original AST; refuse broad schema/load substitution")
        node.name = "_complete_trace_context_v3"
        after = sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        standalone = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        namespace = dict(self.base.__dict__)
        namespace["_valid_complete_trace_load_v3"] = self._valid_complete_trace_load_v3
        exec(compile(standalone, str(source) + "::context_v3_exact_patch", "exec"), namespace)
        return namespace[node.name], dict(counts, original_function_ast_sha256=before,
                                         cloned_function_ast_sha256=after)

    def verify(self, root, *, wrapper_ref, selection_plan_ref, arm, cell_id):
        """Read pinned schema 3 wrapper/trace refs and an unchanged strict v2 plan."""
        self._check_sources()
        b = self.base
        root = Path(root).resolve(strict=True)
        wrapper_ref = b.EvidenceRef.from_mapping(wrapper_ref)
        selection_plan_ref = b.EvidenceRef.from_mapping(selection_plan_ref)
        require(arm in ("baseline", "action"), "one explicit original arm required")
        b._text(cell_id, "exact frozen cell")
        plan = b.load_verification_plan(root, selection_plan_ref.path,
                                        expected_plan_ref=selection_plan_ref)
        b._require(type(plan) is b.PairedVerificationPlan and plan.schema_version == 2,
                   "independently pinned exact original schema 2 selection/timing plan required")
        b._require(dict(plan.timing_contract)["timing_scope"] == "full_decode_step",
                   "standalone full-step trace requires an explicit original full_decode_step timer plan")
        b._require(cell_id in dict(plan.action_operations), "cell absent from frozen plan")
        data = b._measurement(root, wrapper_ref, scope="paired_measurement_wrapper",
                              arm=arm, cell_id=cell_id, context=plan.context, version=3)
        runs = b._v2_runs(data)
        frozen = {(dict(entry)["trace_sha256"], dict(entry)["seed"]): dict(entry)
                  for entry in plan.workload_entries if dict(entry)["cell_id"] == cell_id}
        b._require(len(frozen) == len(runs), "complete runs must exactly cover frozen cell workloads")
        counts = {"calibration": [], "validation": []}
        intervals = []
        for run in runs.values():
            entry = frozen.get((run["trace_sha256"], run["seed"]))
            b._require(entry is not None and all(entry[key] == run[key] for key in entry if key != "cell_id"),
                       "complete run differs from independently frozen workload split")
            counts[run["split"]].append(run)
            intervals.append((run["start_ns"], run["end_ns"]))
        intervals.sort()
        b._require(all(a[1] <= c[0] for a, c in zip(intervals, intervals[1:])),
                   "independent complete runs overlap")
        b._require(len(counts["calibration"]) >= plan.min_calibration_pairs and
                   len(counts["validation"]) >= plan.min_validation_pairs,
                   "missing independently frozen calibration/validation runs")
        orders = [run["arm_order"] for run in counts["calibration"]]
        b._require(orders.count("AB") == orders.count("BA"), "frozen calibration AB/BA balance differs")
        selection = {item[0]: item[1:] for item in plan.selections}[cell_id]
        frozen_load, warmup, measured = selection
        trace_refs, cold = [], []
        steps = output_tokens = 0
        for pair, run in runs.items():
            b._require(run["warmup_windows"] == len(warmup) and run["measured_windows"] == len(measured),
                       "run selected counts differ from frozen plan")
            rows, ref, outputs = self._complete(root, run, data, plan)
            for offset in warmup + measured:
                b._require(offset < len(rows), "frozen selected offset outside complete trace")
                row = rows[offset]
                # This is the original strict positive-context load validator.
                # The schema 3 exception is never visible in selection/cost scope.
                b._valid_load(row["load"])
                b._require(row["load"]["active_decode"] == row["load"]["batch"] and
                           row["load"]["prefill_tokens"] == 0 and "step_kind" not in row,
                           "frozen warmup/measured selection must be actual ordinary pure decode")
                if offset in measured:
                    b._require(row["load"] == dict(frozen_load),
                               "actual measured context/load differs from unchanged frozen selection")
            selected_tokens = sum(sum(len(output["token_ids"]) for output in rows[offset]["outputs"])
                                  for offset in measured)
            b._require(run["selected_output_tokens"] == selected_tokens,
                       "selected output count differs from complete actual token IDs")
            for row in rows:
                if row["load"]["context_length"] == 0:
                    cold.append((pair, row["step_offset"], row["native_step_ordinal"]))
            trace_refs.append(ref)
            steps += len(rows)
            output_tokens += sum(len(tokens) for tokens in outputs.values())
        wrapper_ref.verify(root)
        selection_plan_ref.verify(root)
        self._check_sources()
        return CompleteTraceContextV3Result(wrapper_ref, selection_plan_ref, tuple(trace_refs),
            tuple((name, size, digest) for name, (size, digest) in BASE_REFS.items()),
            data["origin"], len(runs), steps, output_tokens, tuple(cold))

