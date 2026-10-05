"""Bounded CPU semantic recomputation; no claim of authentic GPU execution.

Raw measurement files may have a native-looking origin. Their internal
consistency is verified, but origin text, hashes and reported PASS cannot turn
this result into GPU qualification or authorize production interference.
"""
from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
import json
from .dispatch_budget import Amount, STAGES, ZERO
from .p4_cost_table import CostCell
from .p4_production_table_contract import (
    EvidenceRef, TableContext, BoundCell, BlockedProductionCandidate, TableContractError,
    _require, _keys, _text, _integer, _sha, _read_json, load_production_candidate,
)

ORIGINS = frozenset(("cpu_fixture", "native_gpu_recording"))
FORMULA = "ceil_calibration_means_plus_max_positive_residual_v1"
MAX_PAIRS = 64
MAX_WINDOWS = 4096
MAX_FULL_OUTPUT_TOKENS = 262144
TIMING_SCOPES = frozenset(("model_forward", "full_decode_step"))
MEASUREMENT_ROLES = ("baseline_wrapper", "action_wrapper",
                     "baseline_observations", "action_observations")


@dataclass(frozen=True)
class PairedVerificationPlan:
    context: TableContext
    plan_ref: EvidenceRef
    workload_split_ref: EvidenceRef
    workload_entries: tuple
    action_operations: tuple
    min_calibration_pairs: int = 2
    min_validation_pairs: int = 1
    schema_version: int = 1
    selections: tuple = ()
    timing_contract: tuple = ()

    @property
    def gpu_verified(self):
        return False


@dataclass(frozen=True)
class VerifiedCell:
    cell_id: str
    cost: CostCell
    origin: str
    calibration_pairs: int
    validation_pairs: int
    paired_runs: int
    measured_windows: int
    warmup_windows: int
    max_positive_residual_ns: int
    schema_version: int = 1
    timing_scope: str = "historical_cpu_host_fixture"
    clock_domain: str = "host_monotonic_ns_fixture"
    complete_trace_refs: tuple = ()


@dataclass(frozen=True)
class SemanticPairedVerification:
    context: TableContext
    candidate_ref: EvidenceRef
    plan_ref: EvidenceRef
    cells: tuple
    evidence_refs: tuple
    status: str = "PASS_CPU_SEMANTICS_NOT_GPU_QUALIFICATION"

    @property
    def gpu_verified(self):
        return False

    @property
    def production_qualified(self):
        return False



@dataclass(frozen=True)
class PreparedCellCandidate:
    context: TableContext
    plan_ref: EvidenceRef
    verification: VerifiedCell
    raw_refs: tuple
    analysis_ref: EvidenceRef

    @property
    def status(self):
        return "PREPARED_CPU_CELL_CANDIDATE_GPU_BLOCKED"

    @property
    def gpu_verified(self):
        return False

    @property
    def production_qualified(self):
        return False

    def to_candidate_mapping(self):
        cell = self.verification
        cost = cell.cost
        refs = dict(self.raw_refs)
        refs["pair_analysis"] = self.analysis_ref
        return {
            "cell_id": cell.cell_id,
            "load": dict(zip(("active_decode", "batch", "prefill_tokens", "context_length"),
                             cost.load_signature[4:8])),
            "existing_io": [{"ops": a.ops, "bytes": a.nbytes} for a in cost.existing_io],
            "stage": cost.stage, "physical_bytes": cost.physical_bytes,
            "baseline_ns": cost.baseline_ns,
            "incremental_or_joint_ns": cost.incremental_or_joint_ns,
            "uncertainty_ns": cost.uncertainty_ns,
            "paired_runs_reported": cell.paired_runs,
            "windows_reported": cell.measured_windows,
            "measurement_refs": {role: asdict(ref) for role, ref in refs.items()},
        }


def load_verification_plan(root, plan_path, *, expected_plan_ref):
    root = Path(root).resolve(strict=True)
    _require(type(expected_plan_ref) is EvidenceRef and expected_plan_ref.path == plan_path,
             "independently pinned plan path/ref required")
    path = expected_plan_ref.verify(root)
    data, _, _ = _read_json(path, expected_plan_ref)
    version = data.get("schema_version") if type(data) is dict else None
    _require(type(version) is int and version in (1, 2), "exact semantic plan schema 1 or 2 required")
    extra = ("selections", "timing_contract") if version == 2 else ()
    _keys(data, ("schema_version", "scope", "context", "workload_split_ref",
                 "action_operations", "min_calibration_pairs", "min_validation_pairs",
                 "max_pairs", "max_windows", "formula") + extra, "semantic plan")
    _require(data["scope"] == "p4_paired_semantic_plan" and data["formula"] == FORMULA,
             "frozen CPU semantic plan/formula required")
    context = TableContext.from_mapping(data["context"])
    _require(context.uncertainty_method == "paired_residual_margin",
             "no unspecified interval/bootstrap estimator implemented")
    split_ref = EvidenceRef.from_mapping(data["workload_split_ref"])
    split, _, _ = _read_json(split_ref.verify(root), split_ref)
    _keys(split, ("schema_version", "scope", "entries"), "frozen workload split")
    _require(type(split["schema_version"]) is int and split["schema_version"] == 1 and
             split["scope"] == "p4_frozen_measurement_split", "exact frozen workload split schema")
    entries = split["entries"]
    _require(type(entries) is list and 1 <= len(entries) <= 128 * MAX_PAIRS, "bounded frozen workload entries")
    frozen_entries = []
    identities = set()
    for entry in entries:
        _keys(entry, ("cell_id", "trace_sha256", "prefix_family_sha256", "seed", "split",
                      "workload_sha256", "input_tokens", "output_tokens", "request_count"),
              "frozen workload entry")
        _text(entry["cell_id"], "split cell")
        for key in ("trace_sha256", "prefix_family_sha256", "workload_sha256"):
            _sha(entry[key], key)
        _integer(entry["seed"], "split seed")
        _require(entry["split"] in ("calibration", "validation"), "frozen split label")
        for key in ("input_tokens", "output_tokens", "request_count"):
            _integer(entry[key], "frozen " + key, 1)
        identity = (entry["cell_id"], entry["trace_sha256"], entry["seed"])
        _require(identity not in identities, "duplicate frozen workload")
        identities.add(identity)
        frozen_entries.append(tuple(sorted(entry.items())))
    # Isolation is a property of the whole frozen plan, not only one cell.
    # Relabeling the same content with a new seed/trace/family is not independence.
    for field in ("trace_sha256", "prefix_family_sha256", "workload_sha256"):
        calibration = {e[field] for e in entries if e["split"] == "calibration"}
        validation = {e[field] for e in entries if e["split"] == "validation"}
        _require(not calibration & validation, "global calibration/margin-validation " + field + " leakage")
    for cell_id in {e["cell_id"] for e in entries}:
        cell_entries = [e for e in entries if e["cell_id"] == cell_id]
        _require(len({e["workload_sha256"] for e in cell_entries}) == len(cell_entries),
                 "same workload content is not an independent per-cell pair")
    _integer(data["min_calibration_pairs"], "calibration pairs", 2)
    _integer(data["min_validation_pairs"], "validation pairs", 1)
    _require(data["min_calibration_pairs"] + data["min_validation_pairs"] <= MAX_PAIRS,
             "pair minimum exceeds finite limit")
    _require(type(data["max_pairs"]) is int and data["max_pairs"] == MAX_PAIRS and
             type(data["max_windows"]) is int and data["max_windows"] == MAX_WINDOWS,
             "fixed bounded input limits required")
    ops = data["action_operations"]
    _require(type(ops) is dict and 1 <= len(ops) <= 128, "finite exact cell operation map")
    for name, count in ops.items():
        _text(name, "cell id")
        _integer(count, "action operations", 1)
    _require({e["cell_id"] for e in entries} == set(ops), "frozen split/operation cell map differs")
    selections, timing = _v2_plan(root, data, ops) if version == 2 else ((), ())
    expected_plan_ref.verify(root)
    split_ref.verify(root)
    return PairedVerificationPlan(context, expected_plan_ref, split_ref,
                                  tuple(frozen_entries), tuple(sorted(ops.items())),
                                  data["min_calibration_pairs"], data["min_validation_pairs"],
                                  version, selections, timing)


def _mapping_vector(obj):
    _require(type(obj) is list and len(obj) == 4, "four exact I/O stage counters required")
    result = []
    for amount in obj:
        _keys(amount, ("ops", "bytes"), "raw I/O counters")
        _integer(amount["ops"], "raw operations")
        _integer(amount["bytes"], "raw physical bytes")
        _require((amount["ops"] == 0) == (amount["bytes"] == 0),
                 "physical operation/byte counters inconsistent")
        result.append(Amount(amount["ops"], amount["bytes"]))
    return tuple(result)


def _measurement(root, ref, *, scope, arm, cell_id, context, version=1):
    path = ref.verify(root)
    _require(path.suffix == ".json", "semantic v1 requires bounded JSON objects, not loose JSONL")
    data, _, _ = _read_json(path, ref)
    payload = "runs" if scope == "paired_measurement_wrapper" else "windows"
    _keys(data, ("schema_version", "scope", "origin", "context", "arm", "cell_id", payload),
          "raw " + scope)
    _require(type(data["schema_version"]) is int and data["schema_version"] == version and
             data["scope"] == scope, "raw measurement scope/schema mismatch")
    _require(type(data["origin"]) is str and data["origin"] in ORIGINS, "explicit origin required")
    _require(data["arm"] == arm and data["cell_id"] == cell_id, "raw measurement arm/cell mismatch")
    _require(TableContext.from_mapping(data["context"]) == context, "raw measurement context drift")
    values = data[payload]
    limit = MAX_PAIRS if payload == "runs" else MAX_WINDOWS
    _require(type(values) is list and 1 <= len(values) <= limit, "bounded nonempty raw observations")
    return data


def _runs(data):
    if data["schema_version"] == 2:
        return _v2_runs(data)
    runs = {}
    for run in data["runs"]:
        _keys(run, ("pair_id", "trace_sha256", "prefix_family_sha256", "seed", "split",
                    "arm_order", "warmup_windows", "measured_windows", "input_tokens",
                    "output_tokens", "request_count", "workload_sha256", "start_ns",
                    "end_ns", "exit_code", "accepted_io_drained", "completed_new_io"),
              "paired run")
        _text(run["pair_id"], "pair identity")
        _require(run["pair_id"] not in runs, "duplicate run pair")
        for key in ("trace_sha256", "prefix_family_sha256", "workload_sha256"):
            _sha(run[key], key)
        _integer(run["seed"], "trace seed")
        _require(run["split"] in ("calibration", "validation") and
                 run["arm_order"] in ("AB", "BA"), "explicit split and paired arm order required")
        for key in ("warmup_windows", "measured_windows", "input_tokens",
                    "request_count", "output_tokens"):
            _integer(run[key], key, 1)
        _require(run["output_tokens"] >= 2, "sustained decode work requires two output tokens")
        for key in ("start_ns", "end_ns"):
            _integer(run[key], key)
        _require(run["end_ns"] > run["start_ns"], "nonempty monotonic run interval")
        _require(type(run["exit_code"]) is int and run["exit_code"] == 0 and
                 run["accepted_io_drained"] is True, "successful full accepted-I/O drain required")
        _mapping_vector(run["completed_new_io"])
        runs[run["pair_id"]] = run
    return runs


def _windows(data, runs, cost, plan, action):
    windows = {}
    timeline = {}
    expected_load = dict(zip(("active_decode", "batch", "prefill_tokens", "context_length"),
                             cost.load_signature[4:8]))
    _require(0 < expected_load["active_decode"] <= expected_load["batch"],
             "active decode count must fit the actual execution batch")
    ops = dict(plan.action_operations)[data["cell_id"]]
    _require(ops <= cost.physical_bytes // plan.context.transfer_quantum_bytes,
             "frozen physical operation count exceeds transfer units")
    new = list(ZERO)
    if action:
        new[STAGES.index(cost.stage)] = Amount(ops, cost.physical_bytes)
    expected_new = tuple(new)
    expected_existing = cost.existing_io if action or cost.basis == "existing_io_plus_delta" else ZERO
    for row in data["windows"]:
        _keys(row, ("pair_id", "window_id", "phase", "start_ns", "end_ns", "load",
                    "existing_io", "new_io", "output_tokens"), "paired raw window")
        _text(row["window_id"], "window id")
        pair = row["pair_id"]
        _require(type(pair) is str and pair in runs, "window references unknown pair")
        _require(row["phase"] in ("warmup", "measured"), "explicit warmup/measured window")
        key = (pair, row["window_id"], row["phase"])
        _require(key not in windows, "duplicate pair window")
        _integer(row["start_ns"], "step start")
        _integer(row["end_ns"], "step end")
        _require(runs[pair]["start_ns"] <= row["start_ns"] < row["end_ns"] <= runs[pair]["end_ns"],
                 "window must be a positive interval inside its actual run")
        _keys(row["load"], expected_load, "raw load")
        for name, value in row["load"].items():
            _integer(value, "load " + name)
        _require(row["load"] == expected_load, "window load mismatch")
        _require(_mapping_vector(row["existing_io"]) == expected_existing,
                 "baseline existing-I/O basis/action drift")
        _require(_mapping_vector(row["new_io"]) == expected_new,
                 "raw extra action must exactly match one frozen physical stage")
        _integer(row["output_tokens"], "window output tokens", 1)
        windows[key] = row
        timeline.setdefault(pair, []).append(row)
    for pair, run in runs.items():
        rows = sorted(timeline.get(pair, ()), key=lambda row: row["start_ns"])
        _require(all(a["end_ns"] <= b["start_ns"] for a, b in zip(rows, rows[1:])),
                 "overlapping/double-counted step windows")
        phases = [row["phase"] for row in rows]
        _require(phases == ["warmup"] * run["warmup_windows"] +
                 ["measured"] * run["measured_windows"], "warmup not excluded before measurement")
        _require(sum(row["output_tokens"] for row in rows if row["phase"] == "measured")
                 == run["output_tokens"], "wrapper/token work count mismatch")
        completed = tuple(Amount(expected_new[i].ops * len(rows),
                                 expected_new[i].nbytes * len(rows)) for i in range(4))
        _require(_mapping_vector(run["completed_new_io"]) == completed,
                 "accepted action physical bytes not fully completed/drained")
    return windows


def _ceil_mean(values):
    _require(bool(values), "nonempty calibration values required")
    return -(-sum(values) // len(values))


def _verify_cell(root, bound, plan, *, check_declared=True):
    refs = dict(bound.measurement_refs)
    raw = {}
    for role in MEASUREMENT_ROLES:
        arm = "baseline" if role.startswith("baseline") else "action"
        scope = "paired_measurement_wrapper" if role.endswith("wrapper") else "paired_window_observations"
        raw[role] = _measurement(root, refs[role], scope=scope, arm=arm,
                                 cell_id=bound.cell_id, context=plan.context, version=plan.schema_version)
    origins = {obj["origin"] for obj in raw.values()}
    _require(len(origins) == 1, "mixed measurement origins")
    baseline_runs = _runs(raw["baseline_wrapper"])
    action_runs = _runs(raw["action_wrapper"])
    _require(set(baseline_runs) == set(action_runs), "incomplete baseline/action pairing")
    frozen = { (dict(e)["trace_sha256"], dict(e)["seed"]): dict(e) for e in plan.workload_entries
               if dict(e)["cell_id"] == bound.cell_id }
    _require(len(frozen) == len(baseline_runs), "raw runs must exactly cover frozen cell workloads")
    by_split = {"calibration": [], "validation": []}
    run_intervals = []
    for pair, baseline in baseline_runs.items():
        action = action_runs[pair]
        entry = frozen.get((baseline["trace_sha256"], baseline["seed"]))
        _require(entry is not None and all(entry[k] == baseline[k] for k in entry if k != "cell_id"),
                 "raw workload differs from independently frozen split")
        matching = set(baseline) - {"start_ns", "end_ns", "completed_new_io",
                    "accepted_new_io", "complete_trace_ref", "first_step_ordinal"}
        _require(all(baseline[k] == action[k] for k in matching), "paired seed/trace/work/order mismatch")
        first, second = (baseline, action) if baseline["arm_order"] == "AB" else (action, baseline)
        _require(first["end_ns"] <= second["start_ns"], "paired execution order/time mismatch")
        run_intervals.extend((r["start_ns"], r["end_ns"]) for r in (baseline, action))
        by_split[baseline["split"]].append(baseline)
    intervals = sorted(run_intervals)
    _require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
             "independent paired runs cannot overlap")
    _require(len(by_split["calibration"]) >= plan.min_calibration_pairs and
             len(by_split["validation"]) >= plan.min_validation_pairs,
             "missing independent calibration/held-out validation pairs")
    orders = [r["arm_order"] for r in by_split["calibration"]]
    _require(orders.count("AB") == orders.count("BA"),
             "calibration paired order must be equally balanced AB/BA")
    _require(len({(r["trace_sha256"], r["seed"]) for r in baseline_runs.values()})
             == len(baseline_runs), "duplicate independent trace/seed pair")
    for field in ("trace_sha256", "prefix_family_sha256", "workload_sha256"):
        _require(not ({r[field] for r in by_split["calibration"]} &
                      {r[field] for r in by_split["validation"]}),
                 "calibration/held-out trace or prefix-family leakage")
    extra_refs = ()
    if plan.schema_version == 2:
        base, base_refs, base_outputs = _v2_windows(root, raw["baseline_observations"], baseline_runs, bound.cost, plan, False)
        action, action_refs, action_outputs = _v2_windows(root, raw["action_observations"], action_runs, bound.cost, plan, True)
        _require(base_outputs == action_outputs, "paired complete output token IDs differ")
        extra_refs = base_refs + action_refs
    else:
        base = _windows(raw["baseline_observations"], baseline_runs, bound.cost, plan, False)
        action = _windows(raw["action_observations"], action_runs, bound.cost, plan, True)
    _require(set(base) == set(action), "missing/misaligned paired windows")
    pair_means = {}
    action_step_durations = []
    measured = 0
    warmup = 0
    for pair, run in baseline_runs.items():
        durations_base = []
        durations_action = []
        for key in (key for key in base if key[0] == pair):
            a, b = base[key], action[key]
            _require(a["output_tokens"] == b["output_tokens"], "paired token work differs")
            if key[2] == "warmup":
                warmup += 1
                continue
            durations_base.append(_duration_ns(a, plan))
            durations_action.append(_duration_ns(b, plan))
            action_step_durations.append(_duration_ns(b, plan))
            measured += 1
        # Run means retain trace/seed as the independent statistical unit.
        pair_means[pair] = (_ceil_mean(durations_base), _ceil_mean(durations_action))
    calibration = [pair_means[r["pair_id"]] for r in by_split["calibration"]]
    baseline_ns = _ceil_mean([a for a, _ in calibration])
    incremental = max(0, _ceil_mean([b - a for a, b in calibration]))
    residual = max(0, max(b - baseline_ns - incremental for b in action_step_durations))
    cost = CostCell(bound.cost.load_signature, bound.cost.existing_io,
                    bound.cost.stage, bound.cost.physical_bytes, bound.cost.basis,
                    baseline_ns, incremental, residual)
    result = VerifiedCell(bound.cell_id, cost, next(iter(origins)),
                        len(by_split["calibration"]), len(by_split["validation"]),
                        len(baseline_runs), measured, warmup, residual,
                        plan.schema_version,
                        dict(plan.timing_contract)["timing_scope"] if plan.schema_version == 2 else "historical_cpu_host_fixture",
                        "cuda_event_elapsed" if plan.schema_version == 2 else "host_monotonic_ns_fixture",
                        tuple(extra_refs))
    if not check_declared:
        return result
    _require(cost == bound.cost and bound.paired_runs_reported == len(baseline_runs) and
             bound.windows_reported == measured, "candidate costs/counts do not equal raw recomputation")
    ref = refs["pair_analysis"]
    report, _, _ = _read_json(ref.verify(root), ref)
    _keys(report, ("schema_version", "scope", "origin", "context", "cell_id", "plan_ref",
                  "evidence_refs", "formula", "baseline_ns", "incremental_or_joint_ns",
                  "uncertainty_ns", "calibration_pairs", "validation_pairs",
                  "paired_runs_reported", "windows_reported"), "pair analysis")
    _require(type(report["schema_version"]) is int and report["schema_version"] == plan.schema_version and
             report["scope"] == "p4_paired_analysis_candidate" and report["formula"] == FORMULA,
             "analysis does not identify exact recomputation method")
    _require(report["origin"] == next(iter(origins)) and report["cell_id"] == bound.cell_id and
             TableContext.from_mapping(report["context"]) == plan.context,
             "analysis origin/cell/context mismatch")
    _require(EvidenceRef.from_mapping(report["plan_ref"]) == plan.plan_ref, "analysis plan drift")
    _keys(report["evidence_refs"], MEASUREMENT_ROLES, "analysis evidence chain")
    _require(all(EvidenceRef.from_mapping(report["evidence_refs"][role]) == refs[role]
                 for role in MEASUREMENT_ROLES), "analysis binds other raw evidence")
    expected_counts = {"baseline_ns": baseline_ns, "incremental_or_joint_ns": incremental,
        "uncertainty_ns": residual, "calibration_pairs": len(by_split["calibration"]),
        "validation_pairs": len(by_split["validation"]),
        "paired_runs_reported": len(baseline_runs), "windows_reported": measured}
    for key, value in expected_counts.items():
        _integer(report[key], key)
        _require(report[key] == value, "analysis arithmetic/count differs from raw evidence")
    return result



def build_paired_cell_candidate(root, *, cell_geometry, raw_refs, expected_plan, analysis_path):
    """Recompute one CPU-prepared cell from only four complete raw roles.

    No caller-supplied reported timing/count is accepted. This is preparation,
    not a GPU-origin attestation. The full candidate/qualification loader must
    independently rebind all five roles after the caller assembles that chain.
    """
    root = Path(root).resolve(strict=True)
    _require(type(expected_plan) is PairedVerificationPlan, "exact independently pinned plan required")
    pinned = load_verification_plan(root, expected_plan.plan_ref.path,
                                    expected_plan_ref=expected_plan.plan_ref)
    _require(pinned == expected_plan, "prepared plan fields differ from actual plan bytes")
    _keys(cell_geometry, ("cell_id", "load", "existing_io", "stage", "physical_bytes"),
          "prepared cell geometry")
    cell_id = cell_geometry["cell_id"]
    _text(cell_id, "prepared cell id")
    _require(cell_id in dict(pinned.action_operations), "cell missing frozen operation geometry")
    load = cell_geometry["load"]
    _keys(load, ("active_decode", "batch", "prefill_tokens", "context_length"), "prepared cell load")
    for name, value in load.items():
        _integer(value, name, 1 if name in ("active_decode", "batch", "context_length") else 0)
    _require(load["active_decode"] <= load["batch"], "active decode exceeds execution batch")
    stage = cell_geometry["stage"]
    _require(type(stage) is str and stage in STAGES, "one exact physical stage required")
    physical_bytes = cell_geometry["physical_bytes"]
    _integer(physical_bytes, "prepared physical bytes", 1)
    context = pinned.context
    _require(physical_bytes % context.transfer_quantum_bytes == 0 and
             physical_bytes // context.transfer_quantum_bytes in (1, 2, 4, 8),
             "exact legal storage-unit batch required")
    existing = _mapping_vector(cell_geometry["existing_io"])
    signature = (context.model_sha256, context.gpu_uuid, context.kv_layout_sha256,
        context.kernel_mode, load["active_decode"], load["batch"], load["prefill_tokens"],
        load["context_length"], context.transfer_quantum_bytes)
    # The dummy numbers are never checked, emitted or used as a prediction.
    # This CostCell carries only the frozen geometry into the sole calculator.
    geometry = CostCell(signature, existing, stage, physical_bytes, context.cost_basis, 1, 0, 0)
    _keys(raw_refs, MEASUREMENT_ROLES, "four prepared raw roles")
    _require(all(type(ref) is EvidenceRef for ref in raw_refs.values()), "typed raw byte refs required")
    refs = tuple(sorted(raw_refs.items()))
    all_refs = tuple(raw_refs.values()) + (pinned.plan_ref, pinned.workload_split_ref)
    for ref in all_refs:
        ref.verify(root)
    bound = BoundCell(cell_id, geometry, refs, 0, 0)
    result = _verify_cell(root, bound, pinned, check_declared=False)
    cost = result.cost
    all_refs += result.complete_trace_refs
    if pinned.schema_version == 2:
        all_refs += (dict(pinned.timing_contract)["reference_source_ref"],)
    analysis = {
        "schema_version": pinned.schema_version, "scope": "p4_paired_analysis_candidate", "origin": result.origin,
        "context": asdict(context), "cell_id": cell_id, "plan_ref": asdict(pinned.plan_ref),
        "evidence_refs": {role: asdict(ref) for role, ref in refs}, "formula": FORMULA,
        "baseline_ns": cost.baseline_ns, "incremental_or_joint_ns": cost.incremental_or_joint_ns,
        "uncertainty_ns": cost.uncertainty_ns, "calibration_pairs": result.calibration_pairs,
        "validation_pairs": result.validation_pairs, "paired_runs_reported": result.paired_runs,
        "windows_reported": result.measured_windows,
    }
    analysis_ref = _write_json_evidence(root, analysis, analysis_path, all_refs)
    return PreparedCellCandidate(context, pinned.plan_ref, result, refs, analysis_ref)


def verify_paired_measurements(root, bound_candidate, *, expected_plan):
    root = Path(root).resolve(strict=True)
    _require(type(bound_candidate) is BlockedProductionCandidate and
             type(expected_plan) is PairedVerificationPlan, "bound candidate and frozen plan required")
    pinned = load_verification_plan(root, expected_plan.plan_ref.path,
                                    expected_plan_ref=expected_plan.plan_ref)
    _require(pinned == expected_plan, "semantic plan fields differ from actual frozen plan bytes")
    rebound = load_production_candidate(root, bound_candidate.candidate_ref.path,
        expected_context=expected_plan.context, qualification_ref=bound_candidate.qualification_ref,
        expected_verifier_ref=bound_candidate.expected_verifier_ref)
    _require(rebound == bound_candidate, "bound candidate fields differ from actual candidate bytes")
    _require(bound_candidate.context == expected_plan.context, "candidate/semantic plan context mismatch")
    _require({c.cell_id for c in bound_candidate.cells} ==
             {name for name, _ in expected_plan.action_operations}, "exact cell operation geometry required")
    for c in bound_candidate.cells:
        _require(c.cost.physical_bytes // expected_plan.context.transfer_quantum_bytes in (1, 2, 4, 8),
                 "finite legal storage-unit candidate batch required")
    refs = {bound_candidate.candidate_ref, bound_candidate.qualification_ref,
            bound_candidate.expected_verifier_ref, expected_plan.plan_ref, expected_plan.workload_split_ref}
    refs.update(ref for cell in bound_candidate.cells for _, ref in cell.measurement_refs)
    for ref in refs:
        ref.verify(root)
    cells = tuple(_verify_cell(root, cell, expected_plan) for cell in bound_candidate.cells)
    refs.update(ref for cell in cells for ref in cell.complete_trace_refs)
    if expected_plan.schema_version == 2:
        refs.add(dict(expected_plan.timing_contract)["reference_source_ref"])
    # Reverify every raw byte after arithmetic; no stale values/new source hash.
    for ref in refs:
        ref.verify(root)
    return SemanticPairedVerification(expected_plan.context, bound_candidate.candidate_ref,
        expected_plan.plan_ref, cells, tuple(sorted(refs, key=lambda ref: ref.path)))


def semantic_manifest(verification):
    _require(type(verification) is SemanticPairedVerification, "exact CPU semantic result required")
    return {
        "schema_version": 1, "scope": "p4_cpu_semantic_recomputation",
        "status": verification.status, "gpu_verified": False, "production_qualified": False,
        "gpu_execution_proved": False, "method_is_confidence_interval": False,
        "validation_role": "margin_validation_not_independent_effect_evaluation",
        "formula": FORMULA, "context": asdict(verification.context),
        "candidate_ref": asdict(verification.candidate_ref), "plan_ref": asdict(verification.plan_ref),
        "cells": [asdict(c) for c in verification.cells],
        "evidence_refs": [asdict(ref) for ref in verification.evidence_refs],
        "remaining_gate": "trusted_real_native_GPU_execution_and_control_qualification",
    }


def _write_json_evidence(root, data, relative_path, source_refs):
    root = Path(root).resolve(strict=True)
    _require(type(relative_path) is str and 0 < len(relative_path) <= 512 and
             "\\" not in relative_path and "\x00" not in relative_path, "bounded POSIX output path")
    pure = PurePosixPath(relative_path)
    _require(not pure.is_absolute() and ":" not in pure.parts[0] and
             all(p not in ("", ".", "..") for p in relative_path.split("/")), "unsafe manifest path")
    path = root
    for part in pure.parts:
        path = path / part
        _require(not path.is_symlink(), "symlink output path rejected")
    _require(not path.exists(), "CPU evidence manifest is append-only")
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(root in path.parent.resolve().parents or path.parent.resolve() == root, "output outside root")
    for ref in source_refs:
        ref.verify(root)
    raw = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    for ref in source_refs:
        ref.verify(root)
    return EvidenceRef(relative_path, len(raw), sha256(raw).hexdigest())


def write_semantic_verification_manifest(root, verification, relative_path):
    data = semantic_manifest(verification)
    return _write_json_evidence(root, data, relative_path, verification.evidence_refs)


def _v2_plan(root, data, operations):
    """Version 2 is finite, preregistered selection, never GPU qualification."""
    timing = data["timing_contract"]
    _keys(timing, ("timing_scope", "clock_domain", "reference_source_ref",
                  "reference_valid", "fallback_used", "clock_domain_valid"), "frozen timing contract")
    _require(timing["timing_scope"] in TIMING_SCOPES and
             timing["clock_domain"] == "cuda_event_elapsed" and
             timing["reference_valid"] is True and timing["fallback_used"] is False and
             timing["clock_domain_valid"] is True, "explicit valid GPU elapsed duration contract required")
    source_ref = EvidenceRef.from_mapping(timing["reference_source_ref"])
    source_ref.verify(root)
    selections = data["selections"]
    _keys(selections, operations, "exact frozen selection cell map")
    frozen = []
    for cell_id, selection in selections.items():
        _keys(selection, ("load", "warmup_step_offsets", "measured_step_offsets"),
              "frozen step selection")
        _valid_load(selection["load"])
        warmup, measured = selection["warmup_step_offsets"], selection["measured_step_offsets"]
        for values in (warmup, measured):
            _require(type(values) is list and 1 <= len(values) <= MAX_WINDOWS,
                     "finite nonempty preregistered step offsets required")
            for ordinal in values:
                _integer(ordinal, "frozen step offset")
                _require(ordinal < MAX_WINDOWS, "step offset exceeds bounded complete trace")
            _require(values == sorted(set(values)), "selected step offsets must be ordered and unique")
        _require(max(warmup) < min(measured), "warmup selection must precede measured selection")
        frozen.append((cell_id, tuple(sorted(selection["load"].items())),
                       tuple(warmup), tuple(measured)))
    pinned_timing = dict(timing)
    pinned_timing["reference_source_ref"] = source_ref
    return tuple(sorted(frozen)), tuple(sorted(pinned_timing.items()))


def _valid_load(load):
    _keys(load, ("active_decode", "batch", "prefill_tokens", "context_length"), "actual step load")
    for name, value in load.items():
        _integer(value, "actual " + name)
    _require(load["batch"] >= 1 and 0 <= load["active_decode"] <= load["batch"] and
             load["context_length"] >= 1, "actual decode/batch/context geometry invalid")


def _v2_timing(row, plan):
    timing = row["timing"]
    _keys(timing, ("timing_scope", "clock_domain", "reference_source_ref",
                  "reference_valid", "fallback_used", "clock_domain_valid", "gpu_elapsed_ns"),
          "actual step timing")
    expected = dict(plan.timing_contract)
    for name in ("timing_scope", "clock_domain", "reference_valid", "fallback_used", "clock_domain_valid"):
        _require(type(timing[name]) is type(expected[name]) and timing[name] == expected[name],
                 "actual timing scope/domain/reference/fallback differs from frozen plan")
    _require(EvidenceRef.from_mapping(timing["reference_source_ref"]) == expected["reference_source_ref"],
             "actual timing source differs from frozen byte reference")
    _integer(timing["gpu_elapsed_ns"], "GPU event elapsed duration", 1)
    return timing["gpu_elapsed_ns"]


def _duration_ns(row, plan):
    # V1 remains a historical CPU host-duration fixture contract.
    # V2 host boundaries only establish run membership/order. They never
    # manufacture a GPU time axis or compare host owner events to GPU events.
    return _v2_timing(row, plan) if plan.schema_version == 2 else row["end_ns"] - row["start_ns"]


def _v2_runs(data):
    runs = {}
    for run in data["runs"]:
        _keys(run, ("pair_id", "trace_sha256", "prefix_family_sha256", "seed", "split",
                    "arm_order", "warmup_windows", "measured_windows", "input_tokens",
                    "output_tokens", "request_count", "workload_sha256", "start_ns",
                    "end_ns", "exit_code", "accepted_io_drained", "completed_new_io",
                    "accepted_new_io", "complete_trace_ref", "first_step_ordinal",
                    "full_steps", "full_output_tokens", "selected_output_tokens"), "version 2 paired run")
        _text(run["pair_id"], "pair identity")
        _require(run["pair_id"] not in runs, "duplicate run pair")
        for key in ("trace_sha256", "prefix_family_sha256", "workload_sha256"):
            _sha(run[key], key)
        _integer(run["seed"], "trace seed")
        _require(run["split"] in ("calibration", "validation") and run["arm_order"] in ("AB", "BA"),
                 "explicit split and paired arm order required")
        for key in ("warmup_windows", "measured_windows", "input_tokens", "request_count",
                    "output_tokens", "full_steps", "full_output_tokens", "selected_output_tokens"):
            _integer(run[key], key, 1)
        _integer(run["first_step_ordinal"], "actual first native step ordinal")
        _require(run["output_tokens"] == run["full_output_tokens"] >= 2 and
                 run["full_output_tokens"] <= MAX_FULL_OUTPUT_TOKENS and
                 run["full_steps"] <= MAX_WINDOWS and
                 run["selected_output_tokens"] <= run["full_output_tokens"],
                 "full actual output and finite selected counts must remain separate")
        _require(run["request_count"] <= 128, "bounded actual request count")
        for key in ("start_ns", "end_ns"):
            _integer(run[key], key)
        _require(run["end_ns"] > run["start_ns"], "positive host-monotonic run interval")
        _require(type(run["exit_code"]) is int and run["exit_code"] == 0 and
                 run["accepted_io_drained"] is True, "successful full accepted-I/O drain required")
        accepted = _mapping_vector(run["accepted_new_io"])
        _require(_mapping_vector(run["completed_new_io"]) == accepted,
                 "all actual accepted extra I/O must be fully completed")
        EvidenceRef.from_mapping(run["complete_trace_ref"])
        runs[run["pair_id"]] = run
    _require(sum(r["full_steps"] for r in runs.values()) <= MAX_WINDOWS and
             sum(r["full_output_tokens"] for r in runs.values()) <= MAX_FULL_OUTPUT_TOKENS,
             "whole cell complete trace/output exceeds finite bounds")
    return runs


def _v2_outputs(outputs, request_count):
    _require(type(outputs) is list and len(outputs) <= request_count, "bounded sample output requests")
    result = {}
    total = 0
    for output in outputs:
        _keys(output, ("request_id", "token_ids"), "actual sample token output")
        _text(output["request_id"], "sample request id")
        _require(output["request_id"] not in result, "duplicate sample request id")
        tokens = output["token_ids"]
        _require(type(tokens) is list and len(tokens) <= MAX_FULL_OUTPUT_TOKENS,
                 "bounded actual token IDs")
        for token in tokens:
            _integer(token, "actual token id")
        result[output["request_id"]] = tokens
        total += len(tokens)
    _require(total <= MAX_FULL_OUTPUT_TOKENS, "bounded actual output token sum")
    return result, total


def _v2_complete_trace(root, run, data, plan):
    ref = EvidenceRef.from_mapping(run["complete_trace_ref"])
    trace, _, _ = _read_json(ref.verify(root), ref)
    _keys(trace, ("schema_version", "scope", "origin", "context", "arm", "cell_id",
                  "pair_id", "steps", "outputs", "outside_window_new_io",
                  "accepted_new_io", "completed_new_io", "accepted_io_drained"),
          "complete actual step/output trace")
    _require(type(trace["schema_version"]) is int and trace["schema_version"] == 2 and
             trace["scope"] == "p4_complete_step_output_trace" and
             trace["origin"] == data["origin"] and trace["arm"] == data["arm"] and
             trace["cell_id"] == data["cell_id"] and trace["pair_id"] == run["pair_id"] and
             TableContext.from_mapping(trace["context"]) == plan.context,
             "complete trace version/origin/context/arm/cell/pair differs")
    rows = trace["steps"]
    _require(type(rows) is list and len(rows) == run["full_steps"] and len(rows) <= MAX_WINDOWS,
             "complete trace must include every actual step")
    full_outputs, full_total = _v2_outputs(trace["outputs"], run["request_count"])
    _require(len(full_outputs) == run["request_count"] and full_total == run["full_output_tokens"],
             "complete raw output must cover all actual request token IDs/counts")
    reconstructed = {name: [] for name in full_outputs}
    io_total = list(_mapping_vector(trace["outside_window_new_io"]))
    last_end = run["start_ns"]
    for offset, row in enumerate(rows):
        optional_kind = ("step_kind",) if type(row) is dict and "step_kind" in row else ()
        _keys(row, ("step_offset", "native_step_ordinal", "start_ns", "end_ns", "load",
                    "existing_io", "new_io", "outputs", "timing") + optional_kind, "complete actual step")
        _require(type(row["step_offset"]) is int and row["step_offset"] == offset and
                 type(row["native_step_ordinal"]) is int and
                 row["native_step_ordinal"] == run["first_step_ordinal"] + offset,
                 "actual step ordinal missing/reordered/duplicated")
        for key in ("start_ns", "end_ns"):
            _integer(row[key], "actual host " + key)
        _require(last_end <= row["start_ns"] < row["end_ns"] <= run["end_ns"],
                 "full trace host windows must be ordered within actual run")
        last_end = row["end_ns"]
        _valid_load(row["load"])
        _require(row["load"]["batch"] <= run["request_count"],
                 "actual trace batch cannot exceed complete run request count")
        pure_decode = (row["load"]["active_decode"] == row["load"]["batch"] and
                       row["load"]["prefill_tokens"] == 0)
        _require((pure_decode and not optional_kind) or
                 (row.get("step_kind") == "prefill" and row["load"]["active_decode"] == 0 and
                  row["load"]["prefill_tokens"] > 0),
                 "non-pure trace requires explicit actual prefill kind; mixed/spec/dummy/async unknown unsupported")
        _mapping_vector(row["existing_io"])
        extra = _mapping_vector(row["new_io"])
        if data["arm"] == "baseline":
            _require(extra == ZERO, "baseline cannot contain experimental extra I/O")
        for i in range(4):
            io_total[i] = Amount(io_total[i].ops + extra[i].ops, io_total[i].nbytes + extra[i].nbytes)
        outputs, count = _v2_outputs(row["outputs"], run["request_count"])
        _require(all(len(tokens) <= 1 for tokens in outputs.values()) and count <= row["load"]["batch"],
                 "non-speculative actual forward permits at most one output token per request/batch")
        _require(not pure_decode or count == row["load"]["active_decode"],
                 "every actual pure-decode forward must include one sampled token per active request")
        _require(set(outputs) <= set(full_outputs), "step output references unknown actual request")
        for name, tokens in outputs.items():
            reconstructed[name].extend(tokens)
        _v2_timing(row, plan)
    _require(reconstructed == full_outputs, "step sample token IDs do not reconstruct complete raw outputs")
    _require(trace["accepted_io_drained"] is True and
             _mapping_vector(trace["accepted_new_io"]) == tuple(io_total) and
             _mapping_vector(trace["completed_new_io"]) == tuple(io_total) and
             _mapping_vector(run["accepted_new_io"]) == tuple(io_total) and
             _mapping_vector(run["completed_new_io"]) == tuple(io_total),
             "complete run additional accepted I/O totals/drain are inconsistent")
    if data["arm"] == "baseline":
        _require(tuple(io_total) == ZERO, "baseline outside-window experimental I/O forbidden")
    ref.verify(root)
    return rows, ref, full_outputs


def _v2_windows(root, data, runs, cost, plan, action):
    selection = {item[0]: item[1:] for item in plan.selections}[data["cell_id"]]
    frozen_load, warmup_offsets, measured_offsets = selection
    expected_load = dict(zip(("active_decode", "batch", "prefill_tokens", "context_length"),
                             cost.load_signature[4:8]))
    _require(expected_load == dict(frozen_load) and
             0 < expected_load["active_decode"] == expected_load["batch"] and
             expected_load["prefill_tokens"] == 0,
             "candidate load differs from preregistered pure-decode exact cell")
    ops = dict(plan.action_operations)[data["cell_id"]]
    _require(ops <= cost.physical_bytes // plan.context.transfer_quantum_bytes,
             "frozen physical operation count exceeds transfer units")
    expected_new = list(ZERO)
    if action:
        expected_new[STAGES.index(cost.stage)] = Amount(ops, cost.physical_bytes)
    expected_new = tuple(expected_new)
    expected_existing = cost.existing_io if action or cost.basis == "existing_io_plus_delta" else ZERO
    full = {}
    outputs_by_pair = {}
    extra_refs = []
    for pair, run in runs.items():
        _require(run["warmup_windows"] == len(warmup_offsets) and
                 run["measured_windows"] == len(measured_offsets),
                 "wrapper selected counts differ from preregistered offsets")
        rows, ref, outputs = _v2_complete_trace(root, run, data, plan)
        _require(max(measured_offsets) < len(rows), "preregistered selection outside complete run")
        full[pair] = rows
        outputs_by_pair[pair] = outputs
        extra_refs.append(ref)
    windows = {}
    selected_tokens = {pair: 0 for pair in runs}
    for row in data["windows"]:
        _keys(row, ("pair_id", "window_id", "phase", "step_offset", "native_step_ordinal",
                    "start_ns", "end_ns", "load", "existing_io", "new_io", "output_tokens",
                    "timing"), "version 2 selected raw window")
        pair, offset = row["pair_id"], row["step_offset"]
        _require(type(pair) is str and pair in runs, "selected window references unknown pair")
        _integer(offset, "selected actual step offset")
        _require(offset in warmup_offsets or offset in measured_offsets, "unregistered measured selection")
        phase = "warmup" if offset in warmup_offsets else "measured"
        _require(row["phase"] == phase and row["window_id"] == "step-" + str(offset),
                 "selected phase/window identity differs from frozen offset")
        key = (pair, row["window_id"], phase)
        _require(key not in windows, "duplicate selected actual step")
        actual = full[pair][offset]
        for name in ("native_step_ordinal", "start_ns", "end_ns"):
            _integer(row[name], "selected " + name)
        _valid_load(row["load"])
        _mapping_vector(row["existing_io"])
        _mapping_vector(row["new_io"])
        _v2_timing(row, plan)
        _require(all(row[name] == actual[name] for name in
                     ("step_offset", "native_step_ordinal", "start_ns", "end_ns", "load",
                      "existing_io", "new_io", "timing")), "selected window is not exact full trace projection")
        _, count = _v2_outputs(actual["outputs"], runs[pair]["request_count"])
        _require(type(row["output_tokens"]) is int and row["output_tokens"] == count,
                 "selected output count differs from actual sample token IDs")
        if phase == "measured":
            _require(actual["load"] == expected_load and count == expected_load["active_decode"],
                     "measured non-speculative pure-decode output count must equal actual active batch")
            _require(_mapping_vector(actual["existing_io"]) == expected_existing,
                     "selected existing-I/O basis/action drift")
            _require(_mapping_vector(actual["new_io"]) == expected_new,
                     "selected extra action must exactly match one frozen physical stage")
            selected_tokens[pair] += count
        windows[key] = row
    required = {(pair, "step-" + str(offset), "warmup" if offset in warmup_offsets else "measured")
                for pair in runs for offset in warmup_offsets + measured_offsets}
    _require(set(windows) == required, "selected observations must cover exactly every preregistered offset")
    for pair, count in selected_tokens.items():
        _require(runs[pair]["selected_output_tokens"] == count,
                 "wrapper selected output must not replace complete output count")
    return windows, tuple(extra_refs), outputs_by_pair
