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
    EvidenceRef, TableContext, BlockedProductionCandidate, TableContractError,
    _require, _keys, _text, _integer, _sha, _read_json, load_production_candidate,
)

ORIGINS = frozenset(("cpu_fixture", "native_gpu_recording"))
FORMULA = "ceil_calibration_means_plus_max_positive_residual_v1"
MAX_PAIRS = 64
MAX_WINDOWS = 4096
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


def load_verification_plan(root, plan_path, *, expected_plan_ref):
    root = Path(root).resolve(strict=True)
    _require(type(expected_plan_ref) is EvidenceRef and expected_plan_ref.path == plan_path,
             "independently pinned plan path/ref required")
    path = expected_plan_ref.verify(root)
    data, _, _ = _read_json(path, expected_plan_ref)
    _keys(data, ("schema_version", "scope", "context", "workload_split_ref",
                 "action_operations", "min_calibration_pairs", "min_validation_pairs",
                 "max_pairs", "max_windows", "formula"), "semantic plan")
    _require(type(data["schema_version"]) is int and data["schema_version"] == 1,
             "exact semantic plan schema 1 required")
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
    expected_plan_ref.verify(root)
    split_ref.verify(root)
    return PairedVerificationPlan(context, expected_plan_ref, split_ref,
                                  tuple(frozen_entries), tuple(sorted(ops.items())),
                                  data["min_calibration_pairs"], data["min_validation_pairs"])


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


def _measurement(root, ref, *, scope, arm, cell_id, context):
    path = ref.verify(root)
    _require(path.suffix == ".json", "semantic v1 requires bounded JSON objects, not loose JSONL")
    data, _, _ = _read_json(path, ref)
    payload = "runs" if scope == "paired_measurement_wrapper" else "windows"
    _keys(data, ("schema_version", "scope", "origin", "context", "arm", "cell_id", payload),
          "raw " + scope)
    _require(type(data["schema_version"]) is int and data["schema_version"] == 1 and
             data["scope"] == scope, "raw measurement scope/schema mismatch")
    _require(type(data["origin"]) is str and data["origin"] in ORIGINS, "explicit origin required")
    _require(data["arm"] == arm and data["cell_id"] == cell_id, "raw measurement arm/cell mismatch")
    _require(TableContext.from_mapping(data["context"]) == context, "raw measurement context drift")
    values = data[payload]
    limit = MAX_PAIRS if payload == "runs" else MAX_WINDOWS
    _require(type(values) is list and 1 <= len(values) <= limit, "bounded nonempty raw observations")
    return data


def _runs(data):
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


def _verify_cell(root, bound, plan):
    refs = dict(bound.measurement_refs)
    raw = {}
    for role in MEASUREMENT_ROLES:
        arm = "baseline" if role.startswith("baseline") else "action"
        scope = "paired_measurement_wrapper" if role.endswith("wrapper") else "paired_window_observations"
        raw[role] = _measurement(root, refs[role], scope=scope, arm=arm,
                                 cell_id=bound.cell_id, context=plan.context)
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
        matching = set(baseline) - {"start_ns", "end_ns", "completed_new_io"}
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
            durations_base.append(a["end_ns"] - a["start_ns"])
            durations_action.append(b["end_ns"] - b["start_ns"])
            action_step_durations.append(b["end_ns"] - b["start_ns"])
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
    _require(cost == bound.cost and bound.paired_runs_reported == len(baseline_runs) and
             bound.windows_reported == measured, "candidate costs/counts do not equal raw recomputation")
    ref = refs["pair_analysis"]
    report, _, _ = _read_json(ref.verify(root), ref)
    _keys(report, ("schema_version", "scope", "origin", "context", "cell_id", "plan_ref",
                  "evidence_refs", "formula", "baseline_ns", "incremental_or_joint_ns",
                  "uncertainty_ns", "calibration_pairs", "validation_pairs",
                  "paired_runs_reported", "windows_reported"), "pair analysis")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1 and
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
    return VerifiedCell(bound.cell_id, cost, next(iter(origins)),
                        len(by_split["calibration"]), len(by_split["validation"]),
                        len(baseline_runs), measured, warmup, residual)


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


def write_semantic_verification_manifest(root, verification, relative_path):
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
    for ref in verification.evidence_refs:
        ref.verify(root)
    raw = (json.dumps(semantic_manifest(verification), sort_keys=True, indent=2) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return EvidenceRef(relative_path, len(raw), sha256(raw).hexdigest())
