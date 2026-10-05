"""Finite restore-parent forecasts from an actual guard-closed calibration.

An empirical ready-to-whole-parent-drain estimate is ordering input only.
It is neither an allocator release receipt nor an earliest-time guarantee.
The original owner still finishes and releases all resources.
"""
from dataclasses import dataclass
from pathlib import Path
from statistics import median_high
import hashlib
import json
from .p4_eta import ClosureGeometry, CompletedClosureMeasurement, _values
from .dispatch_budget import integer


class RestoreCalibrationCoverageMissing(ValueError):
    """Only a valid, fully read/closed calibration with no actionable cell."""
    def __init__(self, *, run_id, sample_count, exact_cell_counts):
        super().__init__("no four fresh complete-ready SSD closures in an actionable exact calibration cell")
        self.run_id = run_id
        self.sample_count = sample_count
        self.exact_cell_counts = exact_cell_counts
        self.reason = "valid_closed_calibration_no_actionable_exact_cell"


@dataclass(frozen=True)
class RestoreForecast:
    completion_estimate_ns: int
    empirical_error_ns: int
    sample_count: int
    calibration_run_id: str
    production_qualified: bool = False
    gpu_release_credit: None = None


@dataclass(frozen=True)
class RestoreClosureCalibration:
    run_id: str
    gpu_uuid: str
    common_runtime_domain_sha256: str
    samples: tuple
    result_ref: tuple
    guard_ref: tuple
    runtime_refs: tuple
    max_age_ns: int
    production_qualified: bool = False

    def predict(self, history, parent_id, geometry, context, *, now_ns):
        integer(now_ns, "restore forecast owner time")
        if not all(row[0] == "ssd_read" for row in geometry.rows) or context[5] != 0:
            return None  # the unchanged whole-window continuation gate stays native
        pending = history._pending.get(parent_id)
        if (pending is None or pending.geometry != geometry or
                pending.context != context):
            return None
        fresh = tuple(s for s in self.samples if s.geometry == geometry and
            s.context == context and 0 < now_ns - s.completed_ns <= self.max_age_ns)
        if len(fresh) < 4:
            return None
        point = median_high(s.elapsed_ns for s in fresh)
        error = max(abs(s.elapsed_ns - point) for s in fresh)
        completion = pending.first_observed_ns + point
        if completion <= now_ns:
            return None  # an exceeded empirical forecast cannot be reset
        return RestoreForecast(completion, error, len(fresh), self.run_id)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _ref(value):
    _require(type(value) is dict and set(value) == {"path", "bytes", "sha256"},
        "exact source reference required")
    _require(type(value["path"]) is str and value["path"] and
        type(value["bytes"]) is int and value["bytes"] > 0 and
        type(value["sha256"]) is str and len(value["sha256"]) == 64 and
        all(c in "0123456789abcdef" for c in value["sha256"]), "strict source reference values")
    return (value["path"], value["bytes"], value["sha256"])


def _read(root, refs, reference):
    row = _ref(reference)
    _require(refs.get(row[0]) == reference, "calibration input must be in the current source lock")
    path = (root / row[0]).resolve()
    _require(path.is_relative_to(root), "calibration reference escaped project")
    data = path.read_bytes()
    _require(len(data) == row[1] and hashlib.sha256(data).hexdigest() == row[2],
        "calibration evidence bytes drift")
    return json.loads(data), data


def _samples(histories, run_id, *, now_ns, max_age_ns):
    unique = {}
    for history in histories:
        _require(type(history) is dict and type(history.get("schema_version")) is int and
            history["schema_version"] == 1 and
            history.get("run_id") == run_id and history.get("production_qualified") is False and
            history.get("gpu_qualified") is False and history.get("eta_is_allocator_release") is False and
            history.get("held_job_or_resource_owners") is False and history.get("new_work_queues") == 0 and
            type(history.get("minimum_samples")) is int and history["minimum_samples"] == 4,
            "original bounded owner closure history required")
        records = history.get("recent_measurements")
        _require(type(records) is list and len(records) <= 64, "bounded original measurements required")
        for row in records:
            _require(type(row) is dict and set(row) == set(CompletedClosureMeasurement.__dataclass_fields__),
                "original whole-parent measurement fields required")
            _require(row["run_id"] == run_id and row["gpu_qualified"] is False,
                "wrong calibration run or allocator claim")
            for key in ("parent_id", "first_observed_ns", "completed_ns", "elapsed_ns", "sequence"):
                integer(row[key], key, 1)
            _require(row["completed_ns"] - row["first_observed_ns"] == row["elapsed_ns"] and
                row["completed_ns"] < now_ns, "actual causal whole-parent timing required")
            for key in ("prior_forecast_duration_ns", "prior_forecast_error_ns", "causal_absolute_residual_ns"):
                if row[key] is not None:
                    integer(row[key], key)
            integer(row["prior_forecast_sample_count"], "prior sample count")
            _require(type(row["geometry"]) is dict and set(row["geometry"]) == {"rows"} and
                type(row["geometry"]["rows"]) is list and type(row["context"]) is list,
                "exact native closure geometry/context required")
            geometry = ClosureGeometry(tuple(tuple(r) for r in row["geometry"]["rows"]))
            context = tuple(row["context"])
            _values(context)
            _require(len(context) == 13 and context[0] == "native-ready-closure-v1",
                "unchanged C5 exact closure context required")
            sample = CompletedClosureMeasurement(**dict(row, geometry=geometry, context=context))
            key = (run_id, sample.parent_id, sample.sequence)
            _require(key not in unique or unique[key] == sample,
                "conflicting original measurement across owner snapshots")
            unique[key] = sample
    samples = tuple(sorted(unique.values(), key=lambda s: s.sequence))
    _require(len(samples) <= 64 and len({s.parent_id for s in samples}) == len(samples) and
        len({s.sequence for s in samples}) == len(samples), "unique bounded completed parents required")
    return samples


def load_restore_calibration(root, refs, result_ref, guard_ref, *, expected_gpu_uuid,
        expected_common_runtime_domain_sha256, expected_runtime_refs, now_ns, max_age_ns):
    """Read the actual P316 report/guard; no GPU operations or new receipt.

    ``probe`` metadata is the thin runner's current-device/source binding;
    measurements remain the original owner snapshot's unchanged p4 history.
    Raw outputs and completed guard must be frozen by the calling original
    runner before this loader is used. This parser grants no GPU capability.
    """
    root = Path(root).resolve()
    integer(now_ns, "calibration load time", 1)
    integer(max_age_ns, "fixed calibration age", 1)
    _require(max_age_ns <= 600_000_000_000, "bounded calibration lifetime exceeds ten minutes")
    _require(type(expected_gpu_uuid) is str and expected_gpu_uuid.startswith("GPU-") and
        type(expected_common_runtime_domain_sha256) is str and
        len(expected_common_runtime_domain_sha256) == 64 and
        all(c in "0123456789abcdef" for c in expected_common_runtime_domain_sha256),
        "actual current device and strict resource-domain digest required")
    result, result_bytes = _read(root, refs, result_ref)
    guard, guard_bytes = _read(root, refs, guard_ref)
    probe = result.get("probe")
    _require(type(probe) is dict, "actual original P316 native probe required")
    run_id = probe.get("run_id")
    _require(type(run_id) is str and run_id and guard.get("label") == run_id and
        probe.get("gpu_uuid") == guard.get("gpu_uuid") == expected_gpu_uuid and
        probe.get("common_runtime_domain_sha256") == expected_common_runtime_domain_sha256,
        "actual calibration GPU/run/common-resource domain mismatch")
    _require(type(expected_runtime_refs) is dict and expected_runtime_refs and
        probe.get("runtime_refs") == expected_runtime_refs, "exact common runtime source binding required")
    for path, reference in expected_runtime_refs.items():
        _require(path == _ref(reference)[0] and refs.get(path) == reference,
            "common runtime source not in current lock")
        actual = (root / path).resolve()
        _require(actual.is_relative_to(root), "runtime source escaped project")
        data = actual.read_bytes()
        _require(len(data) == reference["bytes"] and hashlib.sha256(data).hexdigest() == reference["sha256"],
            "current common runtime source drift")
    _require(type(guard.get("exit")) is int and guard["exit"] == 0 and
        type(guard.get("child_exit")) is int and guard["child_exit"] == 0 and
        guard.get("timed_out") is False and guard.get("interrupted_signal") is None and
        guard.get("error") is None and guard.get("gpu_job_attempted") is True and
        guard.get("session_drained") is True and guard.get("session_members_before_cleanup") == [] and
        guard.get("session_members_after_cleanup") == [] and
        type(probe.get("subprocess_sid")) is int and probe["subprocess_sid"] > 0 and
        guard.get("session_id") == probe["subprocess_sid"] and
        type(probe.get("subprocess_pid")) is int and probe["subprocess_pid"] > 0 and
        probe.get("original_engine_shutdown_returned") is True and probe.get("native_tail_drained") is True,
        "actual natural guard exit and original native shutdown required")
    snapshots = probe.get("current_native_snapshots")
    _require(type(snapshots) is list and 1 <= len(snapshots) <= 32,
        "bounded actual owner snapshots required")
    histories = []
    for snapshot in snapshots:
        _require(type(snapshot) is dict and type(snapshot.get("p4")) is dict,
            "actual original owner p4 snapshot missing")
        histories.append(snapshot["p4"]["eta_history"])
    samples = _samples(histories, run_id, now_ns=now_ns, max_age_ns=max_age_ns)
    _require(_read(root, refs, result_ref)[1] == result_bytes and
        _read(root, refs, guard_ref)[1] == guard_bytes, "calibration evidence changed while parsing")
    for reference in expected_runtime_refs.values():
        data = (root / reference["path"]).read_bytes()
        _require(len(data) == reference["bytes"] and hashlib.sha256(data).hexdigest() == reference["sha256"],
            "current common runtime source changed while parsing")
    cells = {}
    for sample in samples:
        if 0 < now_ns - sample.completed_ns <= max_age_ns:
            key = (sample.geometry, sample.context)
            cells[key] = cells.get(key, 0) + 1
    if not any(count >= 4 and all(row[0] == "ssd_read" for row in geometry.rows) and
            context[5] == 0 for (geometry, context), count in cells.items()):
        raise RestoreCalibrationCoverageMissing(run_id=run_id, sample_count=len(samples),
            exact_cell_counts=tuple((geometry.rows, context, count)
                for (geometry, context), count in cells.items()))
    return RestoreClosureCalibration(run_id, expected_gpu_uuid,
        expected_common_runtime_domain_sha256, samples, _ref(result_ref), _ref(guard_ref),
        tuple(sorted(_ref(v) for v in expected_runtime_refs.values())), max_age_ns)
