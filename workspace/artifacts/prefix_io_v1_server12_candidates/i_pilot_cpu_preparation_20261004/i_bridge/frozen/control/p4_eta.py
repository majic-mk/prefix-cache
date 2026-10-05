"""Bounded causal native closure observations; no GPU qualification or owners.

A sample measures first complete-ready observation to successful whole-parent
physical drain. It does not sum per-file timings or infer allocator release.
CPU/fake completion measurements cannot enable a production forecast.
"""
from dataclasses import dataclass, asdict
from collections import OrderedDict, deque
from statistics import median_high
from .dispatch_budget import integer
from .simple_stage_policy import _run_id
from .p4_types import text

def _values(values):
    if type(values) is not tuple or not values or len(values) > 16:
        raise ValueError("bounded exact native context required")
    for item in values:
        if type(item) is int:
            integer(item, "context value")
        elif type(item) is str:
            text(item, "context value")
        else:
            raise ValueError("native context contains scalar values only")

@dataclass(frozen=True)
class ClosureGeometry:
    rows: tuple

    def __post_init__(self):
        if type(self.rows) is not tuple or not 1 <= len(self.rows) <= 64:
            raise ValueError("bounded nonempty native closure geometry required")
        if tuple(sorted(self.rows)) != self.rows or len(set(self.rows)) != len(self.rows):
            raise ValueError("canonical unique closure rows required")
        count = 0
        for row in self.rows:
            if type(row) is not tuple or len(row) != 4:
                raise ValueError("stage, physical bytes, quantum, count required")
            stage, nbytes, quantum, units = row
            if stage not in ("ssd_read", "h2d"):
                raise ValueError("only actual full-ready restore closure geometry supported")
            for value in (nbytes, quantum, units):
                integer(value, "closure geometry", 1)
            if nbytes % quantum:
                raise ValueError("actual native legal quantum required")
            count += units
        if count > 64:
            raise ValueError("native closure work window exceeded")

@dataclass(frozen=True)
class CausalEstimate:
    duration_ns: int
    empirical_error_ns: int
    sample_count: int
    earliest_completed_ns: int
    latest_completed_ns: int
    completed_sequence: int
    source: str = "native_complete_ready_to_successful_parent_drain"
    production_qualified: bool = False

@dataclass(frozen=True)
class CompletedClosureMeasurement:
    run_id: str
    parent_id: int
    geometry: ClosureGeometry
    context: tuple
    first_observed_ns: int
    completed_ns: int
    elapsed_ns: int
    sequence: int
    prior_forecast_duration_ns: int | None
    prior_forecast_error_ns: int | None
    prior_forecast_sample_count: int
    causal_absolute_residual_ns: int | None
    gpu_qualified: bool = False

@dataclass(frozen=True)
class _Pending:
    parent_id: int
    geometry: ClosureGeometry
    context: tuple
    first_observed_ns: int
    prior_estimate: CausalEstimate | None

class NativeClosureHistory:
    """Only bounded scalar metadata; all production estimates remain blocked."""
    def __init__(self, run_id, *, max_age_ns, minimum_samples=4):
        _run_id(run_id)
        integer(max_age_ns, "history age", 1)
        if type(minimum_samples) is not int or not 2 <= minimum_samples <= 16:
            raise ValueError("bounded minimum completed samples required")
        self.run_id = run_id
        self.max_age_ns = max_age_ns
        self.minimum_samples = minimum_samples
        self._pending = {}
        self._cells = OrderedDict()
        self._recent = deque(maxlen=64)
        self._highest_parent = 0
        self._last_clock = None
        self._sequence = 0
        self.observed = self.completed = self.discarded = self.omitted = 0
        self.last_reason = "no_native_closure_measurements"

    @property
    def production_qualified(self):
        return False

    def _clock(self, now_ns):
        integer(now_ns, "native owner time")
        if self._last_clock is not None and now_ns < self._last_clock:
            raise ValueError("native closure owner clock moved backwards")
        self._last_clock = now_ns

    def _binding(self, run_id, parent_id, geometry, context):
        if run_id != self.run_id:
            raise ValueError("native closure run differs")
        integer(parent_id, "accepted parent sequence", 1)
        if type(geometry) is not ClosureGeometry:
            raise TypeError("immutable actual native geometry required")
        _values(context)

    def _estimate(self, key, now_ns):
        samples = self._cells.get(key, ())
        fresh = tuple(s for s in samples if 0 < now_ns-s.completed_ns <= self.max_age_ns)
        if len(fresh) < self.minimum_samples:
            self.last_reason = "insufficient_fresh_completed_native_closures"
            return None
        durations = tuple(s.elapsed_ns for s in fresh)
        point = median_high(durations)
        residuals = tuple(s.causal_absolute_residual_ns for s in fresh
                          if s.causal_absolute_residual_ns is not None)
        margin = max(tuple(abs(d-point) for d in durations) + residuals)
        return CausalEstimate(point, margin, len(fresh),
            min(s.completed_ns for s in fresh), max(s.completed_ns for s in fresh),
            max(s.sequence for s in fresh))

    def observe(self, *, run_id, parent_id, geometry, context, now_ns):
        self._binding(run_id, parent_id, geometry, context)
        self._clock(now_ns)
        pending = self._pending.get(parent_id)
        if pending is not None:
            if pending.geometry != geometry or pending.context != context:
                self.last_reason = "native_closure_context_changed_no_age_reset"
                return False
            self.last_reason = "duplicate_observation_preserves_first_time"
            return False
        if parent_id <= self._highest_parent:
            self.omitted += 1
            self.last_reason = "retired_or_out_of_order_parent_identity"
            return False
        # Scalar high-water mark prevents a retired ID from becoming a new sample.
        self._highest_parent = parent_id
        if len(self._pending) >= 32:
            self.omitted += 1
            self.last_reason = "observation_window_exceeded_native_work_unchanged"
            return False
        estimate = self._estimate((geometry, context), now_ns)
        self._pending[parent_id] = _Pending(parent_id, geometry, context, now_ns, estimate)
        self.observed += 1
        self.last_reason = "native_complete_ready_observed"
        return True

    def estimate(self, *, run_id, parent_id, geometry, context, now_ns,
                 execution="shadow_diagnostic"):
        self._binding(run_id, parent_id, geometry, context)
        self._clock(now_ns)
        if execution not in ("shadow_diagnostic", "cpu_mock", "production"):
            raise ValueError("explicit ETA execution scope required")
        if execution == "production":
            self.last_reason = "production_eta_live_gpu_qualification_blocked"
            return None
        pending = self._pending.get(parent_id)
        if pending is None or pending.geometry != geometry or pending.context != context:
            self.last_reason = "no_compatible_first_native_closure_observation"
            return None
        estimate = self._estimate((geometry, context), now_ns)
        if estimate is None:
            return None
        if pending.first_observed_ns + estimate.duration_ns <= now_ns:
            self.last_reason = "elapsed_native_closure_exceeds_empirical_eta"
            return None
        self.last_reason = "causal_shadow_diagnostic_only"
        return estimate

    def finish(self, *, run_id, parent_id, now_ns, successful, drain_known):
        if run_id != self.run_id:
            raise ValueError("native closure run differs")
        integer(parent_id, "accepted parent sequence", 1)
        self._clock(now_ns)
        if type(successful) is not bool or type(drain_known) is not bool:
            raise ValueError("explicit physical drain result required")
        pending = self._pending.pop(parent_id, None)
        if pending is None:
            self.last_reason = "no_pending_native_closure"
            return None
        if not successful or not drain_known:
            self.discarded += 1
            self.last_reason = "failed_or_unknown_native_drain_not_a_timing_sample"
            return None
        elapsed = now_ns-pending.first_observed_ns
        if elapsed <= 0:
            self.discarded += 1
            self.last_reason = "nonpositive_native_observation_interval"
            return None
        self._sequence += 1
        prior = pending.prior_estimate
        sample = CompletedClosureMeasurement(self.run_id, parent_id, pending.geometry,
            pending.context, pending.first_observed_ns, now_ns, elapsed, self._sequence,
            prior.duration_ns if prior else None, prior.empirical_error_ns if prior else None,
            prior.sample_count if prior else 0,
            abs(elapsed-prior.duration_ns) if prior else None)
        key = (pending.geometry, pending.context)
        cell = self._cells.get(key)
        if cell is None:
            if len(self._cells) >= 32:
                self._cells.popitem(last=False)
            cell = deque(maxlen=16)
            self._cells[key] = cell
        cell.append(sample)
        self._cells.move_to_end(key)
        self._recent.append(sample)
        self.completed += 1
        self.last_reason = "successful_whole_parent_native_drain_measured"
        return sample

    def abort_all(self):
        self.discarded += len(self._pending)
        self._pending.clear()
        self.last_reason = "optional_observer_aborted_no_native_owners_changed"

    def snapshot(self):
        return dict(schema_version=1, run_id=self.run_id, reason=self.last_reason,
            observations=self.observed, successful_measurements=self.completed,
            failed_or_unknown_discarded=self.discarded, omissions=self.omitted,
            pending_count=len(self._pending), cell_count=len(self._cells),
            retained_cell_samples=sum(len(cell) for cell in self._cells.values()),
            history_max_age_ns=self.max_age_ns, minimum_samples=self.minimum_samples,
            pending_limit=32, cell_limit=32, samples_per_cell_limit=16,
            recent_measurements=tuple(asdict(s) for s in self._recent),
            production_qualified=False, gpu_qualified=False,
            held_job_or_resource_owners=False, new_work_queues=0,
            eta_is_allocator_release=False,
            error_is_empirical_not_a_coverage_guarantee=True)
