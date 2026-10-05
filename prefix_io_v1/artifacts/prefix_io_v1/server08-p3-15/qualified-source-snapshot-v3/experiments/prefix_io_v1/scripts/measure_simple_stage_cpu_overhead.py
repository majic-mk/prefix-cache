"""CPU-only isolated control/observation microbenchmark, no native I/O.

Uses the project's public DispatchController and StageAccounting APIs with
explicitly synthetic owner metadata. accepted/completed here are API exercises,
not evidence of real SSD/CUDA transfers, GPU ownership or physical release.
Output/source metadata I/O occurs outside every measured batch.
"""
import argparse
import hashlib
import json
import statistics
import threading
import time
from pathlib import Path

from prefix_io_control.dispatch_budget import Amount, STAGES, ZERO
from prefix_io_control.dispatch_shadow import ShadowState
from prefix_io_control.simple_stage_policy import SimpleStageConfig, make_dispatch_controller
from prefix_io_control.stage_accounting import StageAccounting

MODES = ("off", "direct_accounting", "shadow", "fixed", "pressure")
BATCHES = 3
MAX_TOTAL_DECISIONS_PER_MODE = 10000
DEFAULT_BATCH_DECISIONS = 3333
SYNTHETIC_IO_BYTES = 4096
SYNTHETIC_STAGING_BYTES = 16 * 1024**2

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def config(mode):
    # High complete caps allow each simulated candidate, so this benchmark
    # isolates finite control work rather than waiting/defer behavior.
    amount = Amount(MAX_TOTAL_DECISIONS_PER_MODE,
                    MAX_TOTAL_DECISIONS_PER_MODE * SYNTHETIC_IO_BYTES)
    caps = tuple(amount for _ in STAGES)
    shared = Amount(2 * amount.ops, 2 * amount.nbytes)
    return SimpleStageConfig(
        mode=mode, epoch_ns=1_000_000_000, byte_quantum=SYNTHETIC_IO_BYTES,
        cumulative=caps, inflight=caps, shared_ssd_cumulative=shared,
        shared_ssd_inflight=shared, shared_copy_cumulative_bytes=shared.nbytes,
        shared_copy_inflight_bytes=shared.nbytes, reserve_staging_bytes=0,
        max_accepted_parents=1, sample_max_age_ns=20_000_000,
        max_wait_ns=200_000_000, max_waiting_keys=64, max_records=96,
    )

class OwnerMetadata:
    """Values only; no real queue, worker, FD, Future, slot or GPU reference."""
    def __init__(self, mode):
        self.mode = mode
        self.run_id = "p315-cpu-overhead-" + mode
        self.sequence = 0
        self.accounting = StageAccounting(max_records=4) if mode != "off" else None
        if self.accounting is not None:
            self.accounting.bind()
        policy_mode = mode if mode in ("shadow", "fixed", "pressure") else "off"
        self.controller = make_dispatch_controller(self.run_id, config(policy_mode))
        if self.controller is not None:
            self.controller.bind()

    def sample(self, now_ns):
        # Same four-stage complete/fresh values in every mode. All prior
        # simulated operations complete inside the same iteration. For the
        # modes with accounting, obtain these values through its public snapshot.
        if self.accounting is not None:
            snapshot = self.accounting.snapshot()
            inflight = tuple(
                Amount(snapshot["stages"][stage]["inflight_ops"],
                       snapshot["stages"][stage]["inflight_bytes"])
                for stage in STAGES
            )
        else:
            inflight = ZERO
        return ShadowState(
            self.run_id, now_ns, inflight=inflight,
            free_staging_bytes=SYNTHETIC_STAGING_BYTES, accepted_parents=1,
            native_issue_safe=True,
        )

    def exercise(self):
        seq = self.sequence
        stage = STAGES[seq % len(STAGES)]
        now_ns = time.monotonic_ns()
        sample = self.sample(now_ns)
        decision = None
        if self.controller is not None:
            decision = self.controller.decide(
                stage, SYNTHETIC_IO_BYTES, sample, now_ns=now_ns,
                work_id=("cpu_micro", seq), staging_bytes_needed=SYNTHETIC_IO_BYTES,
            )
            require(decision.action in ("issue", "native_fallback") and
                    decision.attempt is not None, "unexpected synthetic benchmark defer")
            self.controller.accepted(decision.attempt)
        if self.accounting is not None:
            self.accounting.accepted(stage, seq, SYNTHETIC_IO_BYTES)
            self.accounting.completed(stage, seq, SYNTHETIC_IO_BYTES)
        self.sequence += 1

    def final_snapshot(self):
        accounting = self.accounting.snapshot() if self.accounting else None
        controller = self.controller.snapshot() if self.controller else None
        if accounting:
            require(accounting["valid"] and accounting["outstanding_records"] == 0,
                    "simulated accounting left unresolved records")
            require(sum(value["accepted_ops"] for value in accounting["stages"].values()) ==
                    self.sequence, "simulated acceptance count wrong")
            require(all(value["inflight_ops"] == value["inflight_bytes"] == 0
                        for value in accounting["stages"].values()),
                    "simulated four-stage metadata not complete")
        if controller:
            require(not controller["faulted"] and not controller["pending_attempt"] and
                    controller["waiting_keys"] == 0, "bounded owner control did not settle")
            require(sum(value["ops"] for value in controller["observed_api_accepted"].values()) ==
                    self.sequence, "simulated controller acceptance count wrong")
            require(len(controller["records"]) <= 96, "control record window exceeded")
        return dict(
            simulated_operations=self.sequence, accounting=accounting, controller=controller,
            synthetic_snapshot_only=True, native_resource_owner_observed=False,
            physical_release_proved=False, gpu_ownership_credit=False,
            simulated_api_byte_counters_are_not_transferred_bytes=True,
        )

def measured_batch(owner, count, batch_index):
    # Setup/config/module loading/output are excluded; both clocks cover the
    # same Python loop. Process time includes all process threads by definition.
    wall_started = time.perf_counter_ns()
    cpu_started = time.process_time_ns()
    for _ in range(count):
        owner.exercise()
    cpu_ns = time.process_time_ns() - cpu_started
    wall_ns = time.perf_counter_ns() - wall_started
    require(cpu_ns >= 0 and wall_ns >= 0, "clock anomaly")
    return dict(
        mode=owner.mode, batch=batch_index, operations=count,
        process_time_ns=cpu_ns, perf_counter_ns=wall_ns,
        cpu_ns_per_operation=cpu_ns / count, wall_ns_per_operation=wall_ns / count,
        decision_calls=count if owner.controller else 0,
        accounting_accepted_calls=count if owner.accounting else 0,
        accounting_completed_calls=count if owner.accounting else 0,
        real_backend_calls=0, real_file_io_calls_in_measured_loop=0, gpu_calls=0,
    )

def benchmark(count):
    require(type(count) is int and 1 <= count <= MAX_TOTAL_DECISIONS_PER_MODE // BATCHES,
            "three batches must total at most 10000 operations per mode")
    owners = {mode: OwnerMetadata(mode) for mode in MODES}
    rows = []
    # Rotate a deterministic mode order to avoid always measuring off first.
    # This is three batches only, not an adaptive search or performance claim.
    for batch in range(BATCHES):
        order = MODES[batch:] + MODES[:batch]
        for mode in order:
            rows.append(measured_batch(owners[mode], count, batch))
    final = {mode: owner.final_snapshot() for mode, owner in owners.items()}
    summary = {}
    for mode in MODES:
        values = [row for row in rows if row["mode"] == mode]
        cpu = [row["cpu_ns_per_operation"] for row in values]
        wall = [row["wall_ns_per_operation"] for row in values]
        summary[mode] = dict(
            batches=BATCHES, operations=count * BATCHES,
            median_cpu_ns_per_operation=statistics.median(cpu),
            minimum_cpu_ns_per_operation=min(cpu), maximum_cpu_ns_per_operation=max(cpu),
            median_wall_ns_per_operation=statistics.median(wall),
            minimum_wall_ns_per_operation=min(wall), maximum_wall_ns_per_operation=max(wall),
        )
    direct_cpu = summary["direct_accounting"]["median_cpu_ns_per_operation"]
    off_cpu = summary["off"]["median_cpu_ns_per_operation"]
    for mode, record in summary.items():
        record["median_cpu_delta_vs_off_ns_per_operation"] = record["median_cpu_ns_per_operation"] - off_cpu
        record["median_cpu_delta_vs_direct_accounting_ns_per_operation"] = (
            record["median_cpu_ns_per_operation"] - direct_cpu
        )
    return dict(
        status="PASS_CPU_ISOLATED_MICROBENCHMARK", schema_version=1,
        scope="isolated Python owner metadata/control microbenchmark only",
        benchmark_decisions_per_batch=count, batches_per_mode=BATCHES,
        maximum_total_operations_per_mode=MAX_TOTAL_DECISIONS_PER_MODE,
        modes=list(MODES), owner_thread_id=threading.get_ident(),
        same_complete_synthetic_state=True, synthetic_bytes_per_operation=SYNTHETIC_IO_BYTES,
        no_gpu_import_or_operations=True, no_native_io_or_model=True,
        simulated_backend_success=True, gpu_ownership_credit=False,
        physical_resource_release_evidence=False, end_to_end_cpu_decomposition=False,
        end_to_end_model_or_gpu_performance_claim=False,
        output_and_source_metadata_io_outside_timed_batches=True,
        measurement_clocks=dict(
            cpu="time.process_time_ns (entire process CPU, not per-thread CPU)",
            wall="time.perf_counter_ns",
        ),
        rows=rows, summary=summary, final_synthetic_metadata=final,
        interpretation_limitations=[
            "Inputs are synthetic complete metadata; no live native owner was sampled.",
            "Accepted/completed API calls do not establish actual transfer bytes or physical release.",
            "Observed cost includes Python snapshot construction and clock calls in the isolated loop.",
            "Off omits StageAccounting; direct_accounting separates that public-interface cost.",
            "Three batches are descriptive; this is not a formal end-to-end CPU utilization decomposition.",
        ],
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=DEFAULT_BATCH_DECISIONS,
                    help="operations per batch; 3 batches and <=10000 total per mode")
    args = ap.parse_args()
    require(not args.output.exists(), "append-new evidence output already exists")
    require(1 <= args.iterations <= MAX_TOTAL_DECISIONS_PER_MODE // BATCHES,
            "iterations outside frozen CPU microbenchmark bound")
    # Hash Python sources before measurement only; no payload/model is read.
    from prefix_io_control import simple_stage_policy, stage_accounting, dispatch_shadow, dispatch_budget
    modules = {
        name: dict(path=str(Path(module.__file__).resolve()),
                   sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        for name, module in (
            ("simple_stage_policy", simple_stage_policy), ("stage_accounting", stage_accounting),
            ("dispatch_shadow", dispatch_shadow), ("dispatch_budget", dispatch_budget),
        )
    }
    result = benchmark(args.iterations)
    result["python_source_modules"] = modules
    result["benchmark_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], output=str(args.output), gpu_calls=0)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
