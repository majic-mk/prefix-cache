"""CPU scalar-fixture observation costs; no model, GPU or end-to-end claim."""
import json, statistics, time
from pathlib import Path
import pytest
from py_kvcache.reactor import IoReactor
from prefix_io_control.p4_types import P4Config
from tests.prefix_io_v1_p4_02_bridge.test_native_events import owner_model

@pytest.mark.parametrize("parents,files",[(1,1),(32,2)])
def test_cpu_observation_microbenchmark_records_bounded_real_host_time(tmp_path,parents,files):
    enabled=owner_model(parents,files)
    enabled._prefix_p4_bridge.policy.config=P4Config("shadow",200_000_000,1_000_000_000)
    disabled=IoReactor.__new__(IoReactor)
    calls=200;repeats=7
    samples={}
    for label,reactor in (("off",disabled),("shadow_scalar_fixture",enabled)):
        for _ in range(20):reactor._prefix_p4_collect()
        values=[]
        for _ in range(repeats):
            start=time.perf_counter_ns()
            for _ in range(calls):reactor._prefix_p4_collect()
            values.append((time.perf_counter_ns()-start)/calls)
        samples[label]=dict(samples_ns_per_call=values,median_ns_per_call=statistics.median(values),
            min_ns_per_call=min(values),max_ns_per_call=max(values))
    view=enabled._prefix_p4_collect()
    bridge=enabled._prefix_p4_bridge.snapshot()
    assert len(view.works)==parents*files
    assert view.snapshot.gpu_immediately_reusable_bytes is None
    assert not bridge["gpu_qualified"] and not bridge["production_eta_qualified"]
    assert bridge["eta_history"]["pending_count"]<=32
    record=dict(status="CPU_SCALAR_FIXTURE_MICROBENCHMARK_ONLY",parents=parents,works=parents*files,
        calls_per_repeat=calls,repeats=repeats,samples=samples,
        GPU_workloads=0,real_model_inference=False,end_to_end_overhead_unknown=True,
        engineering_two_percent_target_verified=False,
        measurement_scope="same-host wall time of actual original-off/P4 owner collection over immutable fixture facts")
    (tmp_path/"p4-cpu-observation-overhead.json").write_text(json.dumps(record,indent=2)+"\n")
