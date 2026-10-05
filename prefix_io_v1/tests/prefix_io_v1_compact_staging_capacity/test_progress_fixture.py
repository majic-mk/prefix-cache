"""P316 native progress contracts with known and explicitly unknown CPU capacity."""
import pytest
from prefix_io_control.dispatch_budget import STAGES
from tests.prefix_io_v1_start_budget._fixture import rig
from tests.prefix_io_v1_parent_admission._fixture import controlled

@pytest.mark.parametrize("initialize", [True, False])
def test_pressure_zero_quota_uses_known_capacity_or_safe_native_fallback(monkeypatch, initialize):
    with rig(monkeypatch, enabled=False) as x:
        if initialize:
            x.r._prefix_init_capacity_observation()
        accounting, controller = controlled(x, mode="pressure")
        store=x.submit(files=1)
        x.coordinator.request_mandatory([store])
        x.start()
        assert store.result(timeout=2)==4096
        load=x.load()
        x.coordinator.request_mandatory([load])
        assert load.result(timeout=2)==4096
        x.stop()
        result=controller.snapshot(native_shutdown=True)
        assert not result["faulted"] and not result["pending_attempt"]
        for stage in STAGES:
            assert result["observed_api_accepted"][stage]==dict(ops=1,bytes=4096)
            assert accounting.stats[stage]["accepted_bytes"]==4096
        assert accounting.valid and not accounting.records
        assert x.r.parent_admission_snapshot()["accepted_parents"]==0
        if initialize:
            assert result["performance_override_ops"]>=4
            assert result["native_fallbacks"]==0
        else:
            assert result["native_fallbacks"]>0
            assert x.r._prefix_clean_reclaimable_bytes() is None
