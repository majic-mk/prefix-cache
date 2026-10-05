from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from py_kvcache.staging import StagingPool
from py_kvcache.transfer import ParsedKvLayout, TORCH_COPY_AVAILABLE
from py_kvcache import reactor
from py_kvcache.fs_config import SharedFileConfig
from prefix_io_control.bridge import native_boundary
from prefix_io_control.config import ConfigError

def test_actual_backing_fits_approved_budget():
    budget = 8 * 4096
    slots = StagingPool.compute_slot_count(staging_mem_gib=budget / 2**30,
                                           io_size=4096, min_slots=1)
    layout = ParsedKvLayout([None], [4096], 1, False)
    buffer = layout.allocate_staging_buffer(slots)
    actual = buffer.untyped_storage().nbytes()
    assert slots == 7 and actual == slots * 4096 + 4095
    assert actual <= budget and actual + 4096 > budget
    assert not TORCH_COPY_AVAILABLE  # No fallback promotion in this CPU environment.

def test_min_slot_floor_cannot_overallocate():
    with pytest.raises(ValueError, match="minimum"):
        StagingPool.compute_slot_count(staging_mem_gib=(24 * 4096) / 2**30,
                                       io_size=4096, min_slots=24)

@pytest.mark.parametrize("mem", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_staging_budget_fails(mem):
    with pytest.raises(ValueError):
        StagingPool.compute_slot_count(staging_mem_gib=mem, io_size=4096)

def test_reactor_rejects_floor_before_allocating_or_creating_ring(tmp_path, monkeypatch):
    monkeypatch.setattr(reactor, "TORCH_COPY_AVAILABLE", True)  # constructor-only fake backend
    allocate = Mock(side_effect=AssertionError("must not allocate"))
    ring = Mock(side_effect=AssertionError("must not create ring"))
    monkeypatch.setattr(reactor, "LiburingRing", ring)
    layout = SimpleNamespace(storage_block_bytes=4096, allocate_staging_buffer=allocate)
    config = SharedFileConfig(str(tmp_path), iodepth=16, staging_mem=24*4096/2**30)
    with pytest.raises(ValueError, match="minimum"):
        reactor.IoReactor(config=config, file_mapper=Mock(), layout=layout)
    allocate.assert_not_called()
    ring.assert_not_called()

def test_off_bypasses_all_observation():
    native = Mock(return_value=object())
    observer = Mock(side_effect=AssertionError("off must never observe"))
    assert native_boundary("off", native, observe=observer) is native.return_value
    native.assert_called_once_with()
    observer.assert_not_called()

def test_shadow_observer_failure_keeps_native_result():
    native = Mock(return_value="native")
    errors = []
    assert native_boundary("shadow", native,
        observe=Mock(side_effect=RuntimeError("fixture")), record_error=errors.append) == "native"
    assert errors == ["RuntimeError"]
    native.assert_called_once_with()

@pytest.mark.parametrize("mode", ["fixed", "pressure", "interference", "dependency_only", "joint"])
def test_unimplemented_live_policy_never_silently_activates(mode):
    native = Mock()
    with pytest.raises(ConfigError, match="blocked"):
        native_boundary(mode, native)
    native.assert_not_called()

def test_shadow_log_failure_cannot_block_native():
    native = Mock(return_value="native")
    assert native_boundary("shadow", native,
        observe=Mock(side_effect=ValueError("fixture")),
        record_error=Mock(side_effect=RuntimeError("log fixture"))) == "native"
    native.assert_called_once_with()
