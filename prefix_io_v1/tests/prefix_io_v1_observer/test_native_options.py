import pytest
from prefix_io_control.native_options import parse_options,build_observer_kwargs
def test_absent_and_off_do_not_allocate_publisher():
    assert build_observer_kwargs(parse_options({}))=={}
    assert build_observer_kwargs(parse_options({"prefix_io_observation_mode":"off"}))=={}
def test_shadow_is_explicit_and_frozen():
    o=parse_options({"prefix_io_observation_mode":"shadow","prefix_io_observation_run_id":"run"})
    k=build_observer_kwargs(o)
    assert k["progress_run_id"]=="run" and k["observation_sink"].interval_ns==10_000_000
@pytest.mark.parametrize("extra",[
 {"prefix_io_observation_mode":"joint"},
 {"prefix_io_observation_mode":"shadow"},
 {"prefix_io_observation_interval_ns":True},
 {"prefix_io_observation_interval_ns":0},
 {"prefix_io_observation_run_id":"run"},
 {"prefix_io_typo":1}])
def test_invalid_modes_and_unknown_keys_rejected(extra):
    with pytest.raises(ValueError):parse_options(extra)
