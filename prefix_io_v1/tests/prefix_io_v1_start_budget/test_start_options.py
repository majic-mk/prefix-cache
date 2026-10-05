import pytest
from prefix_io_control.start_options import parse_start_options,build_start_kwargs,KEY

def value(mode="fixed"):
    return dict(schema_version=1,mode=mode,run_id="cpu-options",epoch_ns=100_000_000,
        starts_per_epoch=1,max_wait_ns=1_000_000_000,reserve_free_slots=0)

@pytest.mark.parametrize("extra",[{}, {KEY:{"mode":"off"}}])
def test_off_has_no_optional_state(extra):
    assert build_start_kwargs(parse_start_options(extra))=={}

def test_shadow_observes_without_start_gate():
    k=build_start_kwargs(parse_start_options({KEY:dict(schema_version=1,mode="shadow",run_id="s")}))
    assert set(k)=={"progress_run_id","stage_accounting"}

@pytest.mark.parametrize("mode",["fixed","pressure"])
def test_strict_finite_budget_builds_independent_owner_state(mode):
    raw=value(mode);a=parse_start_options({KEY:raw});raw["starts_per_epoch"]=8
    k=build_start_kwargs(a);j=build_start_kwargs(a)
    assert k["start_budget"].config.starts_per_epoch==1
    assert k["start_budget"] is not j["start_budget"]
    assert k["stage_accounting"] is not j["stage_accounting"]
    assert k["progress_run_id"]=="cpu-options"

@pytest.mark.parametrize("key,bad",[("schema_version",True),("mode","joint"),("epoch_ns",100),
    ("max_wait_ns",100),("starts_per_epoch",3),("reserve_free_slots",3),("run_id","")])
def test_invalid_contract_rejected(key,bad):
    v=value();v[key]=bad
    with pytest.raises(ValueError):parse_start_options({KEY:v})

def test_unknown_missing_off_state_and_run_identity_rejected():
    for v in (dict(value(),unknown=1), {"mode":"fixed"}, {"mode":"off","epoch_ns":1}):
        with pytest.raises(ValueError):parse_start_options({KEY:v})
    with pytest.raises(ValueError):
        parse_start_options({KEY:value(),"prefix_io_observation_mode":"shadow","prefix_io_observation_run_id":"other"})
    with pytest.raises(ValueError):parse_start_options({"prefix_io_typo":1})
