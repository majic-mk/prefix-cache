"""Strict production option tests; no backend or GPU qualification."""
import copy
import pytest
from prefix_io_control.p4_options import P4_KEY,parse_p4_options,build_p4_kwargs
from prefix_io_control.simple_stage_options import PARENT_KEY,POLICY_KEY
from prefix_io_control.dispatch_budget import STAGES

def fixed_raw(units=16):
    amount=lambda n:dict(ops=n,bytes=n*4096)
    return dict(schema_version=1,run_id="cpu-run",mode="fixed",epoch_ns=10_000_000,
        byte_quantum=4096,cumulative={s:amount(units) for s in STAGES},
        inflight={s:amount(units) for s in STAGES},
        shared_ssd_cumulative=amount(2*units),shared_ssd_inflight=amount(2*units),
        shared_copy_cumulative_bytes=2*units*4096,shared_copy_inflight_bytes=2*units*4096,
        reserve_staging_bytes=0,max_accepted_parents=64,sample_max_age_ns=200_000_000,
        max_wait_ns=1_000_000_000)

def options(mode="dependency_only"):
    return {PARENT_KEY:dict(schema_version=1,run_id="cpu-run",max_accepted_parents=64),
        P4_KEY:dict(schema_version=1,run_id="cpu-run",mode=mode,
        sample_max_age_ns=200_000_000,max_wait_ns=1_000_000_000,
        internal_step_budget_ns=None,candidate_batches=[1,2,4,8],
        fixed_stage_policy=fixed_raw() if mode=="dependency_only" else None,cost_table=None)}

@pytest.mark.parametrize("mode",["shadow","interference","joint"])
def test_unsupported_and_shadow_install_original_u_not_fixed(mode):
    kwargs=build_p4_kwargs(parse_p4_options(options(mode)))
    assert "dispatch_controller" not in kwargs
    assert kwargs["p4_bridge"].mode==mode
    assert kwargs["max_accepted_parents"]==64
    assert not kwargs["p4_bridge"].policy.production_interference_qualified

def test_dependency_only_uses_existing_fixed_controller_once():
    kwargs=build_p4_kwargs(parse_p4_options(options()))
    assert kwargs["dispatch_controller"].config.mode=="fixed"
    assert kwargs["dispatch_controller"].run_id==kwargs["p4_bridge"].run_id
    assert kwargs["stage_accounting"].valid

def test_off_carries_no_policy_or_controller():
    x=options();x[P4_KEY]={"mode":"off"}
    kwargs=build_p4_kwargs(parse_p4_options(x))
    assert "p4_bridge" not in kwargs and "dispatch_controller" not in kwargs
    assert kwargs["stage_accounting"].valid

@pytest.mark.parametrize("change",[
    lambda x:x[P4_KEY].update(unknown=1),
    lambda x:x[P4_KEY].update(schema_version=True),
    lambda x:x[P4_KEY].update(run_id="other"),
    lambda x:x[P4_KEY].update(cost_table={"scope":"mock_only"}),
    lambda x:x[P4_KEY].update(candidate_batches=[1,2,16]),
    lambda x:x.update({POLICY_KEY:{"mode":"off"}}),
    lambda x:x[P4_KEY]["fixed_stage_policy"].update(mode="pressure"),
    lambda x:x[P4_KEY]["fixed_stage_policy"].update(run_id="other"),
    lambda x:x[P4_KEY].update(mode="off"),
])
def test_strict_options_reject(change):
    x=copy.deepcopy(options());change(x)
    with pytest.raises((ValueError,TypeError)):parse_p4_options(x)

@pytest.mark.parametrize("mode",["shadow","interference","joint"])
def test_declared_common_base_does_not_activate_unqualified_interference(mode):
    x=options(mode);x[P4_KEY]["fixed_stage_policy"]=fixed_raw()
    parsed=parse_p4_options(x)
    assert parsed.fixed is not None
    kwargs=build_p4_kwargs(parsed)
    assert ("dispatch_controller" in kwargs) == (mode == "shadow")
    assert not kwargs["p4_bridge"].policy.production_interference_qualified
