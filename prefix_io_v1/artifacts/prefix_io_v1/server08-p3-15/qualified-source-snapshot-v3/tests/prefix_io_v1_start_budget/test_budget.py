import pytest
from prefix_io_control.start_budget import StartBudget,StartBudgetConfig
from concurrent.futures import Future

def make(mode="fixed",**kwargs):
    values=dict(mode=mode,epoch_ns=100,starts_per_epoch=2,max_wait_ns=300,
                reserve_free_slots=1 if mode=="pressure" else 0)
    values.update(kwargs)
    return StartBudget(StartBudgetConfig(**values))

def grant(b,kind="store",key=None,now=0,free=10,**flags):
    return b.allow(kind,key,now_ns=now,free_slots=free,**flags)

def use(b,**kwargs):
    t=grant(b,**kwargs);assert t is not False
    b.commit(t);return t

@pytest.mark.parametrize("kind",["load","store","preload"])
def test_epochs_are_time_windows_not_pump_counts(kind):
    b=make()
    use(b,kind=kind);use(b,kind=kind)
    for _ in range(30):assert grant(b,kind=kind,now=1) is False
    assert b.used==2 and b.epoch_refreshes==1
    use(b,kind=kind,now=100)
    assert b.used==1 and b.epoch_refreshes==2

def test_joint_read_write_and_preload_share_allowance():
    b=make();use(b,kind="load");use(b,kind="store")
    assert grant(b,kind="preload") is False
    assert b.ordinary_starts==2

def test_idle_windows_do_not_bank_unbounded_credits():
    b=make();use(b);use(b,now=100000)
    use(b,now=100000)
    assert grant(b,now=100000) is False

@pytest.mark.parametrize("flag",["mandatory","support","continuation","shutdown"])
def test_required_progress_can_bypass_exhausted_ordinary_allowance(flag):
    b=make();use(b);use(b)
    t=use(b,**{flag:True})
    assert t.reason!="ordinary" and b.used==2
    assert b.ordinary_starts==2 and b.starts["store"]==3

def test_pressure_preserves_free_slots_for_load_and_age_cannot_starve_store():
    b=make("pressure")
    assert grant(b,key="store",free=1) is False
    use(b,kind="load",key="load",free=1)
    assert grant(b,key="store",now=299,free=0) is False
    t=grant(b,key="store",now=300,free=0);assert t.reason=="age"
    b.commit(t)
    assert grant(b,key="store",now=300,free=0) is False  # age reset after progress
    assert b.reasons["age"]==1

def test_uncommitted_capacity_probe_does_not_consume_allowance():
    b=make()
    for _ in range(20):assert grant(b) is not False
    assert b.used==0 and b.ordinary_starts==0
    t=grant(b);b.commit(t)
    with pytest.raises(RuntimeError,match="duplicate"):b.commit(t)

def test_stale_ticket_cannot_commit_after_another_candidate():
    b=make();old=grant(b);current=grant(b,kind="load")
    with pytest.raises(RuntimeError,match="stale"):b.commit(old)
    b.commit(current);assert b.used==1

def test_run_local_future_keys_do_not_confuse_recycled_numeric_job_ids():
    b=make("pressure");a=Future();c=Future()
    assert grant(b,key=a,free=0) is False
    b.retire(a)
    assert grant(b,key=c,now=300,free=0) is False
    assert b.waiting[("store",c)]==300

def test_bounded_metadata_falls_back_without_new_queue_or_failing_future():
    b=make("pressure",max_waiting_keys=2)
    keys=[Future() for _ in range(3)]
    assert grant(b,key=keys[0],free=0) is False
    assert grant(b,key=keys[1],free=0) is False
    t=grant(b,key=keys[2],free=0)
    assert t.reason=="fallback" and b.faulted and not b.waiting
    assert all(not f.done() for f in keys)
    b.commit(t)

def test_clock_regression_explicitly_degrades_to_native_progress():
    b=make();use(b,now=101)
    t=grant(b,now=100)
    assert t.reason=="fallback" and b.faulted and b.errors==1

@pytest.mark.parametrize("field,value",[
    ("mode","joint"),("mode","off"),("epoch_ns",0),("epoch_ns",False),
    ("starts_per_epoch",3),("starts_per_epoch",0),("max_wait_ns",99),
    ("max_waiting_keys",33),("reserve_free_slots",-1),("reserve_free_slots",1)])
def test_invalid_unfrozen_or_out_of_scope_config_rejected(field,value):
    with pytest.raises(ValueError):make(**{field:value})

def test_clear_drops_only_controller_metadata():
    b=make("pressure");f=Future()
    assert grant(b,key=f,free=0) is False
    b.clear()
    assert not b.waiting and not f.done()
