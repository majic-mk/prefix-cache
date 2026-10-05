from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import pytest
from prefix_io_control.dispatch_budget import (
    Amount, DispatchBudget, NativeState, DispatchLedger, STAGES, ZERO)

def grant(**changes):
    base = DispatchBudget("r", 0, 0, 100, (Amount(2, 8192),)*4, (Amount(2, 8192),)*4,
        Amount(3, 12288), Amount(3, 12288), 12288, 12288, 4096, 2, 10)
    return replace(base, **changes)

def state(**changes):
    return replace(NativeState("r", 0, ZERO, 16384, 1, True), **changes)

def ledger(g=None):
    x = DispatchLedger("r")
    x.publish(g or grant(), now_ns=0)
    return x

def issue(x, stage="h2d", nbytes=4096, sample=None, **kwargs):
    d = x.reserve(stage, nbytes, sample or state(), now_ns=0, **kwargs)
    assert d.action == "issue"
    x.settle(d.permit, accepted=True)
    return d

@pytest.mark.parametrize("stage", STAGES)
def test_epoch_does_not_refill_each_pump_or_completion(stage):
    x = ledger()
    issue(x, stage); issue(x, stage)
    assert x.publish(grant(), now_ns=0) is False
    # A zero in-flight sample proves no bandwidth refund.
    d = x.reserve(stage, 4096, state(), now_ns=0)
    assert d.action == "defer" and "stage_epoch_ops" in d.reasons
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)
    assert x.reserve(stage, 4096, state(captured_ns=100), now_ns=100).action == "issue"

@pytest.mark.parametrize("stage", STAGES)
def test_inflight_independent_of_new_epoch(stage):
    x = ledger(); i = STAGES.index(stage)
    a = list(ZERO); a[i] = Amount(2, 8192)
    x.publish(grant(epoch=1, issued_ns=100, expires_ns=200), now_ns=100)
    d = x.reserve(stage, 1, state(captured_ns=100, inflight=tuple(a)), now_ns=100)
    assert d.action == "defer" and "stage_inflight_ops" in d.reasons

def test_shared_copy_budget_is_not_duplicated_by_direction():
    x = ledger(grant(shared_copy_cumulative_bytes=4096))
    issue(x, "h2d")
    d = x.reserve("d2h", 1, state(), now_ns=0)
    assert d.action == "defer" and d.reasons == ("shared_copy_epoch_bytes",)

@pytest.mark.parametrize("which", ["ops", "bytes"])
def test_shared_read_write_epoch_limit(which):
    x = ledger(grant(shared_ssd_cumulative=Amount(1, 99999) if which=="ops" else Amount(99, 4096)))
    issue(x, "ssd_read")
    d = x.reserve("ssd_write", 1, state(), now_ns=0)
    assert "shared_ssd_epoch_" + which in d.reasons

@pytest.mark.parametrize("stage", ["ssd_write", "d2h"])
def test_opposite_direction_inflight_is_shared(stage):
    x = ledger(grant(shared_ssd_inflight=Amount(1,4096), shared_copy_inflight_bytes=4096))
    s = state(inflight=(Amount(1,4096), Amount(), Amount(1,4096), Amount()))
    d = x.reserve(stage, 1, s, now_ns=0)
    assert d.action == "defer"
    assert any("shared" in r and "inflight" in r for r in d.reasons)

def test_one_fused_batch_charged_once_with_all_physical_destinations():
    x = ledger()
    # Same disk content, two GPU destinations: 4096 + 4096 actual bytes.
    issue(x, "ssd_read", 4096)
    issue(x, "h2d", 8192)
    assert x.snapshot()["used"]["h2d"] == dict(ops=1, bytes=8192)
    assert x.snapshot()["used"]["ssd_read"] == dict(ops=1, bytes=4096)
    assert x.reserve("h2d", 1, state(), now_ns=0).action == "defer"

@pytest.mark.parametrize("reason", ["mandatory", "continuation", "age", "shutdown"])
def test_progress_zero_credit_but_not_native_capacity(reason):
    x = ledger(grant(cumulative=ZERO, inflight=ZERO, shared_copy_cumulative_bytes=0,
                     shared_copy_inflight_bytes=0, shared_ssd_cumulative=Amount(),
                     shared_ssd_inflight=Amount()))
    # For accepted D2H -> SSD write, no new epoch is needed.
    d = issue(x, "ssd_write", progress=reason)
    assert d.reasons and x.snapshot()["progress_overrides"] == 1
    assert x.reserve("d2h", 4096, state(native_issue_safe=False),
        now_ns=0, progress=reason).reasons == ("native_capacity_or_dependency",)
    assert x.reserve("d2h", 4096, state(free_staging_bytes=0),
        now_ns=0, progress=reason, staging_bytes_needed=4096).action == "defer"

def test_staging_reserve_is_separate_from_physical_capacity():
    x=ledger()
    s=state(free_staging_bytes=4096)
    assert x.reserve("ssd_read",4096,s,now_ns=0,staging_bytes_needed=4096).reasons==("staging_reserve",)
    issue(x,"ssd_read",sample=s,progress="mandatory",staging_bytes_needed=4096)

def test_backend_rejection_no_charge_duplicate_and_forged_token_rejected():
    x=ledger()
    d=x.reserve("d2h",4096,state(),now_ns=0)
    with pytest.raises(ValueError): x.settle(replace(d.permit),accepted=True)
    with pytest.raises(ValueError): x.settle(d.permit,accepted=1)
    x.settle(d.permit,accepted=False)
    with pytest.raises(ValueError): x.settle(d.permit,accepted=True)
    assert x.snapshot()["used"]["d2h"]==dict(ops=0,bytes=0)
    issue(x,"d2h")
    # No completion refund API: error CQE still consumed the accepted allowance.
    assert x.snapshot()["used"]["d2h"]==dict(ops=1,bytes=4096)

def test_pending_call_cannot_be_replaced_or_double_reserved():
    x=ledger()
    d=x.reserve("h2d",4096,state(),now_ns=0)
    with pytest.raises(RuntimeError): x.reserve("d2h",4096,state(),now_ns=0)
    with pytest.raises(RuntimeError): x.publish(grant(epoch=1,issued_ns=100,expires_ns=200),now_ns=100)
    x.settle(d.permit,accepted=True)
    assert x.snapshot()["used"]["h2d"]["bytes"]==4096
    x.publish(grant(epoch=1,issued_ns=100,expires_ns=200),now_ns=100)

@pytest.mark.parametrize("g", [
    grant(cumulative=ZERO), grant(epoch=1, issued_ns=50, expires_ns=150),
    grant(run_id="wrong"), grant(epoch=0, expires_ns=99)])
def test_refill_overlap_or_wrong_grant_rejected(g):
    x=ledger()
    with pytest.raises(ValueError): x.publish(g,now_ns=50)

@pytest.mark.parametrize("now,sample", [
    (100,state(captured_ns=100)), (11,state()), (0,state(captured_ns=1))])
def test_expired_or_future_snapshot_returns_native_fallback(now,sample):
    x=ledger()
    d=x.reserve("h2d",1,sample,now_ns=now)
    assert d.action=="native_fallback" and d.permit is None
    assert x.parent_admission(sample,now_ns=now) is None

def test_missing_grant_is_not_unlimited_policy_permission():
    x=DispatchLedger("r")
    assert x.reserve("h2d",1,state(),now_ns=0).action=="native_fallback"

def test_parent_bound_only_applies_before_intake():
    x=ledger()
    assert x.parent_admission(state(accepted_parents=1),now_ns=0)
    assert not x.parent_admission(state(accepted_parents=2),now_ns=0)
    # Already accepted parents are not dropped or prevented from draining.
    issue(x,"ssd_write",sample=state(accepted_parents=3),progress="mandatory")

def test_wrong_owner_and_clock_are_rejected():
    x=ledger()
    with ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError): pool.submit(x.snapshot).result()
    x.parent_admission(state(captured_ns=10),now_ns=10)
    with pytest.raises(ValueError): x.reserve("h2d",1,state(),now_ns=9)

@pytest.mark.parametrize("field,value", [
    ("epoch",True), ("expires_ns",0), ("max_accepted_parents",0),
    ("sample_max_age_ns",None), ("reserve_staging_bytes",-1),
    ("cumulative",list(ZERO)), ("shared_ssd_inflight",None),
    ("shared_copy_inflight_bytes",1.0), ("run_id","")])
def test_strict_frozen_fields(field,value):
    with pytest.raises(ValueError): grant(**{field:value})

@pytest.mark.parametrize("n", [0,-1,True,1.5,None])
def test_physical_bytes_require_positive_integer(n):
    x=ledger()
    with pytest.raises(ValueError): x.reserve("h2d",n,state(),now_ns=0)
