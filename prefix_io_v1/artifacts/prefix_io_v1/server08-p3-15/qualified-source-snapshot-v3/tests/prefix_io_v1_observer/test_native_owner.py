from types import SimpleNamespace as NS
from weakref import ref
import pytest
from prefix_io_control.native_owner import NativeOwnerProbe
class Owner(NS):pass
def setup():
    p=NativeOwnerProbe("run")
    block=Owner(block_id=1,ref_cnt=0,is_null=False)
    block.prev_free_block=Owner(next_free_block=block)
    block.next_free_block=Owner(prev_free_block=block)
    pool=Owner(blocks={1:block});sched=Owner(_block_id_to_pending_jobs={},_jobs={})
    p.pool=ref(pool);p.scheduler=ref(sched);p.generations[1]=1;p.sequence=1
    p.watched[1]=dict(generation=1,parents={10,11},acked={10,11})
    return p,pool,sched,block

@pytest.mark.parametrize("bad",["ref","fence","partial_ack","generation","unlinked","null"])
def test_no_release_without_complete_owner_proof(bad):
    p,pool,sched,b=setup()
    assert p._state(1)["native_reusable"]
    if bad=="ref":b.ref_cnt=1
    elif bad=="fence":sched._block_id_to_pending_jobs[1]={12}
    elif bad=="partial_ack":p.watched[1]["acked"]={10}
    elif bad=="generation":p.generations[1]=2
    elif bad=="unlinked":b.next_free_block.prev_free_block=None
    elif bad=="null":b.is_null=True
    assert not (p._state(1) or {}).get("native_reusable",False)

def test_reallocation_advances_identity_and_resets_parent_evidence():
    p,pool,sched,b=setup();before={1:p._state(1)}
    b.ref_cnt=1;b.prev_free_block=b.next_free_block=None
    p._allocated(pool,[b],before)
    assert p.actual_safe_reallocations==1
    assert p.generations[1]==2 and p.watched[1]["parents"]==set()
    assert not p._state(1)["native_reusable"]

def test_oversize_allocation_fails_closed_without_touching_native_blocks():
    p,pool,sched,b=setup();p._safe(p._allocated,pool,[b]*65,{})
    assert p.faulted and p.errors==1 and b.ref_cnt==0
    assert p.actual_safe_reallocations==0

def test_ack_only_after_original_native_parent_retirement():
    p,pool,sched,b=setup();p.watched[1]["acked"]=set()
    sched._jobs={10:object()}
    p._completed(sched,NS(kv_connector_worker_meta=NS(completed_jobs={10:1,11:1})))
    assert p.watched[1]["acked"]=={11}
    del sched._jobs[10]
    p._completed(sched,NS(kv_connector_worker_meta=NS(completed_jobs={10:1})))
    assert p._state(1)["native_reusable"]

def test_generations_stay_bounded_and_preserve_watched_identity():
    p,pool,sched,b=setup()
    for i in range(2,160):
        other=NS(block_id=i,ref_cnt=1)
        p._allocated(pool,[other],{})
    assert len(p.generations)==64 and p.generations[1]==1
