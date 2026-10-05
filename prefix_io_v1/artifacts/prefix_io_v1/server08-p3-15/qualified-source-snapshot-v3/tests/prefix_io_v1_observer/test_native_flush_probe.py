import ast
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
import sys
import pytest
from prefix_io_control.native_flush_probe import NativeFlushProbe, REASONS

class Owner(NS): pass

def setup():
    blocks=[Owner(block_id=i,ref_cnt=1,is_null=False,prev_free_block=None,next_free_block=None) for i in range(16)]
    pool=Owner(blocks=blocks)
    scheduler=Owner(_jobs={},_block_id_to_pending_jobs={})
    p=NativeFlushProbe("test")
    p.safe(p.allocated,pool,blocks)
    return p,pool,scheduler

def store(p,s,ids=(1,),jid=9):
    s._jobs[jid]=NS(req_id="owner",pending_count=1,is_store=True,non_sliding_window_block_ids=list(ids))
    p.safe(p.stores,s,{jid:object()})

@pytest.mark.parametrize("reason",sorted(REASONS))
def test_precise_reason_owner_and_no_release_credit(reason):
    p,pool,s=setup();store(p,s)
    s._block_id_to_pending_jobs[1]={9}
    p.safe(p.flush,s,reason,"trigger",[1] if reason in ("new_allocation","restore_destination") else None,{9})
    p.safe(p.seal,NS(jobs_to_flush={9}))
    e=p.export()["records"][-1]
    assert e["causes"][0]["reason"]==reason
    parent=e["causes"][0]["parents"][0]
    assert parent["req_id"]=="owner" and parent["source_blocks"][0]["same_generation"] is True
    assert parent["source_blocks"][0]["protecting_jobs"]==[9]
    assert p.export()["immediately_reusable_bytes"] is None

def test_reallocation_preserves_old_parent_generation_as_distinct():
    p,pool,s=setup();store(p,s)
    s._block_id_to_pending_jobs[1]={9}
    p.safe(p.allocated,pool,[pool.blocks[1]])
    p.safe(p.flush,s,"new_allocation","new",[1],None)
    b=p.causes[0]["parents"][0]["source_blocks"][0]
    assert b["same_generation"] is False and b["active_refs"]==1
    assert b["protecting_jobs"]==[9] and not p.faulted

def test_unknown_preinstallation_generation_never_becomes_proof():
    p,pool,s=setup();p.generations[1]=None;store(p,s)
    p.safe(p.flush,s,"cache_reset",None,None,{9})
    assert p.causes[0]["parents"][0]["source_blocks"][0]["same_generation"] is None

def test_completion_requires_native_removal_not_just_one_ack():
    p,pool,s=setup();store(p,s)
    output=NS(kv_connector_worker_meta=NS(completed_jobs={9:1}))
    p.safe(p.completed,s,output)
    assert p.retired_jobs==0
    del s._jobs[9]
    p.safe(p.completed,s,output)
    assert p.retired_jobs==1 and p.records[-1]["kind"]=="native_parent_retired"

def test_long_allocations_supported_but_pool_domain_is_bounded():
    p,pool,s=setup()
    pool.blocks=[Owner(block_id=i) for i in range(2340)]
    p.safe(p.allocated,pool,pool.blocks[:1016])
    assert not p.faulted and p.allocations==1032
    original=p.sequence
    pool.blocks=[Owner(block_id=i) for i in range(4097)]
    p.safe(p.allocated,pool,[])
    assert p.faulted and p.errors==1 and p.sequence==original

def test_source_sampling_and_record_caps_are_explicit():
    p,pool,s=setup();store(p,s,range(16))
    assert list(p.jobs[9]["samples"])==[0,1,2,3,12,13,14,15]
    assert p.jobs[9]["source_truncated"]
    for i in range(160):p.append(dict(kind="synthetic",i=i))
    assert len(p.records)==128 and p.dropped_records==32
    for i in range(20):p.safe(p.flush,s,"all_requests_finished",None,None,{9})
    assert len(p.causes)==16 and p.dropped_causes==4

def test_parent_cap_and_flush_count_do_not_hide_truncation():
    p,pool,s=setup()
    s._block_id_to_pending_jobs[1]=set(range(40))
    p.safe(p.flush,s,"restore_destination","new",[1],None)
    assert p.causes[0]["parents_truncated"] and len(p.causes[0]["parent_ids"])==32
    p.safe(p.seal,NS(jobs_to_flush=set(range(40))))
    assert p.records[-1]["parent_count"]==40 and p.records[-1]["parents_truncated"]

def test_multiple_pool_and_thread_changes_fail_only_observer():
    p,pool,s=setup()
    p.safe(p.allocated,Owner(blocks=pool.blocks),[])
    assert p.faulted
    p,pool,s=setup();p.owner=-1
    assert p.safe(lambda:1) is None and p.errors==1
    assert pool.blocks[1].ref_cnt==1

def scheduler_ast():
    root=Path(__file__).resolve().parents[2]
    path=Path("third_party/work/vllm-author-build/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py")
    return ast.parse((root/path).read_text()),ast.parse((root/"artifacts/prefix_io_v1/server07-p3-07/before"/path).read_text())

def test_native_algorithm_ast_unchanged_after_removing_only_diagnostic_hooks():
    current,old=scheduler_ast()
    class Strip(ast.NodeTransformer):
        def visit_ClassDef(self,node):
            self.generic_visit(node)
            if node.name=="OffloadingConnectorScheduler":
                node.body=[n for n in node.body if not
                    (isinstance(n,ast.FunctionDef) and n.name=="_observe_prefix_flush") and not
                    (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="_prefix_flush_probe" for t in n.targets))]
            return node
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=="_observe_prefix_flush":
                return None
            return self.generic_visit(node)
    assert ast.dump(Strip().visit(current),include_attributes=False)==ast.dump(old,include_attributes=False)

def test_all_five_native_flush_sites_have_exact_reason_hook():
    tree,_=scheduler_ast()
    found=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="_observe_prefix_flush":
            found.append(n.args[0].value)
    assert len(found)==5 and set(found)==REASONS

def test_default_off_native_hook_requires_no_observation_import_or_allocation():
    tree,_=scheduler_ast()
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="OffloadingConnectorScheduler")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_observe_prefix_flush")
    module=ast.Module(body=[fn],type_ignores=[])
    namespace={};exec(compile(ast.fix_missing_locations(module),"<native-hook>","exec"),namespace)
    assert namespace["_observe_prefix_flush"](NS(_prefix_flush_probe=None),"new_allocation") is None

def test_install_calls_native_once_and_uninstall_restores(monkeypatch):
    calls=[]
    class Pool:
        def __init__(self):self.blocks=[Owner(block_id=0)]
        def get_new_blocks(self):calls.append("alloc");return self.blocks
    class Scheduler:
        _prefix_flush_probe=None
        def _build_store_jobs(self):calls.append("store");return {}
        def build_connector_meta(self):calls.append("meta");return NS(jobs_to_flush=set())
        def update_connector_output(self,output):calls.append("complete")
    for name,attr,obj in [
        ("vllm.v1.core.block_pool","BlockPool",Pool),
        ("vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler","OffloadingConnectorScheduler",Scheduler)]:
        m=ModuleType(name);setattr(m,attr,obj);monkeypatch.setitem(sys.modules,name,m)
    old=Pool.get_new_blocks
    p=NativeFlushProbe("test");p.install()
    pool=Pool();s=Scheduler()
    assert pool.get_new_blocks() is pool.blocks
    assert s._build_store_jobs()=={} and not s.build_connector_meta().jobs_to_flush
    s.update_connector_output(NS())
    p.owner=-1  # faulting observation still preserves original call/result
    assert pool.get_new_blocks() is pool.blocks and p.faulted
    p.uninstall()
    assert Pool.get_new_blocks is old and Scheduler._prefix_flush_probe is None
    assert calls==["alloc","store","meta","complete","alloc"]
