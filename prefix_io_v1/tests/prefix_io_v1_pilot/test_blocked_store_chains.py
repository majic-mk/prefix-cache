import copy
import pytest
from analyze_blocked_store_chains import chain,fence

def fixture():
    block=dict(block=3,owner_known=True,generation=9,source_generation=4,same_generation=False,
        active_refs=1,free_queue_linked=False,is_null=False,parents_truncated=False,protecting_jobs=[7])
    parent=dict(job_id=7,req_id="r",is_store=True,pending_worker_acks=1,source_sample_known=True,
        source_truncated=False,source_count=1,source_blocks=[block])
    wait=dict(start=100.,seconds=1.,ids=[7],ids_truncated=False)
    case=dict(unique_match=True,wait=wait,batch=dict(parent_ids=[7],parents_truncated=False,
        causes=[dict(reason="restore_destination",parents_truncated=False,parents=[parent])]))
    def event(name,ts,dur,**extra):
        return dict(name=name,ph="X",ts=ts,dur=dur,pid=21,args=dict(job_id=7,req_id="r",num_bytes=4096,**extra))
    trace=dict(start_ts_ns=99_000_000_000,traceEvents=[
        event("py_kvcache.transfer",0,1_920_000,direction="gpu_to_storage",success=True,num_files=1,num_blocks=1),
        event("py_kvcache.cuda_staging",1_700_000,80,direction="gpu_to_storage",sample=0,batch_size=1),
        event("py_kvcache.file_write",1_710_000,190_000,file_index=0,num_files=1)])
    retired=copy.deepcopy(block);retired["protecting_jobs"]=[]
    records=[dict(kind="native_parent_retired",job_id=7,req_id="r",source_truncated=False,source_blocks=[retired])]
    return case,trace,records

def test_complete_destination_fence_does_not_award_free_capacity():
    c,t,r=fixture();result=fence(c,t,r)
    assert result["qualified"] and result["protected_parent_bytes"]==4096
    assert result["before_first_copy_enqueue_seconds"]==pytest.approx(.7)
    assert result["immediately_reusable_bytes"] is None and not result["ready_before_dispatch_known"]
    assert result["causal_speedup_estimate"] is None
    assert result["parents"][0]["copy_cuda_event_elapsed_seconds"]==pytest.approx(.00008)

@pytest.mark.parametrize("mutation",["wait_truncated","batch_truncated","source_truncated","unknown_owner","same_generation",
    "missing_retirement","different_retired_generation","still_protected","missing_copy","wrong_request","duplicate_write",
    "wrong_direction","byte_mismatch","other_pid","multi_file","nonfinite"])
def test_incomplete_or_ambiguous_data_cannot_qualify(mutation):
    c,t,r=fixture();p=c["batch"]["causes"][0]["parents"][0];b=p["source_blocks"][0]
    if mutation=="wait_truncated":c["wait"]["ids_truncated"]=True
    elif mutation=="batch_truncated":c["batch"]["parents_truncated"]=True
    elif mutation=="source_truncated":p["source_truncated"]=True
    elif mutation=="unknown_owner":b["owner_known"]=False
    elif mutation=="same_generation":b["same_generation"]=True;b["generation"]=b["source_generation"]
    elif mutation=="missing_retirement":r.clear()
    elif mutation=="different_retired_generation":r[0]["source_blocks"][0]["generation"]=10
    elif mutation=="still_protected":r[0]["source_blocks"][0]["protecting_jobs"]=[7]
    elif mutation=="missing_copy":t["traceEvents"].pop(1)
    elif mutation=="wrong_request":t["traceEvents"][1]["args"]["req_id"]="sibling"
    elif mutation=="duplicate_write":t["traceEvents"].append(copy.deepcopy(t["traceEvents"][2]))
    elif mutation=="wrong_direction":t["traceEvents"][1]["args"]["direction"]="storage_to_gpu"
    elif mutation=="byte_mismatch":t["traceEvents"][2]["args"]["num_bytes"]=8192
    elif mutation=="other_pid":t["traceEvents"][1]["pid"]=99
    elif mutation=="multi_file":t["traceEvents"][0]["args"]["num_files"]=2
    elif mutation=="nonfinite":t["traceEvents"][1]["ts"]=float("nan")
    result=fence(c,t,r)
    assert not result["qualified"] and result["immediately_reusable_bytes"] is None
    assert result["causal_speedup_estimate"] is None

def test_other_process_stage_is_not_joined_to_this_parent():
    c,t,r=fixture();decoy=copy.deepcopy(t["traceEvents"][1]);decoy["pid"]=99;t["traceEvents"].append(decoy)
    assert fence(c,t,r)["qualified"]
