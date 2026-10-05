from copy import deepcopy
import pytest
from analyze_store_readiness import diagnose

def inputs():
    parent=dict(parent_id=3,mandatory=True,queued_files=1,next_file_index=0,inflight_files=0,pending_counts_are_lower_bounds=False)
    bg=dict(parent_id=2,mandatory=False,queued_files=10,total_files=100,d2h_pending_observed=1,ssd_write_pending_observed=7)
    state=dict(stores=[bg,parent],truncated=False,data_inflight=8,iodepth=8,free_staging_slots=0)
    case=dict(wait=dict(start=1.0,seconds=1.0))
    chain=dict(qualified=True,parents=[dict(job_id=3,copy_host_enqueue_from_wait_seconds=.8)])
    ob=dict(run_id="r",faulted=False,mandatory_only=True,truncated_samples=0,overwritten=5,
        records=[dict(first_time_ns=1100000000,last_time_ns=1500000000,samples=3,state=state)])
    return case,chain,ob

def test_joins_only_observed_queued_parent_without_invented_gain():
    a,b,c=inputs();r=diagnose(a,b,c,run_id="r");p=r["parents"][0]
    assert p["retained_sample_count"]==3 and p["preceding_background_parent_ids"]==[2]
    assert p["all_selected_iodepth_full"] and p["all_selected_no_free_staging"]
    assert p["minimum_background_stage_ops"]==8 and r["estimated_speedup"] is None
    assert r["overwritten_state_groups"]==5 and not r["complete_wait_coverage"]
    assert not r["reclaimable_staging_known"] and r["immediately_reusable_gpu_bytes"] is None

@pytest.mark.parametrize("change",["outside","unmatched","already_issued","nonmandatory"])
def test_missing_eligible_states_are_unknown_not_zero_wait(change):
    a,b,c=inputs();p=c["records"][0]["state"]["stores"][-1]
    if change=="outside":c["records"][0]["last_time_ns"]=1900000000
    elif change=="unmatched":p["parent_id"]=99
    elif change=="already_issued":p["next_file_index"]=1
    else:p["mandatory"]=False
    x=diagnose(a,b,c,run_id="r")["parents"][0]
    assert x["retained_sample_count"]==0 and x["all_selected_iodepth_full"] is None

@pytest.mark.parametrize("bad",["run","fault","truncated","duplicate","order","unqualified","count"])
def test_rejects_inconsistent_or_partial_evidence(bad):
    a,b,c=inputs()
    if bad=="run":c["run_id"]="x"
    elif bad=="fault":c["faulted"]=True
    elif bad=="truncated":c["truncated_samples"]=1
    elif bad=="duplicate":c["records"][0]["state"]["stores"].append(deepcopy(c["records"][0]["state"]["stores"][-1]))
    elif bad=="order":c["records"]*=2
    elif bad=="unqualified":b["qualified"]=False
    elif bad=="count":c["records"][0]["samples"]=True
    with pytest.raises(ValueError):diagnose(a,b,c,run_id="r")
