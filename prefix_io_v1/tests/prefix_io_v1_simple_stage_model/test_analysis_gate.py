"""Failures and ambiguous timelines must never become development speedups."""
import json
from pathlib import Path
import pytest
from analyze_simple_stage_p3 import audit

def fixture():
    reference={"rows":[dict(family="f",kind="gpu_hot",prompt_token_ids=[1],output_tokens=list(range(128)))]}
    rows=[]
    for i in range(10):
        times=[100.+j/8 for j in range(128)]
        rows.append(dict(request_id="c2-"+str(i),family="f",num_cached_tokens=1,
            output_tokens=list(range(128)),engine_token_timestamps=times,itl_seconds=[.125]*127,
            per_token_complete=True,ambiguous_events=[],metrics=dict(is_corrupted=False,first_token_latency=.2),itl_p95_seconds=.125))
    stats={s:dict(accepted_ops=0,accepted_bytes=0,inflight_ops=0,inflight_bytes=0) for s in ("ssd_read","ssd_write","h2d","d2h")}
    control=dict(physical_drained=True,admission=dict(accepted_parents=0,peak_accepted_parents=1),
        stage_accounting=dict(valid=True,outstanding_records=0,stages=stats),controller=None)
    result=dict(status="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY",rows=rows,engine_shutdown="completed",
        source_preservation=dict(checked_files=3048,changed=[]),final_probe=dict(simple_native_control=[control]),
        probe=dict(flush_wait_calls=0,pending_flush_wait_calls=0,pending_flush_wait_seconds=0.,foreground_slot_unavailable=0),
        policy_mode="off",profile="all_hit",cohort_seconds_including_drain=20.,
        cohort_read_bytes=100,cohort_write_bytes=0)
    return result,reference,dict(families=[dict(name="f",tokens=[1])])

def run(tmp_path,result,reference,manifest):
    a=tmp_path/"result.json"; b=tmp_path/"reference.json"
    a.write_text(json.dumps(result));b.write_text(json.dumps(reference))
    return audit(a,b,manifest)

def test_success_keeps_development_scope_and_complete_drained_tokens(tmp_path):
    result,reference,manifest=fixture();value=run(tmp_path,result,reference,manifest)
    assert value["output_tokens"]==1280 and value["formal_goodput"] is False and value["slo"] is None
    assert value["unseen_evaluation"] is False

@pytest.mark.parametrize("failure",["run","output","timestamp","itl","ambiguous","duplicate","source","drain","admission","stage","quota"])
def test_failure_cannot_be_analyzed_as_effect(tmp_path,failure):
    r,ref,m=fixture();c=r["final_probe"]["simple_native_control"][0]
    if failure=="run":r["status"]="FAILED"
    elif failure=="output":r["rows"][0]["output_tokens"][0]=999
    elif failure=="timestamp":r["rows"][0]["engine_token_timestamps"][1]=100.
    elif failure=="itl":r["rows"][0]["itl_seconds"][0]=.25
    elif failure=="ambiguous":r["rows"][0]["ambiguous_events"]=[1]
    elif failure=="duplicate":r["rows"][0]["request_id"]="c2-1"
    elif failure=="source":r["source_preservation"]["changed"]=["file"]
    elif failure=="drain":c["physical_drained"]=False
    elif failure=="admission":c["admission"]["peak_accepted_parents"]=65
    elif failure=="stage":c["stage_accounting"]["stages"]["h2d"]["inflight_bytes"]=1
    elif failure=="quota":c["controller"]=dict(faulted=True)
    with pytest.raises((ValueError,KeyError)):run(tmp_path,r,ref,m)
