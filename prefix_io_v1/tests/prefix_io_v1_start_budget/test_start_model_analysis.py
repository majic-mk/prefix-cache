import json
import pytest
from analyze_start_budget_model import analyze

def reports(tmp_path):
    paths={}
    for mode in ("off","fixed","pressure"):
        a=dict(valid=True,outstanding_records=0,stages={s:dict(accepted_bytes=1,transferred_bytes=1,failed_ops=0,inflight_bytes=0) for s in ("h2d","d2h","ssd_read","ssd_write")})
        b=dict(faulted=False,errors=0)
        r=dict(status="PASSED_NATIVE",engine_shutdown="completed",policy_mode=mode,full_per_stage_caps=False,
            manifest_sha256="same",cohort_seconds_including_drain=10.,tail_drain_seconds=.1,
            cohort_read_bytes=100,cohort_write_bytes=50,
            rows=[dict(request_id=str(i),output_tokens=[i]*128,per_token_complete=True,
                metrics=dict(first_token_latency=.2),itl_p95_seconds=.01) for i in range(10)],
            probe=dict(pending_flush_wait_calls=0,pending_flush_wait_seconds=0.),
            final_probe=dict(start_budget_final=[dict(drained=True,budget=None if mode=="off" else b,accounting=None if mode=="off" else a)]))
        p=tmp_path/(mode+".json");p.write_text(json.dumps(r));paths[mode]=p
    return paths

def test_equal_output_is_not_promoted_to_research_speedup(tmp_path):
    result=analyze(reports(tmp_path))
    assert result["status"]=="PASS_REAL_MODEL_START_INTEGRATION"
    assert result["speedup_established"] is False and result["confidence_interval"] is None
    assert result["formal_goodput"] is False

@pytest.mark.parametrize("bad",["tokens","unfinished","faulted","bytes","manifest","shutdown"])
def test_false_success_rejected(tmp_path,bad):
    paths=reports(tmp_path);p=paths["fixed"];r=json.loads(p.read_text())
    if bad=="tokens":r["rows"][0]["output_tokens"][0]=999
    if bad=="unfinished":r["rows"].pop()
    if bad=="faulted":r["final_probe"]["start_budget_final"][0]["budget"]["faulted"]=True
    if bad=="bytes":r["final_probe"]["start_budget_final"][0]["accounting"]["stages"]["h2d"]["transferred_bytes"]=0
    if bad=="manifest":r["manifest_sha256"]="different"
    if bad=="shutdown":r["engine_shutdown"]="failed"
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):analyze(paths)
