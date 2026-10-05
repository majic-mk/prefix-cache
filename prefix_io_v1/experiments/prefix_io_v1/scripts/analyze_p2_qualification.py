"""Analyze bounded P2 qualification; never promote latency proxies to goodput."""
import argparse,hashlib,json,statistics
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parents[3]
    assert a.output.resolve().is_relative_to(root/"artifacts/prefix_io_v1")
    runs=root/"experiments/prefix_io_v1/runs"
    inputs={}
    def read(label):
        p=runs/("server07-p2-"+label)/"details/result.json"
        inputs[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    labels=["off-01","shadow-01","shadow-02","off-02"]
    results={label:read(label) for label in labels}
    baseline=results["off-01"]
    comparable=["index","kind","input_tokens","output_tokens","num_cached_tokens"]
    summaries={}
    for label,result in results.items():
        assert result["status"].startswith("PASSED") and result["engine_shutdown"]=="completed"
        assert len(result["rows"])==32
        assert all(all(x[k]==y[k] for k in comparable) for x,y in zip(baseline["rows"],result["rows"]))
        rows=[r for r in result["rows"] if not r["warmup"]]
        handlers=result["probe"]["handlers"]
        assert all(h["observation_failures"]==0 and not h["faulted"] for h in handlers)
        summaries[label]=dict(
            measured_requests=len(rows),
            total_measured_generate_call_seconds=sum(r["wall_seconds"] for r in rows),
            median_ttft_seconds=statistics.median(r["metrics"]["first_token_latency"] for r in rows),
            mean_decode_span_per_interval_seconds=statistics.mean((r["metrics"]["last_token_ts"]-r["metrics"]["first_token_ts"])/15 for r in rows),
            publications=sum(h["publications"] for h in handlers),
            aio_accepted=sum(h["aio"]["accepted"] for h in handlers),
            aio_reaped=sum(h["aio"]["reaped"] for h in handlers),
            outstanding_at_probe=sum(h["aio"]["outstanding"] for h in handlers),
            cold_hot_mismatch_indices=[x["index"] for x in result["cold_hot_token_mismatches"]])
    pairs=[]
    for idx in ["01","02"]:
        off=summaries["off-"+idx]["total_measured_generate_call_seconds"]
        on=summaries["shadow-"+idx]["total_measured_generate_call_seconds"]
        pairs.append(dict(pair=idx,relative_generate_call_time_increase_pct=100*(on/off-1)))
    total_off=sum(summaries["off-"+i]["total_measured_generate_call_seconds"] for i in ["01","02"])
    total_on=sum(summaries["shadow-"+i]["total_measured_generate_call_seconds"] for i in ["01","02"])
    owner=read("owner-02")
    assert all(all(x[k]==y[k] for k in comparable) for x,y in zip(baseline["rows"],owner["rows"]))
    witness=owner["probe"]["owner"]
    assert not witness["faulted"] and not witness["errors"]
    assert witness["actual_safe_reallocations"]>0
    progress=read("progress-05")
    assert progress["status"].startswith("PASSED") and progress["producer_pending_before_wait"]
    assert all(c["aio"]["closed"] and c["aio"]["drained"] and not c["aio"]["outstanding"] and
        c["aio"]["accepted"]==c["aio"]["completed"]==c["aio"]["reaped"] for c in progress["cases"])
    result=dict(status="PASSED_BOUNDED_P2_EVIDENCE",source_hashes=inputs,
        mode_output_equivalence=dict(requests_per_arm=32,arms=5,tokens_per_request=16,all_corresponding_outputs_equal=True),
        mandatory_pending_producer_roundtrip=True,owner_witness={k:v for k,v in witness.items() if k!="records"},
        latency_runs=summaries,pairs=pairs,
        pooled_generate_call_time_increase_pct=100*(total_on/total_off-1),
        engineering_2pct_goodput_target_verified=False,formal_performance_claim=False,
        limitations=[
            "Two runs per mode on one fixed diagnostic token sequence; not independent service-trace seeds.",
            "Wall time covers generate calls; excludes inter-call report writes and final drain. It is NOT cohort goodput.",
            "First/last token span divided by 15 is an average, not individual ITL or ITL P95.",
            "One cold/hot output mismatch reproduces identically with observation off; cause unassigned.",
            "Owner probe is a separate diagnostic excluded from observer overhead. No live controller ownership adapter.",
            "Zero fenced-free observations in this low-pressure trace do not establish absence at production load.",
            "P2 mandatory zero-credit gate is test-only; no production ordinary quotas installed."])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x") as f:json.dump(result,f,indent=2);f.write("\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
