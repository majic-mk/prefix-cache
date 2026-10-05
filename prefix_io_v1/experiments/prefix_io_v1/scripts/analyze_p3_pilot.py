"""Summarize P3 evidence without promoting screening to runtime calibration."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
from prefix_io_control.token_timeline import percentile
ROOT=Path(__file__).resolve().parents[3]
RUNS=ROOT/"experiments/prefix_io_v1/runs"

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def screen_cost(prefix):
    paths={mode:RUNS/(prefix+"-"+mode+"-01")/"details/result.json" for mode in ["cold","populate","paired"]}
    ds={k:json.loads(p.read_text()) for k,p in paths.items()}
    configs=[json.loads((p.parent/"frozen-config.json").read_text()) for p in paths.values()]
    for c in configs[1:]:
        for key in ["gpu_uuid","sizes","reps","sampling","pythonhashseed","model"]:
            assert c[key]==configs[0][key],key
        for key in ["dtype","kv_cache_dtype","kv_cache_memory_bytes","max_model_len","max_num_batched_tokens","attention_config"]:
            assert c["engine"][key]==configs[0]["engine"][key],key
    assert all(d["status"].startswith("PASSED_NATIVE_") for d in ds.values())
    cold={(r["prefix_tokens"],r["rep"]):r for r in ds["cold"]["rows"]}
    store={(r["prefix_tokens"],r["rep"]):r for r in ds["populate"]["rows"] if r["kind"]=="store"}
    paired={(r["prefix_tokens"],r["rep"],r["kind"]):r for r in ds["paired"]["rows"]}
    rows=[];pairs_equal=True;read_bytes=0
    for (n,rep),f in cold.items():
        s=paired[n,rep,"g_ssd"];m=paired[n,rep,"g_mem"];p=store[n,rep]
        same=all(r["prompt_token_ids"]==f["prompt_token_ids"] and r["output_token_ids"]==f["output_token_ids"] for r in [s,m,p])
        pairs_equal &= same
        for r,is_ssd in [(s,True),(m,False)]:
            t=r["trace"];actual=t["foreground_logical_read_bytes"]+t["preload_actual_read_bytes"]
            assert t["transfer_success"] and t["h2d_events"]>0 and r["num_cached_tokens"]==n
            assert actual==(n*57344 if is_ssd else 0)
            if is_ssd:read_bytes+=actual
        rows.append(dict(prefix_tokens=n,rep=rep,warmup=f["warmup"],outputs_equal=same,
            f_seconds=f["metrics"]["first_token_latency"],g_ssd_seconds=s["metrics"]["first_token_latency"],
            g_mem_seconds=m["metrics"]["first_token_latency"]))
    summary={}
    for n in sorted({r["prefix_tokens"] for r in rows}):
        measured=[r for r in rows if r["prefix_tokens"]==n and not r["warmup"]]
        summary[str(n)]={k:dict(n=len(measured),median=statistics.median(r[k] for r in measured),
            samples=[r[k] for r in measured]) for k in ["f_seconds","g_ssd_seconds","g_mem_seconds"]}
        summary[str(n)]["ssd_faster_in_all_measured_pairs"]=all(r["g_ssd_seconds"]<r["f_seconds"] for r in measured)
    return dict(summary=summary,raw_pairs=rows,pairs_equal=pairs_equal,actual_ssd_read_bytes=read_bytes,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in paths.values()},
        runtime_table_exported=False,screening_only=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    assert a.output.resolve().is_relative_to(ROOT/"artifacts/prefix_io_v1")
    summaries={};sources={}
    for label in ["server07-p3-mixed-02","server07-p3-mixed-03","server07-p3-lowreuse-01","server07-p3-lowcontention-01"]:
        path=RUNS/label/"details/result.json";d=json.loads(path.read_text());sources[str(path.relative_to(ROOT))]=digest(path)
        assert d["status"]=="PASSED_NATIVE_P3_DEVELOPMENT_TRACE" and d["engine_shutdown"]=="completed"
        rows=d["rows"];assert all(r["per_token_complete"] and len(r["engine_token_timestamps"])==128 for r in rows)
        assert all(h["aio"]["outstanding"]==0 and h["aio"]["accepted"]==h["aio"]["reaped"] for h in d["probe"]["handlers"])
        profile=RUNS/label/"details/results_engine_core_0.json"
        events=json.loads(profile.read_text())["traceEvents"]
        planner=collections.Counter(e.get("args",{}).get("reason") for e in events if e.get("name")=="py_kvcache.planner.decline")
        before=d["cohort_probe_start"];after=d["probe"]
        def delta(key):return after[key]-before[key] if key in before and key in after else None
        summaries[label]=dict(requests=len(rows),generated_tokens=sum(len(r["output_tokens"]) for r in rows),
            median_ttft_ms=1000*percentile([r["metrics"]["first_token_latency"] for r in rows],.5),
            p95_ttft_ms=1000*percentile([r["metrics"]["first_token_latency"] for r in rows],.95),
            median_of_request_itl_p95_ms=1000*percentile([r["itl_p95_seconds"] for r in rows],.5),
            pooled_token_itl_p95_ms=1000*percentile([t for r in rows for t in r["itl_seconds"]],.95),
            max_client_send_lateness_ms=1000*max(r["client_queue_seconds"] for r in rows),
            cohort_seconds_including_drain=d["cohort_seconds_including_drain"],
            tail_drain_seconds=d["tail_drain_seconds"],completed_throughput=d["completed_throughput_per_second"],
            cohort_write_bytes=delta("write_bytes"),cohort_read_bytes=delta("read_bytes"),
            cohort_pending_flush_wait_calls=delta("pending_flush_wait_calls"),
            cohort_pending_flush_wait_seconds=delta("pending_flush_wait_seconds"),
            cohort_foreground_slot_attempts=delta("foreground_slot_attempts"),
            cohort_foreground_slot_unavailable=delta("foreground_slot_unavailable"),
            full_session_min_free_staging_slots=after["min_free_staging_slots"],
            full_session_planner_decline_reasons=dict(planner),
            no_ordinary_quota=True,goodput=None)
    output=dict(status="P3_DEVELOPMENT_SCREEN_ONLY",native_replays=summaries,source_hashes=sources,
        long_cost_screen=screen_cost("server07-p3-long"),
        eight_k_cost_screen=screen_cost("server07-p3-8k"),
        sixteen_k_cost_screen=screen_cost("server07-p3-16k"),
        limits=["Controlled synthetic-token traces, not production QA or held-out evaluation.",
            "All requests retained including JIT spikes; only predefined warmup excluded.",
            "Per-token timestamps are actual engine output events, not client network token latency.",
            "No SLO registered; completed throughput is not SLO goodput.",
            "P1 curve used unchanged for native short-prompt replay; batch-4 accuracy not requalified.",
            "Extended costs have only two measured trials per point and different frozen budgets/context from short replay.",
            "No ordinary quota, fixed/pressure or joint production integration yet.",
            "Zero observed waits in these traces cannot establish universal absence of a bottleneck."])
    with a.output.open("x") as f:json.dump(output,f,indent=2);f.write("\n")
    print(json.dumps(output,indent=2))
if __name__=="__main__":main()
