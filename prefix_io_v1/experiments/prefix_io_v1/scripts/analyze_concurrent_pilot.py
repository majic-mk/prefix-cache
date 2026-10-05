"""Offline P3 problem screen; association is not a strategy benefit estimate."""
import argparse,bisect,hashlib,json,statistics
from pathlib import Path
from prefix_io_control.token_timeline import percentile

def merged(intervals):
    result=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1][1]=max(b,result[-1][1])
        else:result.append([a,b])
    return result

def overlaps(intervals,a,b):
    if b<=a:return False
    i=bisect.bisect_left(intervals,[b])
    return i>0 and intervals[i-1][1]>a

def distribution(values):
    return dict(n=len(values),median=statistics.median(values) if values else None,
                p95=percentile(values,.95) if values else None,max=max(values) if values else None)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summarize(folder,references):
    r=json.loads((folder/"result.json").read_text());c=json.loads((folder/"frozen-config.json").read_text())
    assert r["status"]=="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY" and r["engine_shutdown"]=="completed"
    assert c["engine"]["max_num_seqs"]==2 and c["engine"]["kv_cache_memory_bytes"]==2147483648
    assert r["source_preservation"]==dict(checked_files=3048,changed=[])
    assert len(r["rows"])==10 and r["cache_resets"]==0 and not r["ordinary_quota_installed"]
    for h in r["final_probe"]["handlers"]:
        a=h["aio"]
        assert a["outstanding"]==0 and a["accepted"]==a["completed"]==a["reaped"]
        assert all(a[k]==0 for k in ("pending","ready","unreaped"))
        assert a["fatal"] is None and h["staging_bytes"]<=1073741824 and h["observation_failures"]==0
    trace=json.loads((folder/"native-cohort.trace.json").read_text());origin=trace["start_ts_ns"]/1e9
    events=trace["traceEvents"];io=[]
    names={"py_kvcache.cuda_staging","py_kvcache.file_read","py_kvcache.file_write","py_kvcache.preload.file_read"}
    for e in events:
        if e.get("name") in names and e.get("ph")=="X":
            start=origin+e["ts"]/1e6
            io.append((start,start+e["dur"]/1e6,e.get("args",{}).get("req_id")))
    before=r["cohort_probe_start"];after=r["probe"]
    counters={k:after[k]-before[k] for k in ["wait_calls","flush_wait_calls","pending_flush_wait_calls",
        "pending_flush_wait_seconds","foreground_slot_attempts","foreground_slot_unavailable"]}
    rows=[];all_itls=[];with_io=[];without_io=[];mismatches=[]
    prefill=merged([(x["metrics"]["scheduled_ts"],x["metrics"]["first_token_ts"]) for x in r["rows"] if x["num_cached_tokens"]==0])
    io_without_cold_prefill=[];quiet_without_cold_prefill=[]
    for row in r["rows"]:
        assert row["per_token_complete"] and not row["metrics"]["is_corrupted"]
        times=row["engine_token_timestamps"];tokens=row["output_tokens"]
        assert len(times)==len(tokens)==128 and len(row["itl_seconds"])==127 and times==sorted(times)
        kind="cold" if row["num_cached_tokens"]==0 else "gpu_hot"
        ref=references[(row["family"],kind)]
        mismatch=[i for i,(a,b) in enumerate(zip(tokens,ref)) if a!=b]
        if mismatch:mismatches.append(dict(request_id=row["request_id"],family=row["family"],cached_tokens=row["num_cached_tokens"],reference_kind=kind,differing_token_indices=mismatch))
        other_io=merged([(a,b) for a,b,rid in io if rid!=row["internal_request_id"]])
        local_io=[];local_quiet=[]
        for a,b in zip(times,times[1:]):
            gap=b-a;all_itls.append(gap);active=overlaps(other_io,a,b)
            (with_io if active else without_io).append(gap)
            (local_io if active else local_quiet).append(gap)
            if not overlaps(prefill,a,b):
                (io_without_cold_prefill if active else quiet_without_cold_prefill).append(gap)
        rows.append(dict(request_id=row["request_id"],family=row["family"],initial_ssd_present=row["initial_ssd_present"],
            num_cached_tokens=row["num_cached_tokens"],native_reference_exact=not mismatch,
            ttft_seconds=row["metrics"]["first_token_latency"],
            scheduled_ttft_seconds=row["client_queue_seconds"]+row["metrics"]["first_token_latency"],
            frontend_late_seconds=row["client_queue_seconds"],itl=distribution(row["itl_seconds"]),
            itl_with_other_request_io=distribution(local_io),itl_without_other_request_io=distribution(local_quiet)))
    return dict(label=folder.parent.name,profile=r["profile"],requests=10,token_events=1280,
        cohort_seconds=r["cohort_seconds_including_drain"],tail_drain_seconds=r["tail_drain_seconds"],
        ssd_read_bytes=r["cohort_read_bytes"],ssd_write_bytes=r["cohort_write_bytes"],
        counters=counters,native_kv=r["native_kv"],staging_actual_bytes=[h["staging_bytes"] for h in r["probe"]["handlers"]],
        steps_emitting_two_requests=sum(len(s["emitted_request_ids"])==2 for s in r["steps"]),
        max_requests_emitted_per_step=max(len(s["emitted_request_ids"]) for s in r["steps"]),
        itl=distribution(all_itls),itl_with_other_request_io=distribution(with_io),itl_without_other_request_io=distribution(without_io),
        itl_with_io_excluding_cold_prefill_window=distribution(io_without_cold_prefill),
        itl_quiet_excluding_cold_prefill_window=distribution(quiet_without_cold_prefill),
        native_reference_exact_count=10-len(mismatches),reference_mismatches=mismatches,rows=rows,
        pending_wait_events=[e for e in r["probe"]["events"] if e["pending_before"]>0 and e["phase"]=="cohort"],
        result_sha256=digest(folder/"result.json"),trace_sha256=digest(folder/"native-cohort.trace.json"),
        outputs={x["request_id"]:x["output_tokens"] for x in r["rows"]})

def analyze(root,plan_path):
    plan=json.loads(plan_path.read_text())
    native=Path(plan["native_reference"]);nr=json.loads((native/"result.json").read_text())
    assert nr["status"]=="PASSED_NATIVE_C2_FULL_REFERENCE" and len(nr["rows"])==10
    refs={(r["family"],r["kind"]):r["output_tokens"] for r in nr["rows"]}
    runs=[summarize(Path(p),refs) for p in plan["runs"]]
    repeats={}
    for profile in ("all_hit","mixed_readwrite"):
        selected=[r for r in runs if r["profile"]==profile]
        assert len(selected)==2
        repeats[profile]=dict(outputs_exact=selected[0]["outputs"]==selected[1]["outputs"],
                             run_labels=[r["label"] for r in selected])
    exact=sum(r["native_reference_exact_count"] for r in runs)
    for r in runs:r.pop("outputs")
    return dict(status="PASSED_C2_NATIVE_OUTPUT_SCREEN" if exact==40 and all(x["outputs_exact"] for x in repeats.values()) else "C2_NUMERIC_DIAGNOSIS_REQUIRED",
        runs=runs,repeats=repeats,native_reference_exact_count=exact,native_reference_sha256=digest(native/"result.json"),
        formal_goodput=False,new_policy_installed=False,research_benefit_measured=False,full_P3_complete=False,
        association_note="Native trace temporal co-occurrence, not new CUPTI validation or causal interference. Request phases, prefill, queueing and batch shape may confound token latency. Unloaded cost checks do not qualify predictions under active concurrency.",
        reference_note="Full-output references are unloaded native cold/GPU-hot shapes. Any differences under concurrent shapes require numerical diagnosis; they are not automatically cache corruption or permissible performance evidence.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    r=analyze(Path.cwd(),a.plan)
    with a.out.open("x") as f:json.dump(r,f,indent=2)
    print(json.dumps(dict(status=r["status"],native_reference_exact_count=r["native_reference_exact_count"],repeats=r["repeats"],runs=[{k:v for k,v in run.items() if k not in ("rows","reference_mismatches","pending_wait_events")} for run in r["runs"]]),indent=2))
