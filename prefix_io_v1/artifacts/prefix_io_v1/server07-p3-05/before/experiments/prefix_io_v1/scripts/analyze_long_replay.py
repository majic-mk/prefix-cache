"""Native replay summaries and actual CUDA activity overlap, without performance claims."""
import argparse,collections,hashlib,json,math
from pathlib import Path

def merge(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not (math.isfinite(start) and math.isfinite(end) and end>=start):
            raise ValueError("invalid interval")
        if end==start:continue
        if result and start<=result[-1][1]:
            result[-1]=(result[-1][0],max(result[-1][1],end))
        else:result.append((start,end))
    return result

def overlap(a,b):
    a=merge(a);b=merge(b);i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total

def is_cache_copy(e):
    n=e.get("args",{}).get("bytes",0)
    return e.get("cat")=="gpu_memcpy" and isinstance(n,int) and n>0 and n%917504==0 and ("htod" in e["name"].lower() or "dtoh" in e["name"].lower())

def count_token_events(rows):
    for row in rows:
        if not row.get("per_token_complete") or "engine_token_timestamps" not in row or len(row["engine_token_timestamps"])!=len(row["output_tokens"]) or row.get("ambiguous_events"):
            raise ValueError("missing actual per-token events")
    return sum(len(row["engine_token_timestamps"]) for row in rows)

def summarize(folder):
    path=folder/"result.json";r=json.loads(path.read_text())
    if r["status"]!="PASSED_NATIVE_LONG_CALIBRATION_REPLAY":
        raise ValueError("replay failed")
    events=json.loads((folder/"native-cohort.trace.json").read_text())["traceEvents"]
    counter=collections.Counter(e["name"] for e in events)
    planner=[e for e in events if "planner" in e.get("name","")]
    result=dict(label=folder.parent.name,requests=len(r["rows"]),
        token_events=count_token_events(r["rows"]),
        output_tokens=sum(len(x["output_tokens"]) for x in r["rows"]),
        cached_tokens=[x["num_cached_tokens"] for x in r["rows"]],
        ssd_read_bytes=r["cohort_read_bytes"],ssd_write_bytes=r["cohort_write_bytes"],
        cohort_seconds=r["cohort_seconds_including_drain"],tail_drain_seconds=r["tail_drain_seconds"],
        native_kv=r["native_kv"],staging_actual_bytes=[h["staging_bytes"] for h in r["probe"]["handlers"]],
        pending_flush_wait_calls=r["probe"]["pending_flush_wait_calls"],
        foreground_slot_unavailable=r["probe"]["foreground_slot_unavailable"],
        planner_event_counts=dict(collections.Counter(e["name"] for e in planner)),
        planner_decision_event_count_note="Repeated planning events, not unique transfers or requests",
        source_preservation=r["source_preservation"],result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        artificial_delay=False,cache_resets=r["cache_resets"],formal_performance_claim=False,
        independent_content_validation=False)
    cuda_path=folder/"native-cohort.trace.json.cuda.json"
    if cuda_path.exists():
        raw=json.loads(cuda_path.read_text());ce=raw["traceEvents"]
        activity=[e for e in ce if e.get("ph")=="X" and e.get("cat") in ("kernel","gpu_memcpy","gpu_memset")]
        kernels=[e for e in activity if e["cat"]=="kernel"]
        gemms=[e for e in kernels if "gemm" in e["name"].lower() or "matmul" in e["name"].lower() or "gemv" in e["name"].lower()]
        copies=[e for e in activity if e["cat"]=="gpu_memcpy"]
        # CUPTI reports fused batch copies in whole 917504-byte storage blocks.
        # Check aggregate direction bytes against native transfers before claiming coverage.
        page=[e for e in copies if is_cache_copy(e)]
        cuda_bytes={d:sum(e["args"]["bytes"] for e in page if tag in e["name"])
                    for d,tag in (("storage_to_gpu","HtoD"),("gpu_to_storage","DtoH"))}
        native_bytes={d:sum(e.get("args",{}).get("num_bytes",0) for e in events
                      if e["name"]=="py_kvcache.transfer" and e.get("args",{}).get("direction")==d)
                      for d in cuda_bytes}
        def intervals(items):return [(e["ts"],e["ts"]+e["dur"]) for e in items]
        result["cuda_timeline"]=dict(trace_bytes=cuda_path.stat().st_size,sha256=hashlib.sha256(cuda_path.read_bytes()).hexdigest(),
            categories=dict(collections.Counter(e.get("cat") for e in ce)),gpu_kernel_events=len(kernels),
            linear_algebra_kernel_events=len(gemms),gpu_memcpy_events=len(copies),cache_copy_events=len(page),cuda_bytes=cuda_bytes,native_bytes=native_bytes,byte_coverage_exact=cuda_bytes==native_bytes and cuda_bytes["storage_to_gpu"]>0,
            cache_copy_compute_overlap_us=overlap(intervals(page),intervals(kernels)),
            cache_copy_linear_algebra_overlap_us=overlap(intervals(page),intervals(gemms)),
            cache_copy_union_us=sum(b-a for a,b in merge(intervals(page))),
            memcpy_names=dict(collections.Counter(e["name"] for e in copies)),
            cache_copy_streams=dict(collections.Counter(str(e.get("args",{}).get("stream")) for e in page)),
            linear_algebra_streams=dict(collections.Counter(str(e.get("args",{}).get("stream")) for e in gemms)),
            attribution="Actual CUPTI intervals; fused block-size/direction filter with exact byte reconciliation; GEMM/GEMV overlap is model compute overlap within the decode cohort, not a speedup or per-token causal effect",
            host_api_flags_used_as_overlap_evidence=False)
    return result

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--folders",nargs="+",type=Path,required=True);ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args();rows=[summarize(p) for p in args.folders]
    data=dict(runs=rows,formal_performance_claim=False)
    if len(args.folders)==2:
        raw=[json.loads((p/"result.json").read_text()) for p in args.folders]
        outputs=[{r["request_id"]:r["output_tokens"] for r in run["rows"]} for run in raw]
        data["corresponding_outputs_exact"]=outputs[0]==outputs[1]
    with args.output.open("x") as f:json.dump(data,f,indent=2)
    print(json.dumps(data,indent=2))
