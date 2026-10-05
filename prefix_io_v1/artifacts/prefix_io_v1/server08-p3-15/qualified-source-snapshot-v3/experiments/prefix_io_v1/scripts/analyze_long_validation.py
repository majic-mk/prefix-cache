"""Compare continuous replay outputs with native GPU-cached 128-token references."""
import argparse,hashlib,json
from pathlib import Path
from analyze_long_replay import summarize
from experiment_storage import details_path
from analyze_repeated_native_costs import require

def analyze(root,plan_path):
    plan=json.loads(plan_path.read_text())
    native_folder=details_path(root,plan["native_reference"],plan)
    native=json.loads((native_folder/"result.json").read_text())
    nc=json.loads((native_folder/"frozen-config.json").read_text())
    require(native["status"]=="PASSED_NATIVE_LONG_CACHED_REFERENCE" and native["engine_shutdown"]=="completed","native reference failed")
    prompts=json.loads((root/plan["prompt_manifest"]).read_text())
    require(hashlib.sha256((root/plan["prompt_manifest"]).read_bytes()).hexdigest()==plan["prompt_manifest_sha256"],"prompt manifest changed")
    family_by_rep={r["rep"]:r["family"] for r in prompts["prompts"]}
    refs={family_by_rep[r["rep"]]:r for r in native["rows"] if r["kind"]=="gpu_hot"}
    require(len(refs)==3 and all(r["num_cached_tokens"]==16256 and len(r["output_tokens"])==128 for r in refs.values()),"native shape/reference coverage")
    runs=[];mismatches=[];outputs=[]
    for label in plan["replays"]:
        folder=details_path(root,label,plan)
        r=json.loads((folder/"result.json").read_text());c=json.loads((folder/"frozen-config.json").read_text())
        s=summarize(folder,independent=True)
        require(r["engine_shutdown"]=="completed" and r["cache_resets"]==0 and r["independent_content_validation"] is True,"lifecycle/scope")
        require(c["sampling"]==nc["sampling"] and c["model"]==nc["model"] and c["gpu_uuid"]==nc["gpu_uuid"],"sampling/model/GPU mismatch")
        require({k:v for k,v in c["engine"].items() if k!="kv_transfer_config"}==
                {k:v for k,v in nc["engine"].items() if k!="kv_transfer_config"},"engine mismatch")
        require(s["requests"]==6 and s["token_events"]==768 and s["ssd_read_bytes"]>0,"service/medium coverage")
        require(r["source_preservation"]==dict(checked_files=3048,changed=[]),"source preservation")
        for h in r["probe"]["handlers"]:
            a=h["aio"];require(a["accepted"]==a["completed"]==a["reaped"] and
                all(a[k]==0 for k in ("outstanding","pending","ready","unreaped")) and
                a["fatal"] is None and h["observation_failures"]==0,"physical I/O not settled")
        for row in r["rows"]:
            family=row["validation_family"];require(family in refs,"unknown family")
            if row["output_tokens"]!=refs[family]["output_tokens"]:
                mismatches.append(dict(label=label,request_id=row["request_id"],family=family,
                    first_difference=next((i for i,(a,b) in enumerate(zip(row["output_tokens"],refs[family]["output_tokens"])) if a!=b),None),
                    native_tokens=refs[family]["output_tokens"],restored_tokens=row["output_tokens"]))
        outputs.append({row["request_id"]:row["output_tokens"] for row in r["rows"]})
        s["per_request"]=[dict(request_id=x["request_id"],family=x["validation_family"],native_ttft_s=x["metrics"]["first_token_latency"],
            client_queue_s=x["client_queue_seconds"],itl_p95_s=x["itl_p95_seconds"]) for x in r["rows"]]
        runs.append(s)
    return dict(status="PASSED_INDEPENDENT_LONG_CACHED_OUTPUTS" if not mismatches else "FAILED_INDEPENDENT_LONG_CACHED_OUTPUTS",
        runs=runs,mismatches=mismatches,native_cached_reference_comparisons=sum(x["requests"] for x in runs),
        all_128_tokens_exact=not mismatches,repeat_outputs_exact=all(x==outputs[0] for x in outputs),
        native_reference_sha256=hashlib.sha256((native_folder/"result.json").read_bytes()).hexdigest(),
        policy_enabled=False,formal_goodput=False,full_P3_complete=False,
        scope="Independent controlled-document sustained decode correctness and native waiting observations; no strategy speedup or workload-wide qualification.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=analyze(Path.cwd(),a.plan)
    with a.out.open("x") as f:json.dump(r,f,indent=2,allow_nan=False)
    compact={k:v for k,v in r.items() if k not in ("runs","mismatches")}
    compact["runs"]=[{k:x[k] for k in ("label","requests","token_events","ssd_read_bytes","ssd_write_bytes","cohort_seconds","tail_drain_seconds","pending_flush_wait_calls","foreground_slot_unavailable","source_preservation")} for x in r["runs"]]
    compact["mismatch_count"]=len(r["mismatches"])
    print(json.dumps(compact,indent=2));raise SystemExit(0 if r["status"].startswith("PASSED") else 1)
