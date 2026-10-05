"""Offline, exact cached-shape numeric diagnosis; never emits runtime curves."""
import argparse, hashlib, json, math
from pathlib import Path
from experiment_storage import details_path,authorized_path
from analyze_repeated_native_costs import check_drains, require

def canonical_top5(row):
    values=row.get("output_logprobs")
    require(isinstance(values,list) and len(values)==1 and isinstance(values[0],dict),
            "one-token logprobs missing")
    values=values[0]
    require(len(values)>=5 and str(row["output_token_ids"][0]) in values,"top5 coverage")
    result={}
    for key,item in values.items():
        value=item["logprob"]
        require(isinstance(value,(int,float)) and math.isfinite(value),"invalid logprob")
        require(isinstance(item["rank"],int) and item["rank"]>0,"invalid rank")
        result[str(int(key))]=value
    return result

def compare_rows(native,external,n):
    require(native["prompt_token_ids"]==external["prompt_token_ids"] and
            len(native["prompt_token_ids"])==n+1,"prompt mismatch")
    require(native["num_cached_tokens"]==external["num_cached_tokens"]==n,"wrong cached shape")
    for row in (native,external):
        require(len(row["output_token_ids"])==1 and not row["metrics"]["is_corrupted"],"corrupted output")
    left=canonical_top5(native);right=canonical_top5(external)
    common=set(left)&set(right)
    return dict(output_equal=native["output_token_ids"]==external["output_token_ids"],
                top5_exact_equal=left==right,
                token_set_equal=set(left)==set(right),
                max_common_logprob_delta=max(abs(left[k]-right[k]) for k in common) if common else None,
                native_tokens=native["output_token_ids"],external_tokens=external["output_token_ids"],
                native_top5=left,external_top5=right)

def validate_external_row(row,n):
    require(row["kind"] in ("g_ssd","g_mem"),"wrong row kind")
    trace=row["trace"]
    actual=trace["foreground_logical_read_bytes"]+trace["preload_actual_read_bytes"]
    require(trace["transfer_success"] and trace["h2d_events"]>0 and
            sum(v["num_bytes"] for v in trace["load_transfers"])==n*57344,"incomplete H2D")
    require(actual==(n*57344 if row["kind"]=="g_ssd" else 0),"wrong medium")
    require(bool(row["drain"].get("handlers")),"no physical drain")

def analyze(root,plan_path):
    plan=json.loads(plan_path.read_text());sources=[]
    def load(label):
        folder=details_path(root,label,plan)
        p=folder/"result.json";c=folder/"frozen-config.json"
        sources.append(dict(label=label,result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                            config_sha256=hashlib.sha256(c.read_bytes()).hexdigest()))
        result=json.loads(p.read_text());config=json.loads(c.read_text());check_drains(result)
        return result,config
    native,nc=load(plan["native_label"])
    require(native["status"]=="PASSED_NATIVE_COLD_ACQUISITION" and native["native_hot_diagnostic"],
            "native reference absent")
    require(nc["engine"]["kv_transfer_config"] is None and nc["sampling"]["logprobs"]==5,
            "native external connector/logprobs mismatch")
    require(nc["sizes"]==plan["sizes"] and nc["reps"]==plan["reps"] and
            nc["acquisition_domain"]==plan["domain"],"native domain mismatch")
    require(nc["gpu_uuid"]==plan["gpu_uuid"] and nc["engine"]["kv_cache_memory_bytes"]==plan["kv_budget_bytes"] and
            nc["engine"]["max_model_len"]==nc["engine"]["max_num_batched_tokens"]==plan["domain"]+16,"frozen native budget mismatch")
    require(len(native["rows"])==len(plan["sizes"])*plan["reps"]*2,"native coverage")
    indexed={}
    cold_differences=[]
    for row in native["rows"]:
        key=(row["prefix_tokens"],row["rep"],row["kind"])
        require(key not in indexed,"duplicate native row")
        require(row["warmup"]==(row["rep"]==0),"native warmup mismatch")
        require(row["num_cached_tokens"]==(0 if row["kind"]=="f" else row["prefix_tokens"]),"native cache shape")
        indexed[key]=row
    for n in plan["sizes"]:
        for rep in range(plan["reps"]):
            cold=indexed[(n,rep,"f")];hot=indexed[(n,rep,"gpu_hot")]
            cold_differences.append(dict(n=n,rep=rep,output_equal=cold["output_token_ids"]==hot["output_token_ids"],
                cold_output=cold["output_token_ids"],hot_output=hot["output_token_ids"],
                top5_equal=canonical_top5(cold)==canonical_top5(hot)))
    comparisons=[];connectors=[]
    for group in plan["external_groups"]:
        ext,ec=load(group["label"])
        require(ext["status"]=="PASSED_NATIVE_PAIRED_ACQUISITION" and ext["cached_reference_logprobs"],
                "paired reference missing")
        for k in ("model","model_alias","alias_target","gpu_uuid","reps","sampling","pythonhashseed","acquisition_domain"):
            require(ec[k]==nc[k],"measurement identity mismatch: "+k)
        require(ec["sizes"]==group["sizes"],"external size mismatch")
        require({k:v for k,v in ec["engine"].items() if k!="kv_transfer_config"}==
                {k:v for k,v in nc["engine"].items() if k!="kv_transfer_config"},"engine mismatch")
        connector=ec["engine"]["kv_transfer_config"]["kv_connector_extra_config"].copy()
        require(connector.pop("shared_storage_path")==str(authorized_path(group.get("storage_path",root/"experiments/prefix_io_v1/runs"/group["storage"]),root)[0]),
                "storage provenance mismatch")
        require(connector["staging_mem"]*1024**3==plan["staging_budget_bytes"],"staging budget mismatch")
        connectors.append(connector)
        require(len(ext["rows"])==len(group["sizes"])*plan["reps"]*2,"external coverage")
        seen=set()
        for row in ext["rows"]:
            n=row["prefix_tokens"];rep=row["rep"];kind=row["kind"];key=(n,rep,kind)
            require(n in group["sizes"] and 0<=rep<plan["reps"] and key not in seen,"unexpected/duplicate pair")
            require(row["warmup"]==(rep==0),"external warmup mismatch")
            seen.add(key);validate_external_row(row,n)
            result=compare_rows(indexed[(n,rep,"gpu_hot")],row,n)
            result.update(n=n,rep=rep,kind=kind,label=group["label"],warmup=row["warmup"])
            comparisons.append(result)
    require(all(c==connectors[0] for c in connectors),"external connector configuration mismatch")
    require(len(comparisons)==len(plan["sizes"])*plan["reps"]*2,"full domain coverage")
    failed=[r for r in comparisons if not r["output_equal"] or not r["top5_exact_equal"]]
    return dict(status="PASSED_EXACT_CACHED_REFERENCE" if not failed else "FAILED_EXACT_CACHED_REFERENCE",
                comparisons=comparisons,failed_comparisons=len(failed),cold_hot=cold_differences,
                cold_hot_output_differences=sum(not r["output_equal"] for r in cold_differences),
                sources=sources,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                cached_output_tolerance=0,latency_fit_allowed=False,runtime_curve_exported=False,
                independent_content_validation=False,
                scope="Exact top5 and selected token comparison; different engine instances and persisted source KV provenance retained; no independent recompute bit equality assertion")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=analyze(Path.cwd(),a.plan)
    with a.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ("comparisons","sources")},indent=2))
    raise SystemExit(0 if result["status"]=="PASSED_EXACT_CACHED_REFERENCE" else 1)
