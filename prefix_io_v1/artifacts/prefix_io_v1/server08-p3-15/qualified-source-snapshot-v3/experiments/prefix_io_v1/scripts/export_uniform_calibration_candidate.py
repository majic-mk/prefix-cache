"""Export a separately gated calibration candidate; the original exporter stays strict."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
from experiment_storage import details_path
from analyze_cached_references import analyze as analyze_references, validate_external_row
from analyze_repeated_native_costs import check_drains, require
from export_native_aio_costs import Pchip, load_curves, build_cost_tables

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_measurement(row, reference, n):
    require(row["prompt_token_ids"]==reference["prompt_token_ids"] and
            row["output_token_ids"]==reference["output_token_ids"],"path-specific token reference mismatch")
    require(len(row["prompt_token_ids"])==n+1 and len(row["output_token_ids"])==1,"token axis")
    require(row["warmup"]==(row["rep"]==0),"warmup mismatch")
    require(not row["metrics"]["is_corrupted"],"corrupted output")
    latency=row["metrics"]["first_token_latency"]
    require(math.isfinite(latency) and latency>0,"invalid latency")
    if row["kind"]=="f": require(row["num_cached_tokens"]==0,"cold hit")
    else:
        require(row["num_cached_tokens"]==n,"wrong cached prefix")
        validate_external_row(row,n)

def build_candidate(root,plan_path,out):
    plan=json.loads(plan_path.read_text())
    reference_plan=root/plan["reference_plan"]
    reference=analyze_references(root,reference_plan)
    require(reference["status"]=="PASSED_EXACT_CACHED_REFERENCE","numeric reference failed")
    saved=json.loads((root/plan["reference_result"]).read_text())
    require(reference==saved,"numeric reference evidence changed")
    rp=json.loads(reference_plan.read_text())
    ref_dir=details_path(root,rp["native_label"],rp)
    rc=json.loads((ref_dir/"frozen-config.json").read_text())
    rr=json.loads((ref_dir/"result.json").read_text())
    refs={(r["prefix_tokens"],r["rep"],r["kind"]):r for r in rr["rows"]}
    expected_sampling={k:v for k,v in rc["sampling"].items() if k!="logprobs"}
    require(plan["domain"]==rp["domain"] and plan["sizes"]==rp["sizes"] and plan["reps"]==rp["reps"],
            "plan/reference mismatch")
    expected_extra={}
    for g in rp["external_groups"]:
        c=json.loads((details_path(root,g["label"],rp)/"frozen-config.json").read_text())
        expected_extra[g["storage"]]=c["engine"]["kv_transfer_config"]
    samples={kind:{n:[] for n in plan["sizes"]} for kind in ("f","g_ssd","g_mem")}
    sources=[];seen_labels=set();counts={kind:{n:0 for n in plan["sizes"]} for kind in samples}
    session_medians=[];checked_rows=0;external_config=None
    for job in plan["jobs"]:
        label=job["label"];require(label not in seen_labels,"duplicate acquisition")
        seen_labels.add(label)
        folder=details_path(root,label,plan)
        report=json.loads((folder/"result.json").read_text());c=json.loads((folder/"frozen-config.json").read_text())
        require(report["mode"]==job["mode"] and report["status"]=="PASSED_NATIVE_"+job["mode"].upper()+"_ACQUISITION","acquisition failed")
        require(not report.get("native_hot_diagnostic") and not report.get("cached_reference_logprobs"),"diagnostic latency excluded")
        require(c["sampling"]==expected_sampling,"sampling differs from non-logprob reference")
        for k in ("model","model_alias","alias_target","gpu_uuid","reps","pythonhashseed","acquisition_domain"):
            require(c[k]==rc[k],"identity mismatch: "+k)
        require(c["sizes"]==job["sizes"],"job sizes")
        require({k:v for k,v in c["engine"].items() if k!="kv_transfer_config"}==
                {k:v for k,v in rc["engine"].items() if k!="kv_transfer_config"},"engine mismatch")
        if job["mode"]=="cold":
            kinds={"f"};require(c["engine"]["kv_transfer_config"] is None,"cold connector active")
        else:
            require(job["mode"]=="paired","paired only")
            kinds={"g_ssd","g_mem"}
            require(c["engine"]["kv_transfer_config"]==expected_extra[job["storage"]],"connector/source mismatch")
            external_config=c
        check_drains(report)
        require(len(report["rows"])==len(job["sizes"])*plan["reps"]*len(kinds),"row coverage")
        seen=set();local={k:{n:[] for n in job["sizes"]} for k in kinds}
        for row in report["rows"]:
            n,rep,kind=row["prefix_tokens"],row["rep"],row["kind"]
            key=(n,rep,kind)
            require(n in job["sizes"] and 0<=rep<plan["reps"] and kind in kinds and key not in seen,"unexpected/duplicate row")
            seen.add(key)
            ref=refs[(n,rep,"f" if kind=="f" else "gpu_hot")]
            verify_measurement(row,ref,n);checked_rows+=1
            if not row["warmup"]:
                local[kind][n].append(row["metrics"]["first_token_latency"])
                samples[kind][n].append(row["metrics"]["first_token_latency"])
        for k in kinds:
            for n in job["sizes"]:
                require(len(local[k][n])==plan["reps"]-1,"measured sample count")
                counts[k][n]+=1
        session_medians.append(dict(label=label,medians_s={k:{str(n):statistics.median(v) for n,v in d.items()} for k,d in local.items()}))
        sources.append(dict(label=label,result_sha256=digest(folder/"result.json"),config_sha256=digest(folder/"frozen-config.json")))
    summary={k:{} for k in samples};curves={}
    for k,points in samples.items():
        for n,values in points.items():
            require(counts[k][n]==plan["fit_sessions"] and len(values)==plan["measurements_per_point_per_path"]>=4,
                    "insufficient independent engine sessions/samples")
            summary[k][str(n)]=dict(n=len(values),sessions=counts[k][n],median_s=statistics.median(values),
                                   min_s=min(values),max_s=max(values),samples_s=values)
        knots={str(n):statistics.median(v) for n,v in points.items()}
        curves[k]=dict(floor=0. if k=="f" else min(knots.values()),knots=knots)
    interp={k:Pchip([int(n) for n in v["knots"]],list(v["knots"].values()),floor=v["floor"]) for k,v in curves.items()}
    blocks=list(range(min(plan["sizes"]),max(plan["sizes"])+1,16))
    require(all(interp["g_ssd"](n)>=interp["g_mem"](n) for n in blocks),"negative paired storage service")
    def threshold(kind):
        profitable=[n for n in blocks if all(interp["f"](m)>interp[kind](m) for m in blocks if m>=n)]
        return min(profitable) if profitable else max(blocks)+16
    payload=dict(schema_version=2,model_name=rc["model_alias"],kv_dtype="auto",kv_bytes_per_token=57344,
                 break_even_ssd_tokens=threshold("g_ssd"),break_even_mem_tokens=threshold("g_mem"),
                 safety_margin_tokens=0,curves=curves,
                 golden=[dict(tokens=n,**{k:v(n) for k,v in interp.items()}) for n in [0,*plan["sizes"]]],
                 provenance=dict(method="Uniform-budget native costs; two fresh engines per path; four measured samples per point; all warmups retained for path-specific correctness",
                     candidate_only=True,allowed_consumer="same-budget calibration integration diagnostic only",
                     independent_content_validation=False,formal_performance_claim=False,
                     minimum_supported_prefix_tokens=min(plan["sizes"]),max_supported_model_len=max(plan["sizes"]),
                     gpu_uuid=rc["gpu_uuid"],model=rc["model"],engine=external_config["engine"],
                     reference_plan_sha256=digest(reference_plan),reference_result_sha256=digest(root/plan["reference_result"]),
                     fit_plan_sha256=digest(plan_path),sources=sources,
                     correctness_protocol="Cold output equals frozen native cold reference; SSD/staging equals frozen native-hot reference; separate exact top5 cached comparison; no cold-hot bit equality claim",
                     unsupported="No predictions below first measured knot; no independent content validation; candidate not production-qualified"))
    out.mkdir(parents=True,exist_ok=False)
    path=out/"curves-v2.json";path.write_text(json.dumps(payload,indent=2))
    parsed=load_curves(str(path),model_name=rc["model_alias"],kv_dtype="auto")
    tables=build_cost_tables(parsed,block_tokens=16,max_model_len=max(plan["sizes"]))
    result=dict(status="PASSED_CALIBRATION_CANDIDATE_ONLY",checked_rows=checked_rows,
                summary=summary,session_medians=session_medians,curves_sha256=digest(path),
                table_max_blocks=tables.max_blocks,runtime_installed=False,formal_qualification=False,
                thresholds={k:payload[k] for k in ("break_even_ssd_tokens","break_even_mem_tokens")})
    (out/"validation.json").write_text(json.dumps(result,indent=2))
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=build_candidate(Path.cwd(),a.plan,a.out)
    print(json.dumps(r,indent=2))
