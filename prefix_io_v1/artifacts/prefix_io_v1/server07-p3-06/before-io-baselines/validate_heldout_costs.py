"""Held-out content checks against a frozen curve; never refits or installs it."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from experiment_storage import details_path
from heldout_manifest import load_manifest
from analyze_cached_references import analyze as analyze_cached
from analyze_repeated_native_costs import check_drains,require
from export_uniform_calibration_candidate import verify_measurement
from export_native_aio_costs import Pchip

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reference(root,plan_path):
    plan=json.loads(plan_path.read_text())
    rows,identity=load_manifest(root/plan["prompt_manifest"],plan["sizes"],plan["reps"])
    require(identity["sha256"]==plan["prompt_manifest_sha256"],"manifest changed")
    result=analyze_cached(root,plan_path)
    for label in [plan["native_label"],*[g["label"] for g in plan["external_groups"]]]:
        folder=details_path(root,label,plan)
        config=json.loads((folder/"frozen-config.json").read_text())
        report=json.loads((folder/"result.json").read_text())
        require(config["prompt_manifest"]==report["prompt_manifest"]==identity,"manifest identity mismatch")
        require(digest(folder/"prompt-manifest.json")==identity["sha256"],"run manifest changed")
        for row in report["rows"]:
            require(row["prompt_token_ids"]==rows[(row["prefix_tokens"],row["rep"])],"run prompt mismatch")
    result.update(independent_content_validation=True,manifest=identity,
        validation_domain=plan["sizes"],formal_evaluation=False,
        limitations=("Three independent document families at the frozen length; one warmup, two measured. No full-vocabulary or unmeasured-length generalization claim." if "run_details" in plan else
                     "Three independent document families at one length; one warmup, two measured. No full-vocabulary or long-prefix generalization claim."))
    return result

def prediction_check(predicted,samples,tolerance):
    require(isinstance(predicted,(int,float)) and math.isfinite(predicted) and predicted>0,"bad prediction")
    require(len(samples)>=4 and all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in samples),"sample coverage")
    require(isinstance(tolerance,(int,float)) and 0<tolerance<1,"bad tolerance")
    observed=statistics.median(samples);error=(observed-predicted)/predicted
    return dict(prediction_s=predicted,observed_median_s=observed,
        relative_error=error,absolute_relative_error=abs(error),passed=abs(error)<=tolerance,
        n=len(samples),samples_s=samples,min_s=min(samples),max_s=max(samples))

def costs(root,plan_path,engine_delta=None):
    plan=json.loads(plan_path.read_text())
    curve_path=root/plan["candidate"];require(digest(curve_path)==plan["candidate_sha256"],"curve changed")
    curve=json.loads(curve_path.read_text())
    require(curve["provenance"]["candidate_only"] and plan["used_for_fit"] is False and plan["formal_evaluation"] is False,
            "validation cannot silently fit/qualify")
    rp=json.loads((root/plan["reference_plan"]).read_text())
    rr=reference(root,root/plan["reference_plan"])
    require(rr["status"]=="PASSED_EXACT_CACHED_REFERENCE" and rr==json.loads((root/plan["reference_result"]).read_text()),"numeric gate failed/changed")
    folder=details_path(root,rp["native_label"],rp)
    nc=json.loads((folder/"frozen-config.json").read_text())
    nr=json.loads((folder/"result.json").read_text())
    refs={(r["prefix_tokens"],r["rep"],r["kind"]):r for r in nr["rows"]}
    ec=json.loads((details_path(root,rp["external_groups"][0]["label"],rp)/"frozen-config.json").read_text())
    engine_without_connector=lambda e:{k:v for k,v in e.items() if k!="kv_transfer_config"}
    expected_engine=engine_without_connector(curve["provenance"]["engine"])
    if engine_delta is not None:
        from concurrent_pilot_contract import expected_variant_engine
        require(plan.get("engine_delta")==engine_delta,"unregistered engine change")
        expected_engine=expected_variant_engine(expected_engine,engine_delta)
    require(engine_without_connector(nc["engine"])==expected_engine,"candidate budget/config mismatch")
    require(nc["model"]==curve["provenance"]["model"] and nc["gpu_uuid"]==curve["provenance"]["gpu_uuid"],"candidate identity mismatch")
    extra=ec["engine"]["kv_transfer_config"]["kv_connector_extra_config"].copy()
    expected=curve["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"].copy()
    extra.pop("shared_storage_path");expected.pop("shared_storage_path")
    require(extra==expected,"candidate storage mechanism/config mismatch")
    samples={k:{n:[] for n in rp["sizes"]} for k in ("f","g_ssd","g_mem")}
    session_counts={k:0 for k in samples};sources=[];seen=set();checked_rows=0;session_medians=[]
    for job in plan["jobs"]:
        label=job["label"];require(label not in seen,"duplicate engine session");seen.add(label)
        f=details_path(root,label,plan)
        c=json.loads((f/"frozen-config.json").read_text());report=json.loads((f/"result.json").read_text())
        require(report["status"]=="PASSED_NATIVE_"+job["mode"].upper()+"_ACQUISITION","acquisition failed")
        require(not report["native_hot_diagnostic"] and not report["cached_reference_logprobs"],"diagnostic cost forbidden")
        require(c["sampling"]=={k:v for k,v in nc["sampling"].items() if k!="logprobs"},"sampling mismatch")
        for key in ("model","model_alias","alias_target","gpu_uuid","reps","pythonhashseed","acquisition_domain","sizes","prompt_manifest"):
            require(c[key]==nc[key],"identity mismatch: "+key)
        require(digest(f/"prompt-manifest.json")==plan["prompt_manifest_sha256"],"measurement manifest changed")
        require(engine_without_connector(c["engine"])==engine_without_connector(nc["engine"]),"engine mismatch")
        if job["mode"]=="cold":
            kinds={"f"};require(c["engine"]["kv_transfer_config"] is None,"cold connector active")
        else:
            require(job["mode"]=="paired" and c["engine"]["kv_transfer_config"]==ec["engine"]["kv_transfer_config"],"connector mismatch")
            kinds={"g_ssd","g_mem"}
        check_drains(report)
        require(len(report["rows"])==len(rp["sizes"])*rp["reps"]*len(kinds),"row coverage")
        local={k:{n:[] for n in rp["sizes"]} for k in kinds};keys=set()
        for row in report["rows"]:
            n,rep,kind=row["prefix_tokens"],row["rep"],row["kind"];key=(n,rep,kind)
            require(n in rp["sizes"] and 0<=rep<rp["reps"] and kind in kinds and key not in keys,"unexpected/duplicate sample")
            keys.add(key)
            verify_measurement(row,refs[(n,rep,"f" if kind=="f" else "gpu_hot")],n);checked_rows+=1
            if not row["warmup"]:
                samples[kind][n].append(row["metrics"]["first_token_latency"])
                local[kind][n].append(row["metrics"]["first_token_latency"])
        for k in kinds:session_counts[k]+=1
        session_medians.append(dict(label=label,median_s={k:{str(n):statistics.median(v) for n,v in d.items()} for k,d in local.items()}))
        sources.append(dict(label=label,result_sha256=digest(f/"result.json"),config_sha256=digest(f/"frozen-config.json")))
    require(all(v==2 for v in session_counts.values()),"two fresh engine sessions per path required")
    checks={}
    for kind,points in samples.items():
        c=curve["curves"][kind];knots=c["knots"];keys=sorted(int(k) for k in knots)
        interpolator=Pchip(keys,[knots[str(k)] for k in keys],floor=c["floor"])
        checks[kind]={}
        for n,values in points.items():
            require(keys[0]<=n<=keys[-1],"outside measured domain")
            checks[kind][str(n)]=prediction_check(interpolator(n),values,plan["maximum_absolute_relative_median_error"])
    passed=all(c["passed"] for d in checks.values() for c in d.values())
    return dict(status="PASSED_HELDOUT_POINT_PREDICTION" if passed else "FAILED_HELDOUT_POINT_PREDICTION",
        checks=checks,checked_rows=checked_rows,session_medians=session_medians,sources=sources,
        candidate_sha256=digest(curve_path),plan_sha256=digest(plan_path),
        independent_content_validation=True,full_domain_validated=False,
        runtime_curve_updated=False,formal_performance_claim=False,
        limitations="Two measured content families and two engine sessions per path at the frozen point only. No confidence interval or full-domain qualification.")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--reference-only",action="store_true");a=p.parse_args()
    result=reference(Path.cwd(),a.plan) if a.reference_only else costs(Path.cwd(),a.plan)
    with a.out.open("x") as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ("comparisons","sources","cold_hot")},indent=2))
    raise SystemExit(0 if result["status"].startswith("PASSED") else 1)
