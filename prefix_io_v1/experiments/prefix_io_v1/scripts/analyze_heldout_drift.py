"""Post-hoc variability description; host spans are not GPU kernel time."""
import argparse,hashlib,json,statistics
from pathlib import Path
from analyze_repeated_native_costs import check_drains,require
from export_uniform_calibration_candidate import verify_measurement

def duration_groups(row):
    p=Path(row["trace_path"]);trace=json.loads(p.read_text());groups={}
    selectors={"host_forward":lambda n:n.startswith("forward("),
               "cache_transfer":lambda n:n=="py_kvcache.transfer",
               "host_cuda_staging_span":lambda n:n=="py_kvcache.cuda_staging",
               "prefix_lookup":lambda n:n=="OffloadingConnector.get_num_new_matched_tokens",
               "load_e2e":lambda n:n.startswith("load_e2e(")}
    for key,test in selectors.items():
        values=[e["dur"] for e in trace["traceEvents"] if e.get("ph")=="X" and test(e.get("name",""))]
        groups[key]=dict(count=len(values),durations_us=values,sum_us=sum(values))
    return dict(kind=row["kind"],rep=row["rep"],warmup=row["warmup"],
                ttft_s=row["metrics"]["first_token_latency"],host_spans=groups,
                trace_sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def analyze(root,plan_path):
    plan=json.loads(plan_path.read_text());n=plan["sizes"][0]
    old=root/"experiments/prefix_io_v1/runs/server07-p3-uniform-long-cost-01/details"
    expected=json.loads((old/"frozen-config.json").read_text())
    refdir=root/"experiments/prefix_io_v1/runs"/plan["native_reference"]/"details"
    refs={(r["prefix_tokens"],r["rep"],r["kind"]):r for r in json.loads((refdir/"result.json").read_text())["rows"]}
    result=[];all_samples={k:[] for k in ("g_ssd","g_mem")}
    for label in plan["labels"]:
        d=root/"experiments/prefix_io_v1/runs"/label/"details"
        r=json.loads((d/"result.json").read_text());c=json.loads((d/"frozen-config.json").read_text())
        require(r["status"]=="PASSED_NATIVE_PAIRED_ACQUISITION" and not r["cached_reference_logprobs"],"control failed")
        check_drains(r)
        for key in ("engine","sampling","reps","acquisition_domain","gpu_uuid","model","pythonhashseed"):
            require(c[key]==expected[key],"control config mismatch: "+key)
        require(c["sizes"]==plan["sizes"] and len(r["rows"])==6,"control coverage")
        seen=set();sample={k:[] for k in all_samples}
        for row in r["rows"]:
            key=(row["prefix_tokens"],row["rep"],row["kind"])
            require(key not in seen and key[0]==n and 0<=key[1]<3 and key[2] in sample,"duplicate/invalid control")
            seen.add(key);verify_measurement(row,refs[(n,key[1],"gpu_hot")],n)
            if not row["warmup"]:sample[row["kind"]].append(row["metrics"]["first_token_latency"])
        for k,v in sample.items():all_samples[k].extend(v)
        result.append(dict(label=label,samples_s=sample,medians_s={k:statistics.median(v) for k,v in sample.items()},
            rows=[duration_groups(row) for row in r["rows"]],
            result_sha256=hashlib.sha256((d/"result.json").read_bytes()).hexdigest()))
    heldout=[]
    for label in ["server07-p3-heldout-paired-cost-01","server07-p3-heldout-paired-cost-02"]:
        r=json.loads((root/"experiments/prefix_io_v1/runs"/label/"details/result.json").read_text())
        heldout.append(dict(label=label,rows=[duration_groups(row) for row in r["rows"]]))
    curve=json.loads((root/plan["candidate"]).read_text());comparison={}
    for k,v in all_samples.items():
        pred=curve["curves"][k]["knots"][str(n)]
        comparison[k]=dict(original_prediction_s=pred,control_median_s=statistics.median(v),
            relative_difference=(statistics.median(v)-pred)/pred,min_s=min(v),max_s=max(v),samples_s=v)
    heldout_path=root/plan["original_heldout_result"]
    require(json.loads(heldout_path.read_text())["status"]=="FAILED_HELDOUT_POINT_PREDICTION","heldout status changed")
    return dict(status="POSTHOC_VARIABILITY_DESCRIPTION_ONLY",controls=result,comparison=comparison,
        heldout_result_sha256=hashlib.sha256(heldout_path.read_bytes()).hexdigest(),
        heldout_host_spans=heldout,heldout_validation_status_unchanged="FAILED_HELDOUT_POINT_PREDICTION",
        scope="Non-interleaved post-hoc controls cannot identify content causality or isolate GPU time; nested host spans must not be summed as disjoint latency.",
        runtime_curve_updated=False,policy_enabled=False)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--plan",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();r=analyze(Path.cwd(),a.plan)
    with a.out.open("x") as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in r.items() if k not in ("controls","heldout_host_spans")},indent=2))
