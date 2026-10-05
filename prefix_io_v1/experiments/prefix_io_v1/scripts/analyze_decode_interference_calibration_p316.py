"""P3 descriptive, conditional calibration audit; standard library only.
Runs, rather than token intervals or repeated windows, are independent units.
No confidence interval, production recommendation, or research benefit is inferred.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SCOPE="P3_SPARSE_DECODE_INTERFERENCE_CALIBRATION_ONLY"
STAGES=("ssd_read","ssd_write","h2d","d2h")
CELLS=[dict(anchor=a,units=8) for a in ("none","warm_h2d","d2h_write","cold_ssd_h2d","joint")]+[
    dict(anchor="warm_h2d",units=1),dict(anchor="d2h_write",units=1)]
CRITERION={
 "schema_version":1,"scope":"P3 conditional sparse owned BF16 native I/O; not production state",
 "calibration_sessions":2,"independent_validation_sessions":1,"sequence":["A","B","B","A"],
 "aggregation":"equal run weight; median two same-role window ITL medians then median two calibration run values",
 "hard_witness":{"output_tokens":128,"cached_tokens":16256,"itl_samples":127,
 "native_stage_acceptance_completion":True,"closed_drained":True,
 "b_non_none_native_span_decode_wall_intersection":True,"none_all_stages_zero":True},
 "validation_loaded_itl_relative_error_max":.25,"calibration_loaded_relative_spread_max":.25,
 "within_run_a_relative_drift_max":.25,"unsupported_value":None,
 "fallback":"original native path; no interference recommendation","p4_connected":False,
 "dma_overlap_claim":False,"confidence_interval":None,
 "effect_delta":"descriptive only; no relative delta prediction gate",
 "low_contention_degradation_target":.02,
 "baseline_selection":"min median full cohort seconds among fixed and pressure; tie fixed; no new tuning",
 "formal_slo":None}

def require(ok,message):
    if not ok:raise ValueError(message)

def same(actual,expected,message):
    require(type(actual) is type(expected),message)
    if isinstance(expected,dict):
        require(set(actual)==set(expected),message)
        for k in expected:same(actual[k],expected[k],message)
    elif isinstance(expected,list):
        require(len(actual)==len(expected),message)
        for a,b in zip(actual,expected):same(a,b,message)
    else:require(actual==expected,message)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    def unique(items):
        out={}
        for k,v in items:
            require(k not in out,"duplicate JSON key: "+k);out[k]=v
        return out
    return json.loads(Path(path).read_text(),object_pairs_hook=unique)

def positive(v):
    return type(v) in (int,float) and math.isfinite(v) and v>0

def percentile(values,q):
    require(bool(values),"empty sample")
    vals=sorted(values);pos=(len(vals)-1)*q;lo=int(pos);hi=min(lo+1,len(vals)-1)
    return vals[lo]+(vals[hi]-vals[lo])*(pos-lo)

def expected_bytes(anchor,units,pulses):
    n=units*pulses*917504
    used={"none":(),"warm_h2d":("h2d",),"d2h_write":("d2h","ssd_write"),
        "cold_ssd_h2d":("ssd_read","h2d"),"joint":STAGES}[anchor]
    return {s:n if s in used else 0 for s in STAGES}

def closed_drained(proof):
    require(proof["worker_joined"] is True and proof["observation_failures"]==0,
            "native worker shutdown unproved")
    require(proof.get("wait_error") is None and type(proof["actual_staging_bytes"]) is int
        and 0<proof["actual_staging_bytes"]<=1024**3,"native drain or staging budget")
    aio=proof["aio"]
    require(aio["closed"] is True and aio["drained"] is True and aio["fatal"] is None,
            "native AIO closed/drained proof missing")
    require(all(aio[k]==0 for k in ("outstanding","pending","ready","unreaped")),
            "native AIO resources unresolved")

def audit_window(window,index,cell,role,golden):
    require(window["index"]==index and window["cell"]==cell and window["role"]==role,
            "window or ABBA order differs")
    require(window["output_exact"] is True and len(window["rows"])==1,"output witness missing")
    row=window["rows"][0]
    require(type(row["num_cached_tokens"]) is int and row["num_cached_tokens"]==16256,
            "different model Prefix cache state")
    require(row["per_token_complete"] is True and row["ambiguous_events"]==[]
        and row["metrics"]["is_corrupted"] is False,"ambiguous token timeline")
    tokens=row["output_tokens"];ts=row["engine_token_timestamps"];itl=row["itl_seconds"]
    require(len(tokens)==len(ts)==128 and len(itl)==127 and tokens==golden,
            "full model output or timing incomplete")
    require(all(type(x) is int and x>=0 for x in tokens),"invalid output token")
    require(all(positive(x) for x in ts+itl) and all(a<b for a,b in zip(ts,ts[1:])),
            "invalid token clock")
    require(all(abs(x-(b-a))<1e-9 for x,a,b in zip(itl,ts,ts[1:])),"ITL differs from token timestamps")
    p=window["probe"];mode=cell["anchor"] if role=="B" else "none"
    require(p["index"]==index and p["anchor"]==mode and p["units"]==cell["units"],
            "native anchor differs")
    issued=p["issued_pulses"];opportunities=p["opportunities"];skipped=p["skipped_busy_opportunities"]
    require(type(issued) is int and 0<=issued<=12 and type(opportunities) is int and opportunities==12,
            "pulse domain differs")
    require(type(skipped) is int and skipped>=0 and len(p["pulses"])==issued,"pulse records incomplete")
    if mode=="none":require(issued==skipped==0,"none window issued native I/O")
    else:require(issued>0 and issued+skipped==opportunities,"no native I/O or deferred pulses")
    parents=p["native_parents"]
    count=issued*(2 if mode=="joint" else 1)
    require(len(parents)==count and len({x["job_id"] for x in parents})==count,"parent witness incomplete")
    by_id={x["job_id"]:x for x in parents};pulse_ids=[]
    for i,pulse in enumerate(p["pulses"]):
        require(pulse["pulse"]==i and type(pulse["decode_ordinal"]) is int
            and pulse["decode_ordinal"]>=16 and (pulse["decode_ordinal"]-16)%8==0,
            "pulse outside original decode boundary")
        pulse_ids.extend(pulse["job_ids"])
        require(len(pulse["job_ids"])==(2 if mode=="joint" else 1),"joint parent count differs")
        hashes=pulse["hashes"]
        require(not set(hashes.get("read",[])).intersection(hashes.get("write",[])),
                "joint sources and destinations alias")
        for jid in pulse["job_ids"]:
            par=by_id[jid]
            require(par["phase"]=="measurement" and par["expected_bytes"]==cell["units"]*917504,
                    "native parent byte domain differs")
            key="write" if par["direction"]=="store" else "read"
            require(par["direction"] in ("load","store") and par["hashes"]==hashes[key]
                and len(par["hashes"])==cell["units"],"parent direction/hash mismatch")
    require(len(set(pulse_ids))==count and set(pulse_ids)==set(by_id),"pulse parent IDs differ")
    expected=expected_bytes(mode,cell["units"],issued)
    same(p["expected_stage_bytes"],expected,"declared stages differ")
    delta=p["stage_delta"];require(set(delta)==set(STAGES),"stage coverage differs")
    for stage in STAGES:
        d=delta[stage]
        require(d["accepted_bytes"]==d["completed_requested_bytes"]==d["transferred_bytes"]==expected[stage]
            and d["failed_ops"]==0 and d["accepted_ops"]==d["completed_ops"],"actual stage mismatch")
        require((expected[stage]==0 and d["accepted_ops"]==0) or
                (expected[stage]>0 and d["accepted_ops"]>0),"accepted stage lacks operation witness")
    idle=p["full_native_idle"]
    require(idle["idle"] is True and idle["observation_failures"]==0,"full native idle missing")
    acc=idle["stage"]["accounting"]
    require(acc["valid"] is True and acc["outstanding_records"]==0 and
        all(v["inflight_ops"]==v["inflight_bytes"]==0 for v in acc["stages"].values()),
        "native stage accounting unresolved")
    closed_drained(p["shutdown"])
    require(p["synthetic_roundtrip_exact"] is True and p["native_wait_in_decode"] is False
        and p["measurement_global_cuda_sync"] is False,"round-trip/async scope differs")
    require(p["observation_cadence_min_ns"]==2_000_000,"observer cadence differs")
    steps=p["decode_steps"]
    require(120<=len(steps)<=128 and [d["ordinal"] for d in steps]==list(range(1,len(steps)+1)),
            "continuous original decode evidence incomplete")
    require(all(type(d[k]) is int and d["end_ns"]>=d["start_ns"] for d in steps
        for k in ("start_ns","end_ns")),"bad decode wall spans")
    spans=p["stage_spans"];totals={s:0 for s in STAGES}
    for span in spans:
        require(span["stage"] in STAGES and type(span["bytes"]) is int and span["bytes"]>0
            and span["result"]==span["bytes"] and type(span["start_ns"]) is int
            and type(span["end_ns"]) is int and span["end_ns"]>=span["start_ns"],"bad native stage span")
        totals[span["stage"]]+=span["bytes"]
    same(totals,expected,"completed span bytes differ")
    overlap=sum(any(s["start_ns"]<d["end_ns"] and s["end_ns"]>d["start_ns"]
        for s in spans) for d in steps)
    require(window["backend_inflight_overlap_decode_steps"]==overlap,"overlap count differs")
    if mode!="none":require(overlap>0,"native occupancy never intersects original decode wall interval")
    require(positive(window["response_seconds"]) and positive(window["duration_including_drain"])
        and type(window["tail_drain_seconds"]) in (int,float) and
        math.isfinite(window["tail_drain_seconds"]) and window["tail_drain_seconds"]>=0,
        "invalid measured duration")
    return dict(index=index,role=role,anchor=mode,units=cell["units"],
        engine_itl_median_seconds=statistics.median(itl),engine_itl_p95_seconds=percentile(itl,.95),
        response_seconds=window["response_seconds"],tail_drain_seconds=window["tail_drain_seconds"],
        duration_including_drain=window["duration_including_drain"],
        issued_pulses=issued,skipped_busy_opportunities=skipped,actual_stage_bytes=expected,
        backend_inflight_overlap_decode_steps=overlap,output_tokens=128,cached_tokens=16256,
        full_output_exact=True,closed_drained=True)

def audit_run(path,partition):
    path=Path(path).resolve();d=read(path);s=d["spec"]
    require(s["native_worktree"]=="third_party/work/py-kvcache-p3-16-cpu","P316 analysis requires exact P316 native source domain")
    require(d["status"]=="PASSED_SPARSE_DECODE_INTERFERENCE" and d["scope"]==SCOPE
        and s["scope"]==SCOPE and s["partition"]==partition,"failed/out-of-domain run")
    require(d["engine_shutdown"]=="completed" and d["inputs_preserved"] is True,"shutdown/input preservation")
    require(d["formal_performance_claim"] is False and d["production_release_witness"] is False
        and d["model_executor_replaced"] is False and d["model_external_connector"] is False
        and d["client_slo"] is None and d["cache_resets"]==0,"production/performance scope differs")
    same(s["cells"],CELLS,"sparse cell domain differs");same(s["sequence"],["A","B","B","A"],"ABBA differs")
    require(s["active_decode_requests"]==1 and s["engine"]["max_num_seqs"]==2
        and s["engine"]["kv_cache_memory_bytes"]==2118123520 and
        s["owned"]["owned_bytes"]==29360128 and s["owned"]["storage_unit_bytes"]==917504,
        "conditional resource allocation domain differs")
    require(s["engine"]["kv_transfer_config"] is None and s["engine"]["enable_prefix_caching"] is True,
            "different original model cache path")
    domain=s["table_domain"]
    require(domain["engine_conditional"] is True and domain["model_kv_budget_bytes"]==2118123520
        and domain["active_decode_requests"]==1 and domain["production_resource_witness"] is False
        and domain["production_state_fallback"] is None and domain["p4_connected"] is False
        and domain["cost_permit_reused"] is False,"production table domain is forbidden")
    require(len(s["prompt_token_ids"])==16257 and len(d["windows"])==28,"incomplete sparse run")
    a=d["allocation"]
    require(a["actual_owned_bytes"]==29360128 and 0<a["actual_model_kv_bytes"]<=2118123520
        and a["combined_gpu_kv_bytes"]==a["actual_model_kv_bytes"]+a["actual_owned_bytes"]
        and a["combined_gpu_kv_bytes"]<=2147483648 and a["production_kv_touched"] is False,
        "actual combined KV budget or owned isolation differs")
    final=d["final_probe"]
    require(final["restored_original_execute"] is True and final["windows"]==28
        and final["owned_released_after_native_shutdown"] is True,"final worker restoration/release unproved")
    for proof in final["all_handler_shutdowns"]:closed_drained(proof)
    require(len(final["all_handler_shutdowns"])==56,"not all setup/measurement handlers shut down")
    golden=d["windows"][0]["rows"][0]["output_tokens"];cells=[];windows=[]
    for ci,cell in enumerate(CELLS):
        group=[audit_window(d["windows"][ci*4+j],ci*4+j,cell,role,golden)
            for j,role in enumerate(("A","B","B","A"))]
        windows.extend(group);av=[group[j]["engine_itl_median_seconds"] for j in (0,3)]
        bv=[group[j]["engine_itl_median_seconds"] for j in (1,2)]
        baseline=statistics.median(av);loaded=statistics.median(bv)
        cells.append(dict(cell=cell,window_indices=[x["index"] for x in group],
            baseline_itl_median_seconds=baseline,loaded_itl_median_seconds=loaded,
            baseline_first_last_relative_drift=abs(av[0]-av[1])/baseline,
            delta_itl_seconds=loaded-baseline,delta_relative_to_baseline=loaded/baseline-1,
            within_run_baseline_window_values=av,within_run_loaded_window_values=bv,
            window_count=4,independent_run_count=1))
    return dict(path=str(path),sha256=digest(path),session_id=s["session_id"],partition=partition,
        spec=s,model=d["model"],cells=cells,windows=windows,requests=28,output_tokens=3584,
        status="PASS_HARD_NATIVE_WITNESS",observer_cpu_overhead_quantified=False)

def aggregate(calibrations,validation,criterion):
    same(criterion,CRITERION,"criterion differs from frozen P3 contract")
    require(len(calibrations)==2,"two independent calibration sessions required")
    runs=calibrations+[validation]
    require(len({r["session_id"] for r in runs})==3 and len({r["path"] for r in runs})==3,
            "independent sessions/paths required")
    shared=("engine","sampling","owned","io","cells","sequence","pulse","active_decode_requests",
            "table_domain","gpu_uuid","native_worktree","source_sha256")
    for r in runs[1:]:
        for key in shared:same(r["spec"][key],runs[0]["spec"][key],"calibration/validation domain differs: "+key)
        same(r["model"],runs[0]["model"],"model identity differs")
    same(calibrations[0]["spec"]["prompt_token_ids"],calibrations[1]["spec"]["prompt_token_ids"],
        "calibration prefixes differ")
    require(validation["spec"]["prompt_family"]!=calibrations[0]["spec"]["prompt_family"] and
        validation["spec"]["prompt_token_ids"][0]!=calibrations[0]["spec"]["prompt_token_ids"][0],
        "validation content is not independent")
    expected_refs={(str((ROOT/r["path"]).resolve()),r["sha256"])
        for r in validation["spec"]["reference_results"]}
    require(expected_refs=={(r["path"],r["sha256"]) for r in calibrations},
            "validation did not freeze these exact calibration results")
    table=[]
    for i,cell in enumerate(CELLS):
        cs=[r["cells"][i] for r in calibrations];v=validation["cells"][i]
        values=[c["loaded_itl_median_seconds"] for c in cs];pred=statistics.median(values)
        observed=v["loaded_itl_median_seconds"]
        error=abs(pred-observed)/observed;spread=abs(values[0]-values[1])/pred
        drifts=[r["cells"][i]["baseline_first_last_relative_drift"] for r in runs]
        failures=[]
        if error>criterion["validation_loaded_itl_relative_error_max"]:failures.append("validation_loaded_error")
        if spread>criterion["calibration_loaded_relative_spread_max"]:failures.append("calibration_run_spread")
        if any(x>criterion["within_run_a_relative_drift_max"] for x in drifts):failures.append("within_run_baseline_drift")
        table.append(dict(cell=cell,supported=not failures,unsupported_reasons=failures,
            conditional_loaded_itl_seconds=pred if not failures else None,
            calibration_run_loaded_values=values,validation_loaded_itl_seconds=observed,
            validation_loaded_relative_error=error,calibration_loaded_relative_spread=spread,
            three_run_baseline_relative_drifts=drifts,run_level_delta_seconds=[
                r["cells"][i]["delta_itl_seconds"] for r in runs],
            run_level_delta_relative_to_baseline=[r["cells"][i]["delta_relative_to_baseline"] for r in runs],
            delta_prediction=None,production_value=None,p4_connected=False,independent_calibration_runs=2,
            independent_validation_runs=1,confidence_interval=None))
    complete=all(c["supported"] for c in table)
    return dict(status="PASS_COMPLETE_CONDITIONAL_TABLE" if complete else "INCOMPLETE_CONDITIONAL_TABLE",
        cells=table,complete_conditional_table=complete,production_state_table_complete=False,
        production_state_fallback=None,p4_connected=False,formal_performance_claim=False,slo=None,
        confidence_interval=None,observer_cpu_overhead_quantified=False,
        sampling_unit="independent run; repeated windows and token intervals are correlated",
        overlap_scope="native accepted-to-completed occupancy intersects worker decode wall; no instantaneous DMA/kernel overlap claim",
        criterion=criterion,denominators=dict(validation_error="observed validation loaded ITL",
            calibration_spread="median calibration run loaded ITL",a_drift="median first/last A-window ITL"),
        inputs=[{k:r[k] for k in ("path","sha256","session_id","partition","requests","output_tokens")} for r in runs],
        domain={k:copy.deepcopy(runs[0]["spec"][k]) for k in shared},
        run_summaries=[{k:r[k] for k in ("session_id","partition","cells","windows")} for r in runs],
        limitations="Three independent sessions only. Positive loaded-ITL prediction qualification does not establish nonzero interference, research advantage, production resource validity, or a statistical confidence interval.")

def lookup(table,domain,cell):
    """Exact conditional lookup only; no interpolation or production fallback value."""
    try:same(domain,table.get("domain"),"out-of-domain lookup")
    except ValueError:return None
    for entry in table.get("cells",[]):
        try:same(cell,entry["cell"],"outside measured sparse cell")
        except ValueError:continue
        return entry["conditional_loaded_itl_seconds"] if entry["supported"] else None
    return None

def summarize_model_runs(development,allhit,allhit_target=.02):
    """Input: audited run dictionaries with explicit frozen arm U/F/P or U/B."""
    require(type(allhit_target) is float and allhit_target==.02,"frozen all-hit target")
    require(len(development)==6 and [r["arm"] for r in development]==["U","F","P","P","F","U"],
            "frozen U/F/P/P/F/U independent run sequence required")
    require(all(r["status"]=="PASS_FULL_MODEL_SIMPLE_STAGE" for r in development+allhit),
            "unaudited/failed model run")
    require(len({r["source"] for r in development+allhit})==len(development+allhit),"duplicated model runs")
    arms={}
    for arm in ("U","F","P"):
        rows=[r for r in development if r["arm"]==arm]
        values=[r["cohort_seconds_including_drain"] for r in rows]
        require(all(positive(x) for x in values),"invalid model run duration")
        arms[arm]=dict(independent_runs=2,cohort_seconds=values,median_seconds=statistics.median(values),
            range_seconds=[min(values),max(values)],confidence_interval=None)
    best=min(("F","P"),key=lambda arm:(arms[arm]["median_seconds"],arm!="F"))
    for arm in arms:arms[arm]["relative_duration_vs_U"]=arms[arm]["median_seconds"]/arms["U"]["median_seconds"]-1
    negative=None
    if allhit:
        require(len(allhit)==4 and [r["arm"] for r in allhit]==["U","B","B","U"],"all-hit ABBA sequence differs")
        a=[r["cohort_seconds_including_drain"] for r in allhit if r["arm"]=="U"]
        b=[r["cohort_seconds_including_drain"] for r in allhit if r["arm"]=="B"]
        require(all(positive(x) for x in a+b),"invalid all-hit duration")
        require(all(r["profile"]=="gpu_hot" and r["actual_read_bytes"]==0 and
                    type(r["actual_write_bytes"]) is int and 0<=r["actual_write_bytes"]<=128*917504 and
                    len(r["comparisons"])==10 and all(x["cached_tokens"]==16256 and
                        x["output_exact"] is True for x in r["comparisons"]) for r in allhit),
                "GPU-hot overhead control lacks hot prefix/read-zero/bounded paid generation witness")
        degradation=statistics.median(b)/statistics.median(a)-1
        negative=dict(independent_runs_per_arm=2,uncontrolled_seconds=a,baseline_seconds=b,
            relative_duration_degradation=degradation,target=allhit_target,
            descriptive_target_met=degradation<=allhit_target,confidence_interval=None,
            io_scope="SSD read zero; original generation writes remain enabled and paid; not zero-I/O control")
    return dict(status="DESCRIPTIVE_DEVELOPMENT_ONLY",arms=arms,strongest_simple_baseline=best,
        baseline_selection="minimum median full-cohort duration; tie fixed",allhit_control=negative,
        formal_goodput=False,slo=None,confidence_interval=None,
        repeatability_scope="two independent runs per arm; no uncertainty interval or significance claim")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--calibration",type=Path,nargs=2,required=True)
    for name in ("validation","criterion","output"):ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args();criterion=read(a.criterion)
    value=dict(status="FAILED_INPUT_AUDIT",formal_performance_claim=False,production_state_fallback=None,
               p4_connected=False,confidence_interval=None)
    try:
        cs=[audit_run(p,"calibration") for p in a.calibration];v=audit_run(a.validation,"validation")
        value=aggregate(cs,v,criterion)
    except Exception as exc:value["error"]=str(exc)
    value["analysis_source_sha256"]=digest(__file__)
    value["criterion_source"]=str(a.criterion.resolve());value["criterion_sha256"]=digest(a.criterion)
    with a.output.open("x") as f:json.dump(value,f,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps(dict(status=value["status"],output=str(a.output),
        complete_conditional_table=value.get("complete_conditional_table",False),
        production_state_table_complete=False,formal_performance_claim=False)))
    return 0 if value["status"]=="PASS_COMPLETE_CONDITIONAL_TABLE" else 1

if __name__=="__main__":raise SystemExit(main())
