"""Audit real model outputs/drain before comparing P3 development cohorts."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
from prefix_io_control.token_timeline import percentile
from concurrent_pilot_contract_p316 import validate_gpu_hot_measurement, validate_pressure_capacity, NATIVE_IO_QUANTUM
from simple_stage_worker_probe_p316 import cpu_delta, CPU_PHASES, CPU_REGIONS

def require(ok, message):
    if not ok: raise ValueError(message)

def audit(path, reference, manifest):
    result=json.loads(Path(path).read_text());ref=json.loads(Path(reference).read_text())
    families={f["name"]:f["tokens"] for f in manifest["families"]}
    indexed={(r["family"],r["kind"]):r for r in ref["rows"]}
    require(result["status"]=="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY","failed cohort cannot imply efficacy")
    require(len(result["rows"])==10,"incomplete cohort")
    require(result["engine_shutdown"]=="completed" and result["source_preservation"]==dict(checked_files=3048,changed=[]),"shutdown/source failure")
    controls=result["final_probe"]["simple_native_control"]
    require(bool(controls) and all(x["physical_drained"] for x in controls),"physical drain not proved")
    require(result["profile"]==manifest["profile"],"manifest/result profile differs")
    comparisons=[];itls=[];ttfts=[]
    for row in result["rows"]:
        kind="cold" if row["num_cached_tokens"]==0 else "gpu_hot"
        golden=indexed[(row["family"],kind)]
        require(golden["prompt_token_ids"]==families[row["family"]],"reference token family differs")
        require(row["output_tokens"]==golden["output_tokens"],"full 128-token output mismatch")
        timestamps=row["engine_token_timestamps"];intervals=row["itl_seconds"]
        require(row["per_token_complete"] and len(timestamps)==len(row["output_tokens"])==128 and len(intervals)==127 and row["ambiguous_events"]==[], "ambiguous/incomplete token timeline")
        require(all(math.isfinite(t) for t in timestamps) and all(a<b for a,b in zip(timestamps,timestamps[1:])),"bad token clock")
        require(all(math.isfinite(x) and x>0 for x in intervals),"bad ITL")
        require(all(abs(x-(b-a))<1e-9 for x,a,b in zip(intervals,timestamps,timestamps[1:])),"ITL differs from token events")
        require(not row["metrics"]["is_corrupted"],"corrupted native output")
        first=row["metrics"]["first_token_latency"]
        require(math.isfinite(first) and first>=0,"bad native TTFT")
        ttfts.append(first);itls.extend(intervals)
        comparisons.append(dict(request_id=row["request_id"],family=row["family"],reference_kind=kind,output_exact=True,cached_tokens=row["num_cached_tokens"]))
    require({r["request_id"] for r in comparisons}=={"c2-"+str(i) for i in range(10)},"duplicated request IDs")
    for control in controls:
        admission=control["admission"];accounting=control["stage_accounting"];policy=control["controller"]
        require(admission["accepted_parents"]==0 and admission["peak_accepted_parents"]<=64,"admission bound violated")
        require(accounting["valid"] and accounting["outstanding_records"]==0,"stage accounting not drained")
        require(all(v["inflight_ops"]==v["inflight_bytes"]==0 for v in accounting["stages"].values()),"stage inflight remains")
        if policy is not None:
            require(not policy["faulted"] and not policy["pending_attempt"] and policy["uncertain_ops"]==policy["completion_unknown_ops"]==0,"control ambiguity")
            for stage,actual in accounting["stages"].items():
                observed=policy["observed_api_accepted"][stage]
                require(observed["ops"]==actual["accepted_ops"] and observed["bytes"]==actual["accepted_bytes"],"controller acceptance differs from actual native API")
    if manifest["profile"]=="gpu_hot":
        golden=indexed[(manifest["families"][0]["name"],"gpu_hot")]
        require(all(r["family"]==manifest["families"][0]["name"] for r in result["rows"]),
                "GPU-hot control switched families")
        require(result.get("gpu_hot_measurement_gate")==
                "PASS_PREFIX16256_GOLDEN128_READ0_PAID_WRITE_CAP",
                "GPU-hot native measurement gate absent")
        require(all(x["native_io_size"]==NATIVE_IO_QUANTUM for x in controls),"GPU-hot native unit changed")
        validate_gpu_hot_measurement(result["rows"],result["cohort_read_bytes"],
            result["cohort_write_bytes"],controls[0]["native_io_size"],golden["output_tokens"])
        require(result["gpu_hot_prewarm"]["external_pipeline_enabled"] is True and
                result["gpu_hot_prewarm"]["output_tokens"]==golden["output_tokens"],
                "GPU-hot prewarm disabled native pipeline or differs from golden")
    sampling=result.get("native_observation","on")
    require(sampling in ("on","off"),"optional sampling mode invalid")
    if sampling=="off":
        require(manifest["profile"]=="gpu_hot" and result["policy_mode"]=="off",
                "sampling-off is not an ordinary pressure/fixed experiment")
    for point in ("native_kv","cohort_probe_start","probe"):
        proof=result[point]["optional_native_sampling"]
        require(proof["mode"]==sampling and
                proof["optional_sink_suppressed"] is (sampling=="off") and
                proof["physical_proof_requires_optional_sampling"] is False,"optional sampling state differs")
    final_sampling=result["final_probe"]["optional_native_sampling"]
    require(final_sampling["mode"]==sampling and final_sampling["optional_sink_suppressed"] is False and
            final_sampling["finish_restore_outside_cohort"] is True,"sampling teardown restoration missing")
    validate_pressure_capacity(result["policy_mode"],sampling,result["probe"]["simple_native_control"])
    metadata_cpu=None
    if result.get("native_cpu_probe_requested",False):
        points={k:result[k]["native_metadata_cpu"] for k in
                ("native_kv","cohort_probe_start","probe","final_probe")}
        for point in points.values():
            require(point.get("valid") is True and point.get("error") is None and
                    point["algorithm_cpu_claim"] is False and
                    point["full_production_cpu_breakdown"] is False and
                    set(point["phase_regions"])==set(CPU_PHASES),"CPU observation scope differs")
            for phase in CPU_PHASES:
                require(set(point["phase_regions"][phase])==set(CPU_REGIONS),"CPU regions differ")
                for row in point["phase_regions"][phase].values():
                    require(set(row)=={"calls","thread_cpu_ns"} and all(type(v) is int and v>=0 for v in row.values()),
                            "invalid metadata CPU scalar")
                require(point["phase_totals"][phase]==dict(
                    calls=sum(r["calls"] for r in point["phase_regions"][phase].values()),
                    thread_cpu_ns=sum(r["thread_cpu_ns"] for r in point["phase_regions"][phase].values())),
                    "metadata CPU total differs")
        expected=dict(
            warmup_and_start_boundary=cpu_delta(points["native_kv"],points["cohort_probe_start"]),
            cohort_and_tail=cpu_delta(points["cohort_probe_start"],points["probe"]),
            finish=cpu_delta(points["probe"],points["final_probe"]))
        actual=result["native_metadata_cpu_intervals"]
        require(all(actual[k]==v for k,v in expected.items()),"CPU start/end difference differs")
        measured=sum(r["thread_cpu_ns"] for r in expected["cohort_and_tail"]["phase_totals"].values())
        metadata_cpu=dict(intervals=expected,cohort_and_tail_thread_cpu_seconds=measured/1e9,
            final_cumulative=points["final_probe"],algorithm_cpu_claim=False,
            full_production_cpu_breakdown=False,
            scope="pure metadata/control/observer hook thread CPU; timer overhead included; native/backend/model CPU excluded")
    seconds=result["cohort_seconds_including_drain"]
    require(math.isfinite(seconds) and seconds>0,"bad cohort duration")
    native=result["probe"]
    return dict(status="PASS_FULL_MODEL_SIMPLE_STAGE",source=str(path),source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        policy_mode=result["policy_mode"],profile=result["profile"],requests=10,output_tokens=1280,comparisons=comparisons,
        cohort_seconds_including_drain=seconds,completed_throughput_per_second=10/seconds,
        pooled_itl_p50_seconds=percentile(itls,.50),pooled_itl_p95_seconds=percentile(itls,.95),
        request_itl_p95_max_seconds=max(row["itl_p95_seconds"] for row in result["rows"]),
        native_ttft_p50_seconds=percentile(ttfts,.50),native_ttft_p95_seconds=percentile(ttfts,.95),
        native_flush_calls=native["flush_wait_calls"],native_pending_flush_calls=native["pending_flush_wait_calls"],
        native_pending_flush_seconds=native["pending_flush_wait_seconds"],foreground_slot_unavailable=native["foreground_slot_unavailable"],
        actual_read_bytes=result["cohort_read_bytes"],actual_write_bytes=result["cohort_write_bytes"],
        final_control=controls,native_observation=sampling,native_metadata_cpu=metadata_cpu,formal_goodput=False,slo=None,unseen_evaluation=False,
        inference_scope="Previously seen P3 development; run is independent sampling unit, token intervals are correlated")

def main():
    ap=argparse.ArgumentParser()
    for name in ("result","reference","manifest","output"):ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args()
    require(json.loads(a.result.read_text())["manifest_sha256"]==
            hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"manifest hash differs")
    value=audit(a.result,a.reference,json.loads(a.manifest.read_text()))
    with a.output.open("x") as f:json.dump(value,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in value.items() if k not in ("comparisons","final_control")},indent=2))
if __name__=="__main__":main()
