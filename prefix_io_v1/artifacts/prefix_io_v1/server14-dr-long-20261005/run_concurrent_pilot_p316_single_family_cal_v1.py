"""P316 C2 replay; true GPU-hot control and optional common metadata CPU timing."""
import argparse,dataclasses,hashlib,json,os,shutil,time,traceback
from pathlib import Path
from experiment_storage import authorized_path,preflight
import native_gpu_prefix_smoke as base
from run_p3_native_pilot import worker_probe
from run_long_calibration_replay import worker_trace,clone_private_storage,verify_source
from concurrent_pilot_contract_p316 import (validate,required_new_bytes,validate_supplement,
    validate_gpu_hot_measurement,validate_optional_sampling,validate_pressure_capacity,GPU_HOT_PREFIX,NATIVE_IO_QUANTUM)
from qualify_concurrent_pilot import verify_gate
from prefix_io_control.token_timeline import TokenTimeline,percentile

"""Strict ancestry for one independent single-family development calibration.

This admits no effect run and does not extend the historical qualification.
The original contract still validates the unchanged historical parent.
"""
import hashlib
import json
from pathlib import Path

CALIBRATION_SCOPE = "CAL_DEV_SINGLE_FAMILY"
PARENT_PATH = "artifacts/prefix_io_v1/server08-p3-15/all_hit-manifest.json"
PARENT_SHA256 = "1dc8f14dcd076f41ce05dacb0244444c4919693c34dda454dc80b9a5cf617d48"
PARENT_BYTES = 1122310


def _cal_require(ok, message):
    if not ok:
        raise ValueError(message)


def _cal_read(path):
    raw = Path(path).read_bytes()
    _cal_require(0 < len(raw) <= 2 * 1024**2, "bounded calibration manifest")
    def unique(items):
        value = {}
        for key, part in items:
            _cal_require(key not in value, "duplicate calibration input key")
            value[key] = part
        return value
    return raw, json.loads(raw, object_pairs_hook=unique)


def validate_calibration_registration(root, permit, manifest, manifest_path, out):
    """Check actual parent bytes and only the declared request-family change."""
    root = Path(root).resolve(strict=True)
    manifest_path, out = Path(manifest_path).resolve(), Path(out).resolve()
    _cal_require(out.parent.name.startswith("server14-dr-cal"), "calibration labels only")
    _cal_require(manifest_path.is_relative_to(root) and out.is_relative_to(root),
                 "calibration inputs and output must stay in project")
    row = permit["manifests"]["all_hit"]
    _cal_require(row == {"path": PARENT_PATH, "sha256": PARENT_SHA256},
                 "unchanged original all-hit parent required")
    parent_path = root / PARENT_PATH
    cursor = parent_path
    while cursor != root:
        _cal_require(not cursor.is_symlink(), "parent manifest symlink")
        cursor = cursor.parent
    parent_raw, parent = _cal_read(parent_path)
    _cal_require(len(parent_raw) == PARENT_BYTES and
                 hashlib.sha256(parent_raw).hexdigest() == PARENT_SHA256,
                 "original all-hit parent bytes changed")
    current_raw, current = _cal_read(manifest_path)
    _cal_require(current == manifest and current["scope"] == CALIBRATION_SCOPE and
                 current["profile"] == parent["profile"] == "all_hit",
                 "explicit single-family calibration scope required")
    _cal_require({key:value for key,value in current.items() if key not in ("scope", "requests")} ==
                 {key:value for key,value in parent.items() if key not in ("scope", "requests")},
                 "calibration changed model, engine, family tokens or other frozen fields")
    _cal_require(len(current["requests"]) == len(parent["requests"]) == 10,
                 "calibration must retain all ten complete requests")
    for child, original in zip(current["requests"], parent["requests"]):
        _cal_require(set(child) == set(original) and type(child["family_index"]) is int and
                     child["family_index"] == 0 and
                     {key:value for key,value in child.items() if key != "family_index"} ==
                     {key:value for key,value in original.items() if key != "family_index"},
                     "calibration changed request identity, arrival or non-family fields")
    _cal_require(current["families"][0]["initial_ssd_present"] is True and
                 len(current["families"][0]["tokens"]) == 16257 and current["output_tokens"] == 128,
                 "calibration needs original full-length SSD family and complete output")
    return dict(parent_manifest=parent, record=dict(scope=CALIBRATION_SCOPE,
        qualification_scope="historical parent only; new stream is independent development calibration",
        parent_manifest=dict(path=PARENT_PATH, bytes=len(parent_raw), sha256=PARENT_SHA256),
        calibration_manifest=dict(path=manifest_path.relative_to(root).as_posix(),
            bytes=len(current_raw), sha256=hashlib.sha256(current_raw).hexdigest()),
        differences=["scope", "requests[*].family_index=0"],
        request_count=10, final_input_ids_per_request=16257, complete_output_tokens=128,
        gpu_hot_prewarm="absent; original ordinary SSD recovery and cost admission retained"))


def validate_calibration_adapter_arm(frozen_config):
    _cal_require(frozen_config["current_device_adapter"]["arm"] == "CAL",
                 "independent calibration runner cannot execute U or method effects")


def main():
    global worker_probe
    ap=argparse.ArgumentParser()
    for name in ["output","model-dir","model-plan","curves","manifest","storage-source"]:
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--qualification",type=Path,required=True)
    ap.add_argument("--storage-registration",type=Path)
    ap.add_argument("--simple-stage-config",type=Path)
    ap.add_argument("--supplemental-manifest-qualification",type=Path)
    ap.add_argument("--native-cpu-probe",action="store_true")
    ap.add_argument("--native-observation",choices=("on","off"),default="on")
    ap.add_argument("--flush-cause-probe",action="store_true")
    ap.add_argument("--store-readiness-probe",action="store_true")
    ap.add_argument("--start-budget-config",type=Path)
    a=ap.parse_args();out=a.output.resolve();source=a.storage_source.resolve()
    simple_extra = None
    if a.simple_stage_config is not None:
        base.require(not a.flush_cause_probe and not a.store_readiness_probe and a.start_budget_config is None,
                     "simple control has its own qualified diagnostic composition")
        from prefix_io_control.simple_stage_options import model_file
        from simple_stage_worker_probe_p316 import worker_probe as simple_probe
        simple_extra = model_file(a.simple_stage_config, out.parent.name)
        worker_probe = simple_probe
    if a.flush_cause_probe:
        from flush_cause_worker_probe import worker_probe as diagnostic_probe
        worker_probe=diagnostic_probe
    if a.store_readiness_probe:
        base.require(a.flush_cause_probe and a.start_budget_config is None,
                     "readiness requires passive flush diagnostics without a start policy")
        from store_readiness_worker_probe import worker_probe as readiness_probe
        worker_probe=readiness_probe
    base.require(not a.native_cpu_probe or simple_extra is not None,"CPU hook timing requires common simple-stage probe")
    authorized_path(out);authorized_path(source)
    start_options = None
    if a.start_budget_config is not None:
        base.require(not a.flush_cause_probe, "combine probes only after separate qualification")
        from start_budget_model_contract import validate_config
        from start_budget_worker_probe import worker_probe as start_probe
        start_options = validate_config(a.start_budget_config)
        worker_probe = start_probe
    permit=verify_gate(a.qualification)
    manifest=json.loads(a.manifest.read_text());candidate=json.loads(a.curves.read_text())
    base.require(permit["candidate_sha256"]==hashlib.sha256(a.curves.read_bytes()).hexdigest(),"candidate changed")
    gpu_hot = None
    if manifest["profile"] == "gpu_hot":
        base.require(simple_extra is not None and a.supplemental_manifest_qualification is not None,
                     "GPU-hot profile needs common probe and frozen supplemental registration")
        gpu_hot = validate_supplement(Path(__file__).resolve().parents[3],
            a.supplemental_manifest_qualification,a.manifest,a.curves,a.qualification,permit,
            os.environ.get("CUDA_VISIBLE_DEVICES"))
    else:
        base.require(a.supplemental_manifest_qualification is None,
                     "supplement must not alter existing profiles")
        calibration_registration=validate_calibration_registration(
            Path(__file__).resolve().parents[3],permit,manifest,a.manifest,out)
    validate_optional_sampling(a.native_observation,manifest["profile"],simple_extra,a.native_cpu_probe)
    source_registration = None
    if a.storage_registration is None:
        base.require(source==Path("/root/prefix-io-v1-validation/storage/heldout-long"),"unexpected published source")
    else:
        from storage_source_registration import validate_registration
        source_registration = validate_registration(
            Path(__file__).resolve().parents[3], a.storage_registration, source, out,
            os.environ.get("CUDA_VISIBLE_DEVICES"))
        root = Path(__file__).resolve().parents[3]
        cost_plan = json.loads((root/permit["cost_plan"]).read_text())
        reference_plan = json.loads((root/cost_plan["reference_plan"]).read_text())
        groups = reference_plan["external_groups"]
        base.require(len(groups)==1 and Path(groups[0]["storage_path"]).resolve()==source,
                     "registered source differs from qualified cached reference")
        base.require(reference_plan["gpu_uuid"]==source_registration["gpu_uuid"],
                     "registered source qualification GPU differs")
    engine_config=validate(calibration_registration["parent_manifest"],candidate,os.environ.get("CUDA_VISIBLE_DEVICES"))
    base.require(manifest.get("io_depth",4)==permit.get("connector_delta",{}).get("iodepth",4),"I/O setting differs from qualification")
    disk_contract=preflight(out,required_new_bytes(manifest["profile"]))
    expected_files=3048
    out.mkdir(parents=True,exist_ok=False);(out/"driver-source.py").write_text(Path(__file__).read_text())
    (out/"probe-source.py").write_text((Path(__file__).parent/("simple_stage_worker_probe_p316.py" if simple_extra is not None else "run_p3_native_pilot.py")).read_text())
    inventory=clone_private_storage(source,out/"storage",expected_files)
    base.write_new_json(out/"storage-source-manifest.json",inventory)
    base.configure_runtime_environment();os.environ["PYTHONHASHSEED"]="0"
    base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("TRANSFORMERS_OFFLINE")=="1","offline wrapper")
    report=dict(status="FAILED",scope=manifest["scope"],
        formal_performance_claim=False,independent_content_validation=False,cache_resets=0,
        ordinary_quota_installed=False,policy_mode="shadow",cuda_timeline_requested=False,
        flush_cause_probe=a.flush_cause_probe,store_readiness_probe=a.store_readiness_probe,
        rows=[],manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        curves_sha256=hashlib.sha256(a.curves.read_bytes()).hexdigest(),
        initial_cache="published SSD corpus via hardlinks; new engine GPU/staging cold; private suffix writes",
        source_shared_physical_bytes=sum(r["bytes"] for r in inventory))
    report["calibration_development_registration"]=calibration_registration["record"]
    report["disk_contract"]=disk_contract
    report["native_cpu_probe_requested"]=a.native_cpu_probe
    report["native_observation"]=a.native_observation
    report["optional_sampling_scope"]="warmup/cohort/tail; restored for finish outside timed cohort; hard protocol/drain/accounting unchanged"
    if gpu_hot is not None:
        report["gpu_hot_control"]=gpu_hot
        report["initial_cache"]="private published SSD corpus; existing family0 warmed through original native pipeline before measurement"
    if simple_extra is not None:
        report.update(common_parent_admission=True, four_stage_hook_available=True,
            full_per_stage_caps=simple_extra["prefix_io_stage_policy"].get("mode") in ("fixed","pressure"),
            simple_stage_options=simple_extra,
            ordinary_quota_installed=simple_extra["prefix_io_stage_policy"].get("mode") in ("fixed","pressure"),
            policy_mode=simple_extra["prefix_io_stage_policy"].get("mode"),
            scope="P3 simple baseline development; not formal research efficacy")
    if source_registration is not None:
        report["source_registration"]=source_registration
    if start_options is not None:
        report.update(start_budget_config=start_options,
            ordinary_quota_installed=start_options["mode"] in ("fixed","pressure"),
            policy_mode=start_options["mode"],full_per_stage_caps=False,
            scope="P3 real-model start-allowance qualification; not research policy efficacy")
    report.update(profile=manifest["profile"],engine_delta={"max_num_seqs":2},
        previously_seen_validation_families=3,new_development_families=2,qualification=permit,steps=[])
    (out/"manifest.json").write_bytes(a.manifest.read_bytes())
    (out/"contract-source.py").write_text((Path(__file__).parent/"concurrent_pilot_contract_p316.py").read_text())
    llm=None;installed=False;tracing=False;trace_path=out/"native-cohort.trace.json"
    try:
        model,identity=base.validate_local_model(a.model_dir,a.model_plan)
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True);alias.symlink_to(model,target_is_directory=True)
        extra=dict(candidate["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"])
        extra.update(shared_storage_path=str(out/"storage"),load_planner="on",iodepth=manifest.get("io_depth",4),
            prefix_cache_break_even_path=str(a.curves.resolve()),
            prefix_io_observation_mode="shadow",prefix_io_observation_run_id=out.parent.name,
            prefix_io_observation_interval_ns=10_000_000)
        if simple_extra is not None:
            extra.update(simple_extra)
        if start_options is not None:
            extra["prefix_io_start_budget"]=dict(start_options["options"],run_id=out.parent.name)
            if start_options["mode"]=="off":
                extra["prefix_io_start_budget"]={"mode":"off"}
        config=dict(engine_config,kv_transfer_config=dict(kv_connector="OffloadingConnector",kv_role="kv_both",
                                                         kv_connector_extra_config=extra))
        sampling=dict(base.SAMPLING,max_tokens=128,min_tokens=128)
        base.write_new_json(out/"frozen-config.json",dict(model=identity,engine=config,sampling=sampling,
            manifest=manifest,gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],pythonhashseed="0",
            flush_cause_probe=a.flush_cause_probe,store_readiness_probe=a.store_readiness_probe))
        validate_calibration_adapter_arm(json.loads((out/"frozen-config.json").read_text()))
        os.chdir(out)
        from vllm import LLM,SamplingParams
        llm=LLM(model=base.MODEL_ID,**config);engine=llm.llm_engine
        probe_limits=dict(kv_budget_bytes=2147483648,staging_budget_bytes=1073741824)
        if a.native_cpu_probe:probe_limits["native_cpu_probe"]=True
        if simple_extra is not None:probe_limits["native_observation"]=a.native_observation
        report["native_kv"]=llm.collective_rpc(worker_probe,timeout=30,
            args=("install",probe_limits))[0];installed=True
        llm.generate([{"prompt_token_ids":[30000]+list(range(1000,1128))}],SamplingParams(**base.SAMPLING),use_tqdm=False)
        report["warmup_drain"]=llm.collective_rpc(worker_probe,timeout=60,
            args=("drain",{"native_cpu_phase":"warmup"}) if simple_extra is not None else ("drain",))[0]
        if gpu_hot is not None:
            warm=llm.generate([{"prompt_token_ids":manifest["families"][0]["tokens"]}],
                SamplingParams(**sampling),use_tqdm=False)
            base.require(len(warm)==1 and len(warm[0].outputs)==1 and
                list(warm[0].outputs[0].token_ids)==gpu_hot["golden_output_tokens"] and
                not warm[0].metrics.is_corrupted,"GPU-hot prewarm golden output differs")
            report["gpu_hot_prewarm"]=dict(family=manifest["families"][0]["name"],
                prompt_tokens=16257,output_tokens=list(warm[0].outputs[0].token_ids),
                num_cached_tokens=warm[0].num_cached_tokens,external_pipeline_enabled=True)
            report["gpu_hot_prewarm_drain"]=llm.collective_rpc(worker_probe,timeout=60,
                args=("drain",{"native_cpu_phase":"warmup"}))[0]
        report["cohort_probe_start"]=llm.collective_rpc(worker_probe,timeout=30,args=("start",))[0]
        tracing=True
        report["trace_start"]=llm.collective_rpc(worker_trace,timeout=45,args=("start",str(trace_path),False))[0]
        requests=manifest["requests"];families=manifest["families"];timelines={};records={};finished=set();cursor=0
        start=time.monotonic();report["cohort_start_monotonic"]=start
        while cursor<len(requests) or engine.has_unfinished_requests():
            now=time.monotonic();base.require(now-start<180,"bounded cohort deadline")
            while cursor<len(requests) and requests[cursor]["scheduled_time"]<=now-start:
                item=requests[cursor];family=families[item["family_index"]];rid="c2-"+str(item["request_id"]);sent=time.monotonic()
                internal=engine.add_request(rid,{"prompt_token_ids":family["tokens"]},SamplingParams(**sampling))
                timelines[rid]=TokenTimeline()
                records[rid]=dict(request_id=rid,internal_request_id=internal,family=family["name"],initial_ssd_present=family["initial_ssd_present"],
                    scheduled_arrival_seconds=item["scheduled_time"],actual_send_seconds=sent-start,
                    client_queue_seconds=sent-start-item["scheduled_time"],prompt_tokens=len(family["tokens"]))
                cursor+=1
            if not engine.has_unfinished_requests():
                if cursor<len(requests):time.sleep(min(.002,max(0,requests[cursor]["scheduled_time"]-(time.monotonic()-start))))
                continue
            step_start=time.monotonic()
            responses=engine.step()
            step_end=time.monotonic()
            report["steps"].append(dict(start=step_start,end=step_end,seconds=step_end-step_start,
                outstanding_frontend_requests=cursor-len(finished),emitted_request_ids=[r.request_id for r in responses]))
            for response in responses:
                base.require(len(response.outputs)==1,"one sample only")
                timeline=timelines[response.request_id]
                timeline.append(response.outputs[0].token_ids,response.metrics.last_token_ts)
                if response.finished:
                    base.require(response.request_id not in finished,"duplicate completion")
                    finished.add(response.request_id);record=records[response.request_id];record.update(timeline.export())
                    record.update(num_cached_tokens=response.num_cached_tokens,metrics=dataclasses.asdict(response.metrics),
                                  response_end_seconds=time.monotonic()-start)
                    base.require(record["per_token_complete"] and len(record["output_tokens"])==128 and not response.metrics.is_corrupted,
                                 "incomplete/corrupted token evidence")
                    if gpu_hot is not None:
                        base.require(record["num_cached_tokens"]==GPU_HOT_PREFIX and
                            record["output_tokens"]==gpu_hot["golden_output_tokens"],
                            "GPU-hot request lost its retained prefix or complete golden output")
                    record["itl_p95_seconds"]=percentile(record["itl_seconds"],.95);report["rows"].append(record)
        report["response_seconds"]=time.monotonic()-start-requests[0]["scheduled_time"]
        drain=time.monotonic();report["probe"]=llm.collective_rpc(worker_probe,timeout=60,args=("drain",))[0]
        report["tail_drain_seconds"]=time.monotonic()-drain
        report["cohort_seconds_including_drain"]=time.monotonic()-start-requests[0]["scheduled_time"]
        base.require(len(finished)==len(requests),"unfinished requests")
        for h in report["probe"]["handlers"]:
            base.require(h["aio"]["outstanding"]==0 and h["aio"]["accepted"]==h["aio"]["reaped"] and
                         h["observation_failures"]==0,"I/O/observation failure")
        if a.flush_cause_probe:
            diagnostic=report["probe"]["flush_diagnostic"]
            base.require(not diagnostic["faulted"] and diagnostic["errors"]==0 and
                         diagnostic["allocations"]>0,"flush ownership diagnostic failed or not co-located")
        before=report["cohort_probe_start"];after=report["probe"]
        if simple_extra is not None:
            validate_pressure_capacity(report["policy_mode"],a.native_observation,after["simple_native_control"])
        report["cohort_read_bytes"]=after["read_bytes"]-before["read_bytes"]
        report["cohort_write_bytes"]=after["write_bytes"]-before["write_bytes"]
        if gpu_hot is not None:
            controls=after["simple_native_control"]
            base.require(bool(controls) and all(x["native_io_size"]==NATIVE_IO_QUANTUM for x in controls),
                         "GPU-hot native storage unit changed")
            validate_gpu_hot_measurement(report["rows"],report["cohort_read_bytes"],
                report["cohort_write_bytes"],controls[0]["native_io_size"],gpu_hot["golden_output_tokens"])
            report["gpu_hot_measurement_gate"]="PASS_PREFIX16256_GOLDEN128_READ0_PAID_WRITE_CAP"
        else:
            base.require(report["cohort_read_bytes"]>0,"no actual SSD read")
        report["trace_stop"]=llm.collective_rpc(worker_trace,timeout=90,args=("stop",str(trace_path),False))[0]
        tracing=False
        report["status"]="PASSED_NATIVE_C2_DEVELOPMENT_REPLAY"
    except BaseException as e:
        report.update(error=str(e),traceback=traceback.format_exc())
        if "timelines" in locals():
            report["unfinished_token_events"]={k:v.export() for k,v in timelines.items() if k not in finished}
    finally:
        if llm is not None:
            try:
                if tracing:report["trace_stop"]=llm.collective_rpc(worker_trace,timeout=90,args=("stop",str(trace_path),False))[0]
                if installed:report["final_probe"]=llm.collective_rpc(worker_probe,timeout=60,args=("finish",))[0]
                llm.llm_engine.engine_core.shutdown(timeout=15);report["engine_shutdown"]="completed"
            except BaseException as e:report.update(status="FAILED",shutdown_error=str(e))
        try:report["source_preservation"]=verify_source(source,inventory)
        except BaseException as e:report.update(status="FAILED",source_verification_error=str(e))
        try:report["storage_after"]=preflight(out,0)
        except BaseException as e:report.update(status="FAILED",storage_budget_error=str(e))
        if a.native_cpu_probe:
            try:
                from simple_stage_worker_probe_p316 import cpu_delta
                points={k:report[k]["native_metadata_cpu"] for k in
                    ("native_kv","cohort_probe_start","probe","final_probe")}
                report["native_metadata_cpu_intervals"]=dict(
                    warmup_and_start_boundary=cpu_delta(points["native_kv"],points["cohort_probe_start"]),
                    cohort_and_tail=cpu_delta(points["cohort_probe_start"],points["probe"]),
                    finish=cpu_delta(points["probe"],points["final_probe"]),
                    scope="outermost metadata/control/observer hook CPU including timing instrumentation; explicit RPC phase boundaries; no whole-model CPU claim")
            except BaseException as e:
                report.update(status="FAILED",cpu_evidence_error=str(e))
        base.write_new_json(out/"result.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","steps","qualification","probe","final_probe","warmup_drain","cohort_probe_start")},indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
