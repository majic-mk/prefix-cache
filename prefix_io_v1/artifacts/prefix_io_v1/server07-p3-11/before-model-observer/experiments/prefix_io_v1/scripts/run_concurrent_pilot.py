"""Two-sequence P3 development replay using the existing native driver helpers."""
import argparse,dataclasses,hashlib,json,os,shutil,time,traceback
from pathlib import Path
from experiment_storage import authorized_path,preflight
import native_gpu_prefix_smoke as base
from run_p3_native_pilot import worker_probe
from run_long_calibration_replay import worker_trace,clone_private_storage,verify_source
from concurrent_pilot_contract import validate,required_new_bytes
from qualify_concurrent_pilot import verify_gate
from prefix_io_control.token_timeline import TokenTimeline,percentile

def main():
    global worker_probe
    ap=argparse.ArgumentParser()
    for name in ["output","model-dir","model-plan","curves","manifest","storage-source"]:
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--qualification",type=Path,required=True)
    ap.add_argument("--flush-cause-probe",action="store_true")
    ap.add_argument("--start-budget-config",type=Path)
    a=ap.parse_args();out=a.output.resolve();source=a.storage_source.resolve()
    if a.flush_cause_probe:
        from flush_cause_worker_probe import worker_probe as diagnostic_probe
        worker_probe=diagnostic_probe
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
    item=permit["manifests"][manifest["profile"]]
    base.require(item["sha256"]==hashlib.sha256(a.manifest.read_bytes()).hexdigest(),"workload changed")
    base.require(source==Path("/root/prefix-io-v1-validation/storage/heldout-long"),"unexpected published source")
    engine_config=validate(manifest,candidate,os.environ.get("CUDA_VISIBLE_DEVICES"))
    base.require(manifest.get("io_depth",4)==permit.get("connector_delta",{}).get("iodepth",4),"I/O setting differs from qualification")
    disk_contract=preflight(out,required_new_bytes(manifest["profile"]))
    expected_files=3048
    out.mkdir(parents=True,exist_ok=False);(out/"driver-source.py").write_text(Path(__file__).read_text())
    (out/"probe-source.py").write_text((Path(__file__).parent/"run_p3_native_pilot.py").read_text())
    inventory=clone_private_storage(source,out/"storage",expected_files)
    base.write_new_json(out/"storage-source-manifest.json",inventory)
    base.configure_runtime_environment();os.environ["PYTHONHASHSEED"]="0"
    base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("TRANSFORMERS_OFFLINE")=="1","offline wrapper")
    report=dict(status="FAILED",scope=manifest["scope"],
        formal_performance_claim=False,independent_content_validation=False,cache_resets=0,
        ordinary_quota_installed=False,policy_mode="shadow",cuda_timeline_requested=False,
        flush_cause_probe=a.flush_cause_probe,
        rows=[],manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        curves_sha256=hashlib.sha256(a.curves.read_bytes()).hexdigest(),
        initial_cache="published SSD corpus via hardlinks; new engine GPU/staging cold; private suffix writes",
        source_shared_physical_bytes=sum(r["bytes"] for r in inventory))
    report["disk_contract"]=disk_contract
    if start_options is not None:
        report.update(start_budget_config=start_options,
            ordinary_quota_installed=start_options["mode"] in ("fixed","pressure"),
            policy_mode=start_options["mode"],full_per_stage_caps=False,
            scope="P3 real-model start-allowance qualification; not research policy efficacy")
    report.update(profile=manifest["profile"],engine_delta={"max_num_seqs":2},
        previously_seen_validation_families=3,new_development_families=2,qualification=permit,steps=[])
    (out/"manifest.json").write_bytes(a.manifest.read_bytes())
    (out/"contract-source.py").write_text((Path(__file__).parent/"concurrent_pilot_contract.py").read_text())
    llm=None;installed=False;tracing=False;trace_path=out/"native-cohort.trace.json"
    try:
        model,identity=base.validate_local_model(a.model_dir,a.model_plan)
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True);alias.symlink_to(model,target_is_directory=True)
        extra=dict(candidate["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"])
        extra.update(shared_storage_path=str(out/"storage"),load_planner="on",iodepth=manifest.get("io_depth",4),
            prefix_cache_break_even_path=str(a.curves.resolve()),
            prefix_io_observation_mode="shadow",prefix_io_observation_run_id=out.parent.name,
            prefix_io_observation_interval_ns=10_000_000)
        if start_options is not None:
            extra["prefix_io_start_budget"]=dict(start_options["options"],run_id=out.parent.name)
            if start_options["mode"]=="off":
                extra["prefix_io_start_budget"]={"mode":"off"}
        config=dict(engine_config,kv_transfer_config=dict(kv_connector="OffloadingConnector",kv_role="kv_both",
                                                         kv_connector_extra_config=extra))
        sampling=dict(base.SAMPLING,max_tokens=128,min_tokens=128)
        base.write_new_json(out/"frozen-config.json",dict(model=identity,engine=config,sampling=sampling,
            manifest=manifest,gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],pythonhashseed="0",
            flush_cause_probe=a.flush_cause_probe))
        os.chdir(out)
        from vllm import LLM,SamplingParams
        llm=LLM(model=base.MODEL_ID,**config);engine=llm.llm_engine
        report["native_kv"]=llm.collective_rpc(worker_probe,timeout=30,
            args=("install",dict(kv_budget_bytes=2147483648,staging_budget_bytes=1073741824)))[0];installed=True
        llm.generate([{"prompt_token_ids":[30000]+list(range(1000,1128))}],SamplingParams(**base.SAMPLING),use_tqdm=False)
        report["warmup_drain"]=llm.collective_rpc(worker_probe,timeout=60,args=("drain",))[0]
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
        report["cohort_read_bytes"]=after["read_bytes"]-before["read_bytes"]
        report["cohort_write_bytes"]=after["write_bytes"]-before["write_bytes"]
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
        base.write_new_json(out/"result.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","steps","qualification","probe","final_probe","warmup_drain","cohort_probe_start")},indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
