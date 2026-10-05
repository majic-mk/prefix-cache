"""Native long-prefix calibration-family replay, never formal development/evaluation."""
import argparse,dataclasses,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import native_gpu_prefix_smoke as base
from run_p3_native_pilot import worker_probe
from long_calibration_contract import validate
from prefix_io_control.token_timeline import TokenTimeline,percentile

def worker_trace(worker,action,path=None,cuda=False):
    from simple_profiler import profiler
    if action=="start":
        profiler.begin_session(path)
        base.require(profiler._active,"native profiler inactive")
        worker._long_trace_profiler=None
        if cuda:
            import torch
            p=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False,with_stack=False)
            p.__enter__();worker._long_trace_profiler=p
        return dict(native_active=True,cuda_requested=cuda)
    p=getattr(worker,"_long_trace_profiler",None)
    try:
        if p is not None:
            p.__exit__(None,None,None)
            p.export_chrome_trace(path+".cuda.json")
            worker._long_trace_profiler=None
    finally:
        if profiler._active:profiler.end_session()
    return dict(native_trace_exists=Path(path).is_file(),
                cuda_trace_exists=Path(path+".cuda.json").is_file())

def clone_private_storage(source,target):
    source=source.resolve();runs=base.ROOT/"experiments/prefix_io_v1/runs"
    base.require(source.is_relative_to(runs) and source.is_dir(),"unsafe/missing storage source")
    files=sorted(source.rglob("*.bin"))
    base.require(len(files)==3072,"16K source inventory changed")
    target.mkdir(parents=True,exist_ok=False);manifest=[]
    for p in files:
        base.require(not p.is_symlink() and p.resolve().is_relative_to(source) and
                     p.stat().st_size==917504,"source layout/path")
        rel=p.relative_to(source);dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        os.link(p,dest)
        manifest.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=digest))
    return manifest

def verify_source(source,manifest):
    changed=[r["path"] for r in manifest if
             hashlib.sha256((source/r["path"]).read_bytes()).hexdigest()!=r["sha256"]]
    base.require(not changed,"published source files changed")
    return dict(checked_files=len(manifest),changed=changed)

def main():
    ap=argparse.ArgumentParser()
    for name in ["output","model-dir","model-plan","curves","manifest","storage-source"]:
        ap.add_argument("--"+name,type=Path,required=True)
    ap.add_argument("--cuda-timeline",action="store_true")
    a=ap.parse_args();out=a.output.resolve();source=a.storage_source.resolve()
    base.require(out.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs"),"unsafe output")
    manifest=json.loads(a.manifest.read_text());candidate=json.loads(a.curves.read_text())
    engine_config=validate(manifest,candidate,os.environ.get("CUDA_VISIBLE_DEVICES"))
    base.require(source==base.ROOT/"experiments/prefix_io_v1/runs/server07-p3-16k-storage-01","unregistered storage source")
    base.require(shutil.disk_usage(base.ROOT).free>=8*1024**3+(1024 if a.cuda_timeline else 128)*1024**2,"reserve 1 GiB for CUDA trace or 128 MiB for unprofiled suffix files")
    out.mkdir(parents=True,exist_ok=False);(out/"driver-source.py").write_text(Path(__file__).read_text())
    (out/"probe-source.py").write_text((Path(__file__).parent/"run_p3_native_pilot.py").read_text())
    inventory=clone_private_storage(source,out/"storage")
    base.write_new_json(out/"storage-source-manifest.json",inventory)
    base.configure_runtime_environment();os.environ["PYTHONHASHSEED"]="0"
    base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("TRANSFORMERS_OFFLINE")=="1","offline wrapper")
    report=dict(status="FAILED",scope="calibration-family native long replay only; not development/evaluation",
        formal_performance_claim=False,independent_content_validation=False,cache_resets=0,
        ordinary_quota_installed=False,policy_mode="shadow",cuda_timeline_requested=a.cuda_timeline,
        rows=[],manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        curves_sha256=hashlib.sha256(a.curves.read_bytes()).hexdigest(),
        initial_cache="published SSD corpus via hardlinks; new engine GPU/staging cold; private suffix writes",
        source_shared_physical_bytes=sum(r["bytes"] for r in inventory))
    llm=None;installed=False;tracing=False;trace_path=out/"native-cohort.trace.json"
    try:
        model,identity=base.validate_local_model(a.model_dir,a.model_plan)
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True);alias.symlink_to(model,target_is_directory=True)
        extra=dict(candidate["provenance"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"])
        extra.update(shared_storage_path=str(out/"storage"),load_planner="on",
            prefix_cache_break_even_path=str(a.curves.resolve()),
            prefix_io_observation_mode="shadow",prefix_io_observation_run_id=out.parent.name,
            prefix_io_observation_interval_ns=10_000_000)
        config=dict(engine_config,kv_transfer_config=dict(kv_connector="OffloadingConnector",kv_role="kv_both",
                                                         kv_connector_extra_config=extra))
        sampling=dict(base.SAMPLING,max_tokens=128,min_tokens=128)
        base.write_new_json(out/"frozen-config.json",dict(model=identity,engine=config,sampling=sampling,
            manifest=manifest,gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],pythonhashseed="0"))
        os.chdir(out)
        from vllm import LLM,SamplingParams
        llm=LLM(model=base.MODEL_ID,**config);engine=llm.llm_engine
        report["native_kv"]=llm.collective_rpc(worker_probe,timeout=30,
            args=("install",dict(kv_budget_bytes=2147483648,staging_budget_bytes=1073741824)))[0];installed=True
        llm.generate([{"prompt_token_ids":[30000]+list(range(1000,1128))}],SamplingParams(**base.SAMPLING),use_tqdm=False)
        report["warmup_drain"]=llm.collective_rpc(worker_probe,timeout=60,args=("drain",))[0]
        report["cohort_probe_start"]=llm.collective_rpc(worker_probe,timeout=30,args=("start",))[0]
        tracing=True
        report["trace_start"]=llm.collective_rpc(worker_trace,timeout=45,args=("start",str(trace_path),a.cuda_timeline))[0]
        requests=manifest["requests"];timelines={};records={};finished=set();cursor=0
        start=time.monotonic()
        while cursor<len(requests) or engine.has_unfinished_requests():
            now=time.monotonic();base.require(now-start<180,"bounded cohort deadline")
            while cursor<len(requests) and requests[cursor]["scheduled_time"]<=now-start:
                item=requests[cursor];rid="long-"+str(item["request_id"]);sent=time.monotonic()
                internal=engine.add_request(rid,{"prompt_token_ids":item["prompt_token_ids"]},SamplingParams(**sampling))
                timelines[rid]=TokenTimeline()
                records[rid]=dict(request_id=rid,internal_request_id=internal,calibration_family=item["calibration_family"],
                    scheduled_arrival_seconds=item["scheduled_time"],actual_send_seconds=sent-start,
                    client_queue_seconds=sent-start-item["scheduled_time"],prompt_tokens=len(item["prompt_token_ids"]))
                cursor+=1
            if not engine.has_unfinished_requests():
                if cursor<len(requests):time.sleep(min(.002,max(0,requests[cursor]["scheduled_time"]-(time.monotonic()-start))))
                continue
            for response in engine.step():
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
        before=report["cohort_probe_start"];after=report["probe"]
        report["cohort_read_bytes"]=after["read_bytes"]-before["read_bytes"]
        report["cohort_write_bytes"]=after["write_bytes"]-before["write_bytes"]
        base.require(report["cohort_read_bytes"]>0,"no actual SSD read")
        report["trace_stop"]=llm.collective_rpc(worker_trace,timeout=90,args=("stop",str(trace_path),a.cuda_timeline))[0]
        tracing=False
        report["status"]="PASSED_NATIVE_LONG_CALIBRATION_REPLAY"
    except BaseException as e:
        report.update(error=str(e),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                if tracing:report["trace_stop"]=llm.collective_rpc(worker_trace,timeout=90,args=("stop",str(trace_path),a.cuda_timeline))[0]
                if installed:report["final_probe"]=llm.collective_rpc(worker_probe,timeout=60,args=("finish",))[0]
                llm.llm_engine.engine_core.shutdown(timeout=15);report["engine_shutdown"]="completed"
            except BaseException as e:report.update(status="FAILED",shutdown_error=str(e))
        try:report["source_preservation"]=verify_source(source,inventory)
        except BaseException as e:report.update(status="FAILED",source_verification_error=str(e))
        base.write_new_json(out/"result.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","probe","final_probe","warmup_drain","cohort_probe_start")},indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
