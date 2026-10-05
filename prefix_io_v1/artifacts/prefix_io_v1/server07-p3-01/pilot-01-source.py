"""P3 bounded native replay: author arrivals, actual engine token events, native waits."""
import argparse,dataclasses,hashlib,json,os,shutil,time,traceback
from pathlib import Path
import native_gpu_prefix_smoke as base
from prefix_io_control.token_timeline import TokenTimeline,percentile

def worker_probe(worker,action):
    from collections import deque
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    cw=get_kv_transfer_group().connector_worker
    if action=="install":
        state=dict(wait_calls=0,flush_wait_calls=0,pending_flush_wait_calls=0,
            flush_wait_seconds=0.,pending_flush_wait_seconds=0.,max_pending_flush_parents=0,
            flush_metadata_calls=0,flush_parent_references=0,read_ops=0,write_ops=0,
            read_bytes=0,write_bytes=0,snapshot_count=0,max_d2h_bytes=0,max_h2d_bytes=0,
            max_active_parents=0,min_free_staging_slots=None,events=deque(maxlen=128),phase="warmup")
        restores=[]
        def replace(obj,name,fn):
            old=getattr(obj,name);setattr(obj,name,fn);restores.append((obj,name,old,fn))
        original=cw.handle_preemptions
        def handle(metadata):
            previous=state.get("in_flush",False);state["in_flush"]=bool(metadata.jobs_to_flush)
            if metadata.jobs_to_flush:
                state["flush_metadata_calls"]+=1
                state["flush_parent_references"]+=len(metadata.jobs_to_flush)
            try:return original(metadata)
            finally:state["in_flush"]=previous
        replace(cw,"handle_preemptions",handle)
        for h in cw.worker.handlers:
            old_wait=h.wait
            def wait(ids,h=h,old_wait=old_wait):
                flush=state.get("in_flush",False)
                pending=sum(j in h._active and not h._active[j][0].done() for j in ids)
                start=time.monotonic();result=old_wait(ids);elapsed=time.monotonic()-start
                state["wait_calls"]+=1
                if flush:
                    state["flush_wait_calls"]+=1;state["flush_wait_seconds"]+=elapsed
                    state["max_pending_flush_parents"]=max(state["max_pending_flush_parents"],pending)
                    if pending:
                        state["pending_flush_wait_calls"]+=1;state["pending_flush_wait_seconds"]+=elapsed
                    state["events"].append(dict(kind="native_jobs_to_flush_wait",start=start,seconds=elapsed,
                        parent_count=len(ids),pending_before=pending,phase=state["phase"],ids=sorted(ids)[:32],ids_truncated=len(ids)>32))
                return result
            replace(h,"wait",wait)
            ring=h.coordinator.reactor.ring;original_rw=ring.queue_rw
            def rw(*args,original_rw=original_rw,**kwargs):
                result=original_rw(*args,**kwargs)
                direction="write" if kwargs["write"] else "read"
                state[direction+"_ops"]+=1;state[direction+"_bytes"]+=kwargs["nbytes"]
                return result
            replace(ring,"queue_rw",rw)
            reactor=h.coordinator.reactor;pub=reactor._observation_sink
            if pub:
                def sink(r,pub=pub):
                    before=pub.published_count;pub(r)
                    if pub.published_count==before:return
                    snap=pub._latest;state["snapshot_count"]+=1
                    state["max_d2h_bytes"]=max(state["max_d2h_bytes"],snap.d2h_inflight_bytes or 0)
                    state["max_h2d_bytes"]=max(state["max_h2d_bytes"],snap.h2d_inflight_bytes or 0)
                    state["max_active_parents"]=max(state["max_active_parents"],snap.active_parent_count)
                    if snap.free_slots is not None:
                        old=state["min_free_staging_slots"]
                        state["min_free_staging_slots"]=snap.free_slots if old is None else min(old,snap.free_slots)
                replace(reactor,"_observation_sink",sink)
        worker._p3_probe=(state,restores)
        runner=worker.model_runner;config=runner.kv_cache_config
        unique={}
        for t in runner.kv_caches:
            base.require(str(t.dtype)=="torch.bfloat16" and t.device.type=="cuda","wrong KV dtype/device")
            s=t.untyped_storage();unique[s.data_ptr()]=s.nbytes()
        actual=sum(unique.values())
        base.require(len(config.kv_cache_groups)==1 and len(runner.kv_caches)==28,"unsupported layout")
        base.require(actual==sum(t.size for t in config.kv_cache_tensors) and 0<actual<=268435456,"KV budget")
        return dict(actual_gpu_kv_bytes=actual,num_blocks=config.num_blocks,layers=len(runner.kv_caches),groups=1)
    state,restores=worker._p3_probe
    if action=="start":
        state["phase"]="cohort"
        return {k:v for k,v in state.items() if k not in ("events","in_flush")}
    if action in ("drain","finish"):
        state["phase"]="drain"
        cw._submit_store_jobs()
        ids=set().union(*(set(h._active) for h in cw.worker.handlers))
        cw.worker.wait(ids)
        for h in cw.worker.handlers:
            for f,_,_ in list(h._active.values()):f.result(timeout=30)
    result={k:(list(v) if k=="events" else v) for k,v in state.items() if k!="in_flush"}
    result["handlers"]=[]
    for h in cw.worker.handlers:
        r=h.coordinator.reactor
        result["handlers"].append(dict(staging_bytes=r.actual_staging_bytes,pinned=r.staging_buffer.is_pinned(),
            progress_enabled=r.progress_bridge_enabled,observation_failures=r._observation_failures,aio=r.ring.snapshot()))
        base.require(r.actual_staging_bytes<=128*1024**2,"staging exceeds frozen budget")
    if action=="finish":
        for obj,name,old,fn in reversed(restores):
            if getattr(obj,name) is fn:setattr(obj,name,old)
    return result

def main():
    ap=argparse.ArgumentParser()
    for flag in ["output","model-dir","model-plan","curves","manifest"]:ap.add_argument("--"+flag,type=Path,required=True)
    a=ap.parse_args();out=a.output.resolve()
    base.require(out.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs"),"unsafe output")
    out.mkdir(parents=True,exist_ok=False);base.configure_runtime_environment()
    base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"),"budget/offline wrapper required")
    os.environ["PYTHONHASHSEED"]="0"
    manifest_path=a.manifest.resolve();manifest=json.loads(manifest_path.read_text())
    base.require(manifest["formal_goodput_enabled"] is False and manifest["slo"] is None,"development only")
    requests=manifest["requests"];base.require(1<=len(requests)<=32,"bounded cohort")
    base.require(all(0<len(r["prompt_token_ids"])<=768 for r in requests),"outside prompt domain")
    base.require(shutil.disk_usage(out).free>=8*1024**3,"disk guard requires 8 GiB free")
    report=dict(status="FAILED",scope=manifest["scope"],formal_performance_claim=False,policy_mode="shadow",
        ordinary_quota_installed=False,artificial_io_delay=False,cache_resets=0,manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        rows=[],metrics_scope="engine token events; TTFT from actual frontend submission; no SLO goodput")
    llm=None;installed=False
    try:
        model,identity=base.validate_local_model(a.model_dir,a.model_plan)
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True);alias.symlink_to(model,target_is_directory=True)
        extra=dict(spec_name="PyKvCacheOffloadingSpec",spec_module_path="py_kvcache.vllm",
            shared_storage_path=str(out/"storage"),block_size=16,staging_mem=128/1024,
            iodepth=4,sync_on_store=False,enable_preload=True,preload_share_staging=True,
            preload_lookahead_requests=1,staging_cache="lru",load_planner="on",
            prefix_cache_break_even_path=str(a.curves.resolve()),io_backend="linux_aio",
            prefix_io_observation_mode="shadow",prefix_io_observation_run_id=out.parent.name,
            prefix_io_observation_interval_ns=10_000_000)
        config=dict(base.ENGINE,**manifest["engine"])
        config.update(prefix_caching_hash_algo="sha256",disable_log_stats=False,
            kv_transfer_config=dict(kv_connector="OffloadingConnector",kv_role="kv_both",kv_connector_extra_config=extra))
        sampling=dict(base.SAMPLING,max_tokens=128,min_tokens=128)
        base.write_new_json(out/"frozen-config.json",dict(model=identity,engine=config,sampling=sampling,manifest=manifest,pythonhashseed="0",
            gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],curves_sha256=hashlib.sha256(a.curves.read_bytes()).hexdigest()))
        os.chdir(out)
        from vllm import LLM,SamplingParams
        llm=LLM(model=base.MODEL_ID,**config);engine=llm.llm_engine
        report["native_kv"]=llm.collective_rpc(worker_probe,timeout=30,args=("install",))[0];installed=True
        llm.generate([{"prompt_token_ids":manifest["warmup_prompt_tokens"]}],SamplingParams(**base.SAMPLING),use_tqdm=False)
        report["warmup_drain"]=llm.collective_rpc(worker_probe,timeout=45,args=("drain",))[0]
        report["cohort_probe_start"]=llm.collective_rpc(worker_probe,timeout=30,args=("start",))[0]
        timelines={};records={};next_request=0;finished=set()
        start=time.monotonic();deadline=start+180
        while next_request<len(requests) or engine.has_unfinished_requests():
            now=time.monotonic();base.require(now<deadline,"cohort deadline")
            while next_request<len(requests) and requests[next_request]["scheduled_time"]<=now-start:
                spec=requests[next_request];rid="p3-"+str(spec["request_id"])
                sent=time.monotonic()
                actual_id=engine.add_request(rid,{"prompt_token_ids":spec["prompt_token_ids"]},SamplingParams(**sampling))
                timelines[actual_id]=TokenTimeline()
                records[actual_id]=dict(request_id=actual_id,scheduled_arrival_seconds=spec["scheduled_time"],
                    actual_send_seconds=sent-start,client_queue_seconds=sent-start-spec["scheduled_time"],
                    prompt_tokens=len(spec["prompt_token_ids"]))
                next_request+=1
            if not engine.has_unfinished_requests():
                if next_request<len(requests):time.sleep(min(.002,max(0,requests[next_request]["scheduled_time"]-(time.monotonic()-start))))
                continue
            outputs=engine.step()
            for response in outputs:
                record=records[response.request_id];timeline=timelines[response.request_id]
                base.require(len(response.outputs)==1,"multiple samples unsupported")
                timeline.append(response.outputs[0].token_ids,response.metrics.last_token_ts)
                if response.finished:
                    base.require(response.request_id not in finished,"duplicate terminal")
                    finished.add(response.request_id)
                    record.update(timeline.export());record.update(num_cached_tokens=response.num_cached_tokens,
                        metrics=dataclasses.asdict(response.metrics),response_end_seconds=time.monotonic()-start)
                    base.require(record["per_token_complete"] and len(record["output_tokens"])==128 and not response.metrics.is_corrupted,"token evidence incomplete/corrupted")
                    record["itl_p95_seconds"]=percentile(record["itl_seconds"],.95)
                    report["rows"].append(record)
        report["response_makespan_seconds"]=time.monotonic()-start
        drain_start=time.monotonic()
        report["probe"]=llm.collective_rpc(worker_probe,timeout=60,args=("drain",))[0]
        report["tail_drain_seconds"]=time.monotonic()-drain_start
        report["cohort_seconds_including_drain"]=time.monotonic()-start-requests[0]["scheduled_time"]
        base.require(len(finished)==len(requests),"unfinished cohort")
        for h in report["probe"]["handlers"]:
            base.require(h["pinned"] and h["observation_failures"]==0,"observer/pinned failure")
            base.require(h["aio"]["outstanding"]==0 and h["aio"]["accepted"]==h["aio"]["reaped"],"AIO not settled")
        report["completed_throughput_per_second"]=len(finished)/report["cohort_seconds_including_drain"]
        report["status"]="PASSED_NATIVE_P3_DEVELOPMENT_TRACE"
    except BaseException as e:report.update(error=str(e),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                if installed:report["final_probe"]=llm.collective_rpc(worker_probe,timeout=45,args=("finish",))[0]
                llm.llm_engine.engine_core.shutdown(timeout=15);report["engine_shutdown"]="completed"
            except BaseException as e:report.update(status="FAILED",shutdown_error=str(e))
        base.write_new_json(out/"result.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","probe","final_probe","warmup_drain","cohort_probe_start")},indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
