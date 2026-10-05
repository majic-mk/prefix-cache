"""P2 native model observer/owner qualification; controlled off/shadow workloads."""
import argparse,json,os,time,traceback,dataclasses,subprocess
from pathlib import Path
import native_gpu_prefix_smoke as base

def worker_probe(worker,action):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    cw=get_kv_transfer_group().connector_worker
    if action=="install_owner":
        from prefix_io_control.native_owner import NativeOwnerProbe
        probe=NativeOwnerProbe("model-owner-"+str(os.getpid()))
        probe.install();worker._prefix_owner_probe=probe
        return dict(installed=True,pid=os.getpid())
    queued=len(cw._unsubmitted_store_jobs)
    cw._submit_store_jobs()
    ids=set()
    for h in cw.worker.handlers:
        for jid,entry in list(h._active.items()):
            entry[0].result(timeout=30);ids.add(jid)
    cw.worker.wait(ids)
    result=dict(queued_submitted=queued,waited=sorted(ids),handlers=[])
    for h in cw.worker.handlers:
        r=h.coordinator.reactor;pub=r._observation_sink
        latest=pub._latest if pub is not None else None
        result["handlers"].append(dict(progress_enabled=r.progress_bridge_enabled,
            observation_failures=r._observation_failures,
            publications=pub.published_count if pub else 0,
            contention_drops=pub.contention_drops if pub else 0,
            faulted=pub._faulted if pub else False,
            actual_staging_bytes=r.actual_staging_bytes,pinned=r.staging_buffer.is_pinned(),
            latest=dataclasses.asdict(latest) if latest else None,aio=r.ring.snapshot()))
    if hasattr(worker,"_prefix_owner_probe"):
        result["owner"]=worker._prefix_owner_probe.export()
        if action=="finish":worker._prefix_owner_probe.uninstall()
    return result

def main():
    ap=argparse.ArgumentParser()
    for flag in ["output","model-dir","model-plan","curves"]:ap.add_argument("--"+flag,type=Path,required=True)
    ap.add_argument("--mode",choices=["off","shadow"],required=True)
    ap.add_argument("--owner-probe",action="store_true")
    args=ap.parse_args()
    out=args.output.resolve();curve=args.curves.resolve()
    base.require(out.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs"),"unsafe output")
    out.mkdir(parents=True,exist_ok=False);base.configure_runtime_environment()
    os.environ["PYTHONHASHSEED"]="0"
    base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"),"budget/offline runner required")
    report=dict(status="FAILED",mode=args.mode,owner_probe=args.owner_probe,
        policy_mode="shadow" if args.mode=="shadow" else "off",ordinary_quota_installed=False,
        formal_performance_claim=False,rows=[],gpu_resets=0)
    llm=None
    try:
        model,identity=base.validate_local_model(args.model_dir,args.model_plan)
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True);alias.symlink_to(model,target_is_directory=True)
        extra=dict(spec_name="PyKvCacheOffloadingSpec",spec_module_path="py_kvcache.vllm",
            shared_storage_path=str(out/"storage"),block_size=16,staging_mem=128/1024,
            iodepth=4,sync_on_store=False,enable_preload=True,preload_share_staging=True,
            preload_lookahead_requests=1,staging_cache="lru",load_planner="on",
            prefix_cache_break_even_path=str(curve),io_backend="linux_aio",
            prefix_io_observation_mode=args.mode)
        if args.mode=="shadow":extra.update(prefix_io_observation_run_id=out.parent.name,prefix_io_observation_interval_ns=10_000_000)
        config=dict(base.ENGINE,prefix_caching_hash_algo="sha256",disable_log_stats=False,
            kv_transfer_config=dict(kv_connector="OffloadingConnector",kv_role="kv_both",kv_connector_extra_config=extra))
        base.write_new_json(out/"frozen-config.json",dict(model=identity,engine=config,sampling=base.SAMPLING,pythonhashseed="0",
            GPU_UUID=os.environ["CUDA_VISIBLE_DEVICES"],scope="P2 mechanics at 64MiB GPU KV; not requalification of costs for performance",
            owner_probe=args.owner_probe,request_count=32,prompt_tokens=129,output_tokens=16))
        os.chdir(out)
        import torch,vllm
        from vllm import LLM,SamplingParams
        base.require(Path(vllm.__file__).resolve().is_relative_to(base.AUTHOR_ROOT),"wrong model engine")
        llm=LLM(model=base.MODEL_ID,**config)
        report["native_kv"]=llm.collective_rpc(base.worker_kv_storage_metadata,timeout=30)[0]
        if args.owner_probe:report["owner_install"]=llm.collective_rpc(worker_probe,timeout=30,args=("install_owner",))[0]
        for i in range(16):
            tokens=[22000+i]+list(range(1000,1128))
            previous=None
            for kind in ["cold","gpu_hot"]:
                start=time.perf_counter()
                r=llm.generate([{"prompt_token_ids":tokens}],SamplingParams(**base.SAMPLING),use_tqdm=False)[0]
                row=dict(index=i,kind=kind,warmup=i<2,input_tokens=tokens,output_tokens=list(r.outputs[0].token_ids),
                    num_cached_tokens=r.num_cached_tokens,wall_seconds=time.perf_counter()-start,
                    metrics=dataclasses.asdict(r.metrics))
                report["rows"].append(row)
                base.require(len(row["output_tokens"])==16 and not row["metrics"]["is_corrupted"],"invalid model output")
                base.require(row["num_cached_tokens"]==(0 if kind=="cold" else 128),"unexpected Prefix path")
                if previous is not None:base.require(row["output_tokens"]==previous,"natural hot output differs")
                previous=row["output_tokens"]
                (out/"partial.json").write_text(json.dumps(report,indent=2))
        # A normal one-token request advances native completion metadata. No reset,
        # fabricated completion, manual fence removal or artificial source release.
        llm.generate([{"prompt_token_ids":[30000]}],SamplingParams(**dict(base.SAMPLING,max_tokens=1,min_tokens=1)),use_tqdm=False)
        report["probe"]=llm.collective_rpc(worker_probe,timeout=60,args=("finish",))[0]
        for h in report["probe"]["handlers"]:
            base.require(h["pinned"] and h["actual_staging_bytes"]<=128*1024**2,"pinned budget")
            base.require(h["observation_failures"]==0 and not h["faulted"],"observer failed")
            if args.mode=="shadow":
                base.require(h["publications"]>0 and h["progress_enabled"],"no live publication/bridge")
                base.require(h["latest"]["gpu_immediately_reusable_bytes"] is None,"reactor invented owner witness")
            else:base.require(h["publications"]==0 and not h["progress_enabled"],"off allocated observation")
        if args.owner_probe:
            p=report["probe"]["owner"]
            base.require(not p["faulted"] and p["errors"]==0,"owner observer failed")
            base.require(p["max_reusable_bytes"]>0 and p["actual_safe_reallocations"]>0,"no native release/reallocation witness")
            base.require(p["active_reference_observations"]>0,"no active-reference negative observation")
        report["status"]="PASSED_NATIVE_P2_"+args.mode.upper()
    except Exception as e:report.update(error=str(e),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                llm.collective_rpc(worker_probe,timeout=45,args=("finish",))
                llm.llm_engine.engine_core.shutdown(timeout=15)
                report["engine_shutdown"]="completed"
            except Exception as e:report.update(status="FAILED",shutdown_error=str(e))
        base.write_new_json(out/"result.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("rows","probe")},indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
