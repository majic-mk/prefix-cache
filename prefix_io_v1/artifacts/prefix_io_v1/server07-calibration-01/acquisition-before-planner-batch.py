"""Bounded native connector calibration acquisition, not a policy benchmark.
Run only through run_gpu_stage.py. No executor/cache algorithm replacement.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import re
from pathlib import Path
import time
import traceback
import native_gpu_prefix_smoke as base

def worker_control(worker, action, path=None):
    from simple_profiler import profiler
    if action == "begin":
        profiler.begin_session(path)
        base.require(profiler._active, "native profiler inactive")
        return {"profile_active":True}
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group, has_kv_transfer_group
    connector=get_kv_transfer_group() if has_kv_transfer_group() else None
    cw=connector.connector_worker if connector else None
    result={}
    if cw is not None:
        queued=len(cw._unsubmitted_store_jobs)
        cw._submit_store_jobs()
        ids=set()
        for handler in cw.worker.handlers:
            for job_id,entry in list(handler._active.items()):
                entry[0].result(timeout=30)
                ids.add(job_id)
        cw.worker.wait(ids)
        # Do not consume get_finished: ordinary engine steps deliver completions
        # and remove scheduler fences through their original metadata path.
        result["queued_stores_submitted"]=queued
        result["waited_job_ids"]=sorted(ids)
        result["handlers"]=[]
        for handler in cw.worker.handlers:
            r=handler.coordinator.reactor
            result["handlers"].append(dict(staging_bytes=r.actual_staging_bytes,
                staging_budget=r.staging_budget_bytes,pinned=r.staging_buffer.is_pinned(),
                io_size=r.file_store.io_size,storage_block_bytes=r.layout.storage_block_bytes,
                aio=r.ring.snapshot()))
    if action == "end":
        profiler.end_session()
        result["profile_flushed"]=Path(path).is_file()
    return result

def prompt(n,rep):
    # Distinct first token avoids unintended shared Prefix across trials.
    return [2000+n+rep*1500]+[1000+(i%500) for i in range(n-1)]+[777]

def trace_summary(path,req_id):
    raw=json.loads(Path(path).read_text())
    events=raw["traceEvents"]
    # Author InputProcessor appends exactly eight random hex chars internally.
    ids={str(e.get("args",{}).get("req_id","")) for e in events}
    matched=[x for x in ids if re.fullmatch(re.escape(str(req_id))+r"-[0-9a-f]{8}",x)]
    base.require(len(matched)<=1,"ambiguous frontend-to-engine request mapping")
    req_id=matched[0] if matched else str(req_id)
    transfers=[e for e in events if e.get("name")=="py_kvcache.transfer" and e.get("args",{}).get("req_id")==req_id]
    loads=[e for e in transfers if e["args"]["direction"]=="storage_to_gpu"]
    reads=[e for e in events if e.get("name")=="py_kvcache.file_read" and e.get("args",{}).get("req_id")==req_id]
    pre=[e for e in events if e.get("name")=="py_kvcache.preload.file_read" and e.get("args",{}).get("success") and e.get("args",{}).get("req_id")==req_id]
    return dict(internal_request_id=req_id,event_count=len(events),load_transfers=[e["args"] for e in loads],
        transfer_success=all(e["args"]["success"] for e in loads),
        foreground_logical_read_bytes=sum(e["args"]["num_bytes"] for e in reads),
        preload_actual_read_bytes=sum(e["args"]["nbytes"] for e in pre),
        src_cache=sum(e["args"]["src_cache"] for e in loads),
        src_file=sum(e["args"]["src_file"] for e in loads),
        src_preload=sum(e["args"]["src_preload"] for e in loads),
        h2d_events=sum(e.get("name")=="py_kvcache.cuda_staging" and
            e.get("args",{}).get("req_id")==req_id and e.get("args",{}).get("direction")=="storage_to_gpu" for e in events),
        planner_events=[{"name":e["name"],"args":e.get("args",{})} for e in events if "planner" in e.get("name","")])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--storage",required=True,type=Path)
    ap.add_argument("--model-dir",required=True,type=Path)
    ap.add_argument("--model-plan",required=True,type=Path)
    ap.add_argument("--mode",choices=["cold","populate","restore","paired","planned"],required=True)
    ap.add_argument("--curves",type=Path)
    ap.add_argument("--sizes",default="16,64,128,256,512,1024")
    ap.add_argument("--reps",type=int,default=6)
    args=ap.parse_args()
    out=args.output_dir.resolve(); storage=args.storage.resolve()
    for p in [out,storage]:
        base.require(p.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs"),"path outside runs")
    base.require(args.mode!="planned" or args.curves,"planned mode requires measured curves")
    sizes=[int(x) for x in args.sizes.split(",")]
    base.require(all(16<=n<=1024 and n%16==0 for n in sizes) and 1<=args.reps<=8,"unbounded acquisition")
    if args.mode=="populate":storage.mkdir(parents=True,exist_ok=False)
    elif args.mode!="cold":base.require(storage.is_dir(),"no published storage")
    out.mkdir(parents=True,exist_ok=False)
    curves=args.curves.resolve() if args.curves else None
    base.configure_runtime_environment()
    os.environ["PYTHONHASHSEED"]="0"  # Author-supported stable NONE_HASH across engine restarts.
    base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("TRANSFORMERS_OFFLINE")=="1","offline runner required")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"),"UUID runner required")
    report=dict(status="FAILED",mode=args.mode,performance_claim=False,full_P1_pass=False,
        scope="native TTFT cost acquisition; resets are diagnostic only",rows=[])
    llm=None
    try:
        model,identity=base.validate_local_model(args.model_dir,args.model_plan)
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True)
        alias.symlink_to(model,target_is_directory=True)
        extra=dict(spec_name="PyKvCacheOffloadingSpec",spec_module_path="py_kvcache.vllm",
            shared_storage_path=str(storage),block_size=16,sync_on_store=False,
            staging_mem=128/1024,iodepth=4,enable_preload=True,
            preload_lookahead_requests=1,preload_share_staging=True,staging_cache="lru",
            load_planner="on" if args.mode=="planned" else "off",io_backend="linux_aio")
        if curves:extra["prefix_cache_break_even_path"]=str(curves)
        config=dict(base.ENGINE,max_model_len=1040,max_num_batched_tokens=1040,
            kv_cache_memory_bytes=268435456,disable_log_stats=False,prefix_caching_hash_algo="sha256")
        if args.mode=="planned":
            measured=json.loads(curves.read_text())
            base.require(measured["model_name"]==base.MODEL_ID and measured["provenance"]["gpu_uuid"]==os.environ["CUDA_VISIBLE_DEVICES"],"curve identity mismatch")
            base.require(measured["provenance"]["max_supported_model_len"]==1024 and max(sizes)<1024,"unsupported planner range")
            config["max_model_len"]=1024
            report["curves_sha256"]=hashlib.sha256(curves.read_bytes()).hexdigest()
        if args.mode!="cold":
            config["kv_transfer_config"]=dict(kv_connector="OffloadingConnector",kv_role="kv_both",kv_connector_extra_config=extra)
        sampling=dict(base.SAMPLING,max_tokens=1,min_tokens=1)
        base.write_new_json(out/"frozen-config.json",dict(model=identity,model_alias=base.MODEL_ID,
            alias_target=str(model),engine=config,sampling=sampling,sizes=sizes,reps=args.reps,
            gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],pythonhashseed="0",cost_acquisition_only=args.mode!="planned",
            measured_axis="reusable prefix N; identical request has N+1 tokens, output one token",
            terminal_knot_note="1040 context only accommodates N=1024 plus query; exported planner domain at most 1024"))
        os.chdir(out)
        import torch,vllm
        from vllm import LLM,SamplingParams
        base.require(Path(vllm.__file__).resolve().is_relative_to(base.AUTHOR_ROOT),"wrong author module")
        base.require(torch.cuda.mem_get_info()[0]>=24*1024**3,"insufficient free VRAM")
        llm=LLM(model=base.MODEL_ID,**config)
        report["model_loaded"]=True
        def generate(n,rep,kind,profile=True):
            path=out/f"{kind}-{n}-{rep}.trace.json"
            if profile:llm.collective_rpc(worker_control,timeout=45,args=("begin",str(path)))
            started=time.perf_counter()
            result=llm.generate([{"prompt_token_ids":prompt(n,rep)}],SamplingParams(**sampling),use_tqdm=False)[0]
            wall=time.perf_counter()-started
            row=dict(kind=kind,prefix_tokens=n,rep=rep,warmup=rep==0,
                request_id=result.request_id,prompt_token_ids=result.prompt_token_ids,
                output_token_ids=list(result.outputs[0].token_ids),
                num_cached_tokens=result.num_cached_tokens,wall_seconds=wall,
                metrics=dataclasses.asdict(result.metrics) if result.metrics else None)
            base.require(len(row["output_token_ids"])==1 and len(row["prompt_token_ids"])==n+1,"token lengths")
            if profile:
                row["drain"]=llm.collective_rpc(worker_control,timeout=45,args=("end",str(path)))[0]
                row["trace"]=trace_summary(path,result.request_id)
                row["trace_path"]=str(path)
            else:llm.collective_rpc(worker_control,timeout=45,args=("drain",None))
            report["rows"].append(row)
            (out/"partial.json").write_text(json.dumps(report,indent=2))
            return row
        for n in sizes:
            for rep in range(args.reps):
                if args.mode=="populate":
                    generate(n,rep,"store")
                    # A real engine step receives prior completion metadata.
                    llm.generate([{"prompt_token_ids":[30000+rep]}],SamplingParams(**sampling),use_tqdm=False)
                    llm.collective_rpc(worker_control,timeout=45,args=("drain",None))
                if args.mode!="planned":
                    base.require(llm.reset_prefix_cache(reset_running_requests=False,reset_connector=False),"GPU-only reset failed")
                kind={"cold":"f","populate":"g_mem","restore":"g_ssd","paired":"g_ssd","planned":"planned"}[args.mode]
                row=generate(n,rep,kind)
                base.require(row["metrics"] and row["metrics"]["first_token_latency"]>0,"native TTFT absent")
                if args.mode=="cold":base.require(row["num_cached_tokens"]==0,"cold cache hit")
                if args.mode=="planned":
                    base.require(row["trace"]["planner_events"],"original planner not called")
                    hot=generate(n,rep,"gpu_hot")
                    base.require(hot["num_cached_tokens"]==n and not hot["trace"]["load_transfers"],"natural GPU hit not proven")
                    base.require(hot["output_token_ids"]==row["output_token_ids"],"natural repeat output differs")
                    report["scope"]="Original measured LoadPlanner plus natural GPU reuse, no per-request reset; calibration-prefix integration check"
                    report["gpu_reset_count"]=0
                if args.mode in ("populate","restore","paired"):
                    tr=row["trace"]
                    base.require(tr["load_transfers"] and tr["transfer_success"] and tr["h2d_events"],"missing successful native load/H2D")
                    base.require(row["num_cached_tokens"]==n,"not full intended external prefix")
                    actual=tr["foreground_logical_read_bytes"]+tr["preload_actual_read_bytes"]
                    base.require(actual==(0 if args.mode=="populate" else n*57344),"wrong medium/read byte coverage")
                    if args.mode=="populate":base.require(tr["src_cache"]>0,"staging source not proven")
                if args.mode=="paired":
                    base.require(llm.reset_prefix_cache(reset_running_requests=False,reset_connector=False),"paired GPU reset failed")
                    mem=generate(n,rep,"g_mem")
                    tr=mem["trace"]
                    base.require(mem["output_token_ids"]==row["output_token_ids"],"paired output differs")
                    base.require(mem["num_cached_tokens"]==n and tr["src_cache"]==n//16 and
                        tr["transfer_success"] and tr["h2d_events"] and
                        tr["foreground_logical_read_bytes"]+tr["preload_actual_read_bytes"]==0,
                        "paired staging path not proven")
        report["status"]="PASSED_NATIVE_"+args.mode.upper()+"_ACQUISITION"
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                report["final_drain"]=llm.collective_rpc(worker_control,timeout=45,args=("drain",None))[0]
                llm.llm_engine.engine_core.shutdown(timeout=15)
                report["engine_shutdown"]="completed"
            except Exception as exc:report.update(status="FAILED",shutdown_error=str(exc))
        base.write_new_json(out/"result.json",report)
    print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
