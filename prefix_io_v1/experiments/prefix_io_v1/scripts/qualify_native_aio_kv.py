"""Real model KV byte roundtrip through the CPU-qualified AIO backend.
Reuses the locked native smoke loader/config; no alternative engine or scheduler.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback
import native_gpu_prefix_smoke as base

def worker_roundtrip(worker, output):
    import torch
    from qualify_aio_gpu_reactor import qualify
    metadata=base.worker_kv_storage_metadata(worker)
    caches=worker.model_runner.kv_caches
    base.require(all(t.is_contiguous() and t.shape[0]==metadata["num_blocks"] for t in caches),
                 "only the frozen contiguous block-leading TRITON layout is qualified")
    views=[t.view(torch.int8).reshape(t.shape[0],-1) for t in caches]
    result=qualify(views,output,production=True)
    result["native_kv_metadata"]=metadata
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output-dir",required=True,type=Path)
    ap.add_argument("--model-dir",required=True,type=Path)
    ap.add_argument("--model-plan",required=True,type=Path)
    args=ap.parse_args()
    out=args.output_dir.resolve()
    base.require(out.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs"),"output outside runs")
    out.mkdir(parents=True,exist_ok=False)
    base.configure_runtime_environment()
    for key in ["HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE"]:
        base.require(os.environ.get(key)=="1","offline budget runner required")
    base.require(os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"),"UUID binding required")
    report=dict(status="FAILED",performance_claim=False,model_loaded=False,
                engine_connector_enabled=False,research_policy_enabled=False,results=[])
    llm=None
    try:
        model,identity=base.validate_local_model(args.model_dir,args.model_plan)
        commit=subprocess.check_output(["git","-C",str(base.AUTHOR_ROOT),"rev-parse","HEAD"],text=True).strip()
        base.require(commit==base.AUTHOR_COMMIT,"author vLLM mismatch")
        base.write_new_json(out/"frozen-config.json",dict(model=identity,engine=base.ENGINE,
              sampling=base.SAMPLING,gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
              author_commit=commit,scope="Model native Prefix + offline diagnostic real handler KV roundtrip",
              backend="linux_aio",staging_MiB=32,io_depth=4,storage_factor=1))
        os.chdir(out)
        import torch,vllm
        from vllm import LLM,SamplingParams
        base.require(Path(vllm.__file__).resolve().is_relative_to(base.AUTHOR_ROOT),"wrong vLLM module")
        free,_=torch.cuda.mem_get_info()
        base.require(free>=24*1024**3,"need at least 24 GiB free for bounded Qwen7B smoke")
        llm=LLM(model=str(model),**base.ENGINE)
        report["model_loaded"]=True
        report.update(base.validate_effective_config(llm.llm_engine.vllm_config,torch.bfloat16))
        first=base.summarize_output(llm.generate([{"prompt_token_ids":base.PROMPT_TOKEN_IDS.copy()}],
                      SamplingParams(**base.SAMPLING),use_tqdm=False))
        report["results"].append(first)
        base.require(first["num_cached_tokens"]==0,"cold unexpectedly cached")
        # No model computation is running during this trusted local worker diagnostic.
        report["production_kv_roundtrip"]=llm.collective_rpc(worker_roundtrip,timeout=90,
                      args=(str(out/"worker-kv-roundtrip"),))[0]
        repeat=base.summarize_output(llm.generate([{"prompt_token_ids":base.PROMPT_TOKEN_IDS.copy()}],
                      SamplingParams(**base.SAMPLING),use_tqdm=False))
        report["results"].append(repeat)
        base.require(repeat["num_cached_tokens"]==base.EXPECTED_HOT_TOKENS,"native Prefix hit differs after restore")
        base.require(first["output_token_ids"]==repeat["output_token_ids"],"tokens differ after restoring same KV bytes")
        report["status"]="PASSED_MODEL_KV_AIO_DIAGNOSTIC"
    except Exception as exc:
        report.update(error=str(exc),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown(timeout=15.0)
                report["engine_shutdown"]="completed"
            except Exception as exc:
                report.update(status="FAILED",shutdown_error=str(exc))
        base.write_new_json(out/"result.json",report)
    print(json.dumps(report,indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
