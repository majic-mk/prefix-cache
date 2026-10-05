"""Native GPU-only 128-token cached reference; diagnostic resets, no cost fitting."""
import argparse,dataclasses,hashlib,json,os,traceback
from pathlib import Path
import native_gpu_prefix_smoke as base
from heldout_manifest import load_manifest
from experiment_storage import preflight
p=argparse.ArgumentParser()
for key in ("output","prompts","native-config","model-dir","model-plan"):
    p.add_argument("--"+key,type=Path,required=True)
a=p.parse_args();out=a.output.resolve();pre=preflight(out,64*1024**2)
rows,identity=load_manifest(a.prompts,[16256],3)
config=json.loads(a.native_config.read_text())["engine"]
base.require(config["kv_transfer_config"] is None and config["max_num_seqs"]==1 and
    config["max_model_len"]==config["max_num_batched_tokens"]==16400 and config["kv_cache_memory_bytes"]==2147483648,
    "frozen GPU reference config")
out.mkdir(parents=True,exist_ok=False);(out/"driver-source.py").write_text(Path(__file__).read_text())
base.configure_runtime_environment();os.environ["PYTHONHASHSEED"]="0"
base.require(os.environ.get("HF_HUB_OFFLINE")=="1" and os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"),"offline UUID runner")
report=dict(status="FAILED",scope="Native GPU-only long cached output reference; no performance claim",
    prompt_manifest=identity,rows=[],gpu_resets=0,ordinary_policy_installed=False)
llm=None
try:
    model,model_identity=base.validate_local_model(a.model_dir,a.model_plan)
    alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True);alias.symlink_to(model,target_is_directory=True)
    sampling=dict(base.SAMPLING,max_tokens=128,min_tokens=128)
    base.write_new_json(out/"frozen-config.json",dict(engine=config,sampling=sampling,model=model_identity,
        gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],prompt_manifest=identity))
    os.chdir(out)
    from vllm import LLM,SamplingParams
    import vllm
    base.require(Path(vllm.__file__).resolve().is_relative_to(base.AUTHOR_ROOT),"wrong author vLLM")
    llm=LLM(model=base.MODEL_ID,**config)
    for rep in range(3):
        base.require(llm.reset_prefix_cache(reset_running_requests=False,reset_connector=False),"diagnostic reset failed")
        report["gpu_resets"]+=1
        for kind in ("cold","gpu_hot"):
            response=llm.generate([{"prompt_token_ids":rows[(16256,rep)]}],SamplingParams(**sampling),use_tqdm=False)[0]
            base.require(response.num_cached_tokens==(0 if kind=="cold" else 16256),"wrong reference shape")
            tokens=list(response.outputs[0].token_ids)
            base.require(len(tokens)==128 and not response.metrics.is_corrupted,"corrupted reference")
            report["rows"].append(dict(rep=rep,kind=kind,num_cached_tokens=response.num_cached_tokens,
                prompt_token_ids=response.prompt_token_ids,output_tokens=tokens,metrics=dataclasses.asdict(response.metrics)))
    report["status"]="PASSED_NATIVE_LONG_CACHED_REFERENCE"
except BaseException as e:report.update(error=str(e),traceback=traceback.format_exc())
finally:
    if llm is not None:
        try:llm.llm_engine.engine_core.shutdown(timeout=15);report["engine_shutdown"]="completed"
        except BaseException as e:report.update(status="FAILED",shutdown_error=str(e))
    report["storage_after"]=preflight(out,0);base.write_new_json(out/"result.json",report)
print(json.dumps({k:v for k,v in report.items() if k!="rows"},indent=2))
raise SystemExit(0 if report["status"].startswith("PASSED") else 1)
