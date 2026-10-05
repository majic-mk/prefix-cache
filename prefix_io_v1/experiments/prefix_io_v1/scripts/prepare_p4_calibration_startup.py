"""CPU startup/source preparation for original LLM single-stage calibration.

No GPU import, event, benchmark, model load, download or budget reservation.
Raw collection must provide exact active-decode and completed step fields; an
old P3 aggregate or scheduled-work metadata is deliberately not converted.
"""
from __future__ import annotations
import argparse,ast,hashlib,json
from pathlib import Path
from prepare_p4_gpu_next_day import (
    ART,OUT,NATIVE,CONTROL,AUTHOR,BINARY,MODEL,MODEL_PLAN,SCRIPTS,
    require,safe_path,ref,inspect_authorization,validate_source_lock,permission_fields
)

OWNED_BYTES=29360128
QUANTUM=917504
ENGINE_SOURCE=SCRIPTS+"/native_gpu_prefix_smoke.py"
CALIBRATION_SOURCE=SCRIPTS+"/run_decode_interference_calibration_p316.py"

def literals(root):
    tree=ast.parse(safe_path(root,ENGINE_SOURCE).read_text())
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            if node.targets[0].id in ("ENGINE","SAMPLING"):
                values[node.targets[0].id]=ast.literal_eval(node.value)
    require(set(values)=={"ENGINE","SAMPLING"},"real original LLM engine/sampling source")
    return values

def prepare(root,name,stage,units,source_lock=None):
    root=root.resolve();authorization=inspect_authorization(root)
    require(type(name) is str and name and len(name)<=80 and all(c.isalnum() or c in "-_" for c in name),
            "bounded calibration name")
    require(stage in ("ssd_read","ssd_write","h2d","d2h"),"original one-stage capture")
    require(type(units) is int and units in (1,2,4,8),"finite aligned native storage units")
    orig=literals(root)
    engine=dict(orig["ENGINE"],max_model_len=16400,max_num_seqs=2,
        max_num_batched_tokens=16400,kv_cache_memory_bytes=2*1024**3-OWNED_BYTES,
        disable_log_stats=False,prefix_caching_hash_algo="sha256")
    sampling=dict(orig["SAMPLING"],max_tokens=128,min_tokens=128)
    require(engine["kv_transfer_config"] is None,"original isolated calibration model executor")
    require(engine["dtype"]=="bfloat16" and engine["quantization"] is None,"original BF16 unquantized domain")
    model=json.loads(safe_path(root,MODEL_PLAN).read_text())
    require(model["model_id"]=="Qwen/Qwen2.5-7B-Instruct" and
            model["revision"]=="16c174980d8a1492910551634b4969e69cdc2444","frozen cached model plan")
    return dict(schema_version=1,scope="P4_CALIBRATION_STARTUP_CPU_PREPARATION",
        name=name,status="SOURCE_AND_ENGINE_PREPARED_NATIVE_GPU_COLLECTOR_UNQUALIFIED",
        current_authorization=authorization,source_lock=validate_source_lock(root,source_lock),
        source_lock_pending=source_lock is None,
        original_engine_source=ref(root,ENGINE_SOURCE),original_calibration_source=ref(root,CALIBRATION_SOURCE),
        new_workspaces=dict(native=NATIVE,control=CONTROL,author=AUTHOR,compiled_binary=BINARY),
        model_dir=MODEL,model_plan=ref(root,MODEL_PLAN),engine=engine,sampling=sampling,
        source_API=dict(LLM="vllm.LLM(model=<verified local model>, **engine)",
                       SamplingParams="vllm.SamplingParams(**sampling)",
                       execute="existing llm.llm_engine.add_request / step",
                       native="existing TransferCoordinator and original handler"),
        action=dict(stage=stage,units=units,transfer_quantum_bytes=QUANTUM,physical_bytes=units*QUANTUM,
                    physical_operations=None,single_stage_only=True,
                    source_owned_bytes=OWNED_BYTES,gpu_KV_total_bytes=2*1024**3,staging_bytes=1024**3),
        collection_domain=dict(signature="exact9 active decode/batch/prefill/context/quantum",
            scheduled_work_v1_is_active_decode=False,context_growth_bucketed=False,
            step_timing_scope=None,raw_new_IO_from_actual_API_acceptance_only=True,
            aggregate_P3_stage_delta_is_per_window_state=False),
        required_new_raw_fields=["actual active_decode/batch/prefill/context at each selected step",
            "exact four-stage existing_IO before action",
            "exact one-stage added physical ops/bytes for every warmup/measured window",
            "native-completed step start/end scope plus full original token/output work",
            "normal run exit and actual full accepted-IO drain",
            "balanced AB/BA calibration and independent trace/prefix validation split"],
        no_automatic_conversion=["old P3 host-only execute timings","old P3 12-pulse aggregate",
                                 "multi-stage cold-SSD/D2H-write/joint windows","scheduled-work-v1"],
        new_gpu_runs=0,gpu_initialized=False,budget_reserved_seconds=0,model_loaded=False,
        GPU_collector_implemented=False,GPU_collector_verified=False,production_qualified=False,
        P5_SLO=None,P5_allowed=False,effect_verified=False)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",type=Path,default=Path("."))
    ap.add_argument("--name",required=True)
    ap.add_argument("--stage",choices=("ssd_read","ssd_write","h2d","d2h"),required=True)
    ap.add_argument("--units",type=int,choices=(1,2,4,8),required=True)
    ap.add_argument("--source-lock")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--check-launch",action="store_true")
    args=ap.parse_args(argv);root=args.project.resolve()
    p=args.output.relative_to(root).as_posix() if args.output.is_absolute() else args.output.as_posix()
    require(p.startswith(OUT+"/"),"bounded calibration preparation receipt")
    target=safe_path(root,p);require(not target.exists(),"append-new preparation")
    result=prepare(root,args.name,args.stage,args.units,args.source_lock)
    if args.check_launch:result["status"]="BLOCKED_CPU_ONLY_NO_GPU_LAUNCH"
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as h:json.dump(result,h,indent=2,allow_nan=False);h.write("\n")
    print(json.dumps({k:result[k] for k in ("status","new_gpu_runs","gpu_initialized","production_qualified")}))
    return 78 if args.check_launch else 0

if __name__=="__main__":raise SystemExit(main())
