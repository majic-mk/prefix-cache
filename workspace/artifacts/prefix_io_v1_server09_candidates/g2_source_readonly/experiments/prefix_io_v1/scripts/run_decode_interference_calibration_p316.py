"""P3 calibration only: original LLM engine plus one owned native I/O handler.
No production KV is accessed by the synthetic handler. No executor is replaced.
Importing this module performs no CUDA import or operation.
"""
from __future__ import annotations
import argparse, copy, dataclasses, hashlib, json, math, os, time, traceback
from pathlib import Path
import native_gpu_prefix_smoke as base
from prefix_io_control.token_timeline import TokenTimeline, percentile

SCOPE = "P3_SPARSE_DECODE_INTERFERENCE_CALIBRATION_ONLY"
ANCHORS = ("none", "warm_h2d", "d2h_write", "cold_ssd_h2d", "joint")
NATIVE_ROOTS = ("third_party/work/py-kvcache-p3-16-cpu",)
TOTAL_KV = 2 * 1024**3
STAGING = 1024**3
OWNED = dict(tensor_count=28, page_bytes=32768, source_units=16,
             destination_units=16, owned_bytes=29360128, storage_unit_bytes=917504)
PULSE = dict(start_decode_step=16, interval_decode_steps=8, max_pulses=12,
             maximum_parent_units=16)
CELLS = [dict(anchor=a, units=8) for a in ANCHORS] + [
    dict(anchor="warm_h2d", units=1), dict(anchor="d2h_write", units=1)]
COMMON_SOURCES = (
 "experiments/prefix_io_v1/scripts/run_decode_interference_calibration_p316.py",
 "experiments/prefix_io_v1/scripts/decode_interference_worker_probe_p316.py",
 "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py",
 "experiments/prefix_io_v1/scripts/analyze_decode_interference_calibration_p316.py",
 "src/prefix_io_control/token_timeline.py",
 "src/prefix_io_control/stage_accounting.py",
 "third_party/work/vllm-author-build/vllm/v1/worker/gpu_worker.py",
 "third_party/work/vllm-author-build/vllm/v1/engine/input_processor.py",
 "third_party/work/vllm-author-build/vllm/v1/engine/llm_engine.py",
 "third_party/work/vllm-author-build/vllm/v1/kv_offload/base.py",
)
TOP_KEYS = {"schema_version", "scope", "partition", "session_id", "gpu_uuid",
 "native_worktree", "source_sha256", "engine", "sampling", "prompt_token_ids",
 "prompt_family", "owned", "io", "cells", "sequence", "pulse",
 "window_deadline_seconds", "reference_results", "active_decode_requests", "table_domain"}

def require(ok, message):
    if not ok: raise ValueError(message)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_load(path):
    def unique(items):
        out = {}
        for k, v in items:
            require(k not in out, "duplicate JSON key: " + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)

def same(actual, expected, message):
    require(type(actual) is type(expected), message)
    if type(expected) is dict:
        require(set(actual) == set(expected), message)
        for key in expected: same(actual[key], expected[key], message)
    elif type(expected) is list:
        require(len(actual) == len(expected), message)
        for a, b in zip(actual, expected): same(a, b, message)
    else: require(actual == expected, message)

def source_names(native_root):
    """Lock all Python modules in the actual author/native/control source trees."""
    require(native_root == NATIVE_ROOTS[-1], "only P316 native sources may be frozen")
    names = set(COMMON_SOURCES)
    for relative in (native_root + "/py_kvcache", "src/prefix_io_control",
                     "third_party/work/vllm-author-build/vllm"):
        folder = base.ROOT / relative
        require(folder.is_dir(), "source tree missing: " + relative)
        names.update(str(p.relative_to(base.ROOT)) for p in folder.rglob("*.py")
                     if "__pycache__" not in p.parts)
    return names

def engine_config():
    return dict(base.ENGINE, max_model_len=16400, max_num_seqs=2,
        max_num_batched_tokens=16400, kv_cache_memory_bytes=TOTAL_KV-OWNED["owned_bytes"],
        disable_log_stats=False, prefix_caching_hash_algo="sha256")

def sampling_config():
    return dict(base.SAMPLING, max_tokens=128, min_tokens=128)

def candidate_spec(session_id, gpu_uuid, partition="calibration",
                   native_worktree=NATIVE_ROOTS[-1]):
    """CPU-only candidate. Root must fill source hashes before a GPU launch."""
    prefix = [31000 if partition == "calibration" else 32000]
    prefix += (list(range(1000, 5096)) * 4)[:16255]
    return dict(schema_version=1, scope=SCOPE, partition=partition,
        session_id=session_id, gpu_uuid=gpu_uuid, native_worktree=native_worktree,
        source_sha256={p: None for p in sorted(source_names(native_worktree))},
        engine=engine_config(), sampling=sampling_config(), prompt_token_ids=prefix+[13],
        prompt_family="owned-calibration" if partition == "calibration" else "owned-validation",
        owned=dict(OWNED), io=dict(iodepth=8, staging_mem=1.0, staging_cache="lru",
        enable_preload=True, preload_share_staging=True, io_backend="linux_aio",
        sync_on_store=False, load_planner="off"),
        cells=copy.deepcopy(CELLS), sequence=["A","B","B","A"], pulse=dict(PULSE),
        window_deadline_seconds=60, reference_results=[], active_decode_requests=1,
        table_domain=dict(engine_conditional=True,model_kv_budget_bytes=TOTAL_KV-OWNED["owned_bytes"],
        active_decode_requests=1,production_resource_witness=False,production_state_fallback=None,
        p4_connected=False,cost_permit_reused=False))

def validate_spec(raw):
    """Pure validation: no runtime import, file population, or GPU operation."""
    require(type(raw) is dict and set(raw) == TOP_KEYS, "missing/unknown spec fields")
    same(raw["schema_version"], 1, "unsupported schema")
    same(raw["scope"], SCOPE, "calibration scope changed")
    require(raw["partition"] in ("calibration","validation"), "unknown partition")
    for key in ("session_id", "gpu_uuid", "prompt_family"):
        value=raw[key]
        require(type(value) is str and 0 < len(value) <= 128 and value.strip()==value,
                "invalid identity: "+key)
        require(all(c.isalnum() or c in "-_." for c in value), "unsafe identity: "+key)
    require(raw["gpu_uuid"].startswith("GPU-"), "explicit GPU UUID required")
    require(raw["native_worktree"] in NATIVE_ROOTS, "unqualified native worktree")
    locks=raw["source_sha256"]
    require(type(locks) is dict and set(locks)==source_names(raw["native_worktree"]),
            "source coverage mismatch")
    for key, value in locks.items():
        require(type(value) is str and len(value)==64 and
                all(c in "0123456789abcdef" for c in value), "unfrozen source hash: "+key)
    same(raw["engine"], engine_config(), "engine/context/KV/attention config changed")
    same(raw["sampling"], sampling_config(), "sampling or output work changed")
    same(raw["owned"], OWNED, "owned layout/budget changed")
    same(raw["pulse"], PULSE, "pulse schedule changed")
    same(raw["io"], candidate_spec("x","GPU-x")["io"], "native I/O config changed")
    same(raw["cells"], CELLS, "sparse cell domain changed")
    same(raw["sequence"], ["A","B","B","A"], "ABBA order changed")
    same(raw["window_deadline_seconds"], 60, "window deadline changed")
    same(raw["active_decode_requests"], 1, "single active request required")
    same(raw["table_domain"], candidate_spec("x","GPU-x")["table_domain"], "conditional table domain changed")
    candidate=candidate_spec("x","GPU-x",raw["partition"],raw["native_worktree"])
    same(raw["prompt_family"],candidate["prompt_family"],"partition prompt family changed")
    same(raw["prompt_token_ids"],candidate["prompt_token_ids"],"frozen partition prefix changed")
    tokens=raw["prompt_token_ids"]
    require(type(tokens) is list and len(tokens)==16257 and all(
        type(t) is int and 0 <= t < 100000 for t in tokens), "invalid prompt tokens")
    require(len(tokens)+raw["sampling"]["max_tokens"] <= raw["engine"]["max_model_len"],
            "prompt plus output exceeds context")
    require(raw["engine"]["kv_cache_memory_bytes"]+raw["owned"]["owned_bytes"]==TOTAL_KV,
            "total model/owned KV budget changed")
    refs=raw["reference_results"]
    require(type(refs) is list and len(refs)==(0 if raw["partition"]=="calibration" else 2),
            "validation requires two frozen calibration references")
    for ref in refs:
        require(type(ref) is dict and set(ref)=={"path","sha256"}, "invalid reference")
        p=Path(ref["path"])
        require(type(ref["path"]) is str and not p.is_absolute() and ".." not in p.parts
            and "\\" not in ref["path"] and "\x00" not in ref["path"]
            and ref["path"].startswith("experiments/prefix_io_v1/runs/")
            and p.name=="result.json", "unsafe reference path")
        h=ref["sha256"]
        require(type(h) is str and len(h)==64 and all(c in "0123456789abcdef" for c in h),
                "unfrozen reference")
    require(len({r["path"] for r in refs})==len(refs), "duplicate calibration reference")
    return copy.deepcopy(raw)

def check_inputs(spec, root=base.ROOT):
    for rel, expected in spec["source_sha256"].items():
        p=root/rel
        require(p.is_file() and not p.is_symlink() and digest(p)==expected,
                "locked source changed: "+rel)
    for ref in spec["reference_results"]:
        p=root/ref["path"]
        require(p.is_file() and not p.is_symlink() and digest(p)==ref["sha256"],
                "calibration reference changed")
        d=json_load(p)
        require(d["status"]=="PASSED_SPARSE_DECODE_INTERFERENCE" and
                d["spec"]["partition"]=="calibration", "unsuccessful calibration reference")
        for key in ("engine","sampling","owned","io","cells","sequence","pulse","active_decode_requests","table_domain","gpu_uuid","native_worktree","source_sha256"):
            same(d["spec"][key], spec[key], "calibration/validation config differs")
        require(d["spec"]["prompt_family"]!=spec["prompt_family"] and
                d["spec"]["prompt_token_ids"][0]!=spec["prompt_token_ids"][0],
                "validation prefix family must be independent")

def validate_measurement_row(row, corrupted):
    """Reject incomplete output/timing and a different native Prefix hot state."""
    require(type(row["num_cached_tokens"]) is int and row["num_cached_tokens"]==16256,
            "measurement native GPU Prefix state is not the frozen hot prefix")
    require(row["per_token_complete"] is True and len(row["output_tokens"])==128
        and row["ambiguous_events"]==[] and corrupted is False,"incomplete token/output evidence")
    ts=row["engine_token_timestamps"];itl=row["itl_seconds"]
    require(len(ts)==128 and len(itl)==127 and
        all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in ts+itl) and
        all(a<b for a,b in zip(ts,ts[1:])),"invalid token times")

def window_metrics(rows, steps):
    intervals=[v for row in rows for v in row["itl_seconds"]]
    return dict(request_count=len(rows), output_tokens=sum(len(r["output_tokens"]) for r in rows),
        itl_samples=len(intervals), itl_median_seconds=percentile(intervals,.5),
        itl_p95_seconds=percentile(intervals,.95),
        request_itl_p95_seconds=[r["itl_p95_seconds"] for r in rows],
        frontend_step_seconds=[r["seconds"] for r in steps],
        scope="engine output ITL and frontend wall step; not client SSE or pure GPU kernel time")

def run_window(llm, spec, index, cell, role, reference_outputs):
    from vllm import SamplingParams
    from decode_interference_worker_probe_p316 import worker_probe
    engine=llm.llm_engine
    prefix="cal-w"+str(index)+"-"
    ids=[prefix+str(i) for i in range(spec["active_decode_requests"])]
    require(not engine.has_unfinished_requests(),"previous original engine request remains active")
    activation=dict(anchor=cell["anchor"] if role=="B" else "none", units=cell["units"],
                    index=index, frontend_request_ids=ids)
    # Arm before request submission. No worker RPC occurs while the core can decode.
    llm.collective_rpc(worker_probe,timeout=60,args=("begin",activation))
    timelines={rid:TokenTimeline() for rid in ids}; rows=[];steps=[];finished=set()
    started=time.monotonic()
    internal=[engine.add_request(rid,{"prompt_token_ids":spec["prompt_token_ids"]},
              SamplingParams(**spec["sampling"])) for rid in ids]
    activation["native_request_ids_returned"]=internal
    while engine.has_unfinished_requests():
        require(time.monotonic()-started < spec["window_deadline_seconds"], "window timeout")
        a=time.monotonic();responses=engine.step();b=time.monotonic()
        steps.append(dict(start=a,end=b,seconds=b-a))
        for response in responses:
            require(response.request_id in timelines and len(response.outputs)==1,"unexpected output")
            t=timelines[response.request_id];t.append(response.outputs[0].token_ids,response.metrics.last_token_ts)
            if response.finished:
                require(response.request_id not in finished,"duplicate request completion")
                finished.add(response.request_id)
                r=dict(request_id=response.request_id, slot=ids.index(response.request_id),
                       num_cached_tokens=response.num_cached_tokens, **t.export())
                r["metrics"]=dataclasses.asdict(response.metrics)
                validate_measurement_row(r,response.metrics.is_corrupted)
                r["itl_p95_seconds"]=percentile(r["itl_seconds"],.95);rows.append(r)
    response_seconds=time.monotonic()-started
    require(finished==set(ids), "unfinished request")
    drain=time.monotonic()
    probe=llm.collective_rpc(worker_probe,timeout=60,args=("end",))[0]
    tail=time.monotonic()-drain
    require(probe["bound_native_request_ids"]==internal,"worker bound a different native request")
    rows.sort(key=lambda r:r["slot"])
    outputs=[r["output_tokens"] for r in rows]
    if reference_outputs is None: reference_outputs=outputs
    require(outputs==reference_outputs, "full model output differs from first baseline")
    spans=probe["stage_spans"]
    pure=probe["decode_steps"]
    overlapping=sum(any(s["start_ns"]<step["end_ns"] and s["end_ns"]>step["start_ns"]
                    for s in spans) for step in pure)
    return dict(index=index, cell=cell, role=role, activation=activation, rows=rows,
        steps=steps, metrics=window_metrics(rows,steps), probe=probe,
        output_exact=True, response_seconds=response_seconds, tail_drain_seconds=tail,
        duration_including_drain=time.monotonic()-started,
        backend_inflight_overlap_decode_steps=overlapping,
        overlap_scope="API accepted-to-native completion overlapping worker decode wall interval; not instantaneous DMA/compute overlap"),reference_outputs

def main():
    ap=argparse.ArgumentParser()
    for key in ("output","model-dir","model-plan","spec"): ap.add_argument("--"+key,type=Path,required=True)
    a=ap.parse_args();spec=validate_spec(json_load(a.spec));check_inputs(spec)
    require(spec["native_worktree"]==NATIVE_ROOTS[-1],"GPU calibration requires P316 qualified native worktree")
    from experiment_storage import permission, preflight
    p=permission()
    require(p["allow_gpu_runs"] and spec["gpu_uuid"] in p["approved_gpu_ids"]
        and os.environ.get("CUDA_VISIBLE_DEVICES")==spec["gpu_uuid"], "GPU identity/permission mismatch")
    require(os.environ.get("HF_HUB_OFFLINE")=="1","offline budget wrapper required")
    out=a.output.resolve()
    require(out.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs") and not out.exists(),
            "new PRIMARY output required")
    storage_before=preflight(out,1536*1024**2)
    out.mkdir(parents=True,exist_ok=False)
    base.write_new_json(out/"spec.json",spec)
    base.configure_runtime_environment();os.environ["PYTHONHASHSEED"]="0"
    report=dict(status="FAILED",scope=SCOPE,spec=spec,spec_sha256=digest(a.spec),
        source_sha256=spec["source_sha256"], storage_before=storage_before, windows=[],
        formal_performance_claim=False,production_release_witness=False,model_executor_replaced=False,
        table_domain=spec["table_domain"],
        cache_resets=0, client_slo=None, model_external_connector=False,
        independent_validation_content=spec["partition"]=="validation",
        expected_measurement_hot_prefix_tokens=16256,
        limitations="Owned BF16 synthetic native-I/O calibration with the original model engine. Different domain from native production cache/controller qualification; no research effect or hardware DMA overlap claim.")
    llm=None;installed=False
    try:
        require(a.spec.resolve().is_relative_to(base.ROOT),"spec must be under project")
    except BaseException:
        raise
    try:
        model,identity=base.validate_local_model(a.model_dir,a.model_plan)
        report["model"]=identity
        alias=out/base.MODEL_ID;alias.parent.mkdir(parents=True,exist_ok=True);alias.symlink_to(model,target_is_directory=True)
        base.write_new_json(out/"frozen-config.json",dict(engine=spec["engine"],
            sampling=spec["sampling"],model=identity,gpu_uuid=spec["gpu_uuid"],spec_sha256=report["spec_sha256"]))
        os.chdir(out)
        from vllm import LLM,SamplingParams
        from decode_interference_worker_probe_p316 import worker_probe
        llm=LLM(model=base.MODEL_ID,**spec["engine"])
        report["allocation"]=llm.collective_rpc(worker_probe,timeout=60,
            args=("install",dict(spec=spec,output=str(out))))[0];installed=True
        # Identical native GPU Prefix warmup, outside every measurement window.
        llm.generate([{"prompt_token_ids":spec["prompt_token_ids"]}],
            SamplingParams(**spec["sampling"]),use_tqdm=False)
        reference=None;index=0
        for cell in spec["cells"]:
            for role in spec["sequence"]:
                llm.collective_rpc(worker_probe,timeout=60,args=("prepare",
                    dict(index=index,anchor=cell["anchor"],units=cell["units"])))
                window,reference=run_window(llm,spec,index,cell,role,reference)
                report["windows"].append(window);index+=1
        require(len(report["windows"])==28,"sparse matrix incomplete")
        require(all(w["output_exact"] for w in report["windows"]),"output mismatch")
        report["status"]="PASSED_SPARSE_DECODE_INTERFERENCE"
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        if llm is not None:
            if installed:
                try:report["final_probe"]=llm.collective_rpc(worker_probe,timeout=60,args=("finish",))[0]
                except BaseException as exc:report.update(status="FAILED",probe_shutdown_error=repr(exc))
            try:
                llm.llm_engine.engine_core.shutdown(timeout=15);report["engine_shutdown"]="completed"
            except BaseException as exc:report.update(status="FAILED",engine_shutdown_error=repr(exc))
        try:
            check_inputs(spec)
            report["inputs_preserved"]=True
            report["storage_after"]=preflight(out,0)
        except BaseException as exc:report.update(status="FAILED",preservation_error=repr(exc))
        base.write_new_json(out/"result.json",report)
    print(json.dumps(dict(status=report["status"],output=str(out),windows=len(report["windows"]),
                         formal_performance_claim=False)))
    return 0 if report["status"].startswith("PASSED") else 1

if __name__=="__main__":raise SystemExit(main())
