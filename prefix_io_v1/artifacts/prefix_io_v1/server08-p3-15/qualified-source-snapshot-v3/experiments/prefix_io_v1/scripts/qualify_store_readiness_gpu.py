"""Bounded real CUDA/AIO validation of passive readiness; no policy efficacy."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from experiment_storage import ROOT, preflight
from prefix_io_control.store_readiness import StoreReadinessProbe
from prefix_io_control.stage_accounting import StageAccounting

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def qualify(out):
    import torch
    import py_kvcache.reactor as native
    from py_kvcache.reactor import TransferCoordinator
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler, SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from vllm.v1.kv_offload.base import (
        CanonicalKVCaches, CanonicalKVCacheTensor, CanonicalKVCacheRef, GPULoadStoreSpec)
    require(Path(native.__file__).resolve().is_relative_to(
        ROOT/"third_party/work/py-kvcache-p3-quota-cpu"), "wrong frozen worktree")
    require(torch.cuda.device_count()==1, "one approved GPU required")
    torch.manual_seed(311)
    tensors=[torch.randint(-128,128,(128,32768),device="cuda",dtype=torch.int8) for _ in range(4)]
    original=[t[:64].cpu().clone() for t in tensors]
    canonical=CanonicalKVCaches(
        [CanonicalKVCacheTensor(t,page_size_bytes=32768) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i,page_size_bytes=32768) for i in range(4)]])
    layout=ParsedKvLayout.from_canonical_kv_caches(
        gpu_block_size=16,storage_block_size=16,kv_caches=canonical)
    payload=64*layout.storage_block_bytes
    results=[]
    for mode in ("off","readiness"):
        directory=out/mode/"files"
        directory.mkdir(parents=True,exist_ok=False)
        hashes=[hashlib.sha256((mode+str(i)).encode()).digest() for i in range(64)]
        disk=SharedStorageLoadStoreSpec(hashes)
        mapper=FileMapper(root_dir=str(directory),model_name="p311-synthetic",
            gpu_block_size=16,gpu_blocks_per_file=1,tp_size=1,pp_size=1,pcp_size=1,
            rank=0,dtype="bfloat16")
        for t,s in zip(tensors,original):
            t[:64].copy_(s)
        torch.cuda.synchronize()
        for phase in ("store","restore"):
            if phase=="restore":
                for t in tensors:t.zero_()
                torch.cuda.synchronize()
            run_id=mode+"-"+phase
            probe=StoreReadinessProbe(run_id=run_id,interval_ns=100_000) if mode=="readiness" else None
            accounting=StageAccounting() if probe else None
            kwargs=dict(progress_run_id=run_id,observation_sink=probe,
                        stage_accounting=accounting) if probe else {}
            c=TransferCoordinator(config=SharedFileConfig(root_dir=str(directory),iodepth=1,
                staging_mem=16/1024,enable_preload=True,preload_share_staging=True,
                io_backend="linux_aio"),file_mapper=mapper,layout=layout,
                storage_block_tokens=16,**kwargs)
            h=NoopSharedStorageOffloadingHandler(coordinator=c)
            try:
                require(c.reactor.staging_buffer.is_pinned(),"staging is not pinned")
                gpu=GPULoadStoreSpec(list(range(64)),[64],[0])
                pair=(gpu,disk) if phase=="store" else (disk,gpu)
                require(h.transfer_async(1,pair,req_id=run_id),"native transfer rejected")
                f=h._active[1][0]
                h.wait({1})
                require(f.result(timeout=1)==payload,"native parent size mismatch")
                if phase=="restore":
                    torch.cuda.synchronize()
                    require(all(torch.equal(t[:64].cpu(),s) for t,s in zip(tensors,original)),
                            "exact same-source GPU bytes changed")
                else:
                    require(all(Path(mapper.get_file_name(b)).stat().st_size==layout.storage_block_bytes
                                for b in hashes),"native cache publication incomplete")
            finally:
                h.shutdown()
            r=c.reactor
            require(not r._worker.is_alive() and not r._active and not r._inflight
                    and not r._pending_copies,"native state did not drain")
            aio=r.ring.snapshot()
            require(aio["closed"] and aio["drained"] and aio["outstanding"]==0,"AIO not drained")
            require(r.actual_staging_bytes<=r.staging_budget_bytes,"staging budget exceeded")
            observation=probe.export() if probe else None
            stages=accounting.snapshot() if accounting else None
            if probe:
                require(not observation["faulted"] and observation["samples"]>0,"observer failed")
                require(stages["valid"] and stages["outstanding_records"]==0,"accounting incomplete")
                for s in stages["stages"].values():
                    require(s["failed_ops"]==0 and s["accepted_bytes"]==s["transferred_bytes"],
                            "physical stage byte imbalance")
                expected=("d2h","ssd_write") if phase=="store" else ("ssd_read","h2d")
                require(all(stages["stages"][s]["accepted_bytes"]==payload for s in expected),
                        "real physical path not exercised")
                if phase=="store":
                    require(observation["counts"]["queued_stores"]>0
                        and observation["counts"]["mandatory_stores"]>0,
                        "queued mandatory native store not observed")
            results.append(dict(mode=mode,phase=phase,payload_bytes=payload,
                observation=observation,accounting=stages,aio=aio,
                actual_staging_bytes=r.actual_staging_bytes,staging_budget_bytes=r.staging_budget_bytes))
    return dict(status="PASS_REAL_GPU_PASSIVE_READINESS",cases=results,
                real_cuda=True,real_linux_aio=True,exact_restore_both_arms=True,
                gpu_model_workload=False,performance_comparison=False,
                dispatch_budget_installed=False,full_stage_caps_qualified=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    a=p.parse_args()
    out=Path(a.output)
    require(not out.exists(),"append-only evidence output required")
    before=preflight(out,128*1024**2)
    out.mkdir(parents=True,exist_ok=False)
    (out/"preflight.json").write_text(json.dumps(before,indent=2))
    started=time.monotonic()
    try:
        result=qualify(out)
        code=0
    except BaseException as exc:
        result=dict(status="FAIL",error=repr(exc),traceback=traceback.format_exc())
        code=1
    result["elapsed_seconds"]=time.monotonic()-started
    try:result["storage_after"]=preflight(out,0)
    except Exception as exc:
        result.update(status="FAIL",storage_error=str(exc));code=1
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(status=result["status"],output=str(out),exit=code)))
    return code

if __name__=="__main__":
    raise SystemExit(main())
