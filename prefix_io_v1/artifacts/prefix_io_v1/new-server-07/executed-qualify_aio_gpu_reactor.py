"""Bounded real author handler/reactor + AIO + CUDA roundtrip qualification."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import threading
import traceback

def check(condition,message):
    if not condition:raise RuntimeError(message)

def qualify(tensors, output, *, factor=1, production=False):
    import torch
    from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler,SharedStorageLoadStoreSpec
    from py_kvcache.transfer import ParsedKvLayout
    from py_kvcache.reactor import TransferCoordinator
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    root=Path(output).resolve()
    approved=Path(__file__).resolve().parents[3]/"experiments/prefix_io_v1/runs"
    check(root.is_relative_to(approved),"output outside run directory")
    root.mkdir(parents=True,exist_ok=False)
    check(len(tensors)==28 and all(t.device.type=="cuda" and t.dtype==torch.int8 and
          t.ndim==2 and t.shape[1]==32768 and t.is_contiguous() for t in tensors),"unexpected KV byte layout")
    n=int(tensors[0].shape[0]);check(all(t.shape[0]==n for t in tensors),"block count mismatch")
    count=n if production else 2*factor
    check(count%factor==0 and count<=n,"invalid block count")
    indices=list(range(count))
    canonical=CanonicalKVCaches(
        [CanonicalKVCacheTensor(tensor=t,page_size_bytes=t.shape[1]) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i,page_size_bytes=t.shape[1]) for i,t in enumerate(tensors)]])
    layout=ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16,storage_block_size=16*factor,kv_caches=canonical)
    check(layout.pin_memory,"pinned staging unavailable")
    # Ordinary CPU snapshots are evidence, not a second caching/staging pool.
    original=[t[:count].cpu().clone() for t in tensors]
    expected_digest=hashlib.sha256(b"".join(t.numpy().tobytes() for t in original)).hexdigest()
    hashes=[hashlib.sha256(("diagnostic-%d"%i).encode()).digest() for i in range(count//factor)]
    mapper=FileMapper(root_dir=str(root/"files"),model_name="diagnostic-production-kv" if production else "diagnostic-synthetic-kv",
        gpu_block_size=16,gpu_blocks_per_file=factor,tp_size=1,pp_size=1,pcp_size=1,rank=0,dtype="bfloat16")
    config=SharedFileConfig(root_dir=str(root/"files"),iodepth=4,staging_mem=32/1024,
        enable_preload=True,preload_share_staging=True,io_backend="linux_aio")
    coordinator=TransferCoordinator(config=config,file_mapper=mapper,layout=layout,storage_block_tokens=16*factor)
    handler=NoopSharedStorageOffloadingHandler(coordinator=coordinator)
    reactor=coordinator.reactor
    check(reactor.staging_buffer.is_pinned(),"staging not actually pinned")
    gpu_spec=GPULoadStoreSpec(indices,[len(indices)],[0])
    disk_spec=SharedStorageLoadStoreSpec(hashes)
    report=dict(scope="production model KV snapshot via real handler, separate diagnostic files" if production else "synthetic GPU byte layout via real handler",
        production_kv=production,block_count=count,file_count=len(hashes),factor=factor,
        bytes_per_file=layout.storage_block_bytes,total_roundtrip_payload_bytes=count*32768*28,
        pinned_staging_bytes=reactor.actual_staging_bytes,staging_budget_bytes=reactor.staging_budget_bytes,
        byte_exact=False,default_runtime_changed=False,research_policy_enabled=False,model_scheduler_connector=False)
    def wait_job(job_id):
        future=handler._active[job_id][0]
        future.result(timeout=20)
        handler.wait({job_id})  # Original mandatory waiting API, no scheduler intervention.
        results=handler.get_finished()
        check(len(results)==1 and results[0].job_id==job_id and results[0].success,"handler failed transfer")
        return future
    def match():
        torch.cuda.synchronize()
        check(all(torch.equal(t[:count].cpu(),saved) for t,saved in zip(tensors,original)),"GPU KV bytes differ")
    try:
        check(handler.transfer_async(1,(gpu_spec,disk_spec),req_id="store-source"),"store not accepted")
        wait_job(1)
        check(all(Path(mapper.get_file_name(h)).stat().st_size==layout.storage_block_bytes for h in hashes),"publication missing")
        for t in tensors:t[:count].zero_()
        torch.cuda.synchronize()
        check(handler.transfer_async(2,(disk_spec,gpu_spec),req_id="foreground"),"load not accepted")
        wait_job(2);match()
        # Two demands share the same preload hash set, then both are consumed.
        if not production:
            check(handler.preload_async("preload-A",disk_spec,req_id="A"),"preload A rejected")
            check(handler.preload_async("preload-B",disk_spec,req_id="B"),"preload B rejected")
            for job_id,req in [(3,"A"),(4,"B")]:
                for t in tensors:t[:count].zero_()
                torch.cuda.synchronize()
                check(handler.load_from_preload_async(job_id,"preload-"+req,disk_spec,gpu_spec,req_id=req),"preload load rejected")
                wait_job(job_id);match()
            report["two_shared_preload_demands_consumed"]=True
        # Queue accepted work and immediately use native shutdown to force its drain.
        final_hashes=[hashlib.sha256(b"drain-"+h).digest() for h in hashes]
        check(handler.transfer_async(5,(gpu_spec,SharedStorageLoadStoreSpec(final_hashes)),req_id="shutdown"),"final store rejected")
        pending=handler._active[5][0]
        handler.shutdown()
        check(pending.done() and pending.result()==len(final_hashes)*layout.storage_block_bytes,"shutdown left accepted store incomplete")
        check(not reactor._worker.is_alive(),"reactor worker remains alive")
        snap=reactor.ring.snapshot()
        check(snap["closed"] and snap["drained"] and snap["outstanding"]==0,"AIO did not fully drain")
        check(not reactor._inflight and not reactor._pending_copies and not reactor._active,"native in-flight state remains")
        match()
        check(not list(root.rglob("*.tmp-*")),"temporary files remain")
        check(not [t for t in threading.enumerate() if t.name.startswith("pykvcache-aio-")],"AIO worker remains")
        report.update(byte_exact=True,sha256=expected_digest,native_wait_passed=True,native_shutdown_drained=True,
                      aio_snapshot=snap,staging_pinned=True,published_files=len(list(root.rglob("*.bin"))))
    finally:
        # No tensor/view/snapshot is released while accepted I/O/copies can still use it.
        handler.shutdown()
        torch.cuda.synchronize()
        if production:
            # Preserve engine-owned storage exactly even on a failed diagnostic.
            for t,saved in zip(tensors,original):t[:count].copy_(saved)
            torch.cuda.synchronize()
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",required=True);a=ap.parse_args()
    output=Path(a.output);output.mkdir(exist_ok=False)
    report={"status":"FAILED","model_loaded":False,"performance_claim":False,"cases":[]}
    try:
        check(os.environ.get("CUDA_VISIBLE_DEVICES","").startswith("GPU-"),"UUID binding required")
        import torch
        check(torch.cuda.device_count()==1,"one GPU required")
        free,_=torch.cuda.mem_get_info()
        check(free>1024**3,"insufficient memory")
        torch.manual_seed(7)
        tensors=[torch.randint(-128,128,(32,32768),device="cuda",dtype=torch.int8) for _ in range(28)]
        for factor in [1,4]:
            report["cases"].append(qualify(tensors,output/("factor-"+str(factor)),factor=factor))
        report["status"]="PASSED_GPU_HANDLER_AIO_SYNTHETIC"
    except Exception as e:
        report.update(error=str(e),traceback=traceback.format_exc())
    (output/"result.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
