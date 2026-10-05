"""Real CUDA/Linux-AIO start-budget diagnostic, not a model performance test."""
import argparse,hashlib,json,os,threading,time,traceback
from pathlib import Path
from experiment_storage import ROOT,preflight,permission
from prefix_io_control.start_options import parse_start_options,build_start_kwargs,KEY
RESERVATION=128*1024**2

def require(ok,message):
    if not ok:raise RuntimeError(message)

def wait(h,jid,f):
    errors=[]
    def target():
        try:h.wait({jid})
        except BaseException as exc:errors.append(repr(exc))
    t=threading.Thread(target=target,daemon=True);t.start();t.join(15)
    require(not t.is_alive() and not errors,"native mandatory wait failed or timed out")
    return f.result(timeout=1)

def until(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while not predicate() and time.monotonic()<deadline:time.sleep(.002)
    require(predicate(),"expected native quota event not observed")

def qualify(out):
    import torch
    import py_kvcache.reactor as reactor_module
    from py_kvcache.reactor import TransferCoordinator
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler,SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
    require(Path(reactor_module.__file__).resolve().is_relative_to(ROOT/"third_party/work/py-kvcache-p3-quota-cpu"),"wrong native worktree")
    require(torch.cuda.device_count()==1,"one bound GPU required")
    torch.manual_seed(309)
    tensors=[torch.randint(-128,128,(8,32768),device="cuda",dtype=torch.int8) for _ in range(28)]
    original=[t[:4].cpu().clone() for t in tensors]
    canonical=CanonicalKVCaches([CanonicalKVCacheTensor(t,page_size_bytes=32768) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i,page_size_bytes=32768) for i in range(28)]])
    layout=ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16,storage_block_size=16,kv_caches=canonical)
    payload=4*layout.storage_block_bytes
    results=[];handlers=[]
    def create(mode,phase):
        directory=out/mode/"files"
        directory.mkdir(parents=True,exist_ok=True)
        mapper=FileMapper(root_dir=str(directory),model_name="p309-synthetic",gpu_block_size=16,
            gpu_blocks_per_file=1,tp_size=1,pp_size=1,pcp_size=1,rank=0,dtype="bfloat16")
        raw=dict(mode=mode)
        if mode!="off":raw.update(schema_version=1,run_id=mode+"-"+phase)
        if mode in ("fixed","pressure"):
            raw.update(epoch_ns=1_000_000_000,starts_per_epoch=1,max_wait_ns=10_000_000_000,
                reserve_free_slots=1 if mode=="pressure" else 0)
        kwargs=build_start_kwargs(parse_start_options({KEY:raw}))
        c=TransferCoordinator(config=SharedFileConfig(root_dir=str(directory),iodepth=4,
            staging_mem=32/1024,enable_preload=True,preload_share_staging=True,io_backend="linux_aio"),
            file_mapper=mapper,layout=layout,storage_block_tokens=16,**kwargs)
        h=NoopSharedStorageOffloadingHandler(coordinator=c);handlers.append(h)
        require(c.reactor.staging_buffer.is_pinned(),"unpinned staging")
        return h,mapper,raw
    def transfer(h,jid,disk,store=False,offset=0):
        gpu=GPULoadStoreSpec(list(range(offset,offset+4)),[4],[0])
        require(h.transfer_async(jid,(gpu,disk) if store else (disk,gpu),req_id="p309-"+str(jid)),"transfer rejected")
        return h._active[jid][0]
    def close(h,mode,phase,config):
        h.shutdown();r=h.coordinator.reactor
        require(not r._worker.is_alive() and not r._active and not r._inflight and not r._pending_copies,"native resources not drained")
        io=r.ring.snapshot()
        require(io["closed"] and io["drained"] and io["outstanding"]==0,"AIO not drained")
        a=r._prefix_stage_accounting.snapshot() if r._prefix_stage_accounting else None
        b=r._prefix_start_budget.snapshot() if r._prefix_start_budget else None
        if a:
            require(a["valid"] and a["outstanding_records"]==0,"invalid or incomplete accounting")
            require(all(v["inflight_bytes"]==0 and v["failed_ops"]==0 and v["accepted_bytes"]==v["transferred_bytes"] for v in a["stages"].values()),"stage byte imbalance")
        if b:require(not b["faulted"],"start allowance fell back")
        result=dict(mode=mode,phase=phase,config=config,accounting=a,budget=b,aio=io,
            staging_bytes=r.actual_staging_bytes,staging_budget_bytes=r.staging_budget_bytes)
        results.append(result);return result
    try:
        for mode in ("off","shadow","fixed","pressure"):
            for t,s in zip(tensors,original):t[:4].copy_(s)
            torch.cuda.synchronize()
            hashes=[hashlib.sha256((mode+"-"+str(i)).encode()).digest() for i in range(4)]
            disk=SharedStorageLoadStoreSpec(hashes)
            h,mapper,cfg=create(mode,"store")
            f=transfer(h,1,disk,True)
            if mode in ("fixed","pressure"):
                until(lambda:h.coordinator.reactor._prefix_start_budget.denied_quota>0 or f.done())
                require(not f.done(),"ordinary quota exhaustion was not exercised")
            require(wait(h,1,f)==payload,"store bytes differ")
            r=close(h,mode,"store",cfg)
            if r["budget"]:require(r["budget"]["reasons"]["mandatory"]>0,"mandatory bypass unexercised")
            require(all(Path(mapper.get_file_name(b)).stat().st_size==layout.storage_block_bytes for b in hashes),"published file size differs")
            # Fresh coordinator: restore must read actual disk, not retained CPU cache.
            for t in tensors:t[:4].zero_()
            torch.cuda.synchronize()
            h,_,cfg=create(mode,"load")
            f=transfer(h,2,disk)
            require(wait(h,2,f)==payload,"load bytes differ")
            torch.cuda.synchronize()
            require(all(torch.equal(t[:4].cpu(),s) for t,s in zip(tensors,original)),"real GPU roundtrip differs")
            r=close(h,mode,"load",cfg)
            if r["accounting"]:
                require(r["accounting"]["stages"]["ssd_read"]["accepted_bytes"]==payload,"disk read bypassed")
                require(r["accounting"]["stages"]["h2d"]["accepted_bytes"]==payload,"H2D bytes differ")
            # Two shared demands into distinct GPU destinations retain native preload/fusion.
            for t in tensors:t.zero_()
            torch.cuda.synchronize()
            h,_,cfg=create(mode,"shared")
            require(h.preload_async("A",disk,req_id="A"),"preload A rejected")
            require(h.preload_async("B",disk,req_id="B"),"preload B rejected")
            for jid,name,offset in ((3,"A",0),(4,"B",4)):
                gpu=GPULoadStoreSpec(list(range(offset,offset+4)),[4],[0])
                require(h.load_from_preload_async(jid,name,disk,gpu,req_id=name),"shared load rejected")
            f=h._active[3][0];g=h._active[4][0]
            require(wait(h,3,f)==wait(h,4,g)==payload,"shared restore size differs")
            torch.cuda.synchronize()
            require(all(torch.equal(t[:4].cpu(),s) and torch.equal(t[4:].cpu(),s) for t,s in zip(tensors,original)),"shared GPU destinations differ")
            r=close(h,mode,"shared",cfg)
            if r["accounting"]:
                require(r["accounting"]["stages"]["ssd_read"]["accepted_bytes"]==payload,"shared read duplicated")
                require(r["accounting"]["stages"]["h2d"]["accepted_bytes"]==2*payload,"shared physical copy bytes miscounted")
        h,_,cfg=create("pressure","shutdown")
        disk=SharedStorageLoadStoreSpec([hashlib.sha256(("shutdown"+str(i)).encode()).digest() for i in range(4)])
        f=transfer(h,5,disk,True)
        r=close(h,"pressure","shutdown",cfg)
        require(f.done() and f.result()==payload,"shutdown incomplete")
        require(r["budget"]["reasons"]["shutdown"]>0,"shutdown progress bypass not exercised")
        return dict(status="PASS_REAL_GPU_START_BUDGET_AND_STAGE_ACCOUNTING",cases=results,
            gpu_backend="real CUDA",io_backend="real Linux AIO",native_worktree=reactor_module.__file__,
            synthetic_kv=True,model_loaded=False,model_scheduler_qualified=False,performance_claim=False,
            full_per_stage_caps=False,payload_per_parent=payload,gpu_byte_exact=True,
            original_sha256=hashlib.sha256(b"".join(s.numpy().tobytes() for s in original)).hexdigest())
    finally:
        for h in handlers:h.shutdown()
        torch.cuda.synchronize()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);ap.add_argument("--dry-run",action="store_true")
    a=ap.parse_args();out=a.output.resolve()
    require(not out.exists(),"output must be fresh")
    before=preflight(out,RESERVATION)
    require(out.is_relative_to(Path("/root/prefix-io-v1-validation/runs")),"dedicated auxiliary run required")
    p=permission()
    require(p["allow_gpu_runs"] and os.environ.get("CUDA_VISIBLE_DEVICES") in p["approved_gpu_ids"],"GPU UUID authorization missing")
    plan=dict(output=str(out),storage=before,modes=["off","shadow","fixed","pressure"],
        total_phases=13,reserved_bytes=RESERVATION,model_download_bytes=0,performance_claim=False)
    if a.dry_run:print(json.dumps(dict(plan,executed=False),indent=2));return 0
    out.mkdir(parents=True,exist_ok=False);(out/"plan.json").write_text(json.dumps(plan,indent=2))
    try:result=qualify(out)
    except BaseException as exc:result=dict(status="FAILED",error=repr(exc),traceback=traceback.format_exc(),performance_claim=False)
    result["storage_after"]=preflight(out,0)
    (out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result["status"].startswith("PASS_") else 1
if __name__=="__main__":raise SystemExit(main())
