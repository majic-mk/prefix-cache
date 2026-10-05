"""Real CUDA/Linux-AIO mandatory-order qualification; no artificial device delays."""
import argparse,hashlib,inspect,json,time,traceback
from pathlib import Path
from experiment_storage import ROOT,preflight
from prefix_io_control.order_options import KEY,parse_order_options,build_order_kwargs
from prefix_io_control.stage_accounting import StageAccounting

SOURCES=[
 "src/prefix_io_control/store_order.py","src/prefix_io_control/order_options.py",
 "third_party/work/py-kvcache-p3-order-cpu/py_kvcache/reactor.py",
 "third_party/work/py-kvcache-p3-order-cpu/py_kvcache/vllm.py"]
def require(ok,message):
    if not ok:raise RuntimeError(message)

def qualify(out):
    import torch
    import py_kvcache.reactor as native
    from py_kvcache.reactor import TransferCoordinator
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler,SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
    require(Path(native.__file__).resolve().is_relative_to(ROOT/"third_party/work/py-kvcache-p3-order-cpu"),"wrong order worktree")
    require(torch.cuda.device_count()==1,"single bound GPU required")
    torch.manual_seed(312)
    tensors=[torch.randint(-128,128,(130,32768),device="cuda",dtype=torch.int8) for _ in range(4)]
    original=[t[:65].cpu().clone() for t in tensors]
    caches=CanonicalKVCaches([CanonicalKVCacheTensor(t,page_size_bytes=32768) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i,page_size_bytes=32768) for i in range(4)]])
    layout=ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16,storage_block_size=16,kv_caches=caches)
    results=[];handlers=[]
    def create(mode,phase):
        directory=out/mode/"files";directory.mkdir(parents=True,exist_ok=True)
        mapper=FileMapper(root_dir=str(directory),model_name="p312-synthetic",gpu_block_size=16,
            gpu_blocks_per_file=1,tp_size=1,pp_size=1,pcp_size=1,rank=0,dtype="bfloat16")
        raw={"mode":"off"} if mode=="off" else dict(schema_version=1,mode="pressure",run_id=mode+"-"+phase,candidate_parents=32)
        kw=build_order_kwargs(parse_order_options({KEY:raw}))
        kw.setdefault("progress_run_id",mode+"-"+phase)
        kw["stage_accounting"]=StageAccounting()
        c=TransferCoordinator(config=SharedFileConfig(root_dir=str(directory),iodepth=1,staging_mem=16/1024,
            enable_preload=True,preload_share_staging=True,io_backend="linux_aio"),
            file_mapper=mapper,layout=layout,storage_block_tokens=16,**kw)
        h=NoopSharedStorageOffloadingHandler(coordinator=c);handlers.append(h)
        require(c.reactor.staging_buffer.is_pinned(),"unpinned staging")
        return h,mapper,raw
    def close(h,mode,phase,raw):
        h.shutdown();r=h.coordinator.reactor
        require(not r._worker.is_alive() and not r._active and not r._inflight and not r._pending_copies,"native resources not drained")
        aio=r.ring.snapshot()
        require(aio["closed"] and aio["drained"] and aio["outstanding"]==0,"AIO not drained")
        a=r._prefix_stage_accounting.snapshot()
        require(a["valid"] and a["outstanding_records"]==0,"stage accounting incomplete")
        require(all(v["failed_ops"]==0 and v["accepted_bytes"]==v["transferred_bytes"] and v["inflight_bytes"]==0 for v in a["stages"].values()),"stage byte imbalance")
        order=r._prefix_store_order.snapshot() if r._prefix_store_order else None
        require(order is None or not order["faulted"],"order fallback occurred")
        require(r.actual_staging_bytes<=r.staging_budget_bytes,"staging over budget")
        record=dict(mode=mode,phase=phase,config=raw,accounting=a,order=order,aio=aio,
                    actual_staging_bytes=r.actual_staging_bytes,staging_budget_bytes=r.staging_budget_bytes)
        results.append(record);return record
    try:
        for mode in ("off","pressure"):
            for t,s in zip(tensors,original):t[:65].copy_(s)
            torch.cuda.synchronize()
            hashes=[hashlib.sha256((mode+"-"+str(i)).encode()).digest() for i in range(65)]
            h,mapper,raw=create(mode,"store")
            large=SharedStorageLoadStoreSpec(hashes[:64]);small=SharedStorageLoadStoreSpec(hashes[64:])
            require(h.transfer_async(1,(GPULoadStoreSpec(list(range(64)),[64],[0]),large),req_id="large"),"large store rejected")
            require(h.transfer_async(2,(GPULoadStoreSpec([64],[1],[0]),small),req_id="small"),"small store rejected")
            a=h._active[1][0];b=h._active[2][0];completion=[]
            a.add_done_callback(lambda f:completion.append(1));b.add_done_callback(lambda f:completion.append(2))
            h.wait({2})
            require(b.result()==layout.storage_block_bytes,"small parent wrong bytes")
            if mode=="pressure":
                require(not a.done(),"mandatory small parent was not prioritized before large completion")
            h.wait({1});require(a.result()==64*layout.storage_block_bytes,"large parent wrong bytes")
            r=close(h,mode,"store",raw);r["parent_completion_order"]=completion
            require(completion==([1,2] if mode=="off" else [2,1]),"unexpected native completion order")
            if mode=="pressure":require(r["order"]["reordered_passes"]>0,"order branch unexercised")
            require(all(Path(mapper.get_file_name(b)).stat().st_size==layout.storage_block_bytes for b in hashes),"native files unpublished")
            for phase in ("restore","shared"):
                for t in tensors:t.zero_()
                torch.cuda.synchronize()
                h,_,raw=create(mode,phase);disk=SharedStorageLoadStoreSpec(hashes)
                if phase=="restore":
                    require(h.transfer_async(3,(disk,GPULoadStoreSpec(list(range(65)),[65],[0])),req_id="restore"),"restore rejected")
                    f=h._active[3][0];h.wait({3});require(f.result()==65*layout.storage_block_bytes,"restore parent bytes")
                else:
                    for jid,name,offset in [(3,"A",0),(4,"B",65)]:
                        require(h.preload_async(name,disk,req_id=name),"preload rejected")
                    for jid,name,offset in [(3,"A",0),(4,"B",65)]:
                        require(h.load_from_preload_async(jid,name,disk,GPULoadStoreSpec(list(range(offset,offset+65)),[65],[0]),req_id=name),"shared consumer rejected")
                    fs=[h._active[j][0] for j in (3,4)];h.wait({3,4})
                    require(all(f.result()==65*layout.storage_block_bytes for f in fs),"shared parent bytes")
                torch.cuda.synchronize()
                require(all(torch.equal(t[:65].cpu(),s) for t,s in zip(tensors,original)),"exact restore mismatch")
                if phase=="shared":require(all(torch.equal(t[65:].cpu(),s) for t,s in zip(tensors,original)),"shared second destination mismatch")
                r=close(h,mode,phase,raw)
                require(r["accounting"]["stages"]["ssd_read"]["accepted_bytes"]==65*layout.storage_block_bytes,"disk shared read duplication or bypass")
                require(r["accounting"]["stages"]["h2d"]["accepted_bytes"]==(2 if phase=="shared" else 1)*65*layout.storage_block_bytes,"physical H2D byte mismatch")
        return dict(status="PASS_REAL_GPU_STORE_ORDER",gpu_byte_exact=True,cases=results,
            native_module=native.__file__,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
            artificial_device_delay=False,ordinary_start_quota=False,full_stage_caps=False,model_performance_claim=False)
    finally:
        for h in handlers:h.shutdown()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    require(not a.output.exists(),"append-only evidence")
    before=preflight(a.output,128*1024**2);a.output.mkdir(parents=True,exist_ok=False)
    (a.output/"preflight.json").write_text(json.dumps(before,indent=2));started=time.monotonic()
    try:r=qualify(a.output);code=0
    except BaseException as exc:r=dict(status="FAIL",error=repr(exc),traceback=traceback.format_exc());code=1
    r["seconds"]=time.monotonic()-started
    try:r["storage_after"]=preflight(a.output,0)
    except Exception as exc:r.update(status="FAIL",storage_error=str(exc));code=1
    (a.output/"result.json").write_text(json.dumps(r,indent=2))
    print(json.dumps(dict(status=r["status"],output=str(a.output),exit=code)))
    return code
if __name__=="__main__":raise SystemExit(main())
