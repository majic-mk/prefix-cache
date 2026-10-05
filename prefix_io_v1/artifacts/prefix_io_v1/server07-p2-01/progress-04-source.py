"""Real CUDA/AIO mandatory mechanism test; zero credits are test-only injection."""
import argparse,json,threading,time,traceback,hashlib
from pathlib import Path
from dataclasses import asdict
import native_gpu_prefix_smoke as base

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();base.require(out.is_relative_to(base.ROOT/"experiments/prefix_io_v1/runs"),"unsafe output")
    out.mkdir(parents=True,exist_ok=False)
    import torch
    from py_kvcache.reactor import IoReactor,TransferCoordinator
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler,SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
    from prefix_io_control.publication import SnapshotPublisher
    base.require(torch.cuda.device_count()==1,"one bound GPU")
    tensors=[torch.randint(-128,128,(32,32768),dtype=torch.int8,device="cuda") for _ in range(28)]
    canonical=CanonicalKVCaches([CanonicalKVCacheTensor(t,page_size_bytes=32768) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i,page_size_bytes=32768) for i in range(28)]])
    layout=ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16,storage_block_size=16,kv_caches=canonical)
    originals=[t.cpu().clone() for t in tensors]
    blocked=threading.Event();counts=dict(blocked=0,mandatory_dispatches=0)
    native=IoReactor._schedule_one
    native_open=IoReactor._can_open
    def test_only_gate(r,job):
        if r.progress_bridge_enabled:
            if not r.is_mandatory(job.future):
                counts["blocked"]+=1;blocked.set();return False
            counts["mandatory_dispatches"]+=1
        return native(r,job)
    def test_only_open_gate(r):
        # This diagnostic has exactly one accepted load at a time. It does not
        # implement a production multi-request admission or resource owner.
        loads=[j for j in r._active if not j.is_store and j.failed is None]
        if r.progress_bridge_enabled and loads and not any(r.is_mandatory(j.future) for j in loads):
            counts["blocked"]+=1;blocked.set();return False
        return native_open(r)
    IoReactor._schedule_one=test_only_gate
    IoReactor._can_open=test_only_open_gate
    report=dict(status="FAILED",scope="real CUDA/AIO synthetic KV layout, test-only zero-credit gate; not production scheduler witness",
        ordinary_credits=0,policy_installed=False,performance_claim=False,cases=[])
    handlers=[]
    def create(name,observed=True):
        root=out/name;root.mkdir()
        mapper=FileMapper(root_dir=str(root/"files"),model_name="p2-diagnostic",gpu_block_size=16,gpu_blocks_per_file=1,tp_size=1,pp_size=1,pcp_size=1,rank=0,dtype="bfloat16")
        pub=SnapshotPublisher(run_id=name,interval_ns=1_000_000) if observed else None
        stats=dict(snapshots=0,max_d2h_bytes=0,max_h2d_bytes=0,max_active_parents=0,gpu_release_reported=False,failed_draining_seen=False)
        def sink(r):
            before=pub.published_count;pub(r)
            if pub.published_count==before:return
            s=pub._latest
            stats["snapshots"]+=1
            stats["max_d2h_bytes"]=max(stats["max_d2h_bytes"],s.d2h_inflight_bytes or 0)
            stats["max_h2d_bytes"]=max(stats["max_h2d_bytes"],s.h2d_inflight_bytes or 0)
            stats["max_active_parents"]=max(stats["max_active_parents"],s.active_parent_count)
            stats["gpu_release_reported"] |= s.gpu_immediately_reusable_bytes is not None
            stats["failed_draining_seen"] |= any(p.lifecycle_state=="FAILED_DRAINING" for p in s.parents)
        c=TransferCoordinator(config=SharedFileConfig(root_dir=str(root/"files"),staging_mem=32/1024,iodepth=4,
            enable_preload=True,preload_share_staging=True,io_backend="linux_aio"),
            file_mapper=mapper,layout=layout,storage_block_tokens=16,
            progress_run_id=name if observed else None,observation_sink=sink if observed else None)
        h=NoopSharedStorageOffloadingHandler(coordinator=c);handlers.append(h)
        return h,mapper,pub,stats
    def submit(h,jid,count,direction,tag):
        hashes=[hashlib.sha256((tag+str(i)).encode()).digest() for i in range(count)]
        gpu=GPULoadStoreSpec(list(range(count)),[count],[0])
        disk=SharedStorageLoadStoreSpec(hashes)
        base.require(h.transfer_async(jid,(gpu,disk) if direction=="store" else (disk,gpu),req_id=tag),"transfer rejected")
        return h._active[jid][0],hashes
    def wait(h,jid,f):
        errors=[]
        def target():
            try:h.wait({jid})
            except BaseException as e:errors.append(str(e))
        thread=threading.Thread(target=target,daemon=True);thread.start();thread.join(30)
        base.require(not thread.is_alive() and not errors,"mandatory handler wait hung")
        return f.result(timeout=1)
    def close_record(h,stats):
        h.shutdown();r=h.coordinator.reactor;snap=r.ring.snapshot()
        base.require(snap["closed"] and snap["drained"] and snap["outstanding"]==0,"AIO not drained")
        base.require(not r._worker.is_alive() and not r._active and not r._pending_copies and not r._inflight,"native work remains")
        base.require(not getattr(r._prefix_progress,"required",set()),"mandatory refs remain")
        base.require(not stats["gpu_release_reported"],"reactor fabricated GPU release")
        return dict(aio=snap,observer=stats,staging_bytes=r.actual_staging_bytes,staging_budget=r.staging_budget_bytes)
    try:
        # Disabled mode preserves native execution and creates no signal/snapshot state.
        h,mapper,pub,stats=create("off",False)
        f,_=submit(h,1,4,"store","off")
        wait(h,1,f);base.require(h.coordinator.reactor._prefix_progress is None and pub is None,"off state")
        report["cases"].append(dict(name="off_native",**close_record(h,stats)))
        h,mapper,pub,stats=create("mandatory")
        blocked.clear()
        # Prepare CPU expectations before enqueueing the delayed producer.
        # CPU fill thread-pool startup must not consume the pending-event window.
        for i,ref in enumerate(originals):ref[:16].fill_((i*7)%256-128)
        torch.cuda._sleep(2_000_000_000) # Bounded mechanism diagnostic, never performance data.
        for i,t in enumerate(tensors):
            t[:16].fill_((i*7)%256-128)
        producer_done=torch.cuda.Event();producer_done.record()
        f,hashes=submit(h,2,16,"store","roundtrip")
        base.require(blocked.wait(5) and not f.done(),"zero-credit gate was not exercised")
        report["producer_pending_before_wait"]=not producer_done.query()
        base.require(report["producer_pending_before_wait"],"compute dependency not pending at test boundary")
        base.require(wait(h,2,f)==16*917504,"store size")
        h.get_finished()
        for t in tensors:t[:16].zero_()
        torch.cuda.synchronize()
        blocked.clear();f,_=submit(h,3,16,"load","roundtrip")
        base.require(blocked.wait(5) and not f.done(),"load gate not exercised")
        base.require(wait(h,3,f)==16*917504,"load size")
        base.require(all(torch.equal(t[:16].cpu(),ref[:16]) for t,ref in zip(tensors,originals)),"real byte roundtrip differs")
        h.get_finished()
        report["cases"].append(dict(name="mandatory_store_load",byte_exact=True,**close_record(h,stats)))
        h,mapper,pub,stats=create("shutdown")
        f,_=submit(h,4,16,"store","parent-A");g,_=submit(h,5,16,"store","parent-B")
        blocked.clear()
        h.shutdown() # STOP must cover all accepted parents without handler.wait or epochs.
        base.require(f.done() and g.done() and f.result()==g.result()==16*917504,"shutdown missed a parent")
        report["cases"].append(dict(name="shutdown_two_parents",**close_record(h,stats)))
        # Actual short file read: failure may resolve before drain; never count it as release.
        h,mapper,pub,stats=create("short_read")
        f,hashes=submit(h,6,8,"store","bad")
        wait(h,6,f);h.get_finished()
        with open(mapper.get_file_name(hashes[0]),"r+b") as stream:stream.truncate(4096)
        f,_=submit(h,7,8,"load","bad")
        failed=False
        try:wait(h,7,f)
        except (IOError,OSError):failed=True
        base.require(failed,"real short read not rejected")
        report["cases"].append(dict(name="short_read_failure_drain",failure_not_release=True,**close_record(h,stats)))
        base.require(counts["blocked"]>0 and counts["mandatory_dispatches"]>0,"mandatory gate not proven")
        report["status"]="PASSED_REAL_GPU_AIO_MANDATORY_DIAGNOSTIC"
    except BaseException as e:report.update(error=str(e),traceback=traceback.format_exc())
    finally:
        for h in handlers:h.shutdown()
        IoReactor._schedule_one=native
        IoReactor._can_open=native_open
        torch.cuda.synchronize()
        report["test_gate_counts"]=counts
        (out/"result.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return 0 if report["status"].startswith("PASSED") else 1
if __name__=="__main__":raise SystemExit(main())
