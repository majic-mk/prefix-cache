"""P3 real native four-stage control qualification; no model efficacy claim."""
import argparse, hashlib, json, time, traceback
from pathlib import Path
from experiment_storage import ROOT, preflight
from prefix_io_control.dispatch_budget import STAGES, Amount
from prefix_io_control.stage_accounting import StageAccounting
from prefix_io_control.simple_stage_policy import DispatchController, SimpleStageConfig

def require(ok, message):
    if not ok: raise RuntimeError(message)

def qualify(out):
    import torch
    import py_kvcache.reactor as native
    from py_kvcache.reactor import TransferCoordinator
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler, SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from vllm.v1.kv_offload.base import CanonicalKVCaches, CanonicalKVCacheTensor, CanonicalKVCacheRef, GPULoadStoreSpec
    require(Path(native.__file__).resolve().is_relative_to(ROOT/"third_party/work/py-kvcache-p3-15-cpu"), "wrong qualified worktree")
    require(torch.cuda.device_count() == 1, "single approved GPU")
    torch.manual_seed(315)
    tensors = [torch.randint(-128, 128, (132, 32768), device="cuda", dtype=torch.int8) for _ in range(4)]
    original = [t[:66].cpu().clone() for t in tensors]
    caches = CanonicalKVCaches([CanonicalKVCacheTensor(t, page_size_bytes=32768) for t in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=32768) for i in range(4)]])
    layout = ParsedKvLayout.from_canonical_kv_caches(gpu_block_size=16, storage_block_size=16, kv_caches=caches)
    cases = []; handlers = []
    def create(mode, phase):
        path = out/mode/"files"; path.mkdir(parents=True, exist_ok=True)
        mapper = FileMapper(root_dir=str(path), model_name="p315-native", gpu_block_size=16,
            gpu_blocks_per_file=1, tp_size=1, pp_size=1, pcp_size=1, rank=0, dtype="bfloat16")
        controller = None
        if mode != "off":
            zero = tuple(Amount() for _ in STAGES)
            cfg = SimpleStageConfig(mode=mode, epoch_ns=10_000_000, byte_quantum=layout.storage_block_bytes,
                cumulative=zero, inflight=zero, shared_ssd_cumulative=Amount(), shared_ssd_inflight=Amount(),
                shared_copy_cumulative_bytes=0, shared_copy_inflight_bytes=0, reserve_staging_bytes=0,
                max_accepted_parents=2, sample_max_age_ns=20_000_000, max_wait_ns=200_000_000)
            controller = DispatchController(mode+"-"+phase, cfg)
        coordinator = TransferCoordinator(config=SharedFileConfig(root_dir=str(path), iodepth=1, staging_mem=16/1024,
            enable_preload=True, preload_share_staging=True, io_backend="linux_aio"),
            file_mapper=mapper, layout=layout, storage_block_tokens=16,
            progress_run_id=mode+"-"+phase, max_accepted_parents=2,
            dispatch_controller=controller, stage_accounting=StageAccounting())
        handler = NoopSharedStorageOffloadingHandler(coordinator=coordinator); handlers.append(handler)
        require(coordinator.reactor.staging_buffer.is_pinned(), "unpinned staging")
        return handler, mapper
    def close(h, mode, phase):
        h.shutdown(); r = h.coordinator.reactor
        require(not r._worker.is_alive() and not r._active and not r._inflight and not r._pending_copies, "native owners not drained")
        aio = r.ring.snapshot(); a = r._prefix_stage_accounting.snapshot()
        require(aio["closed"] and aio["drained"] and aio["outstanding"] == 0, "AIO drain missing")
        require(a["valid"] and a["outstanding_records"] == 0, "stage records not drained")
        require(all(x["failed_ops"] == 0 and x["inflight_bytes"] == 0 and x["accepted_bytes"] == x["transferred_bytes"] for x in a["stages"].values()), "physical stage imbalance")
        c = r._prefix_dispatch_controller
        policy = c.snapshot(native_shutdown=True) if c else None
        require(policy is None or not policy["faulted"], "controller faulted")
        if policy is not None:
            require(not policy["pending_attempt"] and policy["uncertain_ops"] == 0 and policy["completion_unknown_ops"] == 0,
                "unknown or pending control acceptance")
            for stage in STAGES:
                actual = a["stages"][stage]; observed = policy["observed_api_accepted"][stage]
                require(observed["ops"] == actual["accepted_ops"] and observed["bytes"] == actual["accepted_bytes"],
                    "native four-stage API acceptance did not reach controller")
            if mode in ("fixed","pressure"):
                require(policy["performance_override_ops"] > 0, "zero allowance had no explicit progress override")
        admission = r.parent_admission_snapshot()
        require(admission["accepted_parents"] == 0 and admission["peak_accepted_parents"] <= 2, "common parent bound or retirement wrong")
        require(r.actual_staging_bytes <= r.staging_budget_bytes, "staging over budget")
        record = dict(mode=mode, phase=phase, accounting=a, controller=policy, admission=admission,
            aio=aio, actual_staging_bytes=r.actual_staging_bytes, staging_budget_bytes=r.staging_budget_bytes)
        cases.append(record); return record
    try:
        for mode in ("off", "shadow", "fixed", "pressure"):
            for t, s in zip(tensors, original): t[:66].copy_(s)
            torch.cuda.synchronize()
            hashes = [hashlib.sha256((mode+"-"+str(i)).encode()).digest() for i in range(66)]
            h, mapper = create(mode, "store")
            for jid, first, count in ((1,0,64), (2,64,1), (3,65,1)):
                disk = SharedStorageLoadStoreSpec(hashes[first:first+count])
                require(h.transfer_async(jid, (GPULoadStoreSpec(list(range(first,first+count)), [count], [0]), disk), req_id=str(jid)), "native store rejected")
            futures = [h._active[j][0] for j in (1,2,3)]
            h.wait({1,2,3})
            require([f.result() for f in futures] == [64*layout.storage_block_bytes, layout.storage_block_bytes, layout.storage_block_bytes], "store parent bytes wrong")
            close(h, mode, "store")
            h, _ = create(mode, "age_store")
            age_hash = hashlib.sha256((mode+"-age").encode()).digest()
            require(h.transfer_async(6, (GPULoadStoreSpec([0], [1], [0]),
                SharedStorageLoadStoreSpec([age_hash])), req_id="age"), "age store rejected")
            age_future = h._active[6][0]
            require(age_future.result(timeout=5) == layout.storage_block_bytes,
                "zero ordinary allowance did not progress without worker wait/grant")
            age_record = close(h, mode, "age_store")
            require(all(Path(mapper.get_file_name(x)).stat().st_size == layout.storage_block_bytes for x in hashes), "files not published")
            for phase in ("restore", "shared"):
                for t in tensors: t.zero_()
                torch.cuda.synchronize()
                h, _ = create(mode, phase); disk = SharedStorageLoadStoreSpec(hashes)
                if phase == "restore":
                    require(h.transfer_async(4, (disk, GPULoadStoreSpec(list(range(66)), [66], [0])), req_id=phase), "load rejected")
                    futures = [h._active[4][0]]; h.wait({4})
                else:
                    for name in ("A","B"): require(h.preload_async(name, disk, req_id=name), "preload rejected")
                    for jid, name, offset in ((4,"A",0),(5,"B",66)):
                        require(h.load_from_preload_async(jid, name, disk, GPULoadStoreSpec(list(range(offset,offset+66)), [66], [0]), req_id=name), "shared consumer rejected")
                    futures = [h._active[j][0] for j in (4,5)]; h.wait({4,5})
                require(all(f.result() == 66*layout.storage_block_bytes for f in futures), "load parent bytes wrong")
                torch.cuda.synchronize()
                require(all(torch.equal(t[:66].cpu(), s) for t,s in zip(tensors,original)), "restore content mismatch")
                if phase == "shared": require(all(torch.equal(t[66:].cpu(),s) for t,s in zip(tensors,original)), "shared destination mismatch")
                record = close(h, mode, phase)
                require(record["accounting"]["stages"]["ssd_read"]["accepted_bytes"] == 66*layout.storage_block_bytes, "shared disk read duplicated")
                require(record["accounting"]["stages"]["h2d"]["accepted_bytes"] == (2 if phase == "shared" else 1)*66*layout.storage_block_bytes, "H2D actual bytes wrong")
        return dict(status="PASS_REAL_GPU_SIMPLE_STAGE", cases=cases, exact_content=True, common_admission=True,
            full_stage_caps_installed=True, ordinary_zero_allowance=True, progress_may_exceed_performance_allowance=True,
            physical_capacity_hard=True, new_executor=False, artificial_io_delay=False, model_performance_claim=False)
    finally:
        for h in handlers:
            if h.coordinator.reactor._worker.is_alive(): h.shutdown()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    require(not a.output.exists(), "append-only evidence")
    contract=preflight(a.output,128*1024**2); a.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    try: result=qualify(a.output); code=0
    except BaseException as exc: result=dict(status="FAIL",error=repr(exc),traceback=traceback.format_exc()); code=1
    result.update(seconds=time.monotonic()-started, storage_preflight=contract, storage_after=preflight(a.output,0))
    with (a.output/"result.json").open("x") as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result["status"],output=str(a.output)))); return code
if __name__=="__main__": raise SystemExit(main())
