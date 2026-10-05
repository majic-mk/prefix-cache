"""P316 real owned-GPU capacity/STOP qualification; run only under GPU guard.

No model or performance experiment. Original native stores create all tiny
PRIMARY sources. The event observer always calls a real CUDA Event.query and
can only add False while a host gate is held; it never manufactures completion.
This is synthetic host pending, not an artificial GPU/DMA delay.
"""
import argparse
import gc
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

from experiment_storage import ROOT, authorized_path, permission, preflight
from prefix_io_control.dispatch_budget import Amount, STAGES
from prefix_io_control.simple_stage_policy import DispatchController, SimpleStageConfig
from prefix_io_control.stage_accounting import StageAccounting

TIME_ENVELOPE_SECONDS = 240
STAGING_BYTES = 16 * 1024**2
PRIMARY_RESERVATION_BYTES = 128 * 1024**2
SOURCE_FILES = 80
SOURCES = (
    "experiments/prefix_io_v1/scripts/qualify_compact_staging_gpu_p316.py",
    "src/prefix_io_control/simple_stage_policy.py",
    "src/prefix_io_control/stage_accounting.py",
    "src/prefix_io_control/dispatch_budget.py",
    "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py",
    "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging_cache.py",
    "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/staging.py",
    "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/vllm.py",
    "third_party/work/py-kvcache-p3-16-cpu/py_kvcache/linux_aio.py",
    "third_party/work/vllm-author-build/vllm/_custom_ops.py",
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_snapshot():
    return {name: sha_file(ROOT / name) for name in SOURCES}


def normalize_gpu_uuid(value):
    if isinstance(value, (bytes, bytearray)) and len(value) == 16:
        return "GPU-" + str(uuid.UUID(bytes=bytes(value)))
    text = str(value)
    return "GPU-" + str(uuid.UUID(text[4:] if text.startswith("GPU-") else text))


def qualify(out, started, run_state):
    import torch
    import py_kvcache.reactor as native
    from py_kvcache.reactor import TransferCoordinator, NativeDrainUnknown
    from py_kvcache.vllm import NoopSharedStorageOffloadingHandler, SharedStorageLoadStoreSpec
    from py_kvcache.fs_config import SharedFileConfig
    from py_kvcache.file_mapper import FileMapper
    from py_kvcache.transfer import ParsedKvLayout
    from vllm.v1.kv_offload.base import (
        CanonicalKVCaches, CanonicalKVCacheTensor, CanonicalKVCacheRef, GPULoadStoreSpec,
    )

    require(Path(native.__file__).resolve().is_relative_to(
        ROOT / "third_party/work/py-kvcache-p3-16-cpu"), "wrong isolated native worktree")
    require(Path(native.ops.__file__).resolve().is_relative_to(
        ROOT / "third_party/work/vllm-author-build"), "wrong real author operations")
    require(torch.cuda.device_count() == 1, "exactly one approved GPU required")
    raw_uuid = torch.cuda.get_device_properties(0).uuid
    gpu_uuid = normalize_gpu_uuid(raw_uuid)
    permit = permission()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    approved = permit["approved_gpu_ids"]
    require(permit["allow_gpu_runs"] and len(approved) == 1 and "," not in visible and
            normalize_gpu_uuid(visible) == normalize_gpu_uuid(approved[0]) == gpu_uuid,
            "GPU UUID not uniquely and explicitly approved")
    before_sources = source_snapshot()
    write_new(out / "sources-before.json", before_sources)
    write_new(out / "environment.json", dict(
        gpu_uuid=gpu_uuid, gpu_uuid_raw=str(raw_uuid), cuda_visible_devices=visible,
        device_name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda, guarded_run_required=True,
        time_envelope_seconds=TIME_ENVELOPE_SECONDS, model_loaded=False,
        native_module=str(Path(native.__file__).resolve()),
        synthetic_host_pending=True, artificial_dma_delay=False,
        model_performance_claim=False, production_candidates_changed=False,
    ))
    torch.manual_seed(31608)
    tensors = [torch.randint(-128, 128, (132, 32768), device="cuda", dtype=torch.int8)
               for _ in range(4)]
    original = [tensor[:SOURCE_FILES].cpu().clone() for tensor in tensors]
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(tensor, page_size_bytes=32768) for tensor in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=32768) for i in range(4)]],
    )
    layout = ParsedKvLayout.from_canonical_kv_caches(
        gpu_block_size=16, storage_block_size=16, kv_caches=caches)
    require(layout.storage_block_bytes == 131072, "unexpected native test layout")
    active = []
    cases = []
    original_native_torch = native.torch
    original_event = torch.cuda.Event

    def remaining(seconds=30):
        require(time.monotonic() - started < TIME_ENVELOPE_SECONDS - seconds,
                "cannot start another phase within frozen process envelope")

    def create(directory, label, kind=None):
        directory.mkdir(parents=True, exist_ok=True)
        mapper = FileMapper(root_dir=str(directory), model_name="p316-owned-capacity",
            gpu_block_size=16, gpu_blocks_per_file=1, tp_size=1, pp_size=1,
            pcp_size=1, rank=0, dtype="bfloat16")
        controller = None
        if kind is not None:
            quantum = layout.storage_block_bytes
            caps = tuple(Amount(4096, 4096 * quantum) for _ in STAGES)
            controller = DispatchController(label, SimpleStageConfig(
                mode="pressure", epoch_ns=10_000_000, byte_quantum=quantum,
                cumulative=caps, inflight=caps,
                shared_ssd_cumulative=Amount(8192, 8192 * quantum),
                shared_ssd_inflight=Amount(8192, 8192 * quantum),
                shared_copy_cumulative_bytes=8192 * quantum,
                shared_copy_inflight_bytes=8192 * quantum,
                reserve_staging_bytes=127 * quantum, max_accepted_parents=2,
                sample_max_age_ns=20_000_000, max_wait_ns=200_000_000))
        c = TransferCoordinator(
            config=SharedFileConfig(
                root_dir=str(directory), iodepth=4, open_lookahead=4,
                staging_mem=16 / 1024, enable_preload=kind == "shared",
                preload_share_staging=kind == "shared",
                staging_cache="lru" if kind == "cache" else "off",
                io_backend="linux_aio"),
            file_mapper=mapper, layout=layout, storage_block_tokens=16,
            progress_run_id=label, max_accepted_parents=2,
            dispatch_controller=controller, stage_accounting=StageAccounting())
        h = NoopSharedStorageOffloadingHandler(coordinator=c)
        active.append(h)
        r = c.reactor
        require(r.staging_buffer.is_pinned(), "staging is not real pinned memory")
        require(r.staging_pool.slot_count == 127 and
                r.staging_budget_bytes == STAGING_BYTES and
                r.actual_staging_bytes <= STAGING_BYTES, "wrong real 127-slot staging budget")
        return h, mapper

    def close(h):
        h.shutdown()
        r = h.coordinator.reactor
        require(not r._worker.is_alive() and not r._active and not r._inflight and
                not r._pending_copies and not r._copy_ready and
                not r._ready_fds_load and not r._ready_fds_preload,
                "original native owners remain after full shutdown")
        aio = r.ring.snapshot()
        require(aio["closed"] and aio["drained"] and aio["outstanding"] == 0 and
                aio["accepted"] == aio["reaped"] and
                aio["pending"] == aio["ready"] == aio["unreaped"] == 0,
                "actual Linux AIO drain/reap not proved")
        admission = r.parent_admission_snapshot()
        require(admission["count_valid"] and admission["accepted_parents"] == 0 and
                not admission["native_drain_unknown"], "parent physical protection not retired")
        accounting = r._prefix_stage_accounting.snapshot()
        require(accounting["valid"] and accounting["outstanding_records"] == 0 and
                all(row["failed_ops"] == 0 and row["inflight_ops"] == 0 and
                    row["inflight_bytes"] == 0 and
                    row["accepted_bytes"] == row["transferred_bytes"]
                    for row in accounting["stages"].values()),
                "actual native stage drain/content accounting incomplete")
        controller = r._prefix_dispatch_controller
        policy = controller.snapshot(native_shutdown=True) if controller else None
        if policy is not None:
            require(not policy["faulted"] and not policy["pending_attempt"] and
                    policy["uncertain_ops"] == policy["completion_unknown_ops"] == 0,
                    "unknown controller acceptance/completion")
            for stage in STAGES:
                actual = accounting["stages"][stage]
                observed = policy["observed_api_accepted"][stage]
                require(observed["ops"] == actual["accepted_ops"] and
                        observed["bytes"] == actual["accepted_bytes"],
                        "accepted native API not charged exactly")
        capacity = r._prefix_capacity_snapshot(native_shutdown=True)
        require(capacity["valid"] is False and
                capacity["free_reclaimable_staging_bytes"] is None and
                capacity["physical_release_credit"] is False,
                "shutdown observation manufactured resource release")
        active.remove(h)
        return dict(aio=aio, admission=admission, accounting=accounting,
                    controller=policy, staging_capacity=capacity,
                    actual_staging_bytes=r.actual_staging_bytes,
                    full_original_native_shutdown=True,
                    resource_release_inferred_from_accounting=False)

    def wait_snapshot(r, predicate, seconds=60):
        deadline = min(started + TIME_ENVELOPE_SECONDS - 20, time.monotonic() + seconds)
        last = None
        polls = 0
        while time.monotonic() < deadline:
            last = r.inspect_snapshot(timeout=5)
            polls += 1
            if predicate(last):
                return last, polls
            time.sleep(.005)
        raise RuntimeError("bounded actual owner snapshot predicate not reached: " + repr(last))

    class HeldEvent:
        def __init__(self, real, gate, counts):
            self.real = real
            self.gate = gate
            self.counts = counts
        def record(self, stream=None):
            return self.real.record(stream)
        def query(self):
            actual = self.real.query()
            self.counts["real_queries"] += 1
            if self.gate.is_set():
                self.counts["host_false_queries"] += 1
                self.counts["real_ready_while_host_held"] += int(actual)
                return False
            self.counts["real_true_after_release"] += int(actual)
            return actual
        def elapsed_time(self, other):
            return self.real.elapsed_time(other.real if isinstance(other, HeldEvent) else other)
        def __getattr__(self, name):
            return getattr(self.real, name)

    class CudaFacade:
        def __init__(self, r, gate, counts):
            self.r = r
            self.gate = gate
            self.counts = counts
        def Event(self, **kwargs):
            actual = original_event(**kwargs)
            # Only native owner's timed-copy events are wrapped. Main-thread
            # compute events remain genuine Events for Stream.wait_event.
            if self.gate.is_set() and threading.get_ident() == self.r._worker.ident:
                self.counts["wrapped_real_events"] += 1
                return HeldEvent(actual, self.gate, self.counts)
            return actual
        def __getattr__(self, name):
            return getattr(torch.cuda, name)

    class TorchFacade:
        def __init__(self, r, gate, counts):
            self.cuda = CudaFacade(r, gate, counts)
        def __getattr__(self, name):
            return getattr(torch, name)

    try:
        for kind in ("cache", "shared"):
            remaining()
            directory = out / kind
            directory.mkdir(exist_ok=False)
            for tensor, reference in zip(tensors, original):
                tensor[:SOURCE_FILES].copy_(reference)
            torch.cuda.synchronize()
            hashes = [hashlib.sha256(("p316-" + kind + "-" + str(i)).encode()).digest()
                      for i in range(SOURCE_FILES)]
            disk = SharedStorageLoadStoreSpec(hashes)
            source_h, mapper = create(directory / "files", kind + "-native-source")
            require(source_h.transfer_async(1, (
                GPULoadStoreSpec(list(range(SOURCE_FILES)), [SOURCE_FILES], [0]), disk),
                req_id=kind + "-source"), "actual native source store rejected")
            source_future = source_h._active[1][0]
            require(source_future.result(timeout=60) == SOURCE_FILES * layout.storage_block_bytes,
                    "actual native source store bytes differ")
            setup = close(source_h)
            source_files = []
            for file_index, block_hash in enumerate(hashes):
                path = Path(mapper.get_file_name(block_hash))
                require(path.is_file() and not path.is_symlink() and
                        path.stat().st_size == layout.storage_block_bytes,
                        "owned PRIMARY source invalid")
                actual_sha = sha_file(path)
                expected_sha = hashlib.sha256(b"".join(
                    reference[file_index].numpy().tobytes() for reference in original)).hexdigest()
                require(actual_sha == expected_sha, "native source bytes differ from independent CPU golden")
                source_files.append(dict(path=str(path), bytes=path.stat().st_size,
                                         sha256=actual_sha, expected_sha256=expected_sha))
            write_new(directory / "source-setup.json", dict(
                setup=setup, files=source_files, source_created_by_native_store=True,
                heldout_source_imported=False, all_source_bytes_match_independent_cpu_golden=True,
                source_payload_bytes=sum(x["bytes"] for x in source_files)))
            for tensor in tensors:
                tensor.zero_()
            torch.cuda.synchronize()
            h, _ = create(directory / "files", kind + "-target", kind=kind)
            r = h.coordinator.reactor
            if kind == "cache":
                require(h.transfer_async(10, (
                    disk, GPULoadStoreSpec(list(range(SOURCE_FILES)), [SOURCE_FILES], [0])),
                    req_id="cache-fill"), "actual cache fill rejected")
                require(h._active[10][0].result(timeout=60) ==
                        SOURCE_FILES * layout.storage_block_bytes, "cache fill incomplete")
            else:
                # Two consumer demands plus one original native retention
                # demand keep the shared member cached until explicit STOP.
                for owner in ("A", "B", "keeper"):
                    require(h.preload_async(owner, disk, req_id=owner), "native shared demand rejected")
            key = "cache_clean_slots" if kind == "cache" else "shared_clean_slots"
            warm, polls = wait_snapshot(r, lambda snap:
                snap["staging_capacity"]["valid"] is True and
                snap["staging_capacity"][key] == SOURCE_FILES and
                snap["native"]["ring_ops"] == snap["native"]["pending_copies"] == 0 and
                snap["stage_accounting"]["outstanding_records"] == 0)
            require(warm["staging_capacity"]["foreground_reclaimable_slots"] == 127 and
                    warm["stage_accounting"]["stages"]["ssd_read"]["accepted_bytes"] ==
                    SOURCE_FILES * layout.storage_block_bytes,
                    "large registry warmup capacity/read count differs")
            require(warm["controller"]["native_fallbacks"] == 0 and
                    warm["controller"]["denied_performance"] > 0,
                    "high test-only reserve lacked known native pressure decisions")
            write_new(directory / "warmup-owner.json", dict(snapshot=warm, polls=polls,
                large_registry_qualified=True, test_only_pressure_reserve_bytes=127 * layout.storage_block_bytes,
                production_candidates_changed=False))
            if kind == "cache":
                require(all(torch.equal(tensor[:SOURCE_FILES].cpu(), reference)
                            for tensor, reference in zip(tensors, original)),
                        "actual cache fill full content mismatch")
            for tensor in tensors:
                tensor[0].zero_()
                if kind == "shared":
                    tensor[80].zero_()
            torch.cuda.synchronize()
            host_gate = threading.Event()
            host_gate.set()
            counts = dict(real_queries=0, host_false_queries=0,
                          real_ready_while_host_held=0, real_true_after_release=0,
                          wrapped_real_events=0)
            futures = []
            protected = {}
            release_counts = {}
            stop_proof = {}
            stop_recorded = threading.Event()
            observer_errors = []
            original_capture = r._capture_owner_snapshot
            original_intake = r._intake
            original_release = r.staging_pool.release

            def observed_release(index):
                # Native physical release executes first, never conditional on
                # optional diagnostics. This scalar log is bounded by 127 slots.
                result = original_release(index)
                release_counts[index] = release_counts.get(index, 0) + 1
                return result

            def owner_proof():
                require(threading.get_ident() == r._worker.ident, "proof was not native owner")
                member = (r._staging_cache._slots.get(hashes[0]) if kind == "cache"
                          else r._shared_cached.get(hashes[0]))
                if member is not None:
                    protected["member"] = member
                member = protected.get("member")
                if member is None:
                    return dict(member_available=False)
                index = member.slot_index
                return dict(member_available=True, owner_thread=threading.get_ident(),
                    copies_inflight=member.copies_inflight, slot_index=index,
                    slot_in_original_free_list=index in r.staging_pool._free,
                    original_release_count=release_counts.get(index, 0),
                    future_pending=[not future.done() for future in futures],
                    original_pending_copy_batches=len(r._pending_copies),
                    native_stop=r._stop, host_gate_held=host_gate.is_set(),
                    shared_cached=(member.cached if kind == "shared" else None),
                    pinned_count_is_distinct_physical_slots=True,
                    physical_completion_inferred=False)

            def capture(*, native_shutdown=False):
                result = original_capture(native_shutdown=native_shutdown)
                if not native_shutdown:
                    result["qualification_ownership"] = owner_proof()
                return result

            def intake(item):
                result = original_intake(item)
                if item is r._STOP:
                    try:
                        proof = owner_proof()
                        proof["staging_capacity"] = r._prefix_capacity_snapshot()
                        stop_proof.update(proof)
                    except BaseException as exc:
                        observer_errors.append(repr(exc))
                    finally:
                        stop_recorded.set()
                return result

            r._capture_owner_snapshot = capture
            r._intake = intake
            r.staging_pool.release = observed_release
            native.torch = TorchFacade(r, host_gate, counts)
            try:
                one = SharedStorageLoadStoreSpec([hashes[0]])
                destinations = (0,) if kind == "cache" else (0, 80)
                for offset, destination in enumerate(destinations):
                    jid = 20 + offset
                    gpu = GPULoadStoreSpec([destination], [1], [0])
                    accepted = (h.transfer_async(jid, (one, gpu), req_id="cache-busy")
                        if kind == "cache" else
                        h.load_from_preload_async(jid, ("A", "B")[offset], one, gpu,
                                                 req_id=("A", "B")[offset]))
                    require(accepted, "actual busy native consumer rejected")
                    futures.append(h._active[jid][0])
                expected_refs = len(destinations)
                busy, polls = wait_snapshot(r, lambda snap:
                    snap.get("qualification_ownership", {}).get("copies_inflight") == expected_refs and
                    snap["qualification_ownership"]["original_pending_copy_batches"] > 0 and
                    len(snap["qualification_ownership"]["future_pending"]) == expected_refs and
                    all(snap["qualification_ownership"]["future_pending"]))
                capacity = busy["staging_capacity"]
                require(capacity["valid"] is True and
                        capacity[key] == SOURCE_FILES - 1 and
                        capacity["foreground_reclaimable_slots"] == 126,
                        "busy distinct physical slot falsely counted clean")
                pinned_key = "cache_pinned_slots" if kind == "cache" else "shared_cached_pinned_slots"
                require(capacity[pinned_key] == 1, "multiple consumers double-counted physical pin")
                require(not busy["qualification_ownership"]["slot_in_original_free_list"] and
                        busy["qualification_ownership"]["original_release_count"] == 0,
                        "busy slot released before actual event completion")
                write_new(directory / "busy-owner.json", dict(snapshot=busy, polls=polls,
                    event_observer=dict(counts), synthetic_host_pending=True,
                    artificial_dma_delay=False))
                # Native STOP is consumed by the actual owner while real copy
                # events can report only False through the extra host gate.
                r.shutdown(wait=False)
                require(stop_recorded.wait(timeout=10) and not observer_errors and stop_proof,
                        "actual owner STOP proof missing")
                require(stop_proof["copies_inflight"] == expected_refs and
                        all(stop_proof["future_pending"]) and
                        not stop_proof["slot_in_original_free_list"] and
                        stop_proof["original_release_count"] == 0 and
                        stop_proof["staging_capacity"]["valid"] is False and
                        stop_proof["staging_capacity"]["free_reclaimable_staging_bytes"] is None,
                        "STOP reclaimed live copy ownership or invented capacity")
                write_new(directory / "stop-owner.json", dict(
                    ownership=stop_proof, synthetic_host_pending=True,
                    extra_query_false_only=True, real_dma_delayed=False))
                host_gate.clear()
                require(all(future.result(timeout=20) == layout.storage_block_bytes
                            for future in futures), "actual busy copy Future content bytes differ")
                final = close(h)
                torch.cuda.synchronize()
                require(all(torch.equal(tensor[destination].cpu(), reference[0])
                            for tensor, reference in zip(tensors, original)
                            for destination in destinations), "actual final destination bytes differ")
                slot_index = stop_proof["slot_index"]
                require(release_counts.get(slot_index, 0) == 1,
                        "actual native busy slot not released exactly once after complete drain")
                require(counts["host_false_queries"] > 0 and counts["real_true_after_release"] > 0,
                        "real query completion/extra host false witness missing")
                require(final["controller"]["native_fallbacks"] == 0 and
                        final["controller"]["metadata_fallbacks"] == 0,
                        "compact integration became unknown or metadata fallback")
                expected_h2d = (SOURCE_FILES + 1 if kind == "cache" else 2) * layout.storage_block_bytes
                require(final["accounting"]["stages"]["h2d"]["accepted_bytes"] == expected_h2d and
                        final["accounting"]["stages"]["ssd_read"]["accepted_bytes"] ==
                        SOURCE_FILES * layout.storage_block_bytes,
                        "actual cache/shared I/O or copy count differs")
                for row in source_files:
                    require(sha_file(row["path"]) == row["sha256"], "owned source file changed")
                record = dict(status="PASS_REAL_GPU_COMPACT_CAPACITY_STOP", kind=kind,
                    native_final=final, source_files=SOURCE_FILES, cached_members_before_busy=SOURCE_FILES,
                    distinct_busy_slots=1, copy_consumers=expected_refs,
                    event_observer=dict(counts), synthetic_host_pending=True,
                    artificial_dma_delay=False, exact_destination_content=True,
                    busy_slot_original_release_count=release_counts[slot_index],
                    real_cuda=True, real_linux_aio=True, real_author_copy_ops=True,
                    source_files_unchanged=True, all_source_bytes_match_cpu_golden=True, new_executor=False,
                    model_performance_claim=False, production_candidates_changed=False)
                write_new(directory / "final.json", record)
                cases.append(record)
            finally:
                # Release only the extra host gate. Real Event.query still
                # decides completion; restore original module and callbacks.
                host_gate.clear()
                native.torch = original_native_torch
                r._capture_owner_snapshot = original_capture
                r._intake = original_intake
                r.staging_pool.release = original_release
            del h, r, source_h, source_future
            gc.collect()
        require(source_snapshot() == before_sources, "qualified sources changed during GPU run")
        write_new(out / "sources-after.json", source_snapshot())
        return dict(status="PASS_REAL_GPU_P316_COMPACT_STAGING", cases=cases,
                    actual_gpu_run=True, model_loaded=False, cache_registry_over_64=True,
                    shared_registry_over_64=True, compact_known_pressure_qualified=True,
                    exact_content=True, original_stop_and_drain=True,
                    synthetic_host_pending=True, artificial_dma_delay=False,
                    performance_experiment=False, production_candidates_changed=False)
    finally:
        native.torch = original_native_torch
        for handler in list(active):
            reactor = handler.coordinator.reactor
            try:
                handler.shutdown()
                active.remove(handler)
            except BaseException as exc:
                run_state.setdefault("cleanup_errors", []).append(repr(exc))
                if isinstance(exc, NativeDrainUnknown) or getattr(reactor, "_native_drain_unknown", False):
                    run_state["native_drain_unknown"] = True
                    run_state["physical_release_claimed"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    require(not out.exists(), "append-new output must not exist")
    actual, auxiliary = authorized_path(out)
    require(auxiliary is None and actual == out, "owned tiny sources must be approved PRIMARY")
    ancestor = out
    while ancestor != ancestor.parent:
        require(not ancestor.is_symlink(), "PRIMARY output ancestry must not be a symlink")
        ancestor = ancestor.parent
    contract = preflight(out, PRIMARY_RESERVATION_BYTES)
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "preflight.json", contract)
    started = time.monotonic()
    run_state = {}
    try:
        result = qualify(out, started, run_state)
        code = 0
    except BaseException as exc:
        result = dict(status="FAIL", error=repr(exc), traceback=traceback.format_exc(),
                      partial_receipts_retained=True, model_performance_claim=False)
        code = 1
    if run_state:
        result["cleanup_state"] = run_state
        result["status"] = "FAIL"
        code = 1
    result["seconds"] = time.monotonic() - started
    result["storage_preflight"] = contract
    try:
        result["storage_after"] = preflight(out, 0)
    except BaseException as exc:
        result["storage_after_error"] = repr(exc)
        result["status"] = "FAIL"
        code = 1
    write_new(out / "result.json", result)
    print(json.dumps(dict(status=result["status"], output=str(out), exit=code)), flush=True)
    if run_state.get("native_drain_unknown"):
        # Only process/context teardown; no online resource release was asserted.
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
