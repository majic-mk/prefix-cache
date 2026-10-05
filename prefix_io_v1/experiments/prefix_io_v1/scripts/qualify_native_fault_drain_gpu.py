"""Real GPU qualification of P315 native exceptional physical drain.

No model, replacement executor, cache-engine implementation or GPU timing claim.
A temporary observer injects a synthetic exception after one real author copy
mapping; all submission, synchronization, ownership and retirement remain native.
Only newly owned PRIMARY tiny files are read or written. Run under the project's
frozen GPU guard (240-second process envelope); this file never grants GPU use.
"""
import argparse
import gc
import hashlib
import json
import os
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
SOURCES = (
    "experiments/prefix_io_v1/scripts/qualify_native_fault_drain_gpu.py",
    "src/prefix_io_control/simple_stage_policy.py",
    "src/prefix_io_control/stage_accounting.py",
    "src/prefix_io_control/dispatch_budget.py",
    "third_party/work/py-kvcache-p3-15-cpu/py_kvcache/reactor.py",
    "third_party/work/py-kvcache-p3-15-cpu/py_kvcache/vllm.py",
    "third_party/work/py-kvcache-p3-15-cpu/py_kvcache/linux_aio.py",
    "third_party/work/vllm-author-build/vllm/_custom_ops.py",
    "third_party/work/vllm-author-build/csrc/cache_kernels.cu",
)

class SyntheticPartialCopy(RuntimeError):
    pass

class SyntheticEndEventRecord(RuntimeError):
    pass

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

def snapshot_sources():
    return {name: sha_file(ROOT / name) for name in SOURCES}

def normalize_gpu_uuid(value):
    # Torch may expose the full GPU- form, bare UUID, or exactly 16 UUID bytes.
    # Normalize identity only; never substitute a device index or accept MIG IDs.
    if isinstance(value, (bytes, bytearray)) and len(value) == 16:
        return "GPU-" + str(uuid.UUID(bytes=bytes(value)))
    text = str(value)
    if text.startswith("GPU-"):
        text = text[4:]
    return "GPU-" + str(uuid.UUID(text))

def bounded_stage_records(accounting):
    return [dict(stage=stage, native_key=key, requested_bytes=nbytes)
            for (stage, key), nbytes in accounting.records.items()]

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
        ROOT / "third_party/work/py-kvcache-p3-15-cpu"), "wrong native worktree")
    require(Path(native.ops.__file__).resolve().is_relative_to(
        ROOT / "third_party/work/vllm-author-build"), "wrong author operations module")
    require(torch.cuda.device_count() == 1, "exactly one approved GPU required")
    gpu_uuid_raw = torch.cuda.get_device_properties(0).uuid
    gpu_uuid = normalize_gpu_uuid(gpu_uuid_raw)
    permit = permission()
    approved = permit["approved_gpu_ids"]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(permit["allow_gpu_runs"] and len(approved) == 1 and "," not in visible and
            normalize_gpu_uuid(visible) == normalize_gpu_uuid(approved[0]) == gpu_uuid,
            "sole UUID binding/device identity not explicitly approved")
    initial_sources = snapshot_sources()
    modules = {name: dict(path=str(Path(module.__file__).resolve()),
                         sha256=sha_file(module.__file__))
               for name, module in (("native", native), ("author_ops", native.ops), ("torch", torch))}
    write_new(out / "sources-before.json", dict(source_sha256=initial_sources, modules=modules))
    write_new(out / "gpu-environment.json", dict(
        gpu_uuid=gpu_uuid, torch_gpu_uuid_raw=str(gpu_uuid_raw),
        cuda_visible_devices=visible, device_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
        time_envelope_seconds=TIME_ENVELOPE_SECONDS, staging_budget_bytes=STAGING_BYTES,
        model_loaded=False, guarded_run_required=True,
    ))

    torch.manual_seed(31508)
    tensors = [torch.randint(-128, 128, (2, 32768), device="cuda", dtype=torch.int8)
               for _ in range(4)]
    original = [tensor.cpu().clone() for tensor in tensors]
    caches = CanonicalKVCaches(
        [CanonicalKVCacheTensor(tensor, page_size_bytes=32768) for tensor in tensors],
        [[CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=32768) for i in range(4)]],
    )
    layout = ParsedKvLayout.from_canonical_kv_caches(
        gpu_block_size=16, storage_block_size=16, kv_caches=caches)
    require(layout.storage_block_bytes == 4 * 32768, "unexpected tiny-page layout")
    require(sum(t.numel() * t.element_size() for t in tensors) == 262144,
            "unexpected GPU tensor allocation")
    real_torch = native.torch
    original_swap = native.ops.swap_blocks_batch
    original_event = torch.cuda.Event
    active_handlers = []
    cases = []

    # Return real Event objects, never a proxy passed to Stream.wait_event.
    # The facade is local to the native Python module; global torch is unchanged.
    class CudaFacade:
        def __init__(self, event_factory):
            self.Event = event_factory
        def __getattr__(self, name):
            return getattr(torch.cuda, name)
    class TorchFacade:
        def __init__(self, event_factory):
            self.cuda = CudaFacade(event_factory)
        def __getattr__(self, name):
            return getattr(torch, name)

    record_override_supported = False
    record_override_reason = None
    probe = original_event(enable_timing=True)
    try:
        actual_record = probe.record
        def forwarding_record(stream=None):
            return actual_record(stream)
        probe.record = forwarding_record
        require(probe.record is forwarding_record, "Event instance did not retain record observer")
        probe.record = actual_record
        record_override_supported = True
    except (AttributeError, TypeError, RuntimeError) as exc:
        record_override_reason = repr(exc)
    del probe

    def before_new_case():
        require(time.monotonic() - started < TIME_ENVELOPE_SECONDS - 30,
                "remaining process envelope cannot start another case")

    def create(directory, label, audit=False):
        directory.mkdir(parents=True, exist_ok=True)
        mapper = FileMapper(
            root_dir=str(directory), model_name="p315-native-fault",
            gpu_block_size=16, gpu_blocks_per_file=1, tp_size=1, pp_size=1,
            pcp_size=1, rank=0, dtype="bfloat16",
        )
        controller = None
        if audit:
            zero = tuple(Amount() for _ in STAGES)
            controller = DispatchController(label, SimpleStageConfig(
                mode="shadow", epoch_ns=10_000_000, byte_quantum=layout.storage_block_bytes,
                cumulative=zero, inflight=zero, shared_ssd_cumulative=Amount(),
                shared_ssd_inflight=Amount(), shared_copy_cumulative_bytes=0,
                shared_copy_inflight_bytes=0, reserve_staging_bytes=0,
                max_accepted_parents=2, sample_max_age_ns=20_000_000,
                max_wait_ns=200_000_000,
            ))
        c = TransferCoordinator(
            config=SharedFileConfig(
                root_dir=str(directory), iodepth=2, open_lookahead=2,
                staging_mem=16 / 1024, enable_preload=False,
                preload_share_staging=False, staging_cache="off", io_backend="linux_aio",
            ),
            file_mapper=mapper, layout=layout, storage_block_tokens=16,
            progress_run_id=label, max_accepted_parents=2,
            dispatch_controller=controller, stage_accounting=StageAccounting(),
        )
        h = NoopSharedStorageOffloadingHandler(coordinator=c)
        active_handlers.append(h)
        require(c.reactor.staging_buffer.is_pinned(), "staging is not real pinned memory")
        require(c.reactor.staging_budget_bytes == STAGING_BYTES and
                c.reactor.actual_staging_bytes <= STAGING_BYTES, "staging exceeds 16 MiB")
        return h, mapper

    def close(h, expected_fault=False):
        h.shutdown()
        r = h.coordinator.reactor
        require(not r._worker.is_alive(), "native owner still running")
        require(not r._active and not r._inflight and not r._pending_copies and
                not r._copy_ready and not r._ready_fds_load and not r._ready_fds_preload,
                "original native resource owners did not drain")
        aio = r.ring.snapshot()
        require(aio["closed"] and aio["drained"] and aio["outstanding"] == 0 and
                aio["pending"] == 0 and aio["ready"] == 0 and aio["unreaped"] == 0,
                "real Linux AIO did not fully drain/reap")
        admission = r.parent_admission_snapshot()
        require(admission["count_valid"] and admission["accepted_parents"] == 0 and
                not admission["native_drain_unknown"], "parent protection retired without proof")
        accounting = r._prefix_stage_accounting.snapshot()
        if expected_fault:
            require(not accounting["valid"], "expected ambiguity was falsely marked valid")
        else:
            require(accounting["valid"] and accounting["outstanding_records"] == 0,
                    "setup stage accounting did not drain")
        controller = r._prefix_dispatch_controller
        result = dict(
            aio=aio, admission=admission, accounting=accounting,
            known_accepted_records=bounded_stage_records(r._prefix_stage_accounting),
            controller=controller.snapshot(native_shutdown=True) if controller else None,
            actual_staging_bytes=r.actual_staging_bytes,
            expected_fault=expected_fault, resource_release_inferred_from_accounting=False,
        )
        active_handlers.remove(h)
        return result

    def run_case(stage, kind):
        before_new_case()
        name = stage + "-" + kind
        directory = out / name
        directory.mkdir(exist_ok=False)
        for tensor, reference in zip(tensors, original):
            tensor.copy_(reference)
        torch.cuda.synchronize()
        hashes = [hashlib.sha256((name + "-source-" + str(i)).encode()).digest()
                  for i in range(2)]
        disk = SharedStorageLoadStoreSpec(hashes)
        source_h, mapper = create(directory / "files", name + "-source")
        require(source_h.transfer_async(1, (GPULoadStoreSpec([0, 1], [2], [0]), disk),
                                        req_id=name + "-source"), "source setup rejected")
        source_future = source_h._active[1][0]
        source_h.wait({1})
        require(source_future.result(timeout=20) == 2 * layout.storage_block_bytes,
                "source setup parent size mismatch")
        setup = close(source_h)
        source_files = []
        for block_hash in hashes:
            path = Path(mapper.get_file_name(block_hash))
            require(path.is_file() and not path.is_symlink() and
                    path.stat().st_size == layout.storage_block_bytes, "owned source file invalid")
            source_files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha_file(path)))
        write_new(directory / "setup.json", dict(
            native_setup=setup, owned_primary_source_files=source_files,
            heldout_source_imported=False, source_created_by_native_store=True,
        ))
        del source_future, source_h
        gc.collect()

        h, _ = create(directory / "files", name + "-fault", audit=True)
        r = h.coordinator.reactor
        controller = r._prefix_dispatch_controller
        target_call = 2 if stage == "d2h" else 1
        error = SyntheticPartialCopy(name) if kind == "partial" else SyntheticEndEventRecord(name)
        state = dict(
            swap_attempts=0, full_api_returns=0, actual_prefix_api_returns=0,
            prefix_bytes=0, main_requested_bytes=0, injected=False, syncs=[],
            completion_callbacks=[], injection_kind=kind, expected_stage=stage,
        )
        event_stream = (directory / "events.jsonl").open("x")

        def emit(value):
            event_stream.write(json.dumps(value, allow_nan=False) + "\n")
            event_stream.flush()
            os.fsync(event_stream.fileno())

        def swap(src, dst, sizes):
            state["swap_attempts"] += 1
            requested = int(sizes.sum().item())
            if kind == "partial" and state["swap_attempts"] == target_call:
                require(int(sizes.numel()) > 1, "partial injection needs multiple real mappings")
                prefix_bytes = int(sizes[0].item())
                require(prefix_bytes == 32768 and requested > prefix_bytes,
                        "wrong first physical mapping size")
                original_swap(src[:1], dst[:1], sizes[:1])
                state["actual_prefix_api_returns"] += 1
                state["prefix_bytes"] = prefix_bytes
                state["main_requested_bytes"] = requested
                state["injected"] = True
                emit(dict(event="real_author_prefix_api_returned", stage=stage,
                          actual_mappings=1, actual_requested_bytes=prefix_bytes,
                          outer_requested_bytes=requested, synthetic_exception=True,
                          genuine_driver_failure=False, completion_proved=False))
                raise error
            original_swap(src, dst, sizes)
            state["full_api_returns"] += 1
            if state["swap_attempts"] == target_call:
                state["main_requested_bytes"] = requested

        def event_factory(**kwargs):
            event = original_event(**kwargs)
            real_record = event.record
            def record(stream=None):
                if (kind == "end_event" and not state["injected"] and
                        state["swap_attempts"] == target_call and
                        state["full_api_returns"] == target_call):
                    state["injected"] = True
                    emit(dict(event="end_event_record_synthetic_failure", stage=stage,
                              real_copy_api_returned=True, requested_bytes=state["main_requested_bytes"],
                              genuine_driver_failure=False, event_completion_witness=False))
                    raise error
                return real_record(stream)
            event.record = record
            return event

        real_sync_failure = r._sync_copy_failure
        def observed_sync(stream, actual_stage, original_error):
            diagnostic_error = None
            try:
                require(actual_stage == stage and original_error is error, "wrong injected failure boundary")
                require(state["injected"], "native synchronized before synthetic injection")
                job = next(job for job in r._active if job.failed is None)
                require(not job.future.done() and not job.future_set and
                        job.inflight_files > 0, "Future released GPU protection before sync")
                held = set(range(r.staging_pool.slot_count)) - set(r.staging_pool._free)
                require(held and r.accepted_parent_count() == 1, "native source/staging owners missing")
                if stage == "d2h":
                    slot = next(i for i, item in enumerate(r._streams) if item is stream)
                    require(slot in held, "D2H actual stream slot already released")
                    require(len(r._pending_copies) == 1 and job.inflight_files == 2,
                            "D2H sibling was not actually accepted before failure")
                    proof_index = 1
                else:
                    require(stream is r._copy_stream and r._copy_ready,
                            "H2D original fused ownership list missing")
                    slot = r._copy_ready[0].slot_index
                    require(slot in held and all(not item.job.future.done() for item in r._copy_ready),
                            "H2D original consumer protection missing")
                    proof_index = int(r._copy_ready[0].job.block_chunks[r._copy_ready[0].file_index][0])
                before = dict(
                    event="before_actual_native_stream_synchronize", stage=stage,
                    accepted_parents=r.accepted_parent_count(), parent_future_pending=True,
                    inflight_files=job.inflight_files, held_slots=sorted(held), actual_slot=slot,
                    known_accepted_records=bounded_stage_records(r._prefix_stage_accounting),
                    accounting=r._prefix_stage_accounting.snapshot(),
                    native_cumulative_syncs=r._native_failure_sync_total,
                )
                emit(before)
                def parent_done(future):
                    require(job.inflight_files == 0 and job.accepted_parent_retired,
                            "parent Future published before full native drain/retirement")
                    require(not (held & (set(range(r.staging_pool.slot_count)) -
                                         set(r.staging_pool._free))),
                            "fault source/staging slot reused before real drain")
                    require(future.exception() is error, "native exception identity changed")
                    state["completion_callbacks"].append(dict(
                        future_exception=type(error).__name__, inflight_files=job.inflight_files,
                        accepted_parent_retired=job.accepted_parent_retired,
                        fault_slots_released=True,
                    ))
                    emit(dict(event="parent_failure_callback_after_native_drain",
                              **state["completion_callbacks"][-1]))
                job.future.add_done_callback(parent_done)
            except BaseException as diagnostic_exc:
                # An optional observation must never bypass the unchanged
                # physical safety boundary after a real/partial copy launch.
                diagnostic_error = diagnostic_exc
            # Invoke the unchanged native helper. Its real stream.synchronize()
            # supplies the physical witness; this observer cannot fabricate it.
            real_sync_failure(stream, actual_stage, original_error)
            # NativeDrainUnknown propagates above unchanged and takes priority.
            # An ordinary diagnostic error can escape only after real sync
            # returned, so native exceptional cleanup cannot reuse live DMA.
            if diagnostic_error is not None:
                raise RuntimeError("pre-sync diagnostic failed after actual stream drain") from diagnostic_error
            require(not job.future.done() and held <=
                    (set(range(r.staging_pool.slot_count)) - set(r.staging_pool._free)),
                    "native owners changed inside synchronous proof")
            require(r._native_failure_sync_total == before["native_cumulative_syncs"] + 1,
                    "real native stream-sync proof not recorded")
            if stage == "d2h":
                actual = r._slot_view(slot)[:32768].tobytes()
                expected = original[0][proof_index].numpy().tobytes()
                exact_prefix = actual == expected
                exact_full = None
                if kind == "end_event":
                    full_expected = b"".join(value[proof_index].numpy().tobytes() for value in original)
                    exact_full = r._slot_view(slot).tobytes() == full_expected
                    require(exact_full, "D2H accepted full-copy content mismatch after real sync")
            else:
                exact_prefix = torch.equal(tensors[0][proof_index].cpu(), original[0][proof_index])
                exact_full = None
                if kind == "end_event":
                    # Independent real CQEs may form one or two original
                    # fused batches. Check only this accepted batch's actual
                    # destinations; never pretend a sibling read already copied.
                    exact_full = all(
                        torch.equal(tensor[int(rc.job.block_chunks[rc.file_index][0])].cpu(),
                                    reference[rc.file_index])
                        for rc in r._copy_ready
                        for tensor, reference in zip(tensors, original)
                    )
                    require(exact_full, "H2D accepted batch content mismatch after real sync")
            require(exact_prefix, "actual first copy mapping content mismatch after real sync")
            after = dict(
                event="after_actual_native_stream_synchronize", stage=stage,
                real_stream_sync_returned_successfully=True, parent_future_pending=True,
                held_slots=sorted(held), exact_first_mapping_content=bool(exact_prefix),
                exact_full_copy_content=exact_full,
                native_cumulative_syncs=r._native_failure_sync_total,
                inferred_from_accounting=False,
            )
            state["syncs"].append(after)
            emit(after)

        try:
            if stage == "h2d":
                for tensor in tensors:
                    tensor.zero_()
                torch.cuda.synchronize()
                specification = (disk, GPULoadStoreSpec([0, 1], [2], [0]))
            else:
                fault_hashes = [hashlib.sha256((name + "-failed-" + str(i)).encode()).digest()
                                for i in range(2)]
                specification = (GPULoadStoreSpec([0, 1], [2], [0]),
                                 SharedStorageLoadStoreSpec(fault_hashes))
            write_new(directory / "fault-begin.json", dict(
                name=name, stage=stage, kind=kind, synthetic_failure=True,
                genuine_driver_failure=False, replacement_executor=False,
                target_author_api_call=target_call, source_tensor_bytes=262144,
                native_original_helper="_sync_copy_failure", sync_witness="actual CUDA Stream.synchronize",
            ))
            native.ops.swap_blocks_batch = swap
            if kind == "end_event":
                native.torch = TorchFacade(event_factory)
            r._sync_copy_failure = observed_sync
            require(h.transfer_async(2, specification, req_id=name), "native fault parent rejected")
            future = h._active[2][0]
            try:
                future.result(timeout=20)
            except (SyntheticPartialCopy, SyntheticEndEventRecord) as caught:
                require(caught is error, "failure Future did not preserve injected native exception")
            else:
                raise RuntimeError("injected partial/event copy falsely returned success")
            require(state["injected"] and len(state["syncs"]) == 1 and
                    len(state["completion_callbacks"]) == 1, "native drain/failure callback evidence missing")
            results = h.get_finished()
            require(len(results) == 1 and results[0].job_id == 2 and not results[0].success,
                    "worker did not report one protected failed transfer")
            final = close(h, expected_fault=True)
            observed = final["controller"]["observed_api_accepted"][stage]
            actual_account = final["accounting"]["stages"][stage]
            require(final["admission"]["failure_sync_by_stage"][stage] == 1,
                    "wrong native exceptional proof count")
            if kind == "partial":
                require(final["controller"]["uncertain_ops"] == 1 and
                        final["controller"]["rejected_ops"] == 0,
                        "partial physical acceptance misclassified as rejection")
                require(state["actual_prefix_api_returns"] == 1 and
                        observed["ops"] == state["full_api_returns"],
                        "outer ambiguity was falsely classified as a full successful API return")
            else:
                require(final["controller"]["uncertain_ops"] == 0 and
                        final["controller"]["completion_unknown_ops"] == 1,
                        "failed end-event witness misclassified")
                require(observed["ops"] == state["full_api_returns"] and
                        actual_account["accepted_ops"] == target_call,
                        "successful backend return missing from native acceptance records")
                require(any(row["stage"] == stage for row in final["known_accepted_records"]),
                        "known accepted copy record erased on failed event")
                require(not final["controller"]["completion_accounting_complete"],
                        "unknown event witness falsely declared complete")
            require(not final["controller"]["physical_drain_inferred"],
                    "audit ledger manufactured a physical proof")
            for row in source_files:
                require(sha_file(row["path"]) == row["sha256"], "owned PRIMARY source changed")
            record = dict(
                name=name, status="PASS_REAL_GPU_NATIVE_FAULT_DRAIN_CASE",
                real_cuda=True, real_linux_aio=True, real_author_ops=True,
                synthetic_failure=True, genuine_driver_failure=False,
                actual_prefix_api_accepted_bytes=state["prefix_bytes"],
                outer_requested_bytes=state["main_requested_bytes"],
                full_api_returns_total=state["full_api_returns"],
                failure_preserved=True, exact_sync_content=True, native_state=final,
                callback_proofs=state["completion_callbacks"], sync_proofs=state["syncs"],
                source_file_hashes_unchanged=True, accounting_is_expected_invalid=True,
                false_completion_or_release_from_accounting=False,
            )
            write_new(directory / "fault-final.json", record)
            cases.append(record)
        finally:
            native.ops.swap_blocks_batch = original_swap
            native.torch = real_torch
            r._sync_copy_failure = real_sync_failure
            event_stream.close()
        del h, r
        gc.collect()

    try:
        for stage in ("d2h", "h2d"):
            run_case(stage, "partial")
        if record_override_supported:
            for stage in ("d2h", "h2d"):
                run_case(stage, "end_event")
        else:
            for stage in ("d2h", "h2d"):
                case = dict(
                    name=stage + "-end_event", status="CPU_ONLY_EVENT_RECORD_FAILURE",
                    reason=record_override_reason, real_gpu_event_failure_injected=False,
                    cpu_contract="tests/prefix_io_v1_copy_failure_drain/test_native_faults.py",
                )
                write_new(out / (stage + "-end-event-cpu-only.json"), case)
                cases.append(case)
        final_sources = snapshot_sources()
        require(final_sources == initial_sources, "source changed during real-GPU qualification")
        write_new(out / "sources-after.json", dict(source_sha256=final_sources, unchanged=True))
        return dict(
            status="PASS_REAL_GPU_NATIVE_FAULT_DRAIN", cases=cases,
            gpu_uuid=gpu_uuid, source_sha256=initial_sources, modules=modules,
            synthetic_failure_only=True, genuine_driver_fault_tested=False,
            independent_unknown_gpu_case=False,
            unknown_sync_safety_cpu_only="tests/prefix_io_v1_copy_failure_drain/test_native_faults.py",
            event_record_injection_supported=record_override_supported,
            source_gpu_tensor_bytes=262144, staging_budget_bytes=STAGING_BYTES,
            primary_owned_tiny_sources=True, original_cache_executor=True,
            replacement_executor=False, model_loaded=False, model_performance_claim=False,
            guard_time_envelope_seconds=TIME_ENVELOPE_SECONDS,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    finally:
        native.ops.swap_blocks_batch = original_swap
        native.torch = real_torch
        for handler in list(active_handlers):
            try:
                handler.shutdown()
            except NativeDrainUnknown as exc:
                # No online release/reuse: retained native ownership lasts until
                # this independent process context is torn down below/by guard.
                run_state["unexpected_drain_unknown"] = repr(exc)
            except BaseException as exc:
                run_state["cleanup_error"] = repr(exc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve()
    require(not out.exists(), "append-new output must not exist")
    actual, auxiliary = authorized_path(out)
    require(auxiliary is None and actual == out, "tiny sources/evidence must be approved PRIMARY")
    ancestor = out
    while ancestor != ancestor.parent:
        require(not ancestor.is_symlink(), "PRIMARY output ancestor may not be a symlink")
        ancestor = ancestor.parent
    contract = preflight(out, 128 * 1024**2)
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
    if run_state.get("unexpected_drain_unknown"):
        # OS process/CUDA-context teardown only, after append-new failure receipt.
        # No native owner, Future or buffer was declared released online.
        os._exit(code)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
