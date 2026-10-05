"""Existing owner reactor with real Linux AIO/direct files and CPU CUDA fakes.

Only aligned host bytes, native queues, accepted counters, parent drain and
observation isolation are verified. There is no GPU DMA or effect measurement.
"""
import contextlib
import ctypes
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from py_kvcache import reactor as mod
from py_kvcache.file_mapper import FileMapper
from py_kvcache.fs_config import SharedFileConfig
from py_kvcache.liburing_file import DirectIoFileStore
from py_kvcache.linux_aio import LinuxAioRing
from py_kvcache.reactor import IoReactor
from py_kvcache.transfer import ParsedKvLayout
from tests.prefix_io_v1_p4_02_bridge.test_native_events import NativeRig, assert_accepted_equal
from tests.prefix_io_v1_start_budget._fixture import eventually


@contextlib.contextmanager
def hybrid(monkeypatch, directory, mode):
    rig = NativeRig(monkeypatch, mode="shadow")
    r = rig.r
    # Replace only the existing fake storage and fake bytes, keeping all native
    # intake, pump, copy mapping, acceptance, terminal and shutdown code.
    monkeypatch.setattr(mod.time, "monotonic_ns", time.perf_counter_ns)
    rig.host_device = torch.empty((4, 4096), dtype=torch.uint8, device="cpu")
    r.layout = ParsedKvLayout([rig.host_device], [4096], 1, False)
    rig.coordinator.layout = r.layout
    r.staging_buffer = r.layout.allocate_staging_buffer(4)
    r._staging_base_ptr = r.staging_buffer.data_ptr()
    r._slot_view = IoReactor._slot_view.__get__(r)
    r._build_mapping = IoReactor._build_mapping.__get__(r)
    r.file_mapper = FileMapper(
        root_dir=str(directory), model_name="cpu-hybrid", gpu_block_size=1,
        gpu_blocks_per_file=1, tp_size=1, pp_size=1, pcp_size=1, rank=0,
        dtype="cpu-fixture",
    )
    r.file_store = DirectIoFileStore(
        SharedFileConfig(root_dir=str(directory), io_backend="linux_aio",
                         sync_on_store=False, iodepth=2),
        payload_size=4096,
    )
    r.ring = LinuxAioRing(depth=2)
    # The existing fixture pump publishes this on its original owner thread.
    rig.observe = lambda owner, **_: owner.parent_admission_snapshot()
    if mode == "off":
        r._prefix_p4_bridge = None
    ranges = [
        (rig.host_device.data_ptr(), rig.host_device.data_ptr() + rig.host_device.numel()),
        (r._staging_base_ptr, r._staging_base_ptr + r.staging_buffer.numel()),
    ]

    def cpu_copy(src, dst, sizes):
        entries = list(zip(src.tolist(), dst.tolist(), sizes.tolist()))
        for start, target, size in entries:
            assert size > 0
            for pointer in (start, target):
                assert any(lo <= pointer and pointer + size <= hi for lo, hi in ranges), (
                    "CPU fake refused a pointer outside its owned host allocations"
                )
            ctypes.memmove(target, start, size)
        rig.h2d_launches.append(dict(entries=len(entries), bytes=sum(z for _, _, z in entries)))

    monkeypatch.setattr(mod, "ops", NS(swap_blocks_batch=cpu_copy))
    try:
        yield rig
    finally:
        rig.cleanup()
        r.shutdown(wait=True)
        # Actual kernel/metadata drain is checked separately from fake CUDA.
        ring = r.ring.snapshot()
        assert ring["closed"] and ring["drained"] and ring["fatal"] is None
        assert ring["outstanding"] == ring["unreaped"] == 0


@pytest.mark.parametrize("mode", ["off", "shadow"])
def test_real_direct_store_load_roundtrip_preserves_bytes_and_parent_fences(
        monkeypatch, tmp_path, record_property, mode):
    with hybrid(monkeypatch, tmp_path, mode) as rig:
        r = rig.r
        patterns = [
            ((np.arange(4096, dtype=np.uint16) * 17 + offset) % 256).astype(np.uint8).tobytes()
            for offset in (13, 91)
        ]
        for index, value in enumerate(patterns):
            rig.host_device[index].copy_(torch.from_numpy(np.frombuffer(value, dtype=np.uint8).copy()))
        hashes = [b"7-0", b"7-1"]
        paths = [Path(r.file_mapper.get_file_name(h)) for h in hashes]
        rig.copy_ready.clear()
        store = rig.submit(files=2)
        rig.start()
        eventually(lambda: bool(r._pending_copies))
        held_store = [op.slot_index for op in r._pending_copies]
        assert not store.done() and all(not p.exists() for p in paths)
        assert all(index not in r.staging_pool._free for index in held_store)
        rig.copy_ready.set()
        assert store.result(timeout=5) == 8192
        assert [p.read_bytes() for p in paths] == patterns
        eventually(lambda: bool(rig.snapshots) and rig.snapshots[-1]["accepted_parents"] == 0)

        # The CPU copy executes immediately but the original event remains held:
        # this checks owner drain, never a claim of GPU completion.
        rig.host_device.zero_()
        rig.copy_ready.clear()
        before_releases = len(r.staging_pool.releases)
        load = rig.load(files=2, hashes=hashes)
        eventually(lambda: bool(r._pending_copies) and len(rig.h2d_launches) > 2)
        held_load = [op.slot_index for op in r._pending_copies]
        assert not load.done() and rig.snapshots[-1]["accepted_parents"] == 1
        assert all(index not in r.staging_pool._free for index in held_load)
        assert len(r.staging_pool.releases) == before_releases
        rig.copy_ready.set()
        assert load.result(timeout=5) == 8192
        assert [rig.host_device[i].numpy().tobytes() for i in range(2)] == patterns
        rig.stop()
        assert r.parent_admission_snapshot(native_shutdown=True)["accepted_parents"] == 0
        accounting = assert_accepted_equal(rig)
        assert {stage: row["accepted_bytes"] for stage, row in accounting["stages"].items()} == {
            "d2h": 8192, "ssd_write": 8192, "ssd_read": 8192, "h2d": 8192,
        }
        kernel = r.ring.snapshot()
        assert kernel["accepted"] == kernel["completed"] == kernel["reaped"] > 0
        assert kernel["max_outstanding"] <= kernel["capacity"]
        snapshot = rig.bridge.snapshot(native_shutdown=True)
        assert not snapshot["gpu_qualified"] and snapshot["gpu_release_credit"] is None
        assert snapshot["new_work_queues"] == 0
        assert not torch.cuda.is_initialized()
        record_property("cpu_fake_cuda", True)
        record_property("gpu_workloads_run", 0)
        record_property("real_linux_aio", json.dumps(kernel, sort_keys=True))
        record_property("accepted_bytes", json.dumps({
            stage: row["accepted_bytes"] for stage, row in accounting["stages"].items()
        }, sort_keys=True))
        record_property("payload_sha256", json.dumps([
            hashlib.sha256(p).hexdigest() for p in patterns
        ]))


def test_real_short_read_reports_failure_after_original_parent_drain(
        monkeypatch, tmp_path, record_property):
    with hybrid(monkeypatch, tmp_path, "shadow") as rig:
        r = rig.r
        path = Path(r.file_mapper.get_file_name(b"short-real-file"))
        path.parent.mkdir(parents=True)
        path.write_bytes(b"real-short-read" * 16)
        rig.host_device.fill_(77)
        load = rig.load(hashes=[b"short-real-file"])
        rig.start()
        with pytest.raises(Exception, match="read transferred 240 bytes, expected 4096"):
            load.result(timeout=5)
        eventually(lambda: bool(rig.snapshots) and rig.snapshots[-1]["accepted_parents"] == 0)
        rig.stop()
        assert not rig.h2d_launches
        assert rig.host_device.numpy().tobytes() == bytes([77]) * rig.host_device.numel()
        accounting = assert_accepted_equal(rig)
        assert accounting["stages"]["ssd_read"]["accepted_bytes"] == 4096
        assert accounting["stages"]["h2d"]["accepted_bytes"] == 0
        ring = r.ring.snapshot()
        assert ring["accepted"] == ring["completed"] == ring["reaped"] > 0
        assert r.parent_admission_snapshot(native_shutdown=True)["accepted_parents"] == 0 and not r._inflight and not r._pending_copies
        assert not rig.bridge.snapshot(native_shutdown=True)["gpu_qualified"]
        assert not torch.cuda.is_initialized()
        record_property("real_linux_aio", json.dumps(ring, sort_keys=True))
        record_property("gpu_workloads_run", 0)
