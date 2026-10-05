#!/usr/bin/env python3
"""Qualify the author's actual copy ABI or a tiny, explicit GPU byte copy.

This script loads the supplied extension directly. It never imports vLLM,
allocates a model, downloads data, or substitutes another transfer operator.
The caller must run GPU mode through the approved budget/UUID runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback

EXPECTED_SCHEMA = (
    "_C_cache_ops::swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs, "
    "Tensor sizes, bool is_src_access_order_any=False) -> ()"
)
ALLOCATION_BYTES = 1024
H2D_SEGMENTS = ((31, 7, 13), (128, 96, 64), (511, 640, 257))
D2H_SEGMENTS = ((7, 12, 13), (96, 256, 64), (640, 512, 257))


def expected_bytes(source: bytes, fill: int, segments) -> bytes:
    output = bytearray([fill]) * ALLOCATION_BYTES
    for src_offset, dst_offset, size in segments:
        if not (0 <= src_offset < ALLOCATION_BYTES and
                0 <= dst_offset < ALLOCATION_BYTES and size > 0 and
                src_offset + size <= len(source) and
                dst_offset + size <= ALLOCATION_BYTES):
            raise ValueError("copy exceeds a fixed 1024-byte allocation")
        output[dst_offset:dst_offset + size] = source[src_offset:src_offset + size]
    return bytes(output)


def check_no_vllm_import():
    if any(name == "vllm" or name.startswith("vllm.") for name in sys.modules):
        raise RuntimeError("vLLM was imported; this is not direct-extension qualification")


def qualify_abi(torch, op):
    if str(op._schema) != EXPECTED_SCHEMA:
        raise RuntimeError(f"author op schema mismatch: {op._schema}")
    if not torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C_cache_ops::swap_blocks_batch", "CPU"):
        raise RuntimeError("author op has no CPU metadata dispatch")
    empty = torch.empty(0, dtype=torch.int64, device="cpu")
    op(empty, empty, empty)  # Exercise the production default False argument.
    op(empty, empty, empty, False)
    rejected = []
    for index, name in enumerate(("src_ptrs", "dst_ptrs", "sizes")):
        args = [empty, empty, empty]
        args[index] = torch.empty(0, dtype=torch.int32, device="cpu")
        try:
            op(*args, False)
        except RuntimeError as exc:
            if f"{name} must be int64" not in str(exc):
                raise RuntimeError(f"unexpected {name} rejection: {exc}") from exc
            rejected.append(name)
        else:
            raise RuntimeError(f"author op accepted invalid {name} dtype")
    try:
        op(empty, torch.zeros(1, dtype=torch.int64, device="cpu"), empty, False)
    except RuntimeError as exc:
        if "dst_ptrs length must match src_ptrs" not in str(exc):
            raise RuntimeError(f"unexpected length rejection: {exc}") from exc
        rejected.append("mismatched_length")
    else:
        raise RuntimeError("author op accepted mismatched metadata lengths")
    if torch.cuda.is_initialized():
        raise RuntimeError("zero-length CPU ABI qualification initialized CUDA")
    return {"schema": str(op._schema), "cpu_dispatch": True,
            "zero_length_calls": 2, "invalid_inputs_rejected": rejected,
            "real_copy_verified": False, "cuda_initialized": False}


def qualify_gpu(torch, op, gpu_uuid):
    # CUDA_VISIBLE_DEVICES is set by the external budget runner, never by this script.
    if torch.cuda.device_count() != 1:
        raise RuntimeError("GPU qualification requires exactly one visible CUDA device")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    stream = torch.cuda.Stream(device=0)
    metadata = []  # Keep CPU pointer vectors alive through the final synchronization.
    storage = []   # Keep source/destination allocations alive on all exception paths.

    def submit(src, dst, segments):
        expected_bytes(bytes(ALLOCATION_BYTES), 0, segments)  # Bounds before raw pointers.
        src_ptrs = torch.tensor([src.data_ptr() + s for s, _, _ in segments],
                                dtype=torch.int64, device="cpu")
        dst_ptrs = torch.tensor([dst.data_ptr() + d for _, d, _ in segments],
                                dtype=torch.int64, device="cpu")
        sizes = torch.tensor([n for _, _, n in segments], dtype=torch.int64, device="cpu")
        metadata.extend((src_ptrs, dst_ptrs, sizes))
        # This is the only operator performing host/device payload transfers.
        op(src_ptrs, dst_ptrs, sizes, False)

    try:
        source_bytes = bytes((i * 37 + 19) % 256 for i in range(ALLOCATION_BYTES))
        source = torch.empty(ALLOCATION_BYTES, dtype=torch.uint8,
                             device="cpu", pin_memory=True)
        source.copy_(torch.tensor(list(source_bytes), dtype=torch.uint8, device="cpu"))
        witness = torch.empty(ALLOCATION_BYTES, dtype=torch.uint8,
                              device="cpu", pin_memory=True)
        target = torch.empty(ALLOCATION_BYTES, dtype=torch.uint8,
                             device="cpu", pin_memory=True)
        storage.extend((source, witness, target))
        witness.fill_(0x33)
        target.fill_(0x3C)
        with torch.cuda.stream(stream):
            # Canonical production GPU KV storage is int8; raw byte addresses are used.
            gpu = torch.empty(ALLOCATION_BYTES, dtype=torch.int8, device="cuda:0")
            storage.append(gpu)
            gpu.fill_(-91)  # Exact byte value 0xA5, same stream before H2D.
            submit(source, gpu, H2D_SEGMENTS)
            submit(gpu, witness, ((0, 0, ALLOCATION_BYTES),))
        stream.synchronize()
        expected_gpu = expected_bytes(source_bytes, 0xA5, H2D_SEGMENTS)
        actual_gpu = bytes(witness.tolist())
        if actual_gpu != expected_gpu:
            raise RuntimeError("author H2D + author D2H witness differs from exact bytes/guards")
        with torch.cuda.stream(stream):
            submit(gpu, target, D2H_SEGMENTS)
        stream.synchronize()
        expected_target = expected_bytes(expected_gpu, 0x3C, D2H_SEGMENTS)
        actual_target = bytes(target.tolist())
        if actual_target != expected_target:
            raise RuntimeError("author segmented D2H differs from exact bytes/guards")
        return {
            "requested_gpu_uuid": gpu_uuid, "device_name": device_name,
            "visible_cuda_ordinal": 0, "gpu_payload_allocation_bytes": ALLOCATION_BYTES,
            "pinned_cpu_payload_bytes": 3 * ALLOCATION_BYTES,
            "h2d_bytes": sum(n for _, _, n in H2D_SEGMENTS),
            "d2h_bytes": ALLOCATION_BYTES + sum(n for _, _, n in D2H_SEGMENTS),
            "author_op_calls": 3, "is_src_access_order_any": False,
            "byte_exact": True, "guard_bytes_exact": True,
            "stream_synchronized_before_exit": True,
            "gpu_witness_sha256": hashlib.sha256(actual_gpu).hexdigest(),
            "segmented_d2h_sha256": hashlib.sha256(actual_target).hexdigest(),
            "limitations": "Combined H2D/D2H byte qualification only; no timing, io_uring, reactor, or model acceptance.",
        }
    finally:
        # No allocation or pointer-vector lifetime ends before queued work is synchronized.
        stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path,
                        help="Exact freshly built author _C*.so to load with torch.ops.load_library")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--abi-only", action="store_true", help="CPU zero-length schema check; no real copy")
    mode.add_argument("--gpu-uuid", help="Full GPU UUID already bound by the external budget runner")
    parser.add_argument("--output", type=Path, help="New JSON evidence file; refuses to overwrite")
    args = parser.parse_args()
    if args.abi_only:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            parser.error("--abi-only requires an explicitly empty CUDA_VISIBLE_DEVICES")
    else:
        if not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_uuid):
            parser.error("--gpu-uuid requires a full physical GPU UUID")
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
            parser.error("external runner must bind CUDA_VISIBLE_DEVICES to exactly --gpu-uuid")
    library = args.library.resolve(strict=True)
    if not (library.is_file() and library.name.startswith("_C") and library.name.endswith(".so")):
        parser.error("--library must name an existing author _C*.so")
    if args.output is not None and args.output.exists():
        parser.error("--output already exists")
    report = {
        "mode": "cpu_abi_only" if args.abi_only else "real_gpu_copy",
        "status": "FAILED", "library": str(library),
        "library_sha256": hashlib.file_digest(library.open("rb"), "sha256").hexdigest(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "model_download_bytes": 0, "model_loaded": False, "vllm_imported": False,
        "gpu_operations_attempted": False,
    }
    try:
        check_no_vllm_import()
        import torch
        report["torch_version"] = torch.__version__
        report["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA was initialized before qualification")
        if hasattr(torch.ops._C_cache_ops, "swap_blocks_batch"):
            raise RuntimeError("copy operator was already registered before author library loading")
        torch.ops.load_library(str(library))
        check_no_vllm_import()
        if torch.cuda.is_initialized():
            raise RuntimeError("direct library load unexpectedly initialized CUDA")
        op = torch.ops._C_cache_ops.swap_blocks_batch.default
        report["abi"] = qualify_abi(torch, op)
        if not args.abi_only:
            report["gpu_operations_attempted"] = True
            report["gpu"] = qualify_gpu(torch, op, args.gpu_uuid)
        check_no_vllm_import()
        report["status"] = "PASSED"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0 if report["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
