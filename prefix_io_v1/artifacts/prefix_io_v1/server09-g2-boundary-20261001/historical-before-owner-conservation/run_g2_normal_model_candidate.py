"""Read-only CPU source preflight; no native/GPU run is implemented or enabled."""
import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_REFS = (
 ("experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py", 21990,
  "7d31cce851ca2334fedbe6b66c4e2d9bfff19e8768bafaa52f68e414518d77e2"),
 ("experiments/prefix_io_v1/scripts/run_decode_interference_calibration_p316.py", 19415,
  "aa43af9b2b6a173b66eb2512de86e60a749624cebd47f98e8a6ee43b3aa45db5"),
 ("third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_native_window_journal.py", 9647,
  "3a9ded39846cccf88ee9aceed2e16b86d8a7bbe953adfea3ba1a302b2c5aa8c3"),
 ("third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/stage_accounting.py", 3580,
  "11a736586a9278e2c373d4fc1bfe0faf9cd166b81e9d9c3f658500a51ab954aa"),
 ("third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py", 169906,
  "2b0ac1e8e68f82adb42ade03b74b49843eabe55361a00f0ac1f7a738ac423e18"),
 ("third_party/work/py-kvcache-p4-02-cpu/py_kvcache/vllm.py", 33693,
  "901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1"),
)


def cpu_preflight(source_root):
    root = Path(source_root).resolve()
    checked = []
    for relative, nbytes, digest in EXPECTED_REFS:
        path = root / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size != nbytes:
            raise ValueError("source missing/bytes differ: " + relative)
        with path.open("rb") as source:
            data = source.read(nbytes + 1)
        if len(data) != nbytes or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("source SHA differs: " + relative)
        checked.append(dict(path=relative, bytes=nbytes, sha256=digest))
    return dict(status="CPU_BOUNDARY_SOURCE_PREFLIGHT_ONLY", checked_sources=checked,
        source_imported=False, cache_created=False, gpu_initialized=False,
        model_bytes_verified=False, actual_worker_provenance_verified=False,
        runtime_hook_status="not_installed", GPU_collector_verified=False,
        production_qualified=False, executable_native_G2=False,
        blockers=["No original normal-model worker entry is connected by this skeleton.",
            "Real original GPUModelRunner/decorator/runtime/binary/model refs still require G2 freeze.",
            "Real full-step event and cross-clock qualification is not supplied by CPU fake events.",
            "Real NativeWindowJournal injection and post-original-shutdown snapshot provider not installed.",
            "G2 native run needs separate frozen scope and explicit user GPU budget; G1 does not cover it."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--run", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.run:
        # Before source/native/model import, CUDA probing, cache/output mkdir.
        print(json.dumps(dict(status="BLOCKED_NATIVE_G2_NOT_IMPLEMENTED_OR_QUALIFIED",
            gpu_initialized=False, runtime_hook_status="not_installed",
            GPU_collector_verified=False, production_qualified=False)))
        return 2
    if args.source_root is None:
        parser.error("--preflight requires --source-root")
    try:
        print(json.dumps(cpu_preflight(args.source_root)))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status="CPU_SOURCE_PREFLIGHT_REJECTED",
            error=type(exc).__name__ + ": " + str(exc)[:240], gpu_initialized=False)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
