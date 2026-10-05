"""Source/runtime audit; CPU-only io_uring and 4 KiB O_DIRECT probes."""
from pathlib import Path
import ast, hashlib, importlib.util, json, mmap, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/prefix_io_v1/p0"
LOCKS = ROOT / "experiments/prefix_io_v1/locks"
def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()
def main():
    global OUT, LOCKS
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="fresh project artifact directory for a repeated audit")
    args = parser.parse_args()
    if args.output is not None:
        OUT = args.output.resolve()
        if not OUT.is_relative_to(ROOT / "artifacts/prefix_io_v1"):
            raise ValueError("audit output must stay in project artifacts/prefix_io_v1")
        LOCKS = OUT
    if (OUT / "capability-report.json").exists():
        raise RuntimeError("audit evidence exists; choose a fresh --output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    repositories = {}
    for name in ("py-kvcache", "vllm-author", "kvcache-experiments", "simple-profiler"):
        path = ROOT / "third_party/upstream" / name
        repositories[name] = {"path": str(path), "commit": git(path, "rev-parse", "HEAD"),
                              "dirty": git(path, "status", "--porcelain"), "installed": False}
    v = ROOT / "third_party/upstream/vllm-author"
    sources = [
      "py-kvcache/py_kvcache/reactor.py", "py-kvcache/py_kvcache/vllm.py",
      "py-kvcache/py_kvcache/transfer.py", "py-kvcache/py_kvcache/staging.py",
      "py-kvcache/py_kvcache/load_planner.py", "py-kvcache/py_kvcache/liburing_file.py",
      "vllm-author/vllm/v1/kv_offload/base.py",
      "vllm-author/vllm/v1/kv_offload/worker/worker.py",
      "vllm-author/vllm/v1/core/block_pool.py",
      "vllm-author/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py",
      "vllm-author/vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py"]
    locations = {}
    for rel in sources:
        path = ROOT / "third_party/upstream" / rel
        locations[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "symbols": {n.name: n.lineno for n in ast.walk(ast.parse(path.read_text()))
                           if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}}
    old = subprocess.check_output(["git", "-C", str(v), "show",
        "d6eadf416bb5234047760bf55d532f2f038cf697:vllm/v1/kv_offload/base.py"], text=True)
    current = (v / "vllm/v1/kv_offload/base.py").read_text()
    plan = {key: {"original_candidate": ("class " + key) in old,
                  "selected_author_commit": ("class " + key) in current}
            for key in ("PlanCandidate", "PlanDecision", "PlanOutcome")}
    sys.path.insert(0, str(ROOT / "third_party/upstream/py-kvcache"))
    import py_kvcache.vllm as adapter
    from py_kvcache.transfer import TORCH_COPY_AVAILABLE
    from py_kvcache.liburing_file import LiburingRing
    capabilities = {"source_plan_api": plan, "runtime": {
        "VLLM_AVAILABLE": adapter.VLLM_AVAILABLE, "PLAN_API_AVAILABLE": adapter.PLAN_API_AVAILABLE,
        "TORCH_COPY_AVAILABLE": TORCH_COPY_AVAILABLE, "py_kvcache_import_path": adapter.__file__,
        "vllm_installed": importlib.util.find_spec("vllm") is not None},
        "gpu_verified": False, "real_handler_verified": False,
        "independent_source_safe_ack": False,
        "source_fence_release_scope": "all protecting parent jobs; active refcount separate",
        "mandatory_progress_bridge_verified": False, "native_layerwise_consume_verified": False}
    try:
        ring = LiburingRing(2); ring.close()
        capabilities["io_uring"] = {"available": True}
    except OSError as exc:
        capabilities["io_uring"] = {"available": False, "errno": exc.errno, "error": str(exc)}
    probe = OUT / "odirect-probe.bin"
    try:
        with mmap.mmap(-1, 4096) as data:
            data[:] = bytes(range(256)) * 16
            fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
            try:
                written = os.pwrite(fd, data, 0)
                with mmap.mmap(-1, 4096) as loaded:
                    read = os.preadv(fd, [loaded], 0)
                    equal = data[:] == loaded[:]
            finally:
                os.close(fd)
        capabilities["o_direct"] = {"available": written == read == 4096 and equal,
                                    "bytes_written": written, "bytes_read": read}
    except OSError as exc:
        capabilities["o_direct"] = {"available": False, "errno": exc.errno, "error": str(exc)}
    lock = {"schema_version": 1, "status": "source_locked_runtime_not_qualified",
        "compatibility_verified": False, "gpu_verified": False, "repositories": repositories,
        "user_project_head": git(ROOT, "rev-parse", "HEAD"),
        "original_vllm_candidate": "d6eadf416bb5234047760bf55d532f2f038cf697",
        "vllm_selection_reason": "author with_profiling contains existing planned-defer API and call sites",
        "liburing_version": None, "liburing_implementation": "raw io_uring syscalls via ctypes",
        "common_patches": [], "observer_patches": [], "policy_patches": []}
    for file, value in ((OUT/"source-symbols.json", locations),
                        (OUT/"capability-report.json", capabilities),
                        (LOCKS/"dependency-lock.json", lock)):
        file.write_text(json.dumps(value, indent=2) + "\n")
    print(json.dumps(capabilities, indent=2))
if __name__ == "__main__":
    main()
