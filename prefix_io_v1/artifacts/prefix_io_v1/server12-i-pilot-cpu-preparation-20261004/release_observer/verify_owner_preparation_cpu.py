"""Record CPU test execution and SHA-bound native field/method inputs."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
G_LOCAL = PROJECT / "artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate/source/third_party/work"
G_SERVER = PROJECT / "artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/source/third_party/work"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_source(filename, suffix, expected):
    candidates = (HERE.parent / "source_inputs" / filename, G_LOCAL / suffix, G_SERVER / suffix)
    chosen = next((p for p in candidates if p.is_file()), None)
    if chosen is None or chosen.is_symlink() or sha(chosen) != expected:
        raise ValueError("missing or mismatched actual original source: " + filename)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = HERE.parent / "source_inputs"
    entries = json.loads((sources / "SOURCE_INPUTS.json").read_text(encoding="utf-8"))
    extra = json.loads((sources / "KV_OFFLOAD_SOURCE_INPUT.json").read_text(encoding="utf-8"))
    entries.extend([extra] if isinstance(extra, dict) else extra)
    paths = []
    for entry in entries:
        filename = entry["local_path"].replace("\\", "/").rsplit("/", 1)[-1]
        path = sources / filename
        if path.is_symlink() or sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise ValueError("actual server input hash mismatch: " + filename)
        paths.append(path)
    paths.extend((
        bound_source("native_flush_probe.py", "prefix-io-p4-02-cpu/src/prefix_io_control/native_flush_probe.py",
                     "b24902ca76aa998c564218738ec02cfc3c96f5b12523a2df80765a489c04ebda"),
        bound_source("py_kvcache_handler.py", "py-kvcache-p4-02-cpu/py_kvcache/vllm.py",
                     "901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1"),
        bound_source("py_kvcache_reactor.py", "py-kvcache-p4-02-cpu/py_kvcache/reactor.py",
                     "a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47"),
    ))
    before = {str(p): sha(p) for p in paths}
    binding_rows = []
    names = {"KVCacheBlock", "get_new_blocks", "free_blocks", "request_finished", "_build_store_jobs",
             "_remove_pending_job", "update_connector_output", "_observe_prefix_flush", "get_finished",
             "wait", "_future_to_transfer_result", "_drain_cuda_copies", "_file_terminal", "_finish_jobs",
             "install", "uninstall", "state", "completed"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                binding_rows.append(dict(source=str(path), source_sha256=before[str(path)],
                                         symbol=node.name, line=node.lineno, end_line=node.end_lineno,
                                         ast_sha256=hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()))
    command = [sys.executable, "-B", "-I", "-S", "-m", "unittest", "discover", "-s", str(HERE),
               "-p", "test_owner_release_adapter.py", "-v"]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT, env=env, text=True, encoding="utf-8", capture_output=True, timeout=60)
    elapsed = time.monotonic() - started
    after = {str(p): sha(p) for p in paths}
    record = dict(status="PASS" if result.returncode == 0 and before == after else "FAIL",
                  command=command, exit_code=result.returncode, elapsed_seconds=elapsed,
                  stdout=result.stdout, stderr=result.stderr,
                  input_files=len(paths), inputs_before=before, inputs_after=after,
                  inputs_byte_unchanged=before == after, source_bindings=binding_rows,
                  gpu_jobs=0, production_release_capability=False,
                  actual_gpu_owner_release_observed=False,
                  scope="CPU preparation, actual source bindings and exact original scheduler CPU replays only")
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({k: record[k] for k in ("status", "exit_code", "input_files", "inputs_byte_unchanged", "gpu_jobs")}, ensure_ascii=False))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
