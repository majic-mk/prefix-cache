"""Run meaningful CPU fixture tests and preserve actual inputs before/after."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def refs(paths):
    return [{"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controlled-source-dir", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    paths = [here / "prerental_protocol.py", here / "test_prerental_protocol.py", Path(__file__).resolve(),
             args.source_dir / "shared_storage_trace_replay.py", args.source_dir / "prefix_cache_common.py"]
    controlled_source_dir = args.controlled_source_dir or here.parent / "source_inputs"
    paths += [controlled_source_dir / name for name in ("low_contention.json", "index.json", "P3_CONTROLLED_AND_CLOSURE_SOURCE_INPUTS.json")]
    before = refs(paths)
    load("prerental_protocol", here / "prerental_protocol.py")
    tests = load("test_prerental_protocol", here / "test_prerental_protocol.py")
    tests.SOURCE_DIR = args.source_dir
    tests.CONTROLLED_SOURCE_DIR = controlled_source_dir
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    after = refs(paths)
    forbidden = sorted(name for name in sys.modules if name.split(".")[0] in {"torch", "vllm", "openai", "py_kvcache", "numpy"})
    passed = result.wasSuccessful() and before == after and not forbidden
    report = {"schema": "gpu_prerental_protocol_actual_cpu_tests_v1", "status": "PASS_CPU_PRERENTAL_PROTOCOL" if passed else "FAIL_CPU_PRERENTAL_PROTOCOL",
              "test_origin": "actual_local_or_server_CPU_fixture_tests_not_GPU", "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
              "input_sources_before": before, "input_sources_after": after, "input_bytes_unchanged": before == after,
              "forbidden_GPU_or_client_modules_imported": forbidden, "elapsed_seconds": time.monotonic() - started,
              "actual_GPU_operations": 0, "fixture_datasets_are_not_actual_natural_workloads": True,
              "native_model_or_effect_qualification": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "tests_run": result.testsRun, "actual_GPU_operations": 0}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
