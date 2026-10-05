"""Append-only CPU semantic validation, never a timing score or GPU entry."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import unittest

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = dict(preparation=HERE, candidate=args.candidate_root.resolve(strict=True))
    lock_path = HERE / "PREPARATION_SOURCE_LOCK.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["gpu_qualified"] is False and lock["gpu_launch_allowed"] is False
    def verify():
        for row in lock["files"]:
            relative = Path(row["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            path = roots[row["scope"]] / relative
            data = path.read_bytes()
            assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], str(path)
        return len(lock["files"])
    before = verify()
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        SERVER11_C5_CANDIDATE_ROOT=str(roots["candidate"]),
        SERVER11_AUTHOR_SOURCE_ROOT=str(args.author_root.resolve(strict=True)))
    path = HERE / "test_notification_runtime_preparation.py"
    spec = importlib.util.spec_from_file_location("_cpu_notification_preparation_suite", path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    after = verify()
    forbidden = [name for name in sys.modules if name == "torch" or name.startswith("torch.") or name == "vllm" or name.startswith("vllm.")]
    document = dict(status="PASS_CPU_WIRING_ONLY" if result.wasSuccessful() and not forbidden else "FAIL_CPU_WIRING",
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_files_verified_before=before, source_files_verified_after=after,
        source_lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        candidate_root=str(roots["candidate"]), author_root=str(args.author_root.resolve()),
        python=sys.version, platform=platform.platform(), forbidden_modules_imported=forbidden,
        event_origin="SYNTHETIC_CPU_METADATA", actual_vllm_install=False, actual_gpu_runs=0,
        native_execution_verified=False, gpu_qualified=False, cpu_timing_qualification=False,
        performance_claim=False, command=sys.argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "CPU_TEST_STDERR.log").write_text(stream.getvalue(), encoding="utf-8")
    (args.output_dir / "CPU_RESULT.json").write_text(json.dumps(document, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(document, sort_keys=True))
    return 0 if document["status"] == "PASS_CPU_WIRING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
