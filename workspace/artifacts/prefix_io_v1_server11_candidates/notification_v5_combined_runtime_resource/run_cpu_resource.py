"""Run finite parser/source contracts once under an actual CPU source lock."""
import argparse
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_combined_resource_contract", HERE / "resource_preflight.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "source-lock", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--location", choices=("server_cpu", "local_cpu"), default="server_cpu")
    parser.add_argument("--old-protocol", type=Path, help="local archived path with unchanged frozen SHA")
    args = parser.parse_args(); root = args.project_root.resolve(strict=True)
    lockpath = Path(os.path.abspath(args.source_lock))
    R.require(lockpath.is_relative_to(root), "project source lock required")
    lockpath = R.safe_source(root, lockpath.relative_to(root).as_posix())
    output = args.output_dir.resolve()
    R.require(output.is_relative_to(root) and not output.exists(), "new project test output directory")
    protocol = R.closed_protocol(root, args.old_protocol)
    required = [HERE/name for name in R.FILES] + [protocol]
    before = R.verify_lock(root, lockpath, required)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    testspec = importlib.util.spec_from_file_location("_finite_combined_resource_tests", HERE / "test_resource_preflight.py")
    module = importlib.util.module_from_spec(testspec); testspec.loader.exec_module(module)
    log = io.StringIO(); result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    after = R.verify_lock(root, lockpath, required)
    R.require(before == after, "CPU source changed during contract tests")
    forbidden = sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.") or
        name == "vllm" or name.startswith("vllm.") or name == "cupy" or name.startswith("cupy."))
    okay = result.wasSuccessful() and not result.skipped and not forbidden
    document = dict(R.resource_decision(None, None, None), status="PASS_FINITE_CPU_RESOURCE_CONTRACT" if okay else "FAIL_FINITE_CPU_RESOURCE_CONTRACT",
        location=args.location, tests=result.testsRun, passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped), forbidden_imports=forbidden,
        origin="synthetic_cpu_resource_parser_and_source_contract", actual_resource_probe=False,
        source_before=before, source_after=after, source_count=before["source_count"],
        source_lock_sha256=before["source_lock_sha256"], command=[sys.executable]+sys.argv)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "CPU_RESOURCE_TESTS.log").open("x", encoding="utf-8") as stream: stream.write(log.getvalue())
    with (output / "CPU_RESOURCE_RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False); stream.write("\n")
    print(json.dumps(document, sort_keys=True))
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
