"""Execute real CPU tests and append an honest report; no GPU/client import."""
import argparse
import ast
import io
import json
from pathlib import Path
import platform
import sys
import unittest

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
import host_control_observer as h
import reserve_join as r
import test_host_control_observer as tests


def verify_sealed_sources():
    activation = DIRECTORY.parent
    manifest = json.loads((activation / "ACTIVATION_CORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    for row in manifest["files"]:
        ref = dict(row, path=(activation / row["path"]).as_posix())
        h.read_ref(ref)
    protocol = activation.parent / "protocol" / "prerental_protocol.py"
    h.require(h.closed_ref(protocol)["sha256"] == r.PROTOCOL_SHA, "sealed protocol changed")
    return dict(activation_core_files_checked=len(manifest["files"]), sealed_protocol_sha256=r.PROTOCOL_SHA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    h.require(not args.output.exists(), "append-only CPU result")
    before = verify_sealed_sources()
    sources = [h.closed_ref(p) for p in sorted(DIRECTORY.glob("*.py"))]
    forbidden = {"torch", "vllm", "py_kvcache", "cupy", "paramiko", "requests"}
    imported = set()
    for ref in sources:
        tree = ast.parse(h.read_ref(ref), filename=ref["path"])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(n.name.split(".")[0] for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    h.require(not imported & forbidden, "CPU observation source imports forbidden client/GPU namespace")
    output = io.StringIO()
    outcome = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    actual = tests.ObserverCPUTests().observer()
    for category in h.CATEGORIES:
        actual.call(category, 0, category, tests.metadata_boundary, category)
    observation = actual.export()
    h.require(observation["valid"], "actual local host observation invalid")
    after = verify_sealed_sources()
    h.require(before == after and all(h.closed_ref(ref["path"]) == ref for ref in sources), "source changed during CPU validation")
    h.require(not any(name.split(".")[0] in forbidden for name in sys.modules), "GPU/client namespace loaded on CPU")
    result = dict(schema="host_control_observation_CPU_validation_v1", status="PASS_CPU_ONLY" if outcome.wasSuccessful() else "FAILED_CPU_ONLY",
        tests_run=outcome.testsRun, test_failures=len(outcome.failures), test_errors=len(outcome.errors), log=output.getvalue(),
        CPU_only=True, actual_GPU_operations=0, actual_GPU_native_qualifications=0,
        positive_GPU_issuance_fixture=False, formal_goodput_allowed=False,
        natural_trace_bound=False, independent_deadline_available=False, actual_qualified_I_controller_reserve_available=False,
        actual_local_host_observation=observation, source_refs=sources, sealed_sources_verified=after,
        unsupported_shape_policy="missing native frame / empty sampled output / unknown coverage: reject; never trim data into success",
        runtime=dict(python=sys.version, platform=platform.platform(), executable=sys.executable),
        command=[sys.executable, "-B", "-I", "-S", str(Path(__file__).resolve()), "--output", str(args.output.absolute())])
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "tests_run", "test_failures", "test_errors", "actual_GPU_operations")}, sort_keys=True))
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
