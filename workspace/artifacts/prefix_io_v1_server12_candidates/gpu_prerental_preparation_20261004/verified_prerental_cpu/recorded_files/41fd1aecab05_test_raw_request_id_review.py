"""CPU-only actual raw-shape mapping regression; never emits a GPU receipt."""
from __future__ import annotations
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
ARGS = None
NORMALIZE = None


def require(value, reason):
    if not value:
        raise ValueError(reason)


def ref(path):
    raw = path.read_bytes()
    return dict(path=path.as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def fixture():
    # Matches the immutable original execute_window result shape: external
    # request_id/run_id and native UUID-bearing IDs inside raw original frames.
    windows = []
    entries = [dict(pair_id="p" + str(index)) for index in range(3)]
    for index in range(6):
        external = "strong-p%d-%s" % (index // 2, "A" if index % 2 == 0 else "B")
        native = external + "-1234abcd"
        tokens = list(range(128))
        capture = dict(run_id=external, frames=[dict(prepared=dict(rows=[dict(request_id=native)]),
                         outputs=[[native, [token]]]) for token in tokens])
        windows.append(dict(condition="A" if index % 2 == 0 else "B", request_id=external,
            frontend=dict(output=dict(request_id=external, native_request_id=native, output_token_ids=tokens)),
            capture=capture))
    plan = dict(cells=[dict(id="cell", entries=entries)], gpu_uuid="EXPLICIT_CPU_FIXTURE_GPU",
        source_lock_ref=dict(sha256="a" * 64), model_manifest_ref=dict(sha256="b" * 64),
        kv_layout_ref=dict(sha256="c" * 64), common_runtime_domain_sha256="d" * 64)
    return dict(plan=plan, config=dict(plan_ref=dict(path="EXPLICIT_CPU_FIXTURE"))), dict(
        status="PASS_NATIVE_SIX_PROCESS_RAW_CAPTURE_REQUIRES_VALIDATION", windows=windows)


class RawRequestIDReview(unittest.TestCase):
    def test_native_capture_id_and_external_run_id_remain_distinct(self):
        gates, report = fixture()
        before = deepcopy(report)
        result = NORMALIZE(HERE, gates, report)
        for old, row in zip(report["windows"], result["cells"][0]["windows"]):
            self.assertEqual(row["run_id"], old["capture"]["run_id"])
            self.assertEqual(row["request_id"], old["frontend"]["output"]["native_request_id"])
            self.assertEqual(row["capture"], old["capture"])
        self.assertEqual(report, before)
        self.assertIs(result["production_qualified"], False)
        self.assertEqual(result["source_verification_refs"], {})

    def test_frontend_external_id_disagreement_cannot_be_packaged(self):
        gates, report = fixture()
        report["windows"][0]["frontend"]["output"]["request_id"] = "different-frontend"
        with self.assertRaises(ValueError):
            NORMALIZE(HERE, gates, report)

    def test_capture_external_run_id_disagreement_cannot_be_packaged(self):
        gates, report = fixture()
        report["windows"][0]["capture"]["run_id"] = "different-capture"
        with self.assertRaises(ValueError):
            NORMALIZE(HERE, gates, report)


def main():
    global ARGS, NORMALIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-source", type=Path, default=HERE.parent / "runner/strong_native_cost_runner.py")
    parser.add_argument("--output", type=Path, required=True)
    ARGS = parser.parse_args()
    source = ARGS.runner_source.resolve(strict=True)
    before = ref(source)
    tree = ast.parse(source.read_bytes())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "normalize_actual")
    namespace = dict(deepcopy=deepcopy, driver_module=lambda: SimpleNamespace(require=require))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
    NORMALIZE = namespace["normalize_actual"]
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RawRequestIDReview)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = ref(source)
    evidence = dict(schema="independent_cpu_raw_request_id_mapping_review_v1", status="PASS" if result.wasSuccessful() else "FAIL",
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        source_before=before, source_after=after, source_unchanged=before == after,
        actual_gpu_runs=0, RPC_calls=0, native_framework_imports=0, GPU_capabilities_issued=0,
        fixture_kind="explicit_CPU_fixture_of_immutable_original_execute_window_shape",
        real_GPU_positive_acceptance_tested=False, packaged_fixture_measurements_exported=False,
        failure_details=[dict(test=str(test), traceback=trace) for test, trace in result.failures + result.errors])
    with ARGS.output.open("xb") as stream:
        stream.write((json.dumps(evidence, indent=2, sort_keys=True) + "\n").encode())
    return 0 if result.wasSuccessful() and before == after else 1


if __name__ == "__main__":
    raise SystemExit(main())
