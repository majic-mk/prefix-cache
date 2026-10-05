"""Execute original patched assertion/cleanup AST with labelled CPU metadata.

No framework, model, subprocess, CUDA events or production receipt are created.
"""
import ast
from copy import deepcopy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
PREP = HERE.parent
ROOT = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


W = load("_cpu_raw_failure_diagnostics_v5", PREP / "runner/strong_native_cost_runner_v5.py")
OLD = ROOT / "artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/run_native_cost_experiment.py"
if not OLD.is_file():
    OLD = ROOT / "artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/calibration_v2/run_native_cost_experiment.py"
RAW = OLD.read_bytes()
if hashlib.sha256(RAW).hexdigest() != W.OLD_SHA:
    raise ValueError("Original immutable driver SHA mismatch")
PATCHED, PROOF = W.source_patch(RAW)
WINDOW = next(node for node in PATCHED.body if node.name == "execute_window")
OUTER = next(node for node in WINDOW.body if isinstance(node, ast.Try))
HELPER = next(node for node in WINDOW.body if isinstance(node, ast.FunctionDef) and node.name == "preserve_actual_capture")


def guarded_body():
    for node in ast.walk(OUTER):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list):
                continue
            for index, statement in enumerate(body):
                if (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                    and isinstance(statement.value.func, ast.Name) and statement.value.func.id == "require"
                    and len(statement.value.args) == 2 and isinstance(statement.value.args[1], ast.Constant)
                    and statement.value.args[1].value == "all 128 actual frames use the same correctly bound scalar adapter"):
                    return body, index
    raise AssertionError("Original strict frame assertion missing")


BODY, ASSERT_INDEX = guarded_body()
ORIGINAL_WRITE = next(node for node in ast.parse(RAW).body if isinstance(node, ast.FunctionDef) and node.name == "write_json")


def replay_namespace(*, fail_frontend=False, fail_write=False):
    # Compile actual generated helper + assertion + actual except/finally prefix.
    # Only surrounding model setup is replaced by clearly labelled CPU inputs.
    args = ast.parse("def replay(active_capture,scalar_adapter,frontend,out,refs,result,rid): pass").body[0]
    selected = deepcopy(BODY[ASSERT_INDEX - 2:ASSERT_INDEX + 1])
    if fail_frontend:
        selected.insert(0, ast.parse("frontend=CPU_frontend_failure()").body[0])
    protected = ast.Try(body=selected, handlers=deepcopy(OUTER.handlers), orelse=[], finalbody=deepcopy(OUTER.finalbody[:1]))
    args.body = [ast.parse("capture=None").body[0], deepcopy(HELPER), protected,
                 ast.Return(value=ast.Name(id="result", ctx=ast.Load()))]
    namespace = dict(dataclasses=dataclasses, traceback=traceback, json=json, REACTOR="CPU-fixture/native/reactor.py",
                     __name__="CPU_failure_diagnostic_AST_replay")
    def require(ok, reason):
        if not ok:
            raise ValueError(reason)
    def CPU_frontend_failure():
        raise RuntimeError("CPU-labelled frontend failure")
    namespace.update(require=require, CPU_frontend_failure=CPU_frontend_failure)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(ORIGINAL_WRITE), args], type_ignores=[])),
                 "CPU_original_assertion_cleanup_AST", "exec", dont_inherit=True), namespace)
    if fail_write:
        def rejected_write(*unused):
            raise OSError("CPU-labelled diagnostic filesystem failure")
        namespace["write_json"] = rejected_write
    return namespace


@dataclasses.dataclass(frozen=True)
class CPUFrame:
    native_step_ordinal: int
    prepared: dict
    CPU_fixture_only: bool = True
    production_qualified: bool = False


class CPUCapture:
    def __init__(self, *, count=1, export_error=False):
        self.frames = [CPUFrame(index, dict(pre_context=511+index, scheduled_tokens=1)) for index in range(count)]
        self.adapter = NS(frames=tuple(self.frames), native_source_sha256="a"*64, valid=False,
            enabled=True, last_reason="CPU-zero-work-prepared-failure", _pending=dict(ordinal=700,
                phase="begun", start_ns=20, frame=None))
        self.observer = NS(enabled=False,status="observation_invalid",reason="CPU-scalar_execute_invalid",
            scalar=NS(_adapter=self.adapter, enabled=False,status="observation_invalid",
                      last_reason="CPU-scalar_execute_invalid"),
            events=NS(valid=False,reason="CPU-scalar_execute_invalid",pending=[],active=None))
        self.valid = False
        self.failures = ["CPU-labelled real invalid export shape"]
        self.export_error = export_error
        self.export_count = self.detach_count = 0
        self.out = None
        self.files_seen_at_detach = []
    def export(self):
        self.export_count += 1
        if self.export_error:
            raise RuntimeError("CPU-labelled export failure")
        return dict(origin="cpu_fixture",valid=False,frames=[dataclasses.asdict(frame) for frame in self.frames],
            failures=self.failures,actual_GPU_events=0,production_qualified=False)
    def detach(self):
        self.files_seen_at_detach = sorted(path.name for path in self.out.glob("*.json"))
        self.detach_count += 1


def execute_cpu(capture, *, initial_adapter=None, fail_frontend=False, fail_write=False):
    with tempfile.TemporaryDirectory(prefix="CPU-only-failed-capture-") as temp:
        out = Path(temp)
        capture.out = out
        frontend = dict(origin="cpu_fixture", output=dict(num_cached_tokens=512, output_token_ids=list(range(128))),
                        actual_GPU_events=0, production_qualified=False)
        result = dict(status="CPU_FIXTURE_FAILURE_REPLAY_NOT_GPU_QUALIFIED", production_qualified=False)
        namespace = replay_namespace(fail_frontend=fail_frontend,fail_write=fail_write)
        result = namespace["replay"](capture,initial_adapter or capture.adapter,frontend,out,
                                     {"CPU-fixture/native/reactor.py":dict(sha256="a"*64)},result,"CPU-fixture-run")
        rows = {path.name:json.loads(path.read_bytes()) for path in out.glob("*.json")}
        return result, rows


class FailureDiagnosticsCPU(unittest.TestCase):
    def test_strict_failure_export_is_saved_before_original_detach(self):
        capture = CPUCapture(count=1)
        result, rows = execute_cpu(capture)
        self.assertEqual(result["error_message"],"all 128 actual frames use the same correctly bound scalar adapter")
        self.assertEqual(capture.export_count,1) # No extra event query after actual export.
        self.assertEqual(capture.detach_count,1)
        self.assertEqual(len(capture.files_seen_at_detach),2)
        for row in rows.values():
            self.assertFalse(row["production_qualified"])
            self.assertEqual(row["actual_capture_export"]["origin"],"cpu_fixture")
            self.assertEqual(row["actual_frontend"]["output"]["num_cached_tokens"],512)
            self.assertEqual(row["scalar_adapter"]["actual_frame_count"],1)
            self.assertEqual(row["observer_stats"]["scalar_adapter_last_reason"],"CPU-zero-work-prepared-failure")
            self.assertEqual(row["observer_stats"]["event_last_reason"],"CPU-scalar_execute_invalid")
            self.assertEqual(row["actual_prepared_pending"]["phase"],"begun")
    def test_identity_and_source_failure_preserve_current_and_initial_adapter(self):
        capture = CPUCapture(count=128)
        initial = NS(native_source_sha256="b"*64)
        result, rows = execute_cpu(capture,initial_adapter=initial)
        self.assertEqual(result["error_type"],"ValueError")
        row=rows["CPU-fixture-run-preassert-capture-diagnostic.json"]
        self.assertFalse(row["scalar_adapter"]["same_object"])
        self.assertEqual(row["scalar_adapter"]["initial_native_source_sha256"],"b"*64)
        self.assertEqual(row["scalar_adapter"]["current_native_source_sha256"],"a"*64)
    def test_export_exception_still_saves_prepared_failure_and_detaches(self):
        capture = CPUCapture(export_error=True)
        result, rows = execute_cpu(capture)
        self.assertEqual(result["error_message"],"CPU-labelled export failure")
        row=rows["CPU-fixture-run-finally-capture-diagnostic.json"]
        self.assertNotIn("actual_capture_export",row)
        self.assertIn("CPU-labelled export failure",row["diagnostic_errors"][0])
        self.assertEqual(row["scalar_adapter"]["frames"][0]["prepared"]["pre_context"],511)
        self.assertEqual(capture.detach_count,1)
    def test_frontend_or_filesystem_failure_preserves_original_exception_cleanup(self):
        for frontend_failure, write_failure in ((True,False),(False,True)):
            capture=CPUCapture()
            result, rows=execute_cpu(capture,fail_frontend=frontend_failure,fail_write=write_failure)
            self.assertEqual(capture.detach_count,1)
            if frontend_failure:
                self.assertEqual(result["error_message"],"CPU-labelled frontend failure")
                self.assertIn("CPU-fixture-run-finally-capture-diagnostic.json",rows)
            else:
                self.assertIn("capture_diagnostic_write_errors",result)
                self.assertEqual(result["error_type"],"ValueError")
    def test_only_source_patch_changed_and_original_shutdown_assertion_retained(self):
        old=ast.parse((PREP/"runner/strong_native_cost_runner_v4.py").read_bytes())
        new=ast.parse((PREP/"runner/strong_native_cost_runner_v5.py").read_bytes())
        old_nodes={node.name:ast.dump(node,include_attributes=False) for node in old.body if isinstance(node,ast.FunctionDef)}
        new_nodes={node.name:ast.dump(node,include_attributes=False) for node in new.body if isinstance(node,ast.FunctionDef)}
        self.assertEqual([name for name in old_nodes if old_nodes[name]!=new_nodes[name]],["source_patch"])
        self.assertEqual(hashlib.sha256((PREP/"runner/strong_native_cost_runner_v4.py").read_bytes()).hexdigest(),
                         "3f97053f9638a9e58faf8cfa3f560762e294015111b1c92e8b88a379dbc9193a")
        self.assertEqual(BODY[ASSERT_INDEX-1].value.func.id,"preserve_actual_capture")
        self.assertIn("engine_core.shutdown(timeout=15)",ast.unparse(OUTER.finalbody))
        self.assertEqual(PROOF["estimator_modified"],False)
        self.assertFalse(PROOF["GPU_qualification_issued"])
        for stem,sha in (("bounded_native_full_step_collector.py","9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"),
                         ("strong_native_cost_runner_v3.py","d0eca2e77dc1fb2cab5f95fac7f31a41978ea5c560f45f1ead8a8e1b79166240")):
            self.assertEqual(hashlib.sha256((PREP/"runner"/stem).read_bytes()).hexdigest(),sha)


if __name__=="__main__":
    unittest.main(verbosity=2)
