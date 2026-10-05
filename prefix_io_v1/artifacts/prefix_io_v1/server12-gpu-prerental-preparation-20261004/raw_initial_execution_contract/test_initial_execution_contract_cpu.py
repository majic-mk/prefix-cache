"""CPU replay of existing real metadata and original fail-fast parent AST.

No GPU process, model, CUDA event, estimator or private qualification is run.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace as NS
import unittest

HERE=Path(__file__).resolve().parent
PREP=HERE.parent
ROOT=HERE.parents[3]
ACTUAL_DEFAULT=ROOT/"artifacts/prefix_io_v1_server12_candidates/strong_gpu_system_verification_20261005/CAL04_COMPLETED_FIRST_WINDOW_NATIVE_RESULT.json"
ACTUAL=ACTUAL_DEFAULT
SECOND=ROOT/"artifacts/prefix_io_v1_server12_candidates/strong_gpu_system_verification_20261005/CAL04_COMPLETED_SECOND_WINDOW_NATIVE_RESULT.json"
if "--actual-child" in sys.argv:
    index=sys.argv.index("--actual-child")
    ACTUAL=Path(sys.argv[index+1]).resolve()
    del sys.argv[index:index+2]
if "--actual-second" in sys.argv:
    index=sys.argv.index("--actual-second")
    SECOND=Path(sys.argv[index+1]).resolve()
    del sys.argv[index:index+2]
ACTUAL_SHA="a8e1bba92e987335967760c0384be36dcf5d5e483eaf8c5562aea52512808bdd"


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


W=load("_CPU_raw_initial_execution_v7",PREP/"runner/strong_native_cost_runner_v7.py")
R=W.driver_module()
VALIDATION=PREP/"activation/native_conditional_cost.py"
assert hashlib.sha256(VALIDATION.read_bytes()).hexdigest()=="675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"
V=load("_CPU_original_validation_initial496",VALIDATION)
raw=ACTUAL.read_bytes()
if hashlib.sha256(raw).hexdigest()!=ACTUAL_SHA:
    raise ValueError("Existing actual cal04 child bytes differ; no fabricated replacement accepted")
REPORT=json.loads(raw)
second_raw=SECOND.read_bytes()
if hashlib.sha256(second_raw).hexdigest()!="c2b90512467b001b6ba3440c77f329745dabf8e26c391ce42e776f5327b6d05a":
    raise ValueError("Existing actual cal04 B bytes differ; no fabricated replacement accepted")
SECOND_REPORT=json.loads(second_raw)
row_actual=REPORT["windows"][0]
for name in ("native_journal","native_post_shutdown","native_tail_assertions","subprocess_pid","private_storage"):
    row_actual[name]=REPORT[name]
PLAN=dict(cells=[W.cell_descriptor()],journal_run_id=REPORT["label"],
          native_source_ref=dict(sha256=REPORT["native_journal"]["source_sha256"]),
          validation_source_ref=R.ref(ROOT,VALIDATION.relative_to(ROOT).as_posix()))
TREE=ast.parse((PREP/"runner/strong_native_cost_runner_v7.py").read_bytes())
SETUP=next(node for node in TREE.body if isinstance(node,ast.FunctionDef) and node.name=="configure_original")
CALLBACK=next(node for node in SETUP.body if isinstance(node,ast.FunctionDef) and node.name=="strict_completed_window")
namespace=dict(R=R,root=ROOT,plan=PLAN,validation=V,CACHED_TOKENS=512)
exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(CALLBACK)],type_ignores=[])),
             "CPU_actual_original_completed_window_callback_AST","exec",dont_inherit=True),namespace)
STRICT=namespace["strict_completed_window"]
OLD=ROOT/"artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/run_native_cost_experiment.py"
if not OLD.is_file():
    OLD=ROOT/"artifacts/prefix_io_v1_server12_candidates/i_pilot_cpu_preparation_20261004/calibration_v2/run_native_cost_experiment.py"
RAW=OLD.read_bytes()
PATCHED,PROOF=W.source_patch(RAW)
PARENT=next(node for node in PATCHED.body if node.name=="execute_parent")


class CPUStopBeforeNextChild(BaseException):
    """Explicit CPU refusal; no second process or positive GPU receipt exists."""


def capture_check(cached):
    output=row_actual["frontend"]["output"]
    return V.validate_capture(row_actual["capture"],run_id=row_actual["request_id"],
        request_id=output["native_request_id"],output_ids=output["output_token_ids"],
        prompt_tokens=512,measured_offset=16,warmup_offsets=[1],cached_tokens=cached)


def parent_replay(*, mutation=None, exit_code=0, expect_second_refusal=False):
    """Run the actual patched parent; subprocess.run is a labelled CPU refusal stub."""
    calls=[]; validated=[]; metadata=[]
    with tempfile.TemporaryDirectory(prefix="CPU-only-no-model-failfast-") as temp:
        root=Path(temp).resolve()
        config=dict(out="CPU-fixture/run/details",gpu_uuid="CPU_FIXTURE_NOT_GPU_AUTHORITY",
                    gpu_entry_binding_ref=dict(origin="CPU_fixture_only_not_GPU_authority"))
        (root/config["out"]).parent.mkdir(parents=True)
        original=deepcopy(REPORT)
        if mutation is not None:
            mutation(original)
        def CPU_refuse_subprocess(command,**kwargs):
            calls.append(dict(command=list(command),CPU_fixture_only=True))
            if len(calls)>1:
                raise CPUStopBeforeNextChild("CPU test refuses second child; zero actual model/GPU processes")
            return NS(returncode=exit_code,CPU_fixture_only=True,actual_process_started=False)
        def callback(row,index):
            validated.append(index)
            return STRICT(row,index)
        sid=original["subprocess_sid"]
        scope=dict(__name__="CPU_patched_parent_existing_metadata_replay",CONFIG_PATH=root/"CPU-fixture/CONFIG.json",
            SCRIPT="runner/strong_native_cost_runner_v7.py",DELIVERY="CPU-unused",
            PURPOSE="CPU_ONLY_NOT_GPU_QUALIFICATION",LABEL="CPU-only-failfast",
            verify_execution_inputs=lambda *args:dict(CPU_fixture_only=True),safe=R.safe,require=R.require,
            input_groups=lambda *args:None,write_json=lambda path,value:metadata.append((str(path),value)),
            read_json=lambda path:deepcopy(original),file_ref=lambda *args:dict(CPU_fixture_only=True),
            _strict_completed_window=callback,os=NS(environ={},getsid=lambda _:sid,getpgid=lambda _:sid),
            sys=sys,time=time,subprocess=NS(run=CPU_refuse_subprocess),traceback=traceback)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(PARENT)],type_ignores=[])),
                     "CPU_actual_parent_failfast_AST","exec",dont_inherit=True),scope)
        if expect_second_refusal:
            with unittest.TestCase().assertRaises(CPUStopBeforeNextChild):
                scope["execute_parent"](root,config,{},dict(CPU_fixture_only=True))
            result=None
        else:
            result=scope["execute_parent"](root,config,{},dict(CPU_fixture_only=True))
        return result,calls,validated


class InitialExecutionContractCPU(unittest.TestCase):
    def test_existing_actual_B_capture_and_strict_action_io_pass496_reject511(self):
        row=deepcopy(SECOND_REPORT["windows"][0])
        for name in ("native_journal","native_post_shutdown","native_tail_assertions","subprocess_pid","private_storage"):
            row[name]=SECOND_REPORT[name]
        output=row["frontend"]["output"]
        with self.assertRaisesRegex(ValueError,"actual complete cold/decode load differs"):
            V.validate_capture(row["capture"],run_id=row["request_id"],request_id=output["native_request_id"],
                output_ids=output["output_token_ids"],prompt_tokens=512,measured_offset=16,warmup_offsets=[1],cached_tokens=511)
        self.assertTrue(STRICT(row,1))
        self.assertEqual(row["strict_completed_window_validation"]["actual_frame_count"],128)
        self.assertEqual(row["capture"]["frames"][0]["prepared"]["prefill_tokens"],16)
        self.assertFalse(row["strict_completed_window_validation"]["production_qualified"])
    def test_existing_actual_capture_rejects511_passes496_all128(self):
        with self.assertRaisesRegex(ValueError,"actual complete cold/decode load differs"):
            capture_check(511)
        self.assertEqual(len(capture_check(496)),128)
        self.assertEqual(row_actual["capture"]["frames"][0]["prepared"]["prefill_tokens"],16)
        self.assertEqual(row_actual["capture"]["frames"][16]["prepared"]["context_length"],527)
    def test_actual_callback_requires_complete_original_native_io_and_tail(self):
        row=deepcopy(row_actual)
        self.assertTrue(STRICT(row,0))
        evidence=row["strict_completed_window_validation"]
        self.assertEqual(evidence["actual_frame_count"],128)
        self.assertEqual(evidence["initial_execution_pre_context"],496)
        self.assertEqual(evidence["frontend_cached_prompt_tokens"],512)
        self.assertFalse(evidence["production_qualified"])
        self.assertFalse(evidence["estimator_run"])
    def test_closed_first_window_gate_runs_before_second_child_refusal(self):
        result,calls,validated=parent_replay(expect_second_refusal=True)
        self.assertIsNone(result)
        self.assertEqual(validated,[0])
        self.assertEqual(len(calls),2) # both are CPU stubs, second always refuses
    def test_failed_child_exit_or_receipt_stops_before_second_child(self):
        mutations=(None,lambda report:report.update(original_engine_shutdown_returned=False))
        for mutation in mutations:
            result,calls,validated=parent_replay(mutation=mutation,exit_code=7 if mutation is None else 0)
            self.assertEqual(len(calls),1)
            self.assertEqual(validated,[])
            self.assertEqual(result["status"],"FAILED_NATIVE_SIX_PROCESS_DIAGNOSTIC")
            self.assertFalse(result["production_qualified"])
    def test_bad_capture_native_tail_and_io_stop_after_first_closed_window(self):
        def wrong_capture(report):
            report["windows"][0]["capture"]["frames"][0]["prepared"]["context_length"]=511
        def bad_tail(report):
            report["native_tail_assertions"]["worker_alive"]=True
        def bad_io(report):
            report["native_journal"]["valid"]=False
        for mutation in (wrong_capture,bad_tail,bad_io):
            result,calls,validated=parent_replay(mutation=mutation)
            self.assertEqual(len(calls),1)
            self.assertEqual(validated,[0])
            self.assertEqual(result["status"],"FAILED_NATIVE_SIX_PROCESS_DIAGNOSTIC")
            self.assertNotIn("strict_completed_window_validation",result["windows"][0])
    def test_exact_metadata_and_source_change_scope(self):
        old=ast.parse((PREP/"runner/strong_native_cost_runner_v6.py").read_bytes())
        previous={node.name:ast.dump(node,include_attributes=False) for node in old.body if isinstance(node,ast.FunctionDef)}
        current={node.name:ast.dump(node,include_attributes=False) for node in TREE.body if isinstance(node,ast.FunctionDef)}
        self.assertEqual([name for name in previous if previous[name]!=current[name]],["source_patch","configure_original"])
        descriptor=W.cell_descriptor()
        self.assertEqual(descriptor["cached_prompt_tokens"],496)
        self.assertEqual(descriptor["initial_execution_pre_context"],496)
        self.assertEqual(descriptor["frontend_cached_prompt_tokens"],512)
        self.assertEqual(descriptor["measured_offset"],16)
        self.assertEqual(descriptor["warmup_offsets"],[1])
        self.assertEqual([entry["arm_order"] for entry in descriptor["entries"]],["AB","BA","AB"])
        self.assertEqual([entry["split"] for entry in descriptor["entries"]],["calibration","calibration","validation"])
        self.assertEqual(PROOF["initial_execution_pre_context"],496)
        self.assertEqual(PROOF["initial_scheduled_tokens"],16)
        self.assertFalse(PROOF["estimator_modified"])
        self.assertFalse(PROOF["GPU_qualification_issued"])
    def test_previous_sources_validator_collector_and_parent_gate_order_preserved(self):
        pins={"runner/strong_native_cost_runner_v6.py":"be0660712e503cdf496cab6fa7aa48ce206aca78d944803f4862e0f67a540ca7",
            "runner/bounded_native_full_step_collector_v2.py":"a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0",
            "runner/bounded_native_full_step_collector.py":"9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d",
            "activation/native_conditional_cost.py":"675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8"}
        for path,sha in pins.items():
            self.assertEqual(hashlib.sha256((PREP/path).read_bytes()).hexdigest(),sha)
        loop=next(node for node in ast.walk(PARENT) if isinstance(node,ast.For) and isinstance(node.target,ast.Name)
                  and node.target.id=="index" and isinstance(node.iter,ast.Call) and node.iter.func.id=="range"
                  and node.iter.args[0].value==6)
        self.assertEqual(ast.unparse(loop.body[-2]),"rows.append(row)")
        self.assertEqual(ast.unparse(loop.body[-1]),"_strict_completed_window(row, index)")


if __name__=="__main__":
    unittest.main(verbosity=2)
