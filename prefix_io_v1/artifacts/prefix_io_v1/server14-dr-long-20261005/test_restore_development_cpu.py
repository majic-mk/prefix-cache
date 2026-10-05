"""CPU fixtures only: no actual calibration, GPU run, or qualification."""
import ast
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINAL = Path(os.environ.get("DR_CPU_ORIGINAL_CONTROL", str(ROOT / "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/activation/source/prefix_io_control")))
PATCH_CONTROL = Path(os.environ.get("DR_CPU_PATCH_CONTROL", str(HERE / "control")))
ORIGINAL_NATIVE = Path(os.environ.get("DR_CPU_ORIGINAL_NATIVE", str(ROOT / "artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py")))
PATCH_NATIVE = Path(os.environ.get("DR_CPU_PATCH_NATIVE", str(HERE / "native/py_kvcache/reactor.py")))
PKG = types.ModuleType("prefix_io_control")
PKG.__path__ = [str(PATCH_CONTROL), str(ORIGINAL)]
sys.modules[PKG.__name__] = PKG
from prefix_io_control.p4_eta import ClosureGeometry, CompletedClosureMeasurement, NativeClosureHistory
from prefix_io_control.p4_restore_forecast import load_restore_calibration, RestoreCalibrationCoverageMissing
from prefix_io_control.p4_restore_order_journal import RestoreOrderJournal
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.p4_bridge import NativeP4Bridge, NativePublication
from prefix_io_control.dependencies import Parent, Resource, ResourceId
from prefix_io_control.p4_types import P4Config, WorkDescriptor, WaitingTarget, ReleaseWitness, SystemSnapshot

CAL_RUN = "cpu-fixture-calibration"
NOW = 1_000_000
TTL = 600_000_000_000
CONTEXT = ("native-ready-closure-v1", 917504, 917504, 2, 2, 0, 0, 0, 0, 0, 8,
    "scheduler-source-unknown", "scheduled-work-unknown")
GEOMETRY = ClosureGeometry((("ssd_read", 917504, 917504, 1),))


def history(count=4, *, mixed=False):
    rows = []
    for i in range(count):
        start = 100_000+i*10_000
        geo = ClosureGeometry((("h2d",917504,917504,1),)) if mixed else GEOMETRY
        rows.append(asdict(CompletedClosureMeasurement(CAL_RUN,i+1,geo,CONTEXT,
            start,start+1000+i*10,1000+i*10,i+1,None,None,0,None)))
    return dict(schema_version=1,run_id=CAL_RUN,production_qualified=False,gpu_qualified=False,
        eta_is_allocator_release=False,held_job_or_resource_owners=False,new_work_queues=0,
        minimum_samples=4,recent_measurements=json.loads(json.dumps(rows)))


class RestoreCPU(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="restore-cpu-fixture-")
        self.root = Path(self.temp.name)
        self.refs = {}
        source = self.put("fixture_source.py", {"fixture":True})
        self.sources = {source["path"]:source}
        self.probe = dict(run_id=CAL_RUN,gpu_uuid="GPU-cpu-fixture",
            common_runtime_domain_sha256="c"*64,runtime_refs=self.sources,
            subprocess_sid=123,subprocess_pid=124,original_engine_shutdown_returned=True,
            native_tail_drained=True,current_native_snapshots=[dict(p4=dict(eta_history=history()))])
        self.guard = dict(label=CAL_RUN,gpu_uuid="GPU-cpu-fixture",exit=0,child_exit=0,
            timed_out=False,interrupted_signal=None,error=None,gpu_job_attempted=True,
            session_drained=True,session_members_before_cleanup=[],session_members_after_cleanup=[],
            session_id=123)

    def tearDown(self):
        self.temp.cleanup()

    def put(self, name, value):
        data=json.dumps(value,sort_keys=True).encode()
        (self.root/name).write_bytes(data)
        row=dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        self.refs[name]=row
        return row

    def load(self, *, now_ns=NOW):
        return load_restore_calibration(self.root,self.refs,
            self.put("fixture_report.json",dict(probe=self.probe)),self.put("fixture_guard.json",self.guard),
            expected_gpu_uuid="GPU-cpu-fixture",expected_common_runtime_domain_sha256="c"*64,
            expected_runtime_refs=self.sources,now_ns=now_ns,max_age_ns=TTL)

    def test_exact_forecast_drift_and_overrun(self):
        calibration=self.load()
        actual=NativeClosureHistory("cpu-fixture-method",max_age_ns=TTL)
        actual.observe(run_id=actual.run_id,parent_id=1,geometry=GEOMETRY,context=CONTEXT,now_ns=NOW)
        forecast=calibration.predict(actual,1,GEOMETRY,CONTEXT,now_ns=NOW)
        self.assertEqual(forecast.completion_estimate_ns,NOW+1020)
        self.assertIs(forecast.production_qualified,False)
        self.assertIsNone(forecast.gpu_release_credit)
        self.assertIsNone(calibration.predict(actual,1,GEOMETRY,CONTEXT[:-1]+("changed",),now_ns=NOW))
        self.assertIsNone(calibration.predict(actual,1,GEOMETRY,CONTEXT,now_ns=NOW+1020))
        self.assertEqual(actual._pending[1].first_observed_ns,NOW)

    def test_only_coverage_is_typed(self):
        with self.assertRaises(ValueError) as expired:
            self.load(now_ns=NOW+TTL+1)
        self.assertNotIsInstance(expired.exception,RestoreCalibrationCoverageMissing)
        self.probe["current_native_snapshots"][0]["p4"]["eta_history"]=history(3)
        with self.assertRaises(RestoreCalibrationCoverageMissing) as caught:
            self.load()
        self.assertEqual(caught.exception.sample_count,3)
        self.guard["session_drained"]=False
        with self.assertRaises(ValueError) as caught:
            self.load()
        self.assertNotIsInstance(caught.exception,RestoreCalibrationCoverageMissing)

    def test_mixed_only_coverage_is_missing(self):
        self.probe["current_native_snapshots"][0]["p4"]["eta_history"]=history(4,mixed=True)
        with self.assertRaises(RestoreCalibrationCoverageMissing): self.load()

    def test_source_device_and_causal_timing_fail_closed(self):
        for mutate in (lambda:self.probe.update(gpu_uuid="GPU-other"),
                lambda:self.probe["current_native_snapshots"][0]["p4"]["eta_history"]["recent_measurements"][0].update(elapsed_ns=7),
                lambda:(self.root/"fixture_source.py").write_bytes(b"changed")):
            old=json.loads(json.dumps(self.probe)); original=(self.root/"fixture_source.py").read_bytes()
            mutate()
            with self.assertRaises(ValueError) as caught: self.load()
            self.assertNotIsInstance(caught.exception,RestoreCalibrationCoverageMissing)
            self.probe=old;(self.root/"fixture_source.py").write_bytes(original)

    def publication(self, *, h2d=False, stale=False):
        run="cpu-fixture-method"
        parent=Parent(run,1,1,0,1,False,False,2,110)
        other=Parent(run,2,1,0,1,False,False,2,None)
        rid=ResourceId(run,"restore-parent",0,1,1)
        works=(WorkDescriptor(run,1,2,0,"ssd_read",917504,0,90,False,minimum_unit_bytes=917504),
            WorkDescriptor(run,1,1,0,"ssd_read",917504,1,90,False,minimum_unit_bytes=917504))
        if h2d:
            works=works+(WorkDescriptor(run,1,3,0,"h2d",917504,2,90,False,progress="continuation",minimum_unit_bytes=917504),)
        witness=ReleaseWitness("restore:1",Resource(rid,917504,0,frozenset((1,)),"observed_blocking"),
            (parent,),"restore",(works[1].work_id,),"original-owner",110,completed_restore_parent_id=1)
        snapshot=SystemSnapshot(run,1,100,frozenset(("native_ready_work","owner_release_protocol","restore_parent_completion")),
            parents=(parent,other),targets=(WaitingTarget("restore:1",0,90,restore_parent_id=1),),
            generations=((rid,2 if stale else 1),))
        return NativePublication(snapshot,works,(witness,))

    def test_real_policy_restore_permutation_and_guards(self):
        policy=P4Policy("cpu-fixture-method",P4Config("dependency_only",1000,1000))
        bridge=NativeP4Bridge(policy)
        journal=RestoreOrderJournal(policy.run_id)
        bridge.configure_restore_development(None,history_max_age_ns=TTL,order_journal=journal)
        view=self.publication()
        choice=policy.choose(view.snapshot,view.works,view.witnesses,now_ns=100,expected_epoch=1)
        self.assertEqual(choice.action,"selected")
        self.assertEqual(choice.selected_work_ids,(view.works[1].work_id,view.works[0].work_id))
        self.assertIsNone(choice.gpu_release_credit_bytes)
        journal.record_choice(view,choice,at_ns=100)
        self.assertEqual(journal.snapshot()["actual_order_changes"][0]["dependency_parent_ids"],(1,))
        for bad in (self.publication(h2d=True),self.publication(stale=True)):
            actual=policy.choose(bad.snapshot,bad.works,bad.witnesses,now_ns=100,expected_epoch=1)
            self.assertNotEqual(actual.action,"selected")
            self.assertEqual(actual.selected_work_ids,tuple(w.work_id for w in bad.works))

    def test_setup_and_empty_calibration_do_not_publish_eta(self):
        bridge=NativeP4Bridge(P4Policy("cpu-fixture-method",P4Config("dependency_only",1000,1000)))
        bridge.configure_restore_development(None,history_max_age_ns=TTL)
        bridge.bind()
        self.assertIsNone(bridge.observe_native_closure(1,GEOMETRY,CONTEXT,now_ns=NOW))
        with self.assertRaises(ValueError): bridge.configure_restore_development(None,history_max_age_ns=TTL)
        self.assertFalse(bridge.snapshot()["production_eta_qualified"])

    def test_common_identity_binding_and_overflow(self):
        journal=RestoreOrderJournal("cpu-fixture-method",submission_limit=1)
        with self.assertRaises(ValueError):
            journal.record_native_ready_order(((1,2,"ssd_read"),),((1,2,"ssd_read"),),
                ((2,1,0,1,1,False,False),),h2d_ready_count=0,at_ns=1)
        journal.record_native_submission((1,2,"ssd_read"),user_data=7,fd=8,slot_index=0,at_ns=2)
        journal.record_native_submission((1,3,"ssd_read"),user_data=9,fd=8,slot_index=0,at_ns=3)
        self.assertFalse(journal.snapshot()["valid"])
        self.assertEqual(len(journal.snapshot()["actual_accepted_read_submission_order"]),1)

    def test_accepted_hook_calls_original_before_logging_and_exception_does_not_log(self):
        tree=ast.parse(PATCH_NATIVE.read_text(encoding="utf-8"))
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="_submit_read_from_ready")
        namespace=dict(now_ns=lambda:10,time=types.SimpleNamespace(monotonic_ns=lambda:11),
            _RingOp=lambda **kw:types.SimpleNamespace(**kw))
        exec(compile("from __future__ import annotations\n"+ast.unparse(node),"cpu-native-function-fixture","exec"),namespace)
        function=namespace[node.name]
        for fail in (False,True):
            calls=[];journal=RestoreOrderJournal("cpu-fixture-method")
            def queue(*args,**kwargs):
                calls.append("original-queue")
                if fail: raise OSError("CPU fixture enqueue failure")
            obj=types.SimpleNamespace(_slot_view=lambda _:bytearray(8),_new_user_data=lambda:17,
                _prefix_shadow_begin=lambda *a,**k:None,file_store=types.SimpleNamespace(queue_read=queue),ring=object(),
                _prefix_shadow_settle=lambda *a:None,_prefix_stage_settle=lambda *a:None,
                _prefix_stage_unknown=lambda *a:None,_prefix_stage=lambda *a:calls.append("original-accepted-stage"),
                _prefix_restore_order_journal=journal,_data_inflight=0,_inflight={})
            ready=types.SimpleNamespace(preload_info=None,preload_hash=None,fd=8,file_index=2,
                job=types.SimpleNamespace(accepted_parent_sequence=1))
            if fail:
                with self.assertRaises(OSError): function(obj,ready,0)
                self.assertEqual(journal.snapshot()["actual_accepted_read_submission_order"],())
            else:
                function(obj,ready,0)
                row=journal.snapshot()["actual_accepted_read_submission_order"][0]
                self.assertEqual(calls,["original-queue","original-accepted-stage"])
                self.assertEqual(row["work_id"],(obj._inflight[17].job.accepted_parent_sequence,
                    obj._inflight[17].file_index,"ssd_read"))
                self.assertEqual(row["user_data"],17)
            self.assertEqual(obj._data_inflight,1)  # original uncertainty owner retained too

    def test_release_and_order_functions_ast_unchanged(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        before,after=functions(ORIGINAL_NATIVE),functions(PATCH_NATIVE)
        changed={name for name in before if before[name]!=after[name]}
        self.assertEqual(changed,{"_capture_owner_snapshot","_prefix_p4_collect","_drain_ready_load_fds","_submit_read_from_ready"})
        for name in ("_prefix_p4_order","_prefix_p4_parent_terminal","_file_terminal","_queue_load_copy"):
            self.assertEqual(before[name],after[name])


if __name__=="__main__": unittest.main()
