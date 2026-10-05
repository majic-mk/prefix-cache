"""Independent pure-value ABI cases, not backend/GPU or performance evidence."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import sys
import threading
import time
import unittest
import xml.etree.ElementTree as ET

# Standalone qualification forbids even backend imports. A shared suite may
# install its own guard; this module does not mutate shared-suite import policy.
FORBIDDEN_IMPORTS=[]
if __name__=="__main__":
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-01-cpu/src"))
    class ImportGuard:
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split(".")[0] in ("torch","vllm","py_kvcache","cupy","cuda","numpy"):
                FORBIDDEN_IMPORTS.append(fullname)
                raise RuntimeError("backend import forbidden in pure bridge ABI test: "+fullname)
    sys.meta_path.insert(0,ImportGuard())

from prefix_io_control.dependencies import Parent, Resource, ResourceId
from prefix_io_control.dispatch_budget import ZERO
from prefix_io_control.dispatch_shadow import ShadowState
from prefix_io_control.p4_types import (
    P4Config,SystemSnapshot,WorkDescriptor,ReleaseWitness,WaitingTarget,
)
from prefix_io_control.p4_bridge import make_native_bridge,NativePublication
from prefix_io_control.p4_cost_table import CostTable

CAPS=frozenset(("native_ready_work","owner_release_protocol","cpu_owner_generation",
                "cpu_active_refs","cpu_protectors"))
NOW=100

class IndependentValueABITests(unittest.TestCase):
    def setUp(self):
        self.config=P4Config("dependency_only",100,1000)
        self.bridge=make_native_bridge("r",self.config)
        self.bridge.bind()
        self.epoch=self.bridge.epoch(("actual-owner-values",1),NOW)
        self.p1=Parent("r",1,2,0,0,False,False,2,None)
        self.p2=Parent("r",2,2,0,0,False,False,2,None)
        self.rid=ResourceId("r","cpu-staging",0,3,1)
        self.w1=WorkDescriptor("r",self.epoch,1,0,"ssd_write",8,1,90,False,
                               minimum_unit_bytes=8)
        self.w2=WorkDescriptor("r",self.epoch,2,0,"ssd_write",8,0,90,False,
                               minimum_unit_bytes=8)
        target=WaitingTarget("cpu-wait",0,90,need_cpu_bytes=16)
        snapshot=SystemSnapshot("r",self.epoch,NOW,CAPS,(self.p1,self.p2),
            ShadowState("r",NOW,ZERO,0,2,True),(target,),((self.rid,1),),None,0)
        resource=Resource(self.rid,16,0,frozenset((1,)),"observed_blocking")
        witness=ReleaseWitness("cpu-wait",resource,(self.p1,),"cpu",(self.w1.work_id,),
                                "native-owner-parent-fence",None)
        self.current=NativePublication(snapshot,(self.w2,self.w1),(witness,))

    def publish(self,pub=None,current=None,now=NOW):
        return self.bridge.publish(self.current if pub is None else pub,
            self.current if current is None else current,now_ns=now)

    def test_off_has_no_bridge_owner_or_state(self):
        self.assertIsNone(make_native_bridge("r",P4Config("off",100,1000)))

    def test_factory_requires_frozen_config(self):
        with self.assertRaises(TypeError):
            make_native_bridge("r",{"mode":"shadow"})

    def test_cpu_mock_and_conditional_cost_tables_never_enter_native(self):
        for scope in ("mock_only","conditional"):
            table=CostTable((),scope=scope,source_sha256="a"*64)
            with self.assertRaises(ValueError):
                make_native_bridge("r",P4Config("joint",100,1000,20),table=table)

    def test_publication_only_exact_value_types(self):
        with self.assertRaises(TypeError):
            NativePublication(object(),())
        with self.assertRaises(ValueError):
            NativePublication(self.current.snapshot,list(self.current.works))
        with self.assertRaises(ValueError):
            NativePublication(self.current.snapshot,(),[self.current.witnesses[0]])

    def test_work_and_witness_metadata_windows_are_bounded(self):
        works=tuple(replace(self.w1,child_id=i) for i in range(65))
        with self.assertRaises(ValueError):
            NativePublication(self.current.snapshot,works)
        with self.assertRaises(ValueError):
            NativePublication(self.current.snapshot,self.current.works,
                              self.current.witnesses*9)

    def test_duplicate_physical_work_cannot_alias(self):
        with self.assertRaises(ValueError):
            NativePublication(self.current.snapshot,(self.w1,self.w1))

    def test_foreign_run_and_epoch_work_rejected(self):
        for w in (replace(self.w1,run_id="other"),replace(self.w1,snapshot_epoch=self.epoch+1)):
            with self.assertRaises(ValueError):
                NativePublication(self.current.snapshot,(w,))

    def test_gpu_capability_or_even_zero_gpu_field_not_supported(self):
        for s in (replace(self.current.snapshot,gpu_immediately_reusable_bytes=0),
                  replace(self.current.snapshot,capabilities=CAPS|frozenset(("gpu_protectors",)))):
            with self.assertRaises(ValueError):
                NativePublication(s,self.current.works)

    def test_gpu_witness_is_rejected(self):
        with self.assertRaises(ValueError):
            NativePublication(self.current.snapshot,self.current.works,
                              (replace(self.current.witnesses[0],resource_kind="gpu"),))

    def test_bridge_must_bind_before_any_owner_access(self):
        bridge=make_native_bridge("r",self.config)
        with self.assertRaises(RuntimeError):
            bridge.note_reservation(0)
        with self.assertRaises(RuntimeError):
            bridge.epoch((),100)

    def test_bridge_can_bind_only_one_native_reactor(self):
        with self.assertRaises(RuntimeError):
            self.bridge.bind()

    def test_native_slot_reservation_has_new_scalar_generation(self):
        self.assertIsNone(self.bridge.generation(3))
        self.bridge.note_reservation(3)
        old=self.bridge.generation(3)
        self.bridge.note_reservation(3)
        self.assertGreater(self.bridge.generation(3),old)

    def test_bool_slot_id_cannot_read_integer_alias(self):
        self.bridge.note_reservation(1)
        with self.assertRaises(ValueError):
            self.bridge.generation(True)
        with self.assertRaises(ValueError):
            self.bridge.note_reservation(True)

    def test_epoch_is_stable_for_same_actual_facts(self):
        self.assertEqual(self.bridge.epoch(("actual-owner-values",1),101),self.epoch)

    def test_changed_owner_facts_advance_epoch(self):
        self.assertGreater(self.bridge.epoch(("actual-owner-values",2),101),self.epoch)

    def test_expired_owner_signature_advances_epoch_without_new_values(self):
        self.assertEqual(self.bridge.epoch(("actual-owner-values",1),200),self.epoch)
        self.assertGreater(self.bridge.epoch(("actual-owner-values",1),201),self.epoch)

    def test_clock_regression_rejected_even_when_signature_changes(self):
        with self.assertRaises(ValueError):
            self.bridge.epoch(("different",),99)

    def test_signature_never_retains_native_reference(self):
        with self.assertRaises((TypeError,ValueError)):
            self.bridge.epoch((object(),),101)

    def test_signature_is_recursively_bounded(self):
        with self.assertRaises(ValueError):
            self.bridge.epoch(tuple(range(65)),101)

    def test_owner_thread_cannot_be_replaced(self):
        errors=[]
        def foreign():
            try:self.bridge.snapshot()
            except Exception as exc:errors.append(exc)
        worker=threading.Thread(target=foreign)
        worker.start();worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors),1)
        self.assertIsInstance(errors[0],RuntimeError)

    def test_shutdown_snapshot_precondition_requires_exact_bool(self):
        with self.assertRaises(ValueError):
            self.bridge.snapshot(native_shutdown=1)

    def test_exact_duplicate_publication_is_idempotent(self):
        self.assertTrue(self.publish())
        self.assertFalse(self.publish())
        self.assertEqual(self.bridge.publications,1)

    def test_only_one_frozen_publication_per_owner_epoch(self):
        self.publish()
        changed=replace(self.current,snapshot=replace(self.current.snapshot,monotonic_ns=101))
        with self.assertRaises(ValueError):
            self.publish(changed,now=101)

    def test_publish_after_dispatch_boundary_cannot_change_epoch_advice(self):
        self.bridge.order(self.current,self.current.works,now_ns=NOW)
        with self.assertRaises(ValueError):
            self.publish()

    def test_epoch_change_retires_old_publication(self):
        self.publish()
        e=self.bridge.epoch(("new-native-facts",2),101)
        current=replace(self.current,
            snapshot=replace(self.current.snapshot,snapshot_epoch=e,monotonic_ns=101,
                             native_state=replace(self.current.snapshot.native_state,captured_ns=101)),
            works=tuple(replace(w,snapshot_epoch=e) for w in self.current.works))
        self.assertTrue(self.publish(current,current,now=101))
        self.assertEqual(self.bridge.publications,2)

    def test_foreign_run_publication_rejected(self):
        s=SystemSnapshot("foreign",self.epoch,NOW,CAPS)
        pub=NativePublication(s,())
        with self.assertRaises(ValueError):
            self.publish(pub)

    def test_stale_epoch_and_time_publications_rejected(self):
        with self.assertRaises(ValueError):
            self.publish(now=201)
        with self.assertRaises(ValueError):
            self.publish(now=99)
        s=replace(self.current.snapshot,snapshot_epoch=self.epoch+1)
        pub=replace(self.current,snapshot=s,
                    works=tuple(replace(w,snapshot_epoch=self.epoch+1) for w in self.current.works))
        with self.assertRaises(ValueError):
            self.publish(pub)

    def test_slot_generation_change_invalidates_old_publication(self):
        s=replace(self.current.snapshot,generations=((self.rid,2),))
        actual=replace(self.current,snapshot=s)
        with self.assertRaises(ValueError):
            self.publish(current=actual)

    def test_changed_parent_lifecycle_invalidates_old_publication(self):
        actual=replace(self.current,snapshot=replace(self.current.snapshot,
                       parents=(replace(self.p1,inflight_files=1),self.p2)))
        with self.assertRaises(ValueError):
            self.publish(current=actual)

    def test_changed_ready_physical_bytes_invalidates_publication(self):
        actual=replace(self.current,works=(self.w2,replace(self.w1,nbytes=16)))
        with self.assertRaises(ValueError):
            self.publish(current=actual)

    def test_changed_live_inflight_values_rejected(self):
        state=replace(self.current.snapshot.native_state,free_staging_bytes=8)
        actual=replace(self.current,snapshot=replace(self.current.snapshot,native_state=state))
        with self.assertRaises(ValueError):
            self.publish(current=actual)

    def test_fresh_identical_owner_facts_accept_new_capture_timestamp(self):
        state=replace(self.current.snapshot.native_state,captured_ns=101)
        actual=replace(self.current,snapshot=replace(self.current.snapshot,monotonic_ns=101,
                                                   native_state=state))
        self.assertTrue(self.publish(current=actual,now=101))

    def test_injected_parent_and_witness_forecasts_cannot_open_dependency_priority(self):
        forecast=replace(self.p1,completion_estimate_ns=140)
        snap=replace(self.current.snapshot,parents=(forecast,self.p2))
        witness=replace(self.current.witnesses[0],parents=(forecast,),estimated_unblock_ns=140)
        pub=replace(self.current,snapshot=snap,witnesses=(witness,))
        with self.assertRaises(ValueError):
            self.publish(pub)
        ordered=self.bridge.order(self.current,self.current.works,now_ns=NOW)
        self.assertEqual(ordered,tuple(w.work_id for w in self.current.works))

    def test_injected_error_or_interference_estimate_alone_rejected(self):
        for updates in ({"error_ns":8},{"interference_ns":1},{"estimated_unblock_ns":140}):
            pub=replace(self.current,witnesses=(replace(self.current.witnesses[0],**updates),))
            with self.assertRaises(ValueError):
                self.publish(pub)

    def test_parent_witness_must_match_actual_owner_completion_tuple(self):
        witness=replace(self.current.witnesses[0],parents=(replace(self.p1,done_files=1),))
        pub=replace(self.current,witnesses=(witness,))
        with self.assertRaises(ValueError):
            self.publish(pub)

    def test_choice_cache_computes_once_per_owner_epoch(self):
        original=tuple(w.work_id for w in self.current.works)
        self.assertEqual(self.bridge.order(self.current,self.current.works,now_ns=100),original)
        self.assertEqual(self.bridge.order(self.current,self.current.works,now_ns=101),original)
        self.assertEqual(self.bridge.choices,1)
        self.assertFalse(self.bridge.last_choice.gpu_qualified)

    def test_new_epoch_never_reuses_old_choice(self):
        self.bridge.order(self.current,self.current.works,now_ns=100)
        e=self.bridge.epoch(("changed-owner-state",2),101)
        current=replace(self.current,snapshot=replace(self.current.snapshot,snapshot_epoch=e,
                                                     monotonic_ns=101),
                        works=tuple(replace(w,snapshot_epoch=e) for w in self.current.works))
        self.bridge.order(current,current.works,now_ns=101)
        self.assertEqual(self.bridge.choices,2)

    def test_changed_actual_state_retires_external_publication(self):
        self.publish()
        actual=replace(self.current,snapshot=replace(self.current.snapshot,
                                                    cpu_clean_reclaimable_bytes=16))
        self.assertEqual(self.bridge.order(actual,actual.works,now_ns=101),
                         tuple(w.work_id for w in actual.works))
        self.assertEqual(self.bridge.last_reason,"stale_or_changed_native_publication")

    def test_subset_hint_keeps_only_original_native_work(self):
        ordered=self.bridge.order(self.current,(self.w1,),now_ns=100)
        self.assertEqual(ordered,(self.w1.work_id,))

    def test_foreign_subset_cannot_be_inserted_into_existing_queue(self):
        with self.assertRaises(ValueError):
            self.bridge.order(self.current,(replace(self.w1,child_id=99),),now_ns=100)

    def test_metadata_fault_is_sticky_and_never_claims_resource_drain(self):
        self.bridge.fail("first diagnostic")
        self.bridge.fail("second diagnostic")
        self.assertEqual(self.bridge.fault,"first diagnostic")
        original=tuple(w.work_id for w in self.current.works)
        self.assertEqual(self.bridge.order(self.current,self.current.works,now_ns=100),original)
        report=self.bridge.snapshot()
        self.assertFalse(report["held_job_or_resource_owners"])
        self.assertEqual(report["new_work_queues"],0)
        self.assertIsNone(report["gpu_release_credit"])
        self.assertFalse(report["gpu_qualified"])

    def test_interference_joint_advice_and_preview_keep_baseline_fallback(self):
        for mode in ("interference","joint"):
            bridge=make_native_bridge("r",P4Config(mode,100,1000,20))
            bridge.bind()
            bridge.epoch(("fixture",),100)
            ordered=bridge.order(self.current,self.current.works,now_ns=100)
            self.assertEqual(ordered,tuple(w.work_id for w in self.current.works))
            self.assertEqual(bridge.last_choice.action,"native_fallback")
            preview=bridge.preview_issue(self.w1,self.current.snapshot,now_ns=100)
            self.assertEqual(preview.action,"native_fallback")
            self.assertFalse(preview.production_qualified)

    def test_no_bridge_allowance_ledger_or_settlement_owner(self):
        for name in ("pending","used","accepted","settle","queue"):
            self.assertFalse(hasattr(self.bridge,name))
        self.assertFalse(self.bridge.snapshot()["interference_production_qualified"])


def _standalone():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--name",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    out=Path(args.output).resolve()
    if root not in out.parents:
        raise ValueError("evidence output must remain inside project")
    out.mkdir(parents=True,exist_ok=True)
    paths={s:out/(args.name+s) for s in (".json",".xml",".log","-command.json")}
    if any(p.exists() for p in paths.values()):
        raise FileExistsError("fresh evidence names required")
    class Result(unittest.TextTestResult):
        def startTest(self,test):
            self.started=time.monotonic_ns()
            super().startTest(test)
        def stopTest(self,test):
            self.rows.append((test.id(),time.monotonic_ns()-self.started))
            super().stopTest(test)
    class Runner(unittest.TextTestRunner):
        def _makeResult(self):
            r=Result(self.stream,self.descriptions,self.verbosity);r.rows=[];return r
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentValueABITests)
    with paths[".log"].open("x",encoding="utf-8") as f:
        result=Runner(stream=f,verbosity=2).run(suite)
    xml=ET.Element("testsuite",name="Independent native P4 value ABI",tests=str(result.testsRun),
        failures=str(len(result.failures)),errors=str(len(result.errors)),skipped=str(len(result.skipped)))
    bad={t.id():text for t,text in result.failures}
    errors={t.id():text for t,text in result.errors}
    for name,ns in result.rows:
        case=ET.SubElement(xml,"testcase",classname=name.rsplit(".",1)[0],
                           name=name.rsplit(".",1)[1],time=str(ns/1e9))
        if name in bad:ET.SubElement(case,"failure").text=bad[name]
        if name in errors:ET.SubElement(case,"error").text=errors[name]
    ET.ElementTree(xml).write(paths[".xml"],encoding="utf-8",xml_declaration=True)
    receipt={"status":"PASS_INDEPENDENT_CPU_VALUE_ABI" if result.wasSuccessful()
                       else "FAIL_INDEPENDENT_CPU_VALUE_ABI",
             "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "skipped":len(result.skipped),"forbidden_import_attempts":FORBIDDEN_IMPORTS,
             "gpu_initialized":False,"gpu_workloads":0,
             "scope":"pure CPU scalar ABI; not real native I/O, DMA or performance",
             "bridge_source_ref":{"path":"third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control/p4_bridge.py",
                 "sha256":sha256((root/"third_party/work/prefix-io-p4-01-cpu/src/prefix_io_control/p4_bridge.py").read_bytes()).hexdigest()}}
    paths[".json"].write_text(json.dumps(receipt,indent=2))
    paths["-command.json"].write_text(json.dumps({
        "executable":sys.executable,"argv":sys.argv,"cwd":str(Path.cwd()),
        "isolated":sys.flags.isolated,"no_site":sys.flags.no_site,
        "exit":0 if result.wasSuccessful() else 1,
        "environment_required":{"CUDA_VISIBLE_DEVICES":"","PYTHONDONTWRITEBYTECODE":"1"},
        "backend_import_guard":"Torch/vLLM/py_kvcache/CUDA/CuPy/NumPy forbidden"
    },indent=2))
    print(json.dumps(receipt))
    if not result.wasSuccessful():print(paths[".log"].read_text()[-7000:])
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__=="__main__":
    _standalone()
