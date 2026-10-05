"""V2 full-output/selection fixtures; all GPU-looking durations are CPU fixtures."""
from copy import deepcopy
from dataclasses import asdict
import json
import tempfile
import unittest
from unittest.mock import patch
from tests.prefix_io_v1_p4_measurements.test_semantic_verifier import Fixture, io
from prefix_io_control.dispatch_budget import STAGES
from prefix_io_control.p4_production_table_contract import EvidenceRef, TableContractError
from prefix_io_control.p4_paired_measurement_verifier import (
    FORMULA, load_verification_plan, build_paired_cell_candidate, semantic_manifest,
)
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table


class V2Fixture(Fixture):
    def __init__(self, root, *, batch=1, origin="cpu_fixture", scope="model_forward",
                 basis="existing_io_plus_delta"):
        self.ready = False
        super().__init__(root, origin=origin, basis=basis)
        self.load_state = {"active_decode":batch,"batch":batch,"prefill_tokens":0,"context_length":128}
        self.cell.update(load=dict(self.load_state), windows_reported=3)
        self.scope = scope
        self.traces = {}
        self.source_name = "cuda-event-reference-source.py"
        (self.root/self.source_name).write_text("# CPU fixture: never execute or claim real CUDA events.\n")
        self.source_ref = self.ref(self.source_name)
        self.timing = {"timing_scope":scope,"clock_domain":"cuda_event_elapsed",
            "reference_source_ref":asdict(self.source_ref),"reference_valid":True,
            "fallback_used":False,"clock_domain_valid":True}
        for arm in ("baseline","action"):
            self.docs[arm+"_wrapper"]["schema_version"]=2
            self.docs[arm+"_observations"].update(schema_version=2,windows=[])
        for index,entry in enumerate(self.entries):
            entry.update(output_tokens=128,request_count=batch)
            pair="p"+str(index)
            for arm in ("baseline","action"):
                run=self.docs[arm+"_wrapper"]["runs"][index]
                first=(arm=="baseline") == (run["arm_order"]=="AB")
                start=index*2000000+(0 if first else 500000)
                full_steps=128//batch
                native_first=100+index*full_steps*2+(0 if first else full_steps)
                run.update(output_tokens=128,request_count=batch,start_ns=start,
                    end_ns=start+full_steps*1000+1000,warmup_windows=1,measured_windows=1,
                    first_step_ordinal=native_first,full_steps=full_steps,full_output_tokens=128,
                    selected_output_tokens=batch,accepted_new_io=io())
                trace={"schema_version":2,"scope":"p4_complete_step_output_trace","origin":origin,
                    "context":asdict(self.context),"arm":arm,"cell_id":"cell-1","pair_id":pair,
                    "steps":[],"outputs":[{"request_id":"r"+str(r),"token_ids":list(range(full_steps))}
                                          for r in range(batch)],
                    "outside_window_new_io":io(1,8) if arm=="action" else io(),
                    "accepted_new_io":io(),"completed_new_io":io(),"accepted_io_drained":True}
                duration=((100,110),(102,116),(104,119))[index][arm=="action"]
                for offset in range(full_steps):
                    extra=io()
                    if arm=="action":
                        if offset==0:extra=io(4,32)
                        elif offset==2:extra=io(2,16)
                        elif offset==5:extra=io(1,8,"ssd_read")
                    row={"step_offset":offset,"native_step_ordinal":native_first+offset,
                        "start_ns":start+offset*1000+1,"end_ns":start+offset*1000+101,
                        "load":dict(self.load_state,context_length=126+offset),
                        "existing_io":io(1 if offset==2 else 2,8 if offset==2 else 16,"ssd_read")
                            if arm=="action" or basis=="existing_io_plus_delta" else io(),
                        "new_io":extra,"outputs":[{"request_id":"r"+str(r),"token_ids":[offset]}
                                                  for r in range(batch)],
                        "timing":dict(self.timing,gpu_elapsed_ns=duration if offset==2 else 9999)}
                    trace["steps"].append(row)
                    if offset in (0,2):
                        window={key:deepcopy(row[key]) for key in
                            ("step_offset","native_step_ordinal","start_ns","end_ns","load",
                             "existing_io","new_io","timing")}
                        window.update(pair_id=pair,window_id="step-"+str(offset),
                            phase="warmup" if offset==0 else "measured",output_tokens=batch)
                        self.docs[arm+"_observations"]["windows"].append(window)
                self.traces[(arm,pair)]=trace
                self.recount(arm,pair)
        self.ready=True
        self.refresh()

    def recount(self, arm, pair):
        trace=self.traces[(arm,pair)]
        total=deepcopy(trace["outside_window_new_io"])
        for step in trace["steps"]:
            for i in range(4):
                total[i]["ops"]+=step["new_io"][i]["ops"]
                total[i]["bytes"]+=step["new_io"][i]["bytes"]
        trace["accepted_new_io"]=deepcopy(total)
        trace["completed_new_io"]=deepcopy(total)
        run=next(r for r in self.docs[arm+"_wrapper"]["runs"] if r["pair_id"]==pair)
        run["accepted_new_io"]=deepcopy(total)
        run["completed_new_io"]=deepcopy(total)

    def refresh(self):
        if not self.ready:return super().refresh()
        self.split_ref=self.dump("split.json",{"schema_version":1,"scope":"p4_frozen_measurement_split",
                                             "entries":self.entries})
        self.plan={"schema_version":2,"scope":"p4_paired_semantic_plan",
            "context":asdict(self.context),"workload_split_ref":asdict(self.split_ref),
            "action_operations":{"cell-1":2},"min_calibration_pairs":2,"min_validation_pairs":1,
            "max_pairs":64,"max_windows":4096,"formula":FORMULA,
            "selections":{"cell-1":{"load":dict(self.load_state),
                "warmup_step_offsets":[0],"measured_step_offsets":[2]}},
            "timing_contract":deepcopy(self.timing)}
        if hasattr(self,"plan_override"):self.plan_override(self.plan)
        self.plan_ref=self.dump("plan.json",self.plan)
        for (arm,pair),trace in self.traces.items():
            tref=self.dump("complete-"+arm+"-"+pair+".json",trace)
            run=next(r for r in self.docs[arm+"_wrapper"]["runs"] if r["pair_id"]==pair)
            run["complete_trace_ref"]=asdict(tref)
        if hasattr(self,"wrapper_override"):self.wrapper_override(self.docs)
        refs={name:self.dump(name+".json",value) for name,value in self.docs.items()}
        self.analysis={"schema_version":2,"scope":"p4_paired_analysis_candidate",
            "origin":self.docs["baseline_wrapper"]["origin"],"context":asdict(self.context),
            "cell_id":"cell-1","plan_ref":asdict(self.plan_ref),
            "evidence_refs":{name:asdict(ref) for name,ref in refs.items()},"formula":FORMULA,
            "baseline_ns":101,"incremental_or_joint_ns":12,"uncertainty_ns":6,
            "calibration_pairs":2,"validation_pairs":1,"paired_runs_reported":3,"windows_reported":3}
        refs["pair_analysis"]=self.dump("analysis.json",self.analysis)
        self.cell["measurement_refs"]={name:asdict(ref) for name,ref in refs.items()}
        self.candidate={"schema_version":1,"scope":"production_candidate",
                        "context":asdict(self.context),"cells":[self.cell]}
        self.candidate_ref=self.dump("candidate.json",self.candidate)
        self.verifier_ref=self.ref(self.verifier_name)
        self.qualification={"schema_version":1,"scope":"production_qualification_candidate",
            "status":"PASS_REPORTED","candidate_ref":asdict(self.candidate_ref),
            "context":asdict(self.context),"verifier_ref":asdict(self.verifier_ref),
            "measurement_refs":[asdict(ref) for ref in refs.values()]}
        self.qualification_ref=self.dump("qualification.json",self.qualification)


class V2VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.f=V2Fixture(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def reject(self):
        with self.assertRaises(TableContractError):self.f.load()
    def mutate_trace(self,fn,*,arm="action",pair="p0"):
        fn(self.f.traces[(arm,pair)])
        self.f.refresh()
        self.reject()
    def mutate_window(self,fn):
        fn(self.f.docs["action_observations"]["windows"][1])
        self.f.refresh()
        self.reject()
    def mutate_run(self,fn):
        fn(self.f.docs["action_wrapper"]["runs"][0])
        self.f.refresh()
        self.reject()
    def build(self):
        geometry={k:deepcopy(self.f.cell[k]) for k in
                  ("cell_id","load","existing_io","stage","physical_bytes")}
        refs={k:EvidenceRef.from_mapping(self.f.cell["measurement_refs"][k]) for k in
              ("baseline_wrapper","action_wrapper","baseline_observations","action_observations")}
        plan=load_verification_plan(self.f.root,"plan.json",expected_plan_ref=self.f.plan_ref)
        return build_paired_cell_candidate(self.f.root,cell_geometry=geometry,raw_refs=refs,
                                          expected_plan=plan,analysis_path="prepared/analysis-v2.json")

    def test_continuous_context_full128_selected1_gpu_duration_only(self):
        loaded=self.f.load();cell=loaded.verification.cells[0]
        self.assertEqual((cell.cost.baseline_ns,cell.cost.incremental_or_joint_ns,cell.cost.uncertainty_ns),
                         (101,12,6))
        self.assertEqual((cell.measured_windows,cell.warmup_windows),(3,3))
        self.assertEqual((cell.schema_version,cell.timing_scope,cell.clock_domain),
                         (2,"model_forward","cuda_event_elapsed"))
        self.assertEqual(self.f.docs["action_wrapper"]["runs"][0]["full_output_tokens"],128)
        self.assertEqual(self.f.docs["action_wrapper"]["runs"][0]["selected_output_tokens"],1)
        self.assertEqual([r["load"]["context_length"] for r in self.f.traces[("action","p0")]["steps"]][:4],
                         [126,127,128,129])
        self.assertEqual(len(cell.complete_trace_refs),6)
        self.assertFalse(loaded.gpu_verified)
        self.assertFalse(loaded.production_qualified)
        cost=cell.cost
        self.assertIsNone(loaded.lookup(cost.load_signature,cost.existing_io,"h2d",16))
        self.assertFalse(semantic_manifest(loaded.verification)["gpu_execution_proved"])
    def test_full128_selected2_actual_batch_token_ids(self):
        self.f=V2Fixture(self.temp.name,batch=2)
        loaded=self.f.load()
        run=self.f.docs["action_wrapper"]["runs"][0]
        self.assertEqual((run["full_steps"],run["full_output_tokens"],run["selected_output_tokens"]),(64,128,2))
        self.assertEqual(loaded.verification.cells[0].cost.total_ns,119)
    def test_raw_builder_v2_full_loader_roundtrip(self):
        built=self.build()
        self.assertEqual(built.verification.cost.total_ns,119)
        self.assertFalse(built.production_qualified)
        cell=built.to_candidate_mapping()
        cref=self.f.dump("prepared-candidate.json",{"schema_version":1,"scope":"production_candidate",
            "context":asdict(self.f.context),"cells":[cell]})
        report=dict(self.f.qualification,candidate_ref=asdict(cref),
                    measurement_refs=list(cell["measurement_refs"].values()))
        qref=self.f.dump("prepared-qualification.json",report)
        args=self.f.args();args["qualification_ref"]=qref
        loaded=load_semantically_verified_table(self.f.root,"prepared-candidate.json",**args)
        self.assertEqual(loaded.verification.cells[0].cost,built.verification.cost)
        self.assertFalse(loaded.production_qualified)
    def test_native_tag_and_full_decode_scope_still_cannot_qualify(self):
        self.f=V2Fixture(self.temp.name,origin="native_gpu_recording",scope="full_decode_step")
        loaded=self.f.load()
        self.assertFalse(loaded.gpu_verified);self.assertFalse(loaded.production_qualified)
        self.assertIsNone(loaded.lookup(None,None,None,None,execution="production"))
    def test_no_io_plus_joint_v2(self):
        self.f=V2Fixture(self.temp.name,basis="no_io_plus_joint")
        self.assertEqual(self.f.load().verification.cells[0].cost.total_ns,119)
    def test_missing_complete_trace_reference(self):
        self.f.wrapper_override=lambda docs:docs["action_wrapper"]["runs"][0].pop("complete_trace_ref")
        self.f.refresh();self.reject()
    def test_selected_subset_cannot_replace_full128(self):
        self.mutate_run(lambda r:r.update(output_tokens=1,full_output_tokens=1))
    def test_missing_nonselected_trace_step(self):
        self.mutate_trace(lambda t:t["steps"].pop(50))
    def test_duplicate_native_ordinal(self):
        self.mutate_trace(lambda t:t["steps"][1].update(native_step_ordinal=t["steps"][0]["native_step_ordinal"]))
    def test_reordered_native_ordinal(self):
        self.mutate_trace(lambda t:t["steps"].reverse())
    def test_missing_offset(self):
        self.mutate_trace(lambda t:t["steps"][3].pop("step_offset"))
    def test_unregistered_selected_step(self):
        self.mutate_window(lambda r:r.update(step_offset=3,window_id="step-3"))
    def test_selected_geometry_drift(self):
        self.mutate_window(lambda r:r["load"].update(context_length=129))
    def test_preregistered_geometry_drift(self):
        self.f.plan_override=lambda p:p["selections"]["cell-1"]["load"].update(context_length=129)
        self.f.refresh();self.reject()
    def test_selected_output_token_count_forged(self):
        self.mutate_window(lambda r:r.update(output_tokens=2))
    def test_complete_output_token_ids_changed(self):
        self.mutate_trace(lambda t:t["outputs"][0]["token_ids"].__setitem__(60,999))
    def test_full_paired_output_ids_differ_even_when_projection_consistent(self):
        trace=self.f.traces[("action","p0")]
        trace["outputs"][0]["token_ids"][60]=999
        trace["steps"][60]["outputs"][0]["token_ids"][0]=999
        self.f.refresh();self.reject()
    def test_nonselected_actual_outputs_cannot_be_omitted(self):
        self.mutate_trace(lambda t:t["steps"][60].update(outputs=[]))
    def test_unknown_step_request_rejected(self):
        self.mutate_trace(lambda t:t["steps"][60]["outputs"][0].update(request_id="unknown"))
    def test_wrong_clock_domain(self):
        self.mutate_window(lambda r:r["timing"].update(clock_domain="host_monotonic_ns"))
    def test_host_duration_cannot_substitute_gpu_elapsed(self):
        self.mutate_trace(lambda t:t["steps"][2]["timing"].pop("gpu_elapsed_ns"))
    def test_fallback_elapsed_cannot_enter_estimator(self):
        self.mutate_trace(lambda t:t["steps"][2]["timing"].update(fallback_used=True))
    def test_invalid_reference_cannot_enter_estimator(self):
        self.mutate_trace(lambda t:t["steps"][2]["timing"].update(reference_valid=False))
    def test_invalid_clock_reference_cannot_enter_estimator(self):
        self.mutate_trace(lambda t:t["steps"][2]["timing"].update(clock_domain_valid=False))
    def test_forward_cannot_claim_frozen_full_decode_scope(self):
        self.mutate_trace(lambda t:t["steps"][2]["timing"].update(timing_scope="full_decode_step"))
    def test_actual_duration_source_drift(self):
        self.mutate_trace(lambda t:t["steps"][2]["timing"]["reference_source_ref"].update(sha256="f"*64))
    def test_timing_source_byte_mutation_after_pinning(self):
        (self.f.root/self.f.source_name).write_text("# changed source\n")
        self.reject()
    def test_nondrained_nonselected_action_io_rejected(self):
        self.mutate_run(lambda r:r.update(completed_new_io=io(2,16)))
    def test_nondrained_outside_window_action_io_rejected(self):
        self.mutate_trace(lambda t:t.update(outside_window_new_io=io(2,16)))
    def test_selected_action_at_wrong_offset_rejected_even_when_total_same(self):
        trace=self.f.traces[("action","p0")]
        trace["steps"][2]["new_io"],trace["steps"][3]["new_io"]=trace["steps"][3]["new_io"],trace["steps"][2]["new_io"]
        self.f.docs["action_observations"]["windows"][1]["new_io"]=deepcopy(trace["steps"][2]["new_io"])
        self.f.refresh();self.reject()
    def test_full_trace_bytes_mutated_after_wrapper_pinning(self):
        (self.f.root/"complete-action-p0.json").write_text("{}")
        self.reject()
    def test_candidate_cost_tamper_still_rejected(self):
        self.f.cell["baseline_ns"]=999
        self.f.refresh();self.reject()
    def test_v2_global_workload_content_isolation_remains_required(self):
        self.f.entries[2]["workload_sha256"]=self.f.entries[0]["workload_sha256"]
        for arm in ("baseline","action"):
            self.f.docs[arm+"_wrapper"]["runs"][2]["workload_sha256"]=self.f.entries[0]["workload_sha256"]
        self.f.refresh();self.reject()
    def test_builder_rechecks_complete_trace_bytes_after_recomputation(self):
        import prefix_io_control.p4_paired_measurement_verifier as module
        original=module._verify_cell
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs)
            (self.f.root/"complete-action-p0.json").write_text("{}")
            return result
        with patch.object(module,"_verify_cell",side_effect=mutate):
            with self.assertRaises(TableContractError):self.build()


if __name__=="__main__":unittest.main()
