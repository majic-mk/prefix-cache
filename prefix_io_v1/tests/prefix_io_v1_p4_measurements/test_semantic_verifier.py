"""CPU-only raw-pair fixtures; successful semantics never certify GPU origin."""
from dataclasses import asdict, replace, FrozenInstanceError
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
from prefix_io_control.dispatch_budget import Amount, STAGES, ZERO
from prefix_io_control.p4_production_table_contract import (
    EvidenceRef, TableContext, TableContractError, load_production_candidate,
)
from prefix_io_control.p4_paired_measurement_verifier import (
    FORMULA, load_verification_plan, verify_paired_measurements,
    write_semantic_verification_manifest, semantic_manifest, build_paired_cell_candidate,
)
from prefix_io_control.p4_verified_cost_loader import load_semantically_verified_table, PreparedCostTable
from prefix_io_control.p4_cost_table import CostTable


def io(ops=0, nbytes=0, stage="h2d"):
    result=[{"ops":0,"bytes":0} for _ in STAGES]
    result[STAGES.index(stage)]={"ops":ops,"bytes":nbytes}
    return result


class Fixture:
    def __init__(self, root, *, basis="existing_io_plus_delta", origin="cpu_fixture"):
        self.root=Path(root)
        self.context=TableContext("cpu-semantic-fixture","a"*64,"GPU-CPU-FIXTURE",
            "b"*64,"c"*64,"d"*64,"eager","torch-fixture","cuda-fixture",
            "driver-fixture",8,128,64,200,basis,"paired_residual_margin")
        self.load_state={"active_decode":2,"batch":2,"prefill_tokens":0,"context_length":128}
        self.docs={}
        self.entries=[]
        for arm in ("baseline","action"):
            self.docs[arm+"_wrapper"]={"schema_version":1,"scope":"paired_measurement_wrapper",
                "origin":origin,"context":asdict(self.context),"arm":arm,"cell_id":"cell-1","runs":[]}
            self.docs[arm+"_observations"]={"schema_version":1,"scope":"paired_window_observations",
                "origin":origin,"context":asdict(self.context),"arm":arm,"cell_id":"cell-1","windows":[]}
        for index,(baseline,action,split,order) in enumerate(
                ((100,110,"calibration","AB"),(102,116,"calibration","BA"),
                 (104,119,"validation","AB"))):
            pair="p"+str(index)
            entry={"cell_id":"cell-1","trace_sha256":sha256(("trace"+pair).encode()).hexdigest(),
                "prefix_family_sha256":sha256(("family"+split).encode()).hexdigest(),"seed":index,
                "split":split,"workload_sha256":sha256(("work"+pair).encode()).hexdigest(),
                "input_tokens":32,"output_tokens":2,"request_count":1}
            self.entries.append(entry)
            for arm,duration in (("baseline",baseline),("action",action)):
                first=(arm=="baseline") == (order=="AB")
                start=index*50000+(0 if first else 20001)
                extra=io(2,16) if arm=="action" else io()
                run=dict(entry);del run["cell_id"]
                run.update(pair_id=pair,arm_order=order,warmup_windows=1,measured_windows=2,
                    start_ns=start,end_ns=start+20000,exit_code=0,accepted_io_drained=True,
                    completed_new_io=io(6,48) if arm=="action" else io())
                self.docs[arm+"_wrapper"]["runs"].append(run)
                for k,(phase,delta) in enumerate((("warmup",9999),("measured",duration),("measured",duration))):
                    stepstart=start+1 if k==0 else start+10001+(k-1)*1000
                    existing=io(1,8,"ssd_read") if arm=="action" or basis=="existing_io_plus_delta" else io()
                    self.docs[arm+"_observations"]["windows"].append({
                        "pair_id":pair,"window_id":"w"+str(k),"phase":phase,"start_ns":stepstart,
                        "end_ns":stepstart+delta,"load":dict(self.load_state),
                        "existing_io":existing,"new_io":extra,"output_tokens":1})
        self.verifier_name="verifier.py"
        (self.root/self.verifier_name).write_text("raise RuntimeError('must never execute fixture verifier')\n")
        self.cell={"cell_id":"cell-1","load":dict(self.load_state),"existing_io":io(1,8,"ssd_read"),
            "stage":"h2d","physical_bytes":16,"baseline_ns":101,"incremental_or_joint_ns":12,
            "uncertainty_ns":6,"paired_runs_reported":3,"windows_reported":6,"measurement_refs":{}}
        self.refresh()

    def dump(self,name,data):
        (self.root/name).write_text(json.dumps(data,sort_keys=True),encoding="utf-8")
        return self.ref(name)

    def ref(self,name):
        raw=(self.root/name).read_bytes()
        return EvidenceRef(name,len(raw),sha256(raw).hexdigest())

    def refresh(self):
        self.split_ref=self.dump("split.json",{"schema_version":1,"scope":"p4_frozen_measurement_split",
                                               "entries":self.entries})
        self.plan={"schema_version":1,"scope":"p4_paired_semantic_plan",
            "context":asdict(self.context),"workload_split_ref":asdict(self.split_ref),
            "action_operations":{"cell-1":2},"min_calibration_pairs":2,"min_validation_pairs":1,
            "max_pairs":64,"max_windows":4096,"formula":FORMULA}
        self.plan_ref=self.dump("plan.json",self.plan)
        refs={name:self.dump(name+".json",value) for name,value in self.docs.items()}
        analysis={"schema_version":1,"scope":"p4_paired_analysis_candidate",
            "origin":self.docs["baseline_wrapper"]["origin"],"context":asdict(self.context),
            "cell_id":"cell-1","plan_ref":asdict(self.plan_ref),
            "evidence_refs":{name:asdict(ref) for name,ref in refs.items()},"formula":FORMULA,
            "baseline_ns":101,"incremental_or_joint_ns":12,"uncertainty_ns":6,
            "calibration_pairs":2,"validation_pairs":1,"paired_runs_reported":3,"windows_reported":6}
        self.analysis=analysis
        refs["pair_analysis"]=self.dump("analysis.json",analysis)
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

    def args(self):
        return dict(expected_context=self.context,qualification_ref=self.qualification_ref,
                    expected_verifier_ref=self.verifier_ref,plan_path="plan.json",
                    expected_plan_ref=self.plan_ref)

    def load(self):
        return load_semantically_verified_table(self.root,"candidate.json",**self.args())


class SemanticVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.f=Fixture(self.temp.name)
    def tearDown(self):
        self.temp.cleanup()
    def reject(self):
        with self.assertRaises(TableContractError):
            self.f.load()
    def mutate_doc(self,role,fn):
        fn(self.f.docs[role]);self.f.refresh();self.reject()

    def build(self, **overrides):
        geometry={name:deepcopy(self.f.cell[name]) for name in
                  ("cell_id","load","existing_io","stage","physical_bytes")}
        raw_refs={name:EvidenceRef.from_mapping(self.f.cell["measurement_refs"][name]) for name in
                  ("baseline_wrapper","action_wrapper","baseline_observations","action_observations")}
        plan=load_verification_plan(self.f.root,"plan.json",expected_plan_ref=self.f.plan_ref)
        args=dict(cell_geometry=geometry,raw_refs=raw_refs,expected_plan=plan,
                  analysis_path="prepared/analysis.json")
        args.update(overrides)
        return build_paired_cell_candidate(self.f.root,**args)

    def assembled(self, prepared, *, tamper_cost=False):
        cell=prepared.to_candidate_mapping()
        if tamper_cost:cell["baseline_ns"]=999
        candidate={"schema_version":1,"scope":"production_candidate",
                   "context":asdict(self.f.context),"cells":[cell]}
        cref=self.f.dump("prepared-candidate.json",candidate)
        report=dict(self.f.qualification)
        report["candidate_ref"]=asdict(cref)
        report["measurement_refs"]=list(cell["measurement_refs"].values())
        qref=self.f.dump("prepared-qualification.json",report)
        args=self.f.args();args["qualification_ref"]=qref
        return load_semantically_verified_table(self.f.root,"prepared-candidate.json",**args)

    def test_raw_builder_roundtrip_reuses_sole_estimator(self):
        built=self.build()
        self.assertFalse(built.production_qualified)
        self.assertFalse(built.gpu_verified)
        self.assertEqual(built.verification.cost.total_ns,119)
        built.analysis_ref.verify(self.f.root)
        loaded=self.assembled(built)
        self.assertEqual(loaded.verification.cells[0].cost,built.verification.cost)
        self.assertFalse(loaded.production_qualified)
    def test_raw_builder_cannot_accept_reported_timing_numbers(self):
        geometry={name:deepcopy(self.f.cell[name]) for name in
                  ("cell_id","load","existing_io","stage","physical_bytes")}
        geometry["baseline_ns"]=999
        with self.assertRaises(TableContractError):
            self.build(cell_geometry=geometry)
    def test_raw_builder_missing_raw_role_rejected(self):
        refs={name:EvidenceRef.from_mapping(self.f.cell["measurement_refs"][name]) for name in
              ("baseline_wrapper","action_wrapper","baseline_observations")}
        with self.assertRaises(TableContractError):
            self.build(raw_refs=refs)
    def test_raw_builder_missing_load_field_rejected(self):
        geometry={name:deepcopy(self.f.cell[name]) for name in
                  ("cell_id","load","existing_io","stage","physical_bytes")}
        del geometry["load"]["active_decode"]
        with self.assertRaises(TableContractError):
            self.build(cell_geometry=geometry)
    def test_raw_builder_wrong_quantum_bytes_rejected(self):
        geometry={name:deepcopy(self.f.cell[name]) for name in
                  ("cell_id","load","existing_io","stage","physical_bytes")}
        geometry["physical_bytes"]=17
        with self.assertRaises(TableContractError):
            self.build(cell_geometry=geometry)
    def test_raw_builder_rechecks_content_isolation(self):
        self.f.entries[2]["workload_sha256"]=self.f.entries[0]["workload_sha256"]
        self.f.refresh()
        with self.assertRaises(TableContractError):
            self.build()
    def test_raw_builder_analysis_is_append_only(self):
        self.build()
        with self.assertRaises(TableContractError):
            self.build()
    def test_raw_builder_changed_source_during_recompute_rejected(self):
        import prefix_io_control.p4_paired_measurement_verifier as module
        original=module._verify_cell
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs)
            (self.f.root/"action_wrapper.json").write_text("{}")
            return result
        with patch.object(module,"_verify_cell",side_effect=mutate):
            with self.assertRaises(TableContractError):
                self.build()
    def test_raw_builder_forged_native_tag_cannot_qualify(self):
        self.f=Fixture(self.temp.name,origin="native_gpu_recording")
        built=self.build()
        self.assertFalse(built.gpu_verified)
        loaded=self.assembled(built)
        self.assertFalse(loaded.production_qualified)
    def test_raw_builder_declared_candidate_tamper_rejected_by_final_loader(self):
        built=self.build()
        with self.assertRaises(TableContractError):
            self.assembled(built,tamper_cost=True)

    def test_prepared_wrapper_rejects_fake_lookup_constructor(self):
        class FakeLookup:
            def lookup(self,*args,**kwargs):
                return {"production_qualified":True}
        with self.assertRaises(TypeError):
            PreparedCostTable(None,FakeLookup())
        with self.assertRaises(TypeError):
            PreparedCostTable(self.f.load().verification,FakeLookup())
    def test_prepared_wrapper_rejects_other_scope(self):
        p=self.f.load()
        table=CostTable(p.cpu_mock_table.cells,scope="conditional",source_sha256=p.source_ref.sha256)
        with self.assertRaises(ValueError):
            PreparedCostTable(p.verification,table)
    def test_prepared_wrapper_rejects_other_source_digest(self):
        p=self.f.load()
        table=CostTable(p.cpu_mock_table.cells,scope="mock_only",source_sha256="0"*64)
        with self.assertRaises(ValueError):
            PreparedCostTable(p.verification,table)
    def test_prepared_wrapper_rejects_other_recomputed_cells(self):
        p=self.f.load()
        table=CostTable((),scope="mock_only",source_sha256=p.source_ref.sha256)
        with self.assertRaises(ValueError):
            PreparedCostTable(p.verification,table)
    def test_prepared_production_lookup_never_calls_underlying_table(self):
        p=self.f.load();cost=p.verification.cells[0].cost
        with patch.object(CostTable,"lookup",side_effect=AssertionError("must not delegate production")):
            self.assertIsNone(p.lookup(cost.load_signature,cost.existing_io,"h2d",16,execution="production"))

    def test_real_arithmetic_ignores_large_warmup(self):
        prepared=self.f.load()
        cell=prepared.verification.cells[0]
        self.assertEqual(prepared.source_ref,self.f.candidate_ref)
        self.assertEqual(prepared.cell_count,1)
        self.assertEqual((cell.cost.baseline_ns,cell.cost.incremental_or_joint_ns,cell.cost.uncertainty_ns),
                         (101,12,6))
        self.assertEqual((cell.paired_runs,cell.measured_windows,cell.warmup_windows),(3,6,3))
        self.assertFalse(prepared.gpu_verified)
        self.assertFalse(prepared.production_qualified)
        self.assertIsNone(prepared.lookup(cell.cost.load_signature,cell.cost.existing_io,"h2d",16))
        estimate=prepared.lookup(cell.cost.load_signature,cell.cost.existing_io,"h2d",16,execution="cpu_mock")
        self.assertEqual(estimate.total_ns,119)
        self.assertTrue(estimate.mock_only)
        self.assertFalse(estimate.production_qualified)

    def test_forged_native_origin_and_reported_pass_never_qualify(self):
        self.f=Fixture(self.temp.name,origin="native_gpu_recording")
        prepared=self.f.load()
        self.assertFalse(prepared.gpu_verified)
        self.assertFalse(prepared.cpu_mock_table.production_qualified)
        self.assertEqual(prepared.verification.cells[0].origin,"native_gpu_recording")
        manifest=semantic_manifest(prepared.verification)
        self.assertFalse(manifest["gpu_execution_proved"])
        self.assertFalse(manifest["method_is_confidence_interval"])

    def test_no_io_plus_joint_basis_does_not_double_add_existing_cost(self):
        self.f=Fixture(self.temp.name,basis="no_io_plus_joint")
        prepared=self.f.load()
        self.assertEqual(prepared.verification.cells[0].cost.total_ns,119)

    def test_exact_missing_cell_returns_unknown(self):
        p=self.f.load();cell=p.verification.cells[0].cost
        self.assertIsNone(p.lookup(cell.load_signature,cell.existing_io,"h2d",8,execution="cpu_mock"))

    def test_immutable_result_and_table(self):
        p=self.f.load()
        with self.assertRaises(FrozenInstanceError):p.gpu_verified=True
        with self.assertRaises(AttributeError):p.cpu_mock_table.production_qualified=True

    def test_wrapper_pair_missing(self):
        self.mutate_doc("action_wrapper",lambda d:d["runs"].pop())
    def test_duplicate_pair_identity(self):
        self.mutate_doc("baseline_wrapper",lambda d:d["runs"].append(deepcopy(d["runs"][0])))
    def test_seed_mismatch(self):
        self.mutate_doc("action_wrapper",lambda d:d["runs"][0].update(seed=99))
    def test_token_work_mismatch(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][1].update(output_tokens=2))
    def test_wrapper_token_count_mismatch(self):
        self.mutate_doc("action_wrapper",lambda d:d["runs"][0].update(output_tokens=3))
    def test_missing_full_accepted_io_drain(self):
        self.mutate_doc("action_wrapper",lambda d:d["runs"][0].update(accepted_io_drained=False))
    def test_incomplete_actual_physical_bytes(self):
        self.mutate_doc("action_wrapper",lambda d:d["runs"][0].update(completed_new_io=io(2,16)))
    def test_failed_exit_not_an_observation(self):
        self.mutate_doc("baseline_wrapper",lambda d:d["runs"][0].update(exit_code=1))
    def test_order_metadata_disagrees_with_clock(self):
        self.mutate_doc("action_wrapper",lambda d:d["runs"][0].update(arm_order="BA"))
    def test_unbalanced_calibration_order(self):
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][1]["arm_order"]="AB"
        self.f.refresh();self.reject()
    def test_duplicate_trace_seed_independent_unit(self):
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][1]["trace_sha256"]=self.f.docs[role]["runs"][0]["trace_sha256"]
            self.f.docs[role]["runs"][1]["seed"]=0
        self.f.refresh();self.reject()
    def test_heldout_trace_leakage(self):
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][2]["trace_sha256"]=self.f.docs[role]["runs"][0]["trace_sha256"]
        self.f.entries[2]["trace_sha256"]=self.f.entries[0]["trace_sha256"]
        self.f.refresh();self.reject()
    def test_heldout_prefix_family_leakage(self):
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][2]["prefix_family_sha256"]=self.f.docs[role]["runs"][0]["prefix_family_sha256"]
        self.f.entries[2]["prefix_family_sha256"]=self.f.entries[0]["prefix_family_sha256"]
        self.f.refresh();self.reject()
    def test_no_validation_pair(self):
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][2]["split"]="calibration"
        self.f.entries[2]["split"]="calibration"
        self.f.refresh();self.reject()
    def test_unknown_window_pair(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][0].update(pair_id="outside"))
    def test_duplicate_raw_window(self):
        self.mutate_doc("action_observations",lambda d:d["windows"].append(deepcopy(d["windows"][0])))
    def test_missing_paired_window(self):
        self.mutate_doc("action_observations",lambda d:d["windows"].pop())
    def test_nonpositive_step_duration(self):
        self.mutate_doc("baseline_observations",lambda d:d["windows"][0].update(end_ns=1))
    def test_window_outside_run(self):
        self.mutate_doc("baseline_observations",lambda d:d["windows"][0].update(end_ns=999999))
    def test_overlapping_steps_rejected(self):
        self.mutate_doc("baseline_observations",lambda d:d["windows"][2].update(start_ns=10050))
    def test_warmup_mixed_into_measurement(self):
        self.mutate_doc("baseline_observations",lambda d:d["windows"][0].update(phase="measured"))
    def test_raw_load_mismatch(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][0]["load"].update(batch=3))
    def test_existing_io_double_accounting_rejected(self):
        self.mutate_doc("baseline_observations",lambda d:d["windows"][0].update(existing_io=io(2,16,"ssd_read")))
    def test_extra_action_wrong_stage(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][0].update(new_io=io(2,16,"d2h")))
    def test_action_wrong_actual_bytes(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][0].update(new_io=io(2,24)))
    def test_baseline_extra_io_rejected(self):
        self.mutate_doc("baseline_observations",lambda d:d["windows"][0].update(new_io=io(2,16)))
    def test_bool_counter_rejected(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][0]["new_io"][2].update(ops=True))
    def test_invalid_counter_zero_bytes(self):
        self.mutate_doc("action_observations",lambda d:d["windows"][0]["new_io"][2].update(bytes=0))
    def test_heldout_window_tail_cannot_be_hidden_in_run_mean(self):
        rows=self.f.docs["action_observations"]["windows"]
        rows[7]["end_ns"]+=20
        rows[8]["end_ns"]-=20
        self.f.refresh();self.reject()
    def test_manual_plan_fields_must_equal_frozen_actual_bytes(self):
        args=self.f.args()
        bound=load_production_candidate(self.f.root,"candidate.json",
            expected_context=args["expected_context"],qualification_ref=args["qualification_ref"],
            expected_verifier_ref=args["expected_verifier_ref"])
        plan=load_verification_plan(self.f.root,"plan.json",expected_plan_ref=self.f.plan_ref)
        with self.assertRaises(TableContractError):
            verify_paired_measurements(self.f.root,bound,expected_plan=replace(plan,min_validation_pairs=0))
    def test_manual_bound_fields_must_equal_actual_candidate_bytes(self):
        self.f.cell["baseline_ns"]=100
        self.f.refresh()
        args=self.f.args()
        bound=load_production_candidate(self.f.root,"candidate.json",
            expected_context=args["expected_context"],qualification_ref=args["qualification_ref"],
            expected_verifier_ref=args["expected_verifier_ref"])
        forged=replace(bound,cells=(replace(bound.cells[0],cost=replace(bound.cells[0].cost,baseline_ns=101)),))
        plan=load_verification_plan(self.f.root,"plan.json",expected_plan_ref=self.f.plan_ref)
        with self.assertRaises(TableContractError):
            verify_paired_measurements(self.f.root,forged,expected_plan=plan)
    def test_same_workload_new_trace_seed_labels_are_not_independent(self):
        value=self.f.entries[0]["workload_sha256"]
        self.f.entries[1]["workload_sha256"]=value
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][1]["workload_sha256"]=value
        self.f.refresh();self.reject()
    def test_heldout_same_workload_cannot_hide_behind_other_trace_family(self):
        value=self.f.entries[0]["workload_sha256"]
        self.f.entries[2]["workload_sha256"]=value
        for role in ("baseline_wrapper","action_wrapper"):
            self.f.docs[role]["runs"][2]["workload_sha256"]=value
        self.f.refresh();self.reject()
    def global_split_rejected(self, field):
        entry=deepcopy(self.f.entries[2])
        entry["cell_id"]="cell-2"
        entry[field]=self.f.entries[0][field]
        self.f.entries.append(entry)
        self.f.refresh()
        self.f.plan["action_operations"]["cell-2"]=2
        self.f.plan_ref=self.f.dump("plan.json",self.f.plan)
        with self.assertRaises(TableContractError):
            load_verification_plan(self.f.root,"plan.json",expected_plan_ref=self.f.plan_ref)
    def test_global_plan_cross_cell_trace_leakage(self):
        self.global_split_rejected("trace_sha256")
    def test_global_plan_cross_cell_prefix_family_leakage(self):
        self.global_split_rejected("prefix_family_sha256")
    def test_global_plan_cross_cell_workload_leakage(self):
        self.global_split_rejected("workload_sha256")
    def test_plan_entry_cell_missing_operation_geometry_rejected(self):
        entry=deepcopy(self.f.entries[2]);entry["cell_id"]="unknown-cell"
        self.f.entries.append(entry);self.f.refresh()
        with self.assertRaises(TableContractError):
            load_verification_plan(self.f.root,"plan.json",expected_plan_ref=self.f.plan_ref)
    def test_active_decode_greater_than_execution_batch_rejected(self):
        self.f.load_state["active_decode"]=3
        self.f.cell["load"]["active_decode"]=3
        for role in ("baseline_observations","action_observations"):
            for row in self.f.docs[role]["windows"]:
                row["load"]["active_decode"]=3
        self.f.refresh();self.reject()
    def test_candidate_declared_cost_not_raw_recomputed(self):
        self.f.cell["uncertainty_ns"]=5;self.f.refresh();self.reject()
    def test_candidate_declared_counts_not_raw_recomputed(self):
        self.f.cell["windows_reported"]=99;self.f.refresh();self.reject()
    def test_source_context_change(self):
        self.mutate_doc("action_observations",lambda d:d["context"].update(native_source_sha256="f"*64))
    def test_mixed_origin_rejected(self):
        self.mutate_doc("action_observations",lambda d:d.update(origin="native_gpu_recording"))
    def test_non_supported_origin_rejected(self):
        self.mutate_doc("action_observations",lambda d:d.update(origin="gpu_verified"))
    def test_frozen_split_real_bytes_changed(self):
        (self.f.root/"split.json").write_text("{}");self.reject()
    def test_frozen_workload_cannot_be_self_declared(self):
        self.f.entries[0]["input_tokens"]=33;self.f.refresh();self.reject()
    def test_changed_raw_bytes_after_reference(self):
        (self.f.root/"action_observations.json").write_text("{}");self.reject()
    def test_duplicate_json_keys_rejected(self):
        p=self.f.root/"plan.json";p.write_text('{"schema_version":1,"schema_version":1}')
        self.f.plan_ref=self.f.ref("plan.json");self.reject()
    def test_nonfinite_json_rejected(self):
        p=self.f.root/"plan.json";p.write_text('{"schema_version":NaN}')
        self.f.plan_ref=self.f.ref("plan.json");self.reject()
    def test_interval_margin_is_not_fabricated(self):
        self.f.context=replace(self.f.context,uncertainty_method="paired_interval_margin")
        self.f.refresh();self.reject()
    def test_analysis_arithmetic_mismatch(self):
        self.f.analysis["baseline_ns"]=999
        ref=self.f.dump("analysis.json",self.f.analysis)
        self.f.cell["measurement_refs"]["pair_analysis"]=asdict(ref)
        self.f.candidate_ref=self.f.dump("candidate.json",self.f.candidate)
        self.f.qualification["candidate_ref"]=asdict(self.f.candidate_ref)
        self.f.qualification["measurement_refs"]=[self.f.cell["measurement_refs"][name] for name in self.f.cell["measurement_refs"]]
        self.f.qualification_ref=self.f.dump("qualification.json",self.f.qualification)
        self.reject()
    def test_analysis_raw_chain_mismatch(self):
        self.f.analysis["evidence_refs"]["action_wrapper"]=self.f.analysis["evidence_refs"]["baseline_wrapper"]
        ref=self.f.dump("analysis.json",self.f.analysis)
        self.f.cell["measurement_refs"]["pair_analysis"]=asdict(ref)
        self.f.candidate_ref=self.f.dump("candidate.json",self.f.candidate)
        self.f.qualification["candidate_ref"]=asdict(self.f.candidate_ref)
        self.f.qualification["measurement_refs"]=[self.f.cell["measurement_refs"][name] for name in self.f.cell["measurement_refs"]]
        self.f.qualification_ref=self.f.dump("qualification.json",self.f.qualification)
        self.reject()
    def test_final_byte_recheck_rejects_timing_toctou(self):
        import prefix_io_control.p4_paired_measurement_verifier as module
        original=module._verify_cell
        def mutate(*args):
            result=original(*args)
            (self.f.root/"action_observations.json").write_text("{}")
            return result
        with patch.object(module,"_verify_cell",side_effect=mutate):
            self.reject()
    def test_manifest_append_only_and_real_sha(self):
        p=self.f.load()
        ref=write_semantic_verification_manifest(self.f.root,p.verification,"receipts/cpu.json")
        ref.verify(self.f.root)
        self.assertFalse(json.loads((self.f.root/ref.path).read_text())["gpu_verified"])
        with self.assertRaises(TableContractError):
            write_semantic_verification_manifest(self.f.root,p.verification,"receipts/cpu.json")
    def test_manifest_traversal_and_symlink_rejected(self):
        p=self.f.load()
        for path in ("../outside.json","/tmp/out.json","a/../out.json","a\\b.json"):
            with self.assertRaises(TableContractError):
                write_semantic_verification_manifest(self.f.root,p.verification,path)
        (self.f.root/"alias").symlink_to(self.f.root,target_is_directory=True)
        with self.assertRaises(TableContractError):
            write_semantic_verification_manifest(self.f.root,p.verification,"alias/out.json")
    def test_manifest_rechecks_actual_input_bytes(self):
        p=self.f.load()
        (self.f.root/"plan.json").write_text("{}")
        with self.assertRaises(TableContractError):
            write_semantic_verification_manifest(self.f.root,p.verification,"receipt.json")


if __name__=="__main__":
    unittest.main()
