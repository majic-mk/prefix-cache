"""Production-candidate parser rejection tests: every result is still CPU-only."""
from dataclasses import asdict, replace, FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from prefix_io_control.p4_production_table_contract import (
    EvidenceRef, TableContext, TableContractError, BlockedProductionCandidate,
    load_production_candidate,
)

class ProductionContractTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name).resolve()
        self.context=TableContext(
            "CPU-METADATA-FIXTURE","a"*64,"GPU-CPU-FIXTURE","b"*64,"c"*64,"d"*64,
            "eager","torch-fixture","cuda-fixture","driver-fixture",8,128,64,20,
            "existing_io_plus_delta","paired_interval_margin")
        self.refs={}
        for role in ("baseline_wrapper","action_wrapper","baseline_observations",
                     "action_observations","pair_analysis"):
            p=self.root/(role+".json")
            p.write_text(json.dumps({"fixture_role":role,"CPU_FIXTURE":True}))
            self.refs[role]=self.ref(p.name)
        verifier=self.root/"independent_verifier.py"
        verifier.write_text("raise RuntimeError('CPU fixture verifier must NEVER execute')\n")
        self.verifier=self.ref(verifier.name)
        self.cell={
            "cell_id":"cell-01",
            "load":{"active_decode":2,"batch":2,"prefill_tokens":0,"context_length":16400},
            "existing_io":[{"ops":1,"bytes":8},{"ops":0,"bytes":0},
                           {"ops":0,"bytes":0},{"ops":0,"bytes":0}],
            "stage":"ssd_read","physical_bytes":16,
            "baseline_ns":12,"incremental_or_joint_ns":3,"uncertainty_ns":2,
            "paired_runs_reported":2,"windows_reported":8,
            "measurement_refs":{key:asdict(ref) for key,ref in self.refs.items()},
        }
        self.candidate={"schema_version":1,"scope":"production_candidate",
                        "context":asdict(self.context),"cells":[self.cell]}
        self.save()
    def tearDown(self):
        self.temp.cleanup()
    def ref(self,name):
        data=(self.root/name).read_bytes()
        return EvidenceRef(name,len(data),sha256(data).hexdigest())
    def save(self):
        (self.root/"candidate.json").write_text(json.dumps(self.candidate))
        self.candidate_ref=self.ref("candidate.json")
        allrefs={EvidenceRef.from_mapping(ref) for cell in self.candidate["cells"]
                 for ref in cell["measurement_refs"].values()}
        self.report={
            "schema_version":1,"scope":"production_qualification_candidate",
            "status":"PENDING_INDEPENDENT_VERIFICATION",
            "candidate_ref":asdict(self.candidate_ref),"context":self.candidate["context"],
            "verifier_ref":asdict(self.verifier),
            "measurement_refs":[asdict(ref) for ref in sorted(allrefs,key=lambda ref:ref.path)],
        }
        self.save_report()
    def save_report(self):
        (self.root/"qualification.json").write_text(json.dumps(self.report))
        self.qualification=self.ref("qualification.json")
    def load(self,**kwargs):
        args=dict(expected_context=self.context,qualification_ref=self.qualification,
                  expected_verifier_ref=self.verifier)
        args.update(kwargs)
        return load_production_candidate(self.root,"candidate.json",**args)
    def rejected(self):
        with self.assertRaises(TableContractError):
            self.load()

    def test_bound_candidate_is_still_blocked_and_not_cost_table(self):
        result=self.load()
        self.assertIsInstance(result,BlockedProductionCandidate)
        self.assertTrue(result.metadata_binding_complete)
        self.assertEqual(result.status,"BLOCKED_PRODUCTION_GPU_QUALIFICATION")
        self.assertFalse(result.production_qualified)
        self.assertFalse(result.gpu_verified)
        self.assertIn("independent_real_paired_gpu_verifier",result.missing_requirements)
        self.assertEqual(result.cells[0].cost.total_ns,17)
        with self.assertRaises(AttributeError):
            result.production_qualified=True

    def test_reported_pass_and_matching_hashes_cannot_authorize_gpu(self):
        self.report["status"]="PASS_REPORTED"
        self.save_report()
        result=self.load()
        self.assertFalse(result.production_qualified)
        self.assertFalse(result.gpu_verified)

    def test_mock_and_p3_conditional_scopes_rejected(self):
        for scope in ("mock_only","conditional","P3_CONDITIONAL_TABLE","production_qualified"):
            self.candidate["scope"]=scope
            self.save()
            self.rejected()

    def test_self_qualification_boolean_not_a_schema_field(self):
        self.candidate["gpu_verified"]=True
        self.save()
        self.rejected()

    def test_report_gpu_boolean_cannot_replace_verifier_chain(self):
        self.report["gpu_verified"]=True
        self.save_report()
        self.rejected()

    def test_schema_version_bool_float_and_unknown_rejected(self):
        for value in (True,1.0,2):
            self.candidate["schema_version"]=value
            self.save()
            self.rejected()

    def test_cross_model_gpu_kernel_source_layout_geometry_context_rejected(self):
        for name,value in (
            ("model_sha256","e"*64),("gpu_uuid","GPU-DIFFERENT"),
            ("native_source_sha256","f"*64),("vllm_source_sha256","0"*64),
            ("kv_layout_sha256","1"*64),("kernel_mode","graph"),
            ("transfer_quantum_bytes",16),("stage_capacity_bytes",128),
            ("gpu_pool_bytes",256),("torch_version","changed"),
            ("cuda_version","changed"),("driver_version","changed")):
            self.candidate["context"]=asdict(replace(self.context,**{name:value}))
            self.save()
            self.rejected()

    def test_context_scalar_types_rejected(self):
        for name,value in (("gpu_pool_bytes",True),("transfer_quantum_bytes",0),
                           ("internal_step_budget_ns",1.0),("cost_basis",[])):
            self.candidate["context"]=asdict(self.context)
            self.candidate["context"][name]=value
            self.save()
            self.rejected()

    def test_uncertainty_and_cost_accounting_modes_are_explicit(self):
        for name,value in (("uncertainty_method","unspecified"),
                           ("cost_basis","existing_io_plus_joint_plus_delta")):
            self.candidate["context"]=asdict(self.context)
            self.candidate["context"][name]=value
            self.save()
            self.rejected()

    def test_negative_float_and_bool_cell_counts_rejected(self):
        for name,value in (("physical_bytes",True),("baseline_ns",0),
                           ("incremental_or_joint_ns",-1),("uncertainty_ns",1.5),
                           ("windows_reported",True),("paired_runs_reported",0)):
            original=self.cell[name]
            self.cell[name]=value
            self.save()
            self.rejected()
            self.cell[name]=original

    def test_load_state_and_existing_io_strict_types(self):
        for name,value in (("batch",True),("context_length",0),("prefill_tokens",-1)):
            original=self.cell["load"][name]
            self.cell["load"][name]=value
            self.save()
            self.rejected()
            self.cell["load"][name]=original
        self.cell["existing_io"][0]["ops"]=False
        self.save()
        self.rejected()

    def test_legal_transfer_quantum_and_known_stage_required(self):
        for name,value in (("physical_bytes",17),("stage","combined_untyped")):
            original=self.cell[name]
            self.cell[name]=value
            self.save()
            self.rejected()
            self.cell[name]=original

    def test_exact_four_stage_io_and_all_pair_roles_required(self):
        self.cell["existing_io"]=self.cell["existing_io"][:3]
        self.save()
        self.rejected()
        self.cell["existing_io"].append({"ops":0,"bytes":0})
        del self.cell["measurement_refs"]["pair_analysis"]
        self.save()
        self.rejected()

    def test_same_control_and_action_evidence_is_not_a_pair(self):
        self.cell["measurement_refs"]["action_wrapper"]=self.cell["measurement_refs"]["baseline_wrapper"]
        self.save()
        self.rejected()

    def test_candidate_missing_and_extra_keys_rejected(self):
        self.candidate["unused"]=0
        self.save()
        self.rejected()
        del self.candidate["unused"]
        del self.candidate["context"]["kernel_mode"]
        self.save()
        self.rejected()

    def test_duplicate_cell_ids_and_exact_cost_keys_rejected(self):
        self.candidate["cells"].append(dict(self.cell))
        self.save()
        self.rejected()
        self.candidate["cells"][-1]["cell_id"]="different-id-same-action"
        self.save()
        self.rejected()

    def test_cell_window_is_bounded(self):
        self.candidate["cells"]=[dict(self.cell) for _ in range(129)]
        self.save()
        self.rejected()

    def test_absolute_traversal_dot_backslash_and_empty_paths_rejected(self):
        for value in ("/tmp/nope","../nope",".","a/../nope","a\\nope","", "C:/outside"):
            with self.assertRaises(TableContractError):
                load_production_candidate(self.root,value,expected_context=self.context,
                    qualification_ref=self.qualification,expected_verifier_ref=self.verifier)

    def test_leaf_symlink_inside_project_is_rejected(self):
        (self.root/"alias.json").symlink_to(self.root/"baseline_wrapper.json")
        ref=asdict(self.refs["baseline_wrapper"])
        ref["path"]="alias.json"
        self.cell["measurement_refs"]["baseline_wrapper"]=ref
        self.save()
        self.rejected()

    def test_parent_directory_symlink_is_rejected(self):
        (self.root/"real").mkdir()
        (self.root/"real"/"value.json").write_text("{}")
        (self.root/"alias").symlink_to(self.root/"real",target_is_directory=True)
        ref=asdict(self.ref("real/value.json"))
        ref["path"]="alias/value.json"
        self.cell["measurement_refs"]["baseline_wrapper"]=ref
        self.save()
        self.rejected()

    def test_changed_measurement_sha_and_size_rejected(self):
        (self.root/"baseline_observations.json").write_text('{"changed":true}')
        self.rejected()

    def test_changed_candidate_not_bound_to_qualification(self):
        self.cell["baseline_ns"]=13
        (self.root/"candidate.json").write_text(json.dumps(self.candidate))
        self.rejected()

    def test_candidate_change_during_cell_binding_cannot_bind_stale_values(self):
        from prefix_io_control import p4_production_table_contract as contract
        original=contract._cell
        changed=dict(self.candidate)
        changed["cells"]=[dict(self.cell,baseline_ns=99)]
        future_bytes=json.dumps(changed).encode("utf-8")
        # Pin a receipt for the later version before the loader starts. It must
        # not silently attach those bytes to the earlier parsed cell values.
        self.report["candidate_ref"]=asdict(EvidenceRef("candidate.json",len(future_bytes),
                                                    sha256(future_bytes).hexdigest()))
        self.save_report()
        def mutate(*args):
            result=original(*args)
            (self.root/"candidate.json").write_bytes(future_bytes)
            return result
        with patch.object(contract,"_cell",side_effect=mutate):
            self.rejected()

    def test_changed_verifier_and_forged_verifier_chain_rejected(self):
        (self.root/"independent_verifier.py").write_text("changed")
        self.rejected()
        new=self.ref("independent_verifier.py")
        self.report["verifier_ref"]=asdict(new)
        self.save_report()
        self.rejected()

    def test_qualification_context_candidate_and_evidence_chain_strict(self):
        self.report["context"]=asdict(replace(self.context,kernel_mode="other"))
        self.save_report()
        self.rejected()
        self.save()
        self.report["measurement_refs"].pop()
        self.save_report()
        self.rejected()
        self.save()
        self.report["candidate_ref"]["sha256"]="e"*64
        self.save_report()
        self.rejected()

    def test_p3_or_cpu_qualification_report_status_rejected(self):
        for scope,status in (("P3_CONDITIONAL","PASS"),
                             ("production_qualification_candidate","CPU_PASS"),
                             ("production_qualification_candidate","PASS_GPU")):
            self.report["scope"]=scope
            self.report["status"]=status
            self.save_report()
            self.rejected()

    def test_verifier_is_not_executed_by_parser(self):
        # The fixture source raises immediately if executed. Parsing only hashes it.
        self.assertFalse(self.load().production_qualified)

    def test_duplicate_json_keys_and_nonfinite_numbers_rejected(self):
        path=self.root/"candidate.json"
        path.write_text('{"schema_version":1,"schema_version":1,"scope":"production_candidate"}')
        self.rejected()
        path.write_text('{"schema_version":NaN}')
        self.rejected()

    def test_metadata_reads_are_bounded_before_hashing(self):
        with patch("prefix_io_control.p4_production_table_contract.MAX_JSON_BYTES",20):
            self.rejected()
        with patch("prefix_io_control.p4_production_table_contract.MAX_EVIDENCE_BYTES",1):
            self.rejected()

    def test_independently_frozen_context_and_verifier_types_required(self):
        with self.assertRaises(TableContractError):
            self.load(expected_context=asdict(self.context))
        with self.assertRaises(TableContractError):
            self.load(expected_verifier_ref=asdict(self.verifier))


if __name__=="__main__":
    unittest.main()
