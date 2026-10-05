"""V2 offline CLI uses complete bound raw roles; no sampling/GPU is invented."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from tests.prefix_io_v1_p4_measurements.test_semantic_v2 import V2Fixture
from prefix_io_control.p4_production_table_contract import EvidenceRef, TableContractError
SCRIPTS=Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("p4_raw_cli_v2_test",SCRIPTS/"prepare_p4_raw_pair.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import prepare_p4_gpu_next_day as base

class OfflineV2CliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.f=V2Fixture(self.temp.name)
        p=self.f.root/base.PERMISSION;p.parent.mkdir(parents=True)
        p.write_text("allow_gpu_runs: false\n")
        auth=dict(schema_version=1,allow_project_local_edits=True,allow_cpu_tests=True,
            allow_gpu_initialization=False,allow_gpu_runs=False,allow_model_downloads=False,
            allow_driver_or_system_changes=False,base_permissions=base.ref(self.f.root,base.PERMISSION))
        p=self.f.root/base.AUTH_CHOICES[0];p.parent.mkdir(parents=True);p.write_text(json.dumps(auth))
        self.repin()
    def tearDown(self):self.temp.cleanup()
    def repin(self):
        self.bundle=dict(schema_version=2,scope="p4_explicit_pair_observations",
            context=asdict(self.f.context),cell_id="cell-1",origin="cpu_fixture",
            cell_geometry={key:deepcopy(self.f.cell[key]) for key in
                           ("cell_id","load","existing_io","stage","physical_bytes")},
            raw_refs={role:deepcopy(self.f.cell["measurement_refs"][role]) for role in
                      ("baseline_wrapper","action_wrapper","baseline_observations","action_observations")})
        self.write_bundle()
        self.f.dump("plan-ref.json",asdict(self.f.plan_ref))
        self.f.dump("verifier-ref.json",asdict(self.f.verifier_ref))
    def write_bundle(self):
        bref=self.f.dump("bundle.json",self.bundle)
        self.f.dump("bundle-ref.json",asdict(bref))
    def argv(self,output="cli-v2"):
        return ["--project",str(self.f.root),"--bundle","bundle.json","--bundle-ref","bundle-ref.json",
            "--plan","plan.json","--plan-ref","plan-ref.json","--verifier-ref","verifier-ref.json",
            "--output-dir",base.OUT+"/"+output]
    def reject(self):
        with self.assertRaises((ValueError,TableContractError)):m.main(self.argv())

    def test_v2_actual_cli_main_raw_builder_full_loader_roundtrip(self):
        self.assertEqual(m.main(self.argv()),0)
        receipt=json.loads((self.f.root/base.OUT/"cli-v2/receipt.json").read_text())
        self.assertEqual((receipt["schema_version"],receipt["measurement_schema_version"]),(2,2))
        self.assertEqual(len(receipt["raw_refs"]),4)
        self.assertEqual(len(receipt["ingested_source_refs"]),11)
        self.assertFalse(receipt["production_qualified"]);self.assertFalse(receipt["gpu_verified"])
        self.assertEqual(receipt["GPU_runs_performed"],0)
        self.assertFalse(receipt["effect_verified"]);self.assertFalse(receipt["GPU_collector_implemented"])
        cell=json.loads((self.f.root/base.OUT/"cli-v2/candidate.json").read_text())["cells"][0]
        self.assertEqual((cell["baseline_ns"],cell["incremental_or_joint_ns"],cell["uncertainty_ns"]),(101,12,6))
        self.assertEqual(len(cell["measurement_refs"]),5)
    def test_output_is_append_new(self):
        m.main(self.argv());self.reject()
    def test_original_source_bytes_drift_rejected(self):
        (self.f.root/"action_wrapper.json").write_text("{}");self.reject()
        self.assertFalse((self.f.root/base.OUT).exists())
    def test_full_trace_bytes_drift_rejected(self):
        (self.f.root/"complete-action-p0.json").write_text("{}");self.reject()
        self.assertFalse((self.f.root/base.OUT).exists())
    def test_event_reference_source_drift_rejected(self):
        (self.f.root/self.f.source_name).write_text("# altered\n");self.reject()
        self.assertFalse((self.f.root/base.OUT).exists())
    def test_raw_role_v1_cannot_enter_v2_bundle(self):
        self.f.docs["action_wrapper"]["schema_version"]=1
        self.f.refresh();self.repin();self.reject()
        self.assertFalse((self.f.root/base.OUT).exists())
    def test_complete_trace_v1_cannot_enter_v2_bundle(self):
        self.f.traces[("action","p0")]["schema_version"]=1
        self.f.refresh();self.repin();self.reject()
        self.assertFalse((self.f.root/base.OUT).exists())
    def test_bundle_v1_shape_does_not_enable_v2_refs(self):
        self.bundle["schema_version"]=1;self.write_bundle();self.reject()
        self.assertFalse((self.f.root/base.OUT).exists())
    def test_v2_bundle_requires_independently_pinned_v2_plan(self):
        plan=deepcopy(self.f.plan);plan["schema_version"]=1
        del plan["selections"];del plan["timing_contract"]
        pref=self.f.dump("plan.json",plan);self.f.dump("plan-ref.json",asdict(pref))
        self.reject();self.assertFalse((self.f.root/base.OUT).exists())
    def test_fifth_raw_role_cannot_be_added(self):
        self.bundle["raw_refs"]["pair_analysis"]=deepcopy(self.f.cell["measurement_refs"]["pair_analysis"])
        self.write_bundle();self.reject()
    def test_claimed_origin_must_match_all_raw_roles(self):
        self.bundle["origin"]="native_gpu_recording";self.write_bundle();self.reject()
    def test_claimed_context_must_match_frozen_plan(self):
        self.bundle["context"]["native_source_sha256"]="f"*64;self.write_bundle();self.reject()
    def test_original_raw_source_changed_after_assembly_rejected(self):
        import prefix_io_control.p4_raw_pair_recorder as recorder
        original=recorder.assemble_raw_pair
        def mutate(*args,**kwargs):
            result=original(*args,**kwargs)
            (self.f.root/"baseline_wrapper.json").write_text("{}")
            return result
        with patch.object(recorder,"assemble_raw_pair",side_effect=mutate):self.reject()
        self.assertFalse((self.f.root/base.OUT/"cli-v2/receipt.json").exists())
    def test_check_launch_stops_before_v2_inputs(self):
        (self.f.root/"bundle.json").unlink()
        self.assertEqual(m.main(self.argv()+["--check-launch"]),78)
        self.assertFalse((self.f.root/base.OUT).exists())

if __name__=="__main__":unittest.main()
