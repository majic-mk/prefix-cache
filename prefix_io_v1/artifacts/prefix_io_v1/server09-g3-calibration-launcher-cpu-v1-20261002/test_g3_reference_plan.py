"""CPU reference contract fixtures; no human/native/GPU authority is created."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--baseline-path", type=Path, default=HERE / "gpu-source-lock-metrics-v3-final.json")
PARSER.add_argument("--g2-baseline-path", type=Path, default=HERE.parents[2] /
    "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/gpu-source-lock-candidate.json")
ARGS, REST = PARSER.parse_known_args()
blocked = []


class BackendBlocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache"):
            blocked.append(fullname)
            raise AssertionError("CPU reference test forbids backend import")


guard = BackendBlocker()
sys.meta_path.insert(0, guard)
SPEC = importlib.util.spec_from_file_location("g3_reference_cpu_fixture", HERE / "g3_reference_plan.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def ref(path, data):
    return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


class ReferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = ARGS.baseline_path.read_bytes()
        cls.g2 = ARGS.g2_baseline_path.read_bytes()
        cls.before = {p: p.read_bytes() for p in (HERE / "g3_reference_plan.py", C.PARENT_PATH,
            ARGS.baseline_path, ARGS.g2_baseline_path)}
        cls.backend_before = {n for n in sys.modules if n.split(".")[0] in
            ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache")}

    @classmethod
    def tearDownClass(cls):
        assert not blocked
        for p, data in cls.before.items():
            assert p.read_bytes() == data
        assert cls.backend_before == {n for n in sys.modules if n.split(".")[0] in
            ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache")}

    def setUp(self):
        self.rows = [ref(C.DELIVERY + "/CPU_REFERENCE_FIXTURE.py", b"CPU_ONLY=True\n")]
        self.input = ref(C.DELIVERY + "/CPU_REFERENCE_INPUT_FIXTURE.json", b'{"CPU_fixture":true}')
        self.analyzer = ref(C.DELIVERY + "/CPU_REFERENCE_ANALYZER_FIXTURE.json", b'{"CPU_analyzer":true}')
        self.plan = C.build_reference_plan(self.rows, self.input, analyzer_plan_ref=self.analyzer)
        self.plan_ref = ref(C.DELIVERY + "/CPU_REFERENCE_PLAN_FIXTURE.json", json.dumps(self.plan).encode())
        self.lock_ref = ref(C.DELIVERY + "/reference-source-lock-fixture.json", b"CPU lock fixture only")
        self.human_ref = ref(C.DELIVERY + "/HUMAN_CPU_REFERENCE_FIXTURE.json", b"CPU fixture; not a human reply")
        self.scope = C.scope_template(self.lock_ref, self.plan_ref, self.input, self.analyzer)
        self.scope.update(status="USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC",
            allow_gpu_initialization=True, allow_gpu_runs=True, human_authorization_record=self.human_ref)
        self.human = {key: copy.deepcopy(value) for key, value in self.scope.items()
            if key not in ("schema_version", "status", "human_authorization_record")}
        self.human.update(authorization_origin="direct_human_reply", question="CPU fixture question",
            verbatim_user_answer="CPU fixture; not authorization", reply_observed_utc="CPU fixture", question_item_id="CPU fixture")
        self.refs = {row["path"]: row for row in (C.PERMISSIONS_REF, self.plan_ref, self.input, self.analyzer)}
        self.ledger = dict(gpu_wall_seconds=28800 - 10990.8596, events=[], active_reservation=None)
        self.snapshot = dict(origin="actual_statvfs", captured_unix=time.time(), primary_root=C.PROJECT_ROOT,
            free_bytes=12 * 1024 ** 3)
        self.command = [".venv/bin/python", "-B", "CPU_REFERENCE_CHILD_FIXTURE.py"]

    def passed(self, mode):
        return dict(mode=mode, label=C.JOB_NAMES[mode], guard_exit=0, child_exit=0, timed_out=False, error=None,
            session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[],
            original_exit_code=0, original_acquisition_status="PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            original_shutdown_completed=True)

    def active(self, mode="cold"):
        self.ledger["active_reservation"] = dict(label=C.JOB_NAMES[mode], gpu_uuid=C.GPU_UUID,
            seconds_limit=300, reserved_seconds=320, session_id=123, process_group=123,
            permissions=copy.deepcopy(C.PERMISSIONS_REF), command=self.command, id="a" * 32)
        return self.ledger

    def test_fixed_two_routes_and_original_acquirer_flags(self):
        context = C.fixed_context()
        self.assertEqual(C.MODES, ("cold", "paired"))
        self.assertEqual(C.JOB_NAMES, dict(cold="server09-g3-reference-native-01", paired="server09-g3-reference-paired-01"))
        self.assertEqual(context["requests_by_mode"], dict(cold=6, paired=6))
        self.assertEqual((context["total_requests"], context["total_outputs"], context["sampling"]["logprobs"]), (12, 12, 5))
        self.assertEqual(context["reference_routes"], dict(cold=["f", "gpu_hot"], paired=["g_ssd", "g_mem"]))
        self.assertEqual(C.acquisition_arguments("cold")[-2:], ["--native-hot-diagnostic", "--diagnostic-logprobs"])
        self.assertEqual(C.acquisition_arguments("paired")[-1], "--cached-reference-logprobs")
        for mode in C.MODES:
            self.assertNotIn("--curves", C.acquisition_arguments(mode))
            self.assertNotIn("--prompt-manifest", C.acquisition_arguments(mode))

    def test_fixed_models_engine_sampling_and_no_effect_qualification(self):
        context = C.fixed_context()
        self.assertEqual((context["acquisition_domain"], context["sizes"], context["reps"], context["max_num_seqs"], context["iodepth"]), (1024, [128], 3, 1, 4))
        self.assertEqual(context["engine_overrides"]["kv_cache_memory_bytes"], 268435456)
        self.assertEqual(context["engine_overrides"]["max_model_len"], 1040)
        self.assertIs(context["engine_overrides"]["async_scheduling"], False)
        self.assertEqual(context["connector"]["staging_mem"], 0.125)
        self.assertTrue(context["connector"]["preload_share_staging"])
        self.assertEqual(context["connector"]["load_planner"], "off")
        self.assertIs(type(context["sampling"]["temperature"]), float)
        self.assertEqual(math.copysign(1.0, context["sampling"]["temperature"]), 1.0)
        for key in ("curve_output", "cost_qualified", "production_qualified", "performance_claim", "budget_reset", "new_executor"):
            self.assertIs(context[key], False)

    def test_cold_unused_storage_is_separate_from_existing_paired_input(self):
        self.assertEqual(C.STORAGE_RELATIVE, "experiments/prefix_io_v1/runs/server09-g3-reference-native-01-unused-storage")
        self.assertEqual(C.PAIRED_STORAGE_RELATIVE, "experiments/prefix_io_v1/runs/server09-g3-calibration-02-private-storage")
        for mode in C.MODES:
            argv = C.acquisition_arguments(mode)
            self.assertEqual(argv[argv.index("--storage") + 1], C.PROJECT_ROOT + "/" + C.storage_relative(mode))
        self.assertTrue(C.fixed_context()["paired_original_writes_permitted"])
        self.assertFalse(C.fixed_context()["deletes_existing_input"])

    def test_templates_never_authorize_with_or_without_refs(self):
        for template in (C.scope_template(), C.scope_template(self.lock_ref, self.plan_ref, self.input, self.analyzer)):
            self.assertFalse(template["allow_gpu_runs"] or template["allow_gpu_initialization"])
            self.assertIsNone(template["human_authorization_record"])
            with self.assertRaises(ValueError): C.validate_scope_records(template, self.human, self.refs)

    def test_simulated_matching_human_records_remain_cpu_consistency_only(self):
        result = C.validate_scope_records(self.scope, self.human, self.refs)
        self.assertTrue(result["human_record_matches"])
        self.assertFalse(result["authorizes_gpu"] or result["independent_human_provenance_verified"] or result["production_qualified"])

    def test_context_exact_float_type_and_positive_zero(self):
        for value in (0, False, -0.0):
            bad = copy.deepcopy(self.scope)
            bad["qualification_context"]["sampling"]["temperature"] = value
            with self.assertRaises(ValueError): C.validate_scope_records(bad, self.human, self.refs)
        C.same(-0.0, -0.0, "same signed zero")

    def test_missing_input_analyzer_or_human_binding_refuses(self):
        for key in ("input_manifest_ref", "analyzer_plan_ref", "human_authorization_record"):
            bad = copy.deepcopy(self.scope); bad[key] = None
            with self.assertRaises(ValueError): C.validate_scope_records(bad, self.human, self.refs)
        for row in (self.input, self.analyzer):
            bad = dict(self.refs); bad.pop(row["path"])
            with self.assertRaises(ValueError): C.validate_scope_records(self.scope, self.human, bad)

    def test_exact_input_analyzer_and_new_source_locked_plan(self):
        for key in ("source_lock", "input_manifest_ref", "analyzer_plan_ref", "cpu_plan_ref"):
            bad = copy.deepcopy(self.human); bad[key]["sha256"] = "0" * 64
            with self.assertRaises(ValueError): C.validate_scope_records(self.scope, bad, self.refs)
        self.assertEqual(self.plan["input_manifest_ref"], self.input)
        self.assertEqual(self.plan["analyzer_plan_ref"], self.analyzer)
        self.assertFalse(self.plan["source_bytes_verified"] or self.plan["authorizes_gpu"])

    def test_old_calibration_scopes_names_and_gpu_identity_rejected(self):
        for key, value in (("purpose", C.P.PURPOSE), ("maximum_jobs", 3),
            ("maximum_total_planned_reserve_seconds", 960), ("gpu_uuid", "GPU-foreign")):
            bad = copy.deepcopy(self.scope); bad[key] = value
            with self.assertRaises(ValueError): C.validate_scope_records(bad, self.human, self.refs)
        bad = copy.deepcopy(self.scope); bad["permitted_run_names"]["cold"] = "server09-g3-calibration-cold-02"
        with self.assertRaises(ValueError): C.validate_scope_records(bad, self.human, self.refs)

    def test_typed_reply_cannot_fabricate_form_id(self):
        human = copy.deepcopy(self.human)
        human.update(reply_medium="direct_typed_user_message", form_response_id_fabricated=False,
            question_item_id="not_applicable:direct_typed_user_message")
        C.validate_scope_records(self.scope, human, self.refs)
        human["question_item_id"] = "invented_form_response"
        with self.assertRaises(ValueError): C.validate_scope_records(self.scope, human, self.refs)

    def test_source_ancestor_sha_and_order_are_preserved_with_parent_validation(self):
        candidate = C.assemble_source_lock(self.baseline, self.rows, g2_baseline_raw=self.g2)
        result = C.validate_source_manifest(candidate, self.baseline, g2_baseline_raw=self.g2)
        self.assertEqual(result["files"], 4077)
        self.assertEqual(candidate["files"][:4075], json.loads(self.baseline)["files"])
        self.assertEqual(candidate["files"][4075], C.BASELINE_REF)
        self.assertFalse(result["full_source_bytes_verified"] or result["authorizes_gpu"])

    def test_new_append_cap_does_not_accumulate_parent_cap(self):
        rows = [ref(C.DELIVERY + f"/CPU_REFERENCE_APPEND_{i}.py", b"CPU=True\n") for i in range(15)]
        candidate = C.assemble_source_lock(self.baseline, rows, g2_baseline_raw=self.g2)
        self.assertEqual(len(candidate["files"]) - 4075, 16)
        self.assertGreater(len(json.loads(self.baseline)["new_source_refs"]) + len(rows), 32)
        C.validate_source_manifest(candidate, self.baseline, g2_baseline_raw=self.g2)
        with self.assertRaises(ValueError): C.assemble_source_lock(self.baseline, rows + [ref(C.DELIVERY + "/CPU_OVERFLOW.py", b"CPU")])

    def test_ancestor_reformat_or_byte_mutation_cannot_pass(self):
        for raw in (json.dumps(json.loads(self.baseline)).encode(), self.baseline[:-1] + bytes([self.baseline[-1] ^ 1])):
            with self.assertRaises(ValueError): C.assemble_source_lock(raw, self.rows)

    def test_replaced_reordered_or_lost_ancestor_and_fake_flags_refuse(self):
        candidate = C.assemble_source_lock(self.baseline, self.rows)
        variants=[]
        bad=copy.deepcopy(candidate); bad["files"][0]["sha256"]="0"*64; variants.append(bad)
        bad=copy.deepcopy(candidate); bad["files"][0],bad["files"][1]=bad["files"][1],bad["files"][0]; variants.append(bad)
        bad=copy.deepcopy(candidate); bad["files"].pop(0); variants.append(bad)
        bad=copy.deepcopy(candidate); bad["allow_gpu_runs"]=True; variants.append(bad)
        bad=copy.deepcopy(candidate); bad["schema_version"]=True; variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError): C.validate_source_manifest(bad, self.baseline)

    def test_no_prior_replacement_duplicates_or_authority_source_cycles(self):
        for rows in ([self.rows[0], self.rows[0]], [C.BASELINE_REF], [json.loads(self.baseline)["files"][-1]],
            [ref(C.DELIVERY + "/HUMAN_AUTHORIZATION.json", b"CPU")],
            [ref(C.DELIVERY + "/new-source-lock.json", b"CPU")], [ref(C.DELIVERY + "/scope.json", b"CPU")]):
            with self.assertRaises(ValueError): C.assemble_source_lock(self.baseline, rows)

    def test_source_verification_cannot_skip_missing_original_bytes(self):
        candidate=C.assemble_source_lock(self.baseline,self.rows)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for row,raw in ((C.BASELINE_REF,self.baseline),(C.P.BASELINE_REF,self.g2)):
                path=root/row["path"]; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
            raw=json.dumps(candidate).encode(); lock=ref(C.DELIVERY+"/CPU_REFERENCE_CANDIDATE.json",raw)
            path=root/lock["path"]; path.write_bytes(raw)
            with self.assertRaises(ValueError): C.verify_source_lock(root,lock)
            self.assertEqual(path.read_bytes(),raw)

    def test_scope_validation_checks_actual_human_bytes_before_any_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): C.validate_scope(Path(directory),self.scope,self.refs)

    def test_budget_two_jobs_640_old_budget_not_reset(self):
        result=C.validate_budget_and_storage(self.ledger,self.snapshot)
        self.assertAlmostEqual(result["remaining_gpu_seconds"],10990.8596)
        self.assertEqual(result["reserved_gpu_seconds"],640)
        self.assertEqual(result["remaining_storage_reserve_bytes"],2*1024**3)
        self.assertFalse(result["authorizes_gpu"])
        for jobs in (0,3,True):
            with self.assertRaises(ValueError): C.validate_budget_and_storage(self.ledger,self.snapshot,remaining_jobs=jobs)

    def test_budget_reservation_old_usage_and_storage_floor_fail_closed(self):
        for used in (28800-639,True,float("nan")):
            ledger=copy.deepcopy(self.ledger); ledger["gpu_wall_seconds"]=used
            with self.assertRaises(ValueError): C.validate_budget_and_storage(ledger,self.snapshot)
        ledger=copy.deepcopy(self.ledger); ledger["active_reservation"]={}
        with self.assertRaises(ValueError): C.validate_budget_and_storage(ledger,self.snapshot)
        bad=copy.deepcopy(self.snapshot);bad["free_bytes"]=10*1024**3-1
        with self.assertRaises(ValueError): C.validate_budget_and_storage(self.ledger,bad)
        bad=copy.deepcopy(self.snapshot);bad["captured_unix"]-=121
        with self.assertRaises(ValueError): C.validate_budget_and_storage(self.ledger,bad)

    def test_only_successful_cold_drained_allows_paired_then_complete(self):
        self.assertEqual(C.next_job([]),"cold")
        self.assertEqual(C.next_job([self.passed("cold")]),"paired")
        self.assertIsNone(C.next_job([self.passed("cold"),self.passed("paired")]))
        with self.assertRaises(ValueError): C.next_job([self.passed("paired")])
        with self.assertRaises(ValueError): C.next_job([self.passed("cold")]*3)

    def test_failed_shutdown_session_or_original_child_stops_without_retry(self):
        for key,value in (("guard_exit",1),("child_exit",1),("original_exit_code",1),("timed_out",True),
            ("error","original fixture failure"),("original_shutdown_completed",False),("session_drained",False),
            ("session_members_before_cleanup",[123]),("session_members_after_cleanup",[123]),
            ("label","server09-g3-calibration-cold-02")):
            row=self.passed("cold");row[key]=value
            with self.assertRaises(ValueError): C.next_job([row])

    def test_original_guard_witness_has_old_thirteen_keys_new_purpose(self):
        witness=C.validate_execution_guard(self.scope,self.active(),"cold",actual_session_id=123,
            expected_command=self.command,scope_sha256="f"*64)
        self.assertEqual(len(witness),13)
        self.assertEqual(witness["purpose"],C.PURPOSE)
        self.assertEqual(witness["label"],C.JOB_NAMES["cold"])
        self.assertEqual((witness["seconds_limit"],witness["reserved_seconds"]),(300,320))

    def test_guard_unknown_identity_argv_or_used_label_refuses(self):
        for key,value in (("gpu_uuid","GPU-foreign"),("session_id",124),("process_group",124),
            ("reserved_seconds",321),("seconds_limit",True),("command",["foreign"]),("id","bad")):
            ledger=copy.deepcopy(self.active());ledger["active_reservation"][key]=value
            with self.assertRaises(ValueError): C.validate_execution_guard(self.scope,ledger,"cold",actual_session_id=123,
                expected_command=self.command,scope_sha256="f"*64)
        ledger=self.active();ledger["events"].append(dict(label=C.JOB_NAMES["cold"],exit=1))
        with self.assertRaises(ValueError): C.validate_execution_guard(self.scope,ledger,"cold",actual_session_id=123,
            expected_command=self.command,scope_sha256="f"*64)

    def test_cli_template_stdout_only_and_no_launch_action(self):
        with tempfile.TemporaryDirectory() as directory:
            done=subprocess.run([sys.executable,"-B","-I","-S",str(HERE/"g3_reference_plan.py"),"--scope-template"],
                cwd=directory,capture_output=True,text=True,timeout=10)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertFalse(json.loads(done.stdout)["allow_gpu_runs"])
            failed=subprocess.run([sys.executable,"-B","-I","-S",str(HERE/"g3_reference_plan.py"),"--launch"],
                cwd=directory,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(list(Path(directory).iterdir()),[])


if __name__ == "__main__":
    try:
        unittest.main(argv=[sys.argv[0], *REST])
    finally:
        sys.meta_path.remove(guard)
