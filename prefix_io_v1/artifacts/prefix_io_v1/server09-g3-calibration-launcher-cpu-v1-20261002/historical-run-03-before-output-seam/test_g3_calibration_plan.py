"""CPU-only contract fixtures; simulated records do not grant human/run authority."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
BASELINE = Path(os.environ.get("PREFIX_G3_CALIBRATION_BASELINE_PATH", str(HERE.parents[2] /
    "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/gpu-source-lock-candidate.json")))
blocked = []
class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache"):
            blocked.append(fullname)
            raise AssertionError("CPU test forbids backend import")
guard = Blocker()
sys.meta_path.insert(0, guard)
spec = importlib.util.spec_from_file_location("g3_calibration_plan", HERE / "g3_calibration_plan.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def ref(path, raw):
    return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = BASELINE.read_bytes()
        cls.baseline_before = hashlib.sha256(cls.baseline).hexdigest()
        cls.helper_before = (HERE / "g3_calibration_plan.py").read_bytes()
        cls.original_backend_modules = {name for name in sys.modules if name.split(".")[0] in
            ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache")}

    @classmethod
    def tearDownClass(cls):
        assert BASELINE.read_bytes() == cls.baseline
        assert (HERE / "g3_calibration_plan.py").read_bytes() == cls.helper_before
        assert not blocked
        assert cls.original_backend_modules == {name for name in sys.modules if name.split(".")[0] in
            ("torch", "vllm", "numpy", "triton", "cupy", "py_kvcache")}

    def setUp(self):
        self.rows = [ref(C.DELIVERY + "/CPU_FIXTURE_RUNTIME.py", b"CPU_FIXTURE_ONLY = True\n")]
        self.candidate = C.assemble_source_lock(self.baseline, self.rows)
        self.plan = C.build_calibration_plan(C.PROJECT_ROOT, new_source_refs=self.rows)
        self.plan_ref = ref(C.DELIVERY + "/G3_CALIBRATION_CPU_PLAN.json", json.dumps(self.plan).encode())
        self.lock_ref = ref(C.DELIVERY + "/gpu-source-lock.json", json.dumps(self.candidate).encode())
        self.scope = C.scope_template(self.lock_ref, plan_ref=self.plan_ref)
        self.scope.update(status="USER_AUTHORIZED_G3_RAW_COST_PILOT", allow_gpu_initialization=True,
                          allow_gpu_runs=True, human_authorization_record=ref(C.DELIVERY + "/HUMAN_FIXTURE.json", b"CPU ONLY"))
        self.human = {key:copy.deepcopy(self.scope[key]) for key in ("purpose", "gpu_uuid", "allowed_modes",
            "permitted_run_names", "maximum_jobs", "maximum_total_planned_reserve_seconds",
            "seconds_limit_per_job", "guard_reserve_seconds_per_job", "qualification_context",
            "source_lock", "cpu_plan_ref", "base_permissions", "baseline_source_lock")}
        self.human.update(authorization_origin="direct_human_reply", question="CPU fixture question; not actual authorization",
            verbatim_user_answer="CPU fixture answer; not actual authorization", reply_observed_utc="CPU fixture",
            question_item_id="CPU fixture", allow_gpu_initialization=True, allow_gpu_runs=True)
        self.refs = {C.PERMISSIONS_REF["path"]:copy.deepcopy(C.PERMISSIONS_REF), self.plan_ref["path"]:self.plan_ref}
        self.ledger = dict(gpu_wall_seconds=17206.34363487875, events=[], active_reservation=None)
        self.snapshot = dict(origin="actual_statvfs", captured_unix=time.time(), primary_root=C.PROJECT_ROOT,
                             free_bytes=12788760576)
        self.command = [C.PROJECT_ROOT + "/.venv/bin/python", "-B", C.PROJECT_ROOT + "/" + C.DELIVERY + "/CPU_FIXTURE_RUNTIME.py"]

    def passed(self, mode):
        return dict(mode=mode, label=C.JOB_NAMES[mode], guard_exit=0, child_exit=0, timed_out=False, error=None,
            session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[],
            original_exit_code=0, original_acquisition_status="PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            original_shutdown_completed=True)

    def active(self):
        self.ledger["active_reservation"] = dict(label=C.JOB_NAMES["cold"], gpu_uuid=C.GPU_UUID,
            seconds_limit=300, reserved_seconds=320, session_id=123, process_group=123,
            permissions=copy.deepcopy(C.PERMISSIONS_REF), command=self.command, id="a"*32)
        return self.ledger

    def test_complete_baseline_inheritance_no_order_or_row_loss(self):
        report = C.validate_source_manifest(self.candidate, self.baseline)
        self.assertEqual(report["files"], 4052)
        self.assertFalse(report["full_source_bytes_verified"] or report["authorizes_gpu"])
        self.assertEqual(self.candidate["files"][:4050], json.loads(self.baseline)["files"])
        self.assertEqual(self.candidate["files"][4050], C.BASELINE_REF)

    def test_finite_plan_actual_request_counts_and_no_authority(self):
        self.assertEqual([job["requests"] for job in self.plan["jobs"]], [3, 9, 6])
        self.assertEqual(sum(job["requests"] for job in self.plan["jobs"]), 18)
        self.assertFalse(self.plan["authorizes_gpu"] or self.plan["source_bytes_verified"])
        ctx = self.plan["qualification_context"]
        self.assertEqual((ctx["acquisition_domain"], ctx["sizes"], ctx["reps"]), (1024, [128], 3))
        self.assertIs(ctx["engine_overrides"]["async_scheduling"], False)
        self.assertIs(ctx["connector"]["sync_on_store"], False)
        self.assertTrue(ctx["connector"]["preload_share_staging"])
        self.assertIsNone(ctx["connector"]["prefix_cache_break_even_path"])
        self.assertFalse(ctx["curve_output"] or ctx["cost_qualified"] or ctx["production_qualified"] or ctx["performance_claim"])

    def test_template_never_authorizes_even_with_pinned_refs(self):
        template = C.scope_template(self.lock_ref, plan_ref=self.plan_ref)
        self.assertFalse(template["allow_gpu_initialization"] or template["allow_gpu_runs"])
        self.assertIsNone(template["human_authorization_record"])
        with self.assertRaises(ValueError): C.validate_scope_records(template, self.human, self.refs)

    def test_simulated_matching_records_remain_cpu_consistency_only(self):
        result = C.validate_scope_records(self.scope, self.human, self.refs)
        self.assertTrue(result["human_record_matches"])
        self.assertFalse(result["authorizes_gpu"] or result["independent_human_provenance_verified"] or result["production_qualified"])

    def test_real_budget_snapshot_fits_fixed_three_job_reservation(self):
        result = C.validate_budget_and_storage(self.ledger, self.snapshot)
        self.assertAlmostEqual(result["remaining_gpu_seconds"], 11593.65636512125)
        self.assertEqual(result["reserved_gpu_seconds"], 960)
        self.assertEqual(result["remaining_storage_reserve_bytes"], 3*1024**3)
        self.assertFalse(result["authorizes_gpu"] or result["actual_free_space_independently_verified"])

    def test_three_ordered_successes_only_then_complete(self):
        receipts=[]
        for mode in C.MODES:
            self.assertEqual(C.next_job(receipts), mode)
            receipts.append(self.passed(mode))
        self.assertIsNone(C.next_job(receipts))

    def test_original_acquirer_argv_no_curves_or_planned_strategy(self):
        argv = C.acquisition_arguments("paired")
        self.assertNotIn("--curves", argv)
        self.assertNotIn("--prompt-manifest", argv)
        self.assertEqual(argv[:12], ["--mode", "paired", "--domain", "1024", "--sizes", "128", "--reps", "3", "--max-num-seqs", "1", "--iodepth", "4"])
        for mode in ("planned", "restore", "shadow"):
            with self.assertRaises(ValueError): C.acquisition_arguments(mode)

    def test_fixed_server_model_and_alias_cannot_drift(self):
        with self.assertRaises(ValueError): C.build_calibration_plan("/another/server", new_source_refs=self.rows)
        self.scope["qualification_context"]["model_alias"]="renamed/model"
        with self.assertRaises(ValueError): C.validate_scope_records(self.scope, self.human, self.refs)

    def test_complete_baseline_sha_rejects_semantic_same_reformat(self):
        altered = json.dumps(json.loads(self.baseline)).encode()
        with self.assertRaises(ValueError): C.assemble_source_lock(altered, self.rows)

    def test_prior_row_missing_or_reordered_or_metadata_change_rejected(self):
        for mutate in (lambda c:c["files"].pop(0), lambda c:c["files"].reverse(),
                       lambda c:c["files"][0].__setitem__("bytes", 0)):
            c=copy.deepcopy(self.candidate);mutate(c)
            with self.assertRaises(ValueError): C.validate_source_manifest(c, self.baseline)

    def test_original_guard_wrong_mirror_or_boolean_meta_rejected(self):
        c=copy.deepcopy(self.candidate)
        original = next(row for row in c["files"] if row["path"] == C.GUARD_REF["path"])
        original["sha256"]="0"*64
        with self.assertRaises(ValueError): C.validate_source_manifest(c, self.baseline)
        c=copy.deepcopy(self.candidate);c["schema_version"]=True
        with self.assertRaises(ValueError): C.validate_source_manifest(c, self.baseline)

    def test_no_old_path_replacement_or_duplicate_new_source(self):
        for rows in ([json.loads(self.baseline)["files"][0]], self.rows+self.rows):
            with self.assertRaises(ValueError): C.assemble_source_lock(self.baseline, rows)

    def test_new_authority_source_lock_or_external_source_disallowed(self):
        for path in (C.DELIVERY+"/HUMAN_AUTHORIZATION_RECORD.json", C.DELIVERY+"/scope.json",
                     C.DELIVERY+"/gpu-source-lock.json", "third_party/new.py", "../escape.py"):
            with self.assertRaises(ValueError): C.assemble_source_lock(self.baseline, [ref(path,b"fixture")])

    def test_claimed_candidate_allow_gpu_cannot_promote(self):
        self.candidate["allow_gpu_runs"]=True
        with self.assertRaises(ValueError): C.validate_source_manifest(self.candidate,self.baseline)

    def test_partial_local_mirror_cannot_pass_full_source_bytes_gate(self):
        with tempfile.TemporaryDirectory(prefix="g3_calibration_fixture_") as directory:
            root=Path(directory)
            for row,raw in ((self.lock_ref,json.dumps(self.candidate).encode()),(C.BASELINE_REF,self.baseline)):
                p=root/row["path"];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,"size drift"): C.verify_source_lock(root,self.lock_ref)

    def test_checked_bytes_detect_same_size_content_drift(self):
        with tempfile.TemporaryDirectory(prefix="g3_calibration_fixture_") as directory:
            root=Path(directory);p=root/"fixture.json";raw=b"1234";p.write_bytes(raw)
            row=ref("fixture.json",raw)
            self.assertEqual(C.checked_ref(root,row),p.resolve())
            p.write_bytes(b"4321")
            with self.assertRaisesRegex(ValueError,"SHA drift"):C.checked_ref(root,row)

    def test_duplicate_json_and_nonfinite_rejected(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}'):
            with self.assertRaises(ValueError):C.parse_json(raw)

    def test_old_G2_authorization_and_fake_human_origin_rejected(self):
        for key,value in (("purpose","NORMAL_MODEL_FULL_OUTPUT_LIFECYCLE_QUALIFICATION_ONLY"),
                          ("maximum_jobs",2),("authorization_origin","cpu_fixture")):
            human=copy.deepcopy(self.human);human[key]=value
            with self.assertRaises(ValueError): C.validate_scope_records(self.scope,human,self.refs)

    def test_empty_human_reply_or_missing_actual_record_rejected(self):
        self.human["verbatim_user_answer"]=""
        with self.assertRaises(ValueError):C.validate_scope_records(self.scope,self.human,self.refs)
        with self.assertRaises(ValueError):C.validate_scope(Path.cwd(),C.scope_template(),self.refs)

    def test_typed_scope_geometry_output_or_strategy_drift_rejected(self):
        for key,value in (("outputs_per_request",128),("reps",True),("curve_output",True),("strategies","joint")):
            scope=copy.deepcopy(self.scope);scope["qualification_context"][key]=value
            with self.assertRaises(ValueError):C.validate_scope_records(scope,self.human,self.refs)
        self.scope["qualification_context"]["sampling"]["temperature"]=0
        with self.assertRaises(ValueError):C.validate_scope_records(self.scope,self.human,self.refs)

    def test_budget_unresolved_nonfinite_bool_and_insufficient_rejected(self):
        for key,value in (("active_reservation",{}),("gpu_wall_seconds",float("nan")),
                          ("gpu_wall_seconds",True),("gpu_wall_seconds",28800-959)):
            ledger=copy.deepcopy(self.ledger);ledger[key]=value
            with self.assertRaises(ValueError):C.validate_budget_and_storage(ledger,self.snapshot)

    def test_storage_aggregate_three_reserves_not_one_rejected(self):
        self.snapshot["free_bytes"]=C.STORAGE_FLOOR_BYTES+2*C.JOB_STORAGE_RESERVE_BYTES
        with self.assertRaises(ValueError):C.validate_budget_and_storage(self.ledger,self.snapshot)
        self.assertEqual(C.validate_budget_and_storage(self.ledger,self.snapshot,remaining_jobs=2)["remaining_storage_reserve_bytes"],2*C.JOB_STORAGE_RESERVE_BYTES)

    def test_stale_estimated_AUX_boolean_storage_rejected(self):
        for key,value in (("captured_unix",time.time()-121),("origin","old_metadata"),
                          ("primary_root","/root/prefix-io-v1-validation"),("free_bytes",True)):
            snapshot=copy.deepcopy(self.snapshot);snapshot[key]=value
            with self.assertRaises(ValueError):C.validate_budget_and_storage(self.ledger,snapshot)

    def test_prior_failure_shutdown_missing_or_live_children_stops(self):
        for key,value in (("guard_exit",1),("child_exit",70),("original_exit_code",1),
                          ("timed_out",True),("error","native error"),("session_drained",False),
                          ("session_members_before_cleanup",[11]),("session_members_after_cleanup",[12]),
                          ("original_shutdown_completed",False),("original_acquisition_status","FAILED")):
            row=self.passed("cold");row[key]=value
            with self.assertRaises(ValueError):C.next_job([row])

    def test_retry_reorder_fourth_or_bool_success_rejected(self):
        for rows in ([self.passed("populate")],[self.passed("cold"),self.passed("cold")],
                     [self.passed(m) for m in C.MODES]+[self.passed("paired")]):
            with self.assertRaises(ValueError):C.next_job(rows)
        row=self.passed("cold");row["guard_exit"]=False
        with self.assertRaises(ValueError):C.next_job([row])

    def test_active_guard_matches_actual_argv_identity_and_no_ledger_mutation(self):
        ledger=self.active();before=copy.deepcopy(ledger)
        result=C.validate_execution_guard(self.scope,ledger,"cold",actual_session_id=123,expected_command=self.command,scope_sha256="f"*64)
        self.assertEqual(result["session_id"],123)
        self.assertEqual(result["guard_command"],self.command)
        self.assertEqual(ledger,before)

    def test_guard_foreign_SID_GPU_argv_permissions_reserve_rejected(self):
        ledger=self.active()
        for key,value in (("session_id",124),("gpu_uuid","GPU-old"),("command",["different"]),
                          ("permissions",{}),("reserved_seconds",300),("seconds_limit",True)):
            drift=copy.deepcopy(ledger);drift["active_reservation"][key]=value
            with self.assertRaises(ValueError):C.validate_execution_guard(self.scope,drift,"cold",actual_session_id=123,expected_command=self.command,scope_sha256="f"*64)

    def test_no_actual_guard_or_CPU_template_cannot_execute(self):
        no_active=copy.deepcopy(self.ledger)
        for scope,ledger in ((self.scope,no_active),(C.scope_template(),self.active())):
            with self.assertRaises(ValueError):C.validate_execution_guard(scope,ledger,"cold",actual_session_id=123,expected_command=self.command,scope_sha256="f"*64)

    def test_guard_scope_context_drift_is_not_authorized_by_caller_boolean(self):
        ledger=self.active()
        self.scope["qualification_context"]["engine_overrides"]["async_scheduling"]=True
        with self.assertRaises(ValueError):C.validate_execution_guard(self.scope,ledger,"cold",actual_session_id=123,expected_command=self.command,scope_sha256="f"*64)

    def test_active_same_scope_retry_cannot_pass_original_guard_record(self):
        ledger=self.active()
        ledger["events"].append(dict(label=C.JOB_NAMES["cold"],exit=1))
        with self.assertRaises(ValueError):C.validate_execution_guard(self.scope,ledger,"cold",actual_session_id=123,expected_command=self.command,scope_sha256="f"*64)

    def test_CLI_template_stdout_only_and_launch_not_supported(self):
        with tempfile.TemporaryDirectory(prefix="g3_calibration_cli_fixture_") as directory:
            before=list(Path(directory).iterdir())
            done=subprocess.run([sys.executable,"-B","-I","-S",str(HERE/"g3_calibration_plan.py"),"--scope-template"],cwd=directory,capture_output=True,text=True,timeout=10)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertFalse(json.loads(done.stdout)["allow_gpu_runs"])
            fail=subprocess.run([sys.executable,"-B","-I","-S",str(HERE/"g3_calibration_plan.py"),"--launch"],cwd=directory,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(fail.returncode,0)
            self.assertEqual(list(Path(directory).iterdir()),before)


if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        sys.meta_path.remove(guard)
