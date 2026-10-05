"""CPU-only negative authority and path tests of the concrete pilot CLI."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("tested_g3_cli", Path(__file__).with_name("run_g3_calibration_pilot_v3.py"))
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class PilotCLI(unittest.TestCase):
    def test_missing_new_scope_cannot_load_runtime_or_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            before = list(root.iterdir())
            for action in ("--launch", "--execute"):
                with mock.patch.object(CLI, "load_module", side_effect=AssertionError("module import")), \
                     mock.patch.object(CLI.subprocess, "run", side_effect=AssertionError("process launch")), \
                     contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(CLI.main(["--project", str(root), action]), 78)
                    self.assertIn("NO_NEW_G3_HUMAN_SCOPE", json.loads(stdout.getvalue())["reason"])
                self.assertEqual(before, list(root.iterdir()))

    def test_existing_scope_template_is_not_a_permission(self):
        # Even an existing record cannot get as far as execution if the full
        # source chain is unavailable. No output or guard reservation appears.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            (root / "template.json").write_text('{"status":"NOT_AUTHORIZED"}')
            with contextlib.redirect_stdout(io.StringIO()), \
                 mock.patch.object(CLI.subprocess, "run", side_effect=AssertionError("launch")):
                self.assertEqual(CLI.main(["--project", str(root), "--scope-record", "template.json", "--launch"]), 78)
            self.assertEqual(["template.json"], [p.name for p in root.iterdir()])

    def test_child_argv_binds_mode_scope_and_source(self):
        root = Path("/approved/project")
        cmd = CLI.child_command(root, "paired", "source.json", "human-scope.json")
        self.assertEqual(cmd, [".venv/bin/python", "-B", CLI.SCRIPT,
            "--project", str(root), "--mode", "paired", "--source-lock", "source.json",
            "--scope-record", "human-scope.json", "--execute"])
        self.assertNotEqual(cmd, CLI.child_command(root, "cold", "source.json", "human-scope.json"))

    def test_paths_cannot_escape_or_be_redirected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for bad in ("../secret", "a/../b", "/absolute", "a\\b", "a//b", "./b", ""):
                with self.assertRaises(ValueError):
                    CLI.safe(root, bad)
            self.assertEqual(CLI.safe(root, "new/result.json"), root / "new/result.json")
            link = root / "redirect"
            try:
                link.symlink_to(root, target_is_directory=True)
            except OSError:
                return
            with self.assertRaises(ValueError):
                CLI.safe(root, "redirect/new/result.json")

    def test_json_cannot_overwrite_or_contain_nonfinite_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "receipt.json"
            CLI.new_json(p, {"fact": "CPU_ONLY"})
            with self.assertRaises(FileExistsError):
                CLI.new_json(p, {"fact": "replacement"})
            self.assertEqual(CLI.read_json(p), {"fact": "CPU_ONLY"})
            p.write_text('{"seconds":NaN}')
            with self.assertRaises(ValueError):
                CLI.read_json(p)

    def test_loaded_helper_requires_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            p = root / "helper.py"
            p.write_text("VALUE=1\n")
            pin = CLI.ref(root, "helper.py")
            self.assertEqual(CLI.load_module(root, "helper.py", pin).VALUE, 1)
            p.write_text("VALUE=2\n")
            with self.assertRaises(ValueError):
                CLI.load_module(root, "helper.py", pin)

    def test_drifted_bootstrap_is_rejected_before_its_code_executes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            p = root / CLI.PLAN_MODULE
            p.parent.mkdir(parents=True)
            marker = root / "FORBIDDEN_EXECUTION"
            p.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").mkdir()\n")
            with self.assertRaises(ValueError):
                CLI.full_sources(root, "nonexistent-lock.json")
            self.assertFalse(marker.exists())

    def test_postlaunch_failure_cannot_be_reported_as_zero_gpu_work(self):
        # The subprocess is a CPU stub. The regression is the reporting path
        # after an actual guard launch has been attempted, not a GPU fixture.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            fake = (types.SimpleNamespace(JOB_NAMES={"cold": "fixture"}), {}, {}, {}, {},
                    root / "out", root / "storage", None)
            with mock.patch.object(CLI, "gates", return_value=fake), \
                 mock.patch.object(CLI.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)), \
                 mock.patch.object(CLI, "full_sources", side_effect=ValueError("post-run drift")), \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(CLI.main(["--project", str(root), "--scope-record", "CPU_STUB", "--launch"]), 78)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["actual_gpu_runs"], "UNKNOWN_READ_ORIGINAL_GUARD_LEDGER")
            self.assertEqual(report["GPU_initialized"], "UNKNOWN_READ_RUNTIME_RECEIPT")
            self.assertFalse((root / "out").exists())


class FrozenStorageBoundary(unittest.TestCase):
    @staticmethod
    def plan():
        p=Path(__file__).with_name("g3_calibration_plan_metrics_v2.py")
        spec=importlib.util.spec_from_file_location("storage_v2_frozen_plan",p)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_all_modes_bind_the_same_frozen_plan_storage(self):
        plan=self.plan()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            storage=CLI.shared_storage(root,plan)
            self.assertEqual(storage,root/plan.STORAGE_RELATIVE)
            self.assertEqual(plan.fixed_context()["shared_storage_relative"],plan.STORAGE_RELATIVE)
            for mode in plan.MODES:
                argv=plan.acquisition_arguments(mode)
                self.assertEqual(argv[argv.index("--storage")+1],plan.PROJECT_ROOT+"/"+plan.STORAGE_RELATIVE)
            self.assertFalse(storage.exists())
            self.assertEqual(list(root.iterdir()),[])

    def test_populate_leaves_directory_creation_to_original_acquirer(self):
        # CPU filesystem/callback fixture only. No authority or GPU qualification.
        plan=self.plan()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            runs=root/CLI.RUNS
            runs.mkdir(parents=True)
            storage=CLI.shared_storage(root,plan)
            out=runs/plan.JOB_NAMES["populate"]/"details"
            out.parent.mkdir()
            calls=[]
            def original_fixture(actual_root,mode,actual_out,actual_storage,refs,witness):
                self.assertEqual(actual_storage,storage)
                self.assertEqual(actual_storage.parent,runs)
                self.assertTrue(actual_storage.parent.is_dir())
                self.assertFalse(actual_storage.exists())
                actual_storage.mkdir(parents=True,exist_ok=False)
                calls.append(mode)
                return {"original_exit_code":0}
            fake=(plan,{CLI.RUNTIME_MODULE:{"fixture":"CPU_ONLY"}}, {},{}, {},out,storage,{"fixture":"CPU_ONLY"})
            runtime=types.SimpleNamespace(execute_original_acquisition=original_fixture)
            with mock.patch.object(CLI,"gates",return_value=fake), \
                 mock.patch.object(CLI,"load_module",return_value=runtime), \
                 mock.patch.object(CLI.subprocess,"run",side_effect=AssertionError("GPU process forbidden")):
                self.assertEqual(CLI.main(["--project",str(root),"--mode","populate","--scope-record","CPU_FIXTURE","--execute"]),0)
            self.assertEqual(calls,["populate"])
            self.assertTrue(storage.is_dir())
            self.assertTrue(out.is_dir())

    def test_child_entry_defaults_to_repaired_lock_and_v2_source(self):
        self.assertTrue(CLI.SCRIPT.endswith("/run_g3_calibration_pilot_v3.py"))
        self.assertTrue(CLI.LOCK.endswith("/gpu-source-lock-metrics-v3.json"))
        self.assertFalse(hasattr(CLI,"SHARED"))


class CommonMetricsProof(unittest.TestCase):
    @staticmethod
    def proof(root):
        spec=importlib.util.spec_from_file_location("stats_proof_runtime",Path(__file__).with_name("g3_calibration_runtime_metrics_v2.py"))
        runtime=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        return {"common_stats_dictionary_fix":dict(installed=True,restored=True,
            original_metrics_ref=runtime.METRICS_REF,helper_ref=runtime.STATS_COMPAT_REF,
            witness=dict(installed=False,source_path=str(root/runtime.METRICS_SOURCE),
                source_bytes=6195,source_sha256=runtime.METRICS_REF["sha256"],
                original_record_calls_per_invocation=1,original_consumers_unchanged=True))}

    def test_bound_and_restored_common_fix_record_consistency(self):
        # CPU record consistency only; a caller dictionary is not GPU provenance.
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            CLI.check_common_stats_dictionary_fix(root,self.proof(root))
            self.assertEqual(list(root.iterdir()),[])

    def test_missing_or_failed_restoration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            for bad in ({},{"common_stats_dictionary_fix":{}},):
                with self.assertRaises(ValueError):CLI.check_common_stats_dictionary_fix(root,bad)
            for field in ("installed","restored"):
                bad=self.proof(root);bad["common_stats_dictionary_fix"][field]=False
                with self.assertRaises(ValueError):CLI.check_common_stats_dictionary_fix(root,bad)
            bad=self.proof(root);bad["common_stats_dictionary_fix"]["witness"]["installed"]=True
            with self.assertRaises(ValueError):CLI.check_common_stats_dictionary_fix(root,bad)

    def test_source_drift_or_changed_original_consumer_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            for section,field,value in (("helper_ref","sha256","0"*64),
                ("original_metrics_ref","bytes",6196),
                ("witness","source_path",str(root/"foreign.py")),
                ("witness","original_consumers_unchanged",False),
                ("witness","original_record_calls_per_invocation",2)):
                bad=self.proof(root);bad["common_stats_dictionary_fix"][section][field]=value
                with self.assertRaises(ValueError):CLI.check_common_stats_dictionary_fix(root,bad)


if __name__ == "__main__":
    unittest.main()
