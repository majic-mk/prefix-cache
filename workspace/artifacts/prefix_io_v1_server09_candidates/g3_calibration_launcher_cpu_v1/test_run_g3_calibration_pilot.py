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

SPEC = importlib.util.spec_from_file_location("tested_g3_cli", Path(__file__).with_name("run_g3_calibration_pilot.py"))
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


if __name__ == "__main__":
    unittest.main()
