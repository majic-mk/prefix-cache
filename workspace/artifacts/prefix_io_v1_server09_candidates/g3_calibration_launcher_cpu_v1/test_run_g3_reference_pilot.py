"""CPU fixtures for the concrete two-job reference CLI; no GPU authority.

The numerical failure test executes the SHA-pinned original analyzer functions.
Only temporary fixture storage permission and guard/source receipts are mocked.
No runtime, GPU backend, subprocess, or human authorization is created here.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root", type=Path)
options, remaining = parser.parse_known_args()
SOURCE = options.source_root or HERE.parents[1] / "prefix_io_v1_server08_primary_qualification/contents"
spec = importlib.util.spec_from_file_location("tested_g3_reference_cli_CPU_only", HERE / "run_g3_reference_pilot.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)

LABELS = {"cold": "server09-g3-reference-native-01", "paired": "server09-g3-reference-paired-01"}
PURPOSE = "CURRENT_CONTEXT_CACHED_NUMERICAL_REFERENCE_ONLY"
PREFIXES = ("yaml", "prefix_io_control", "experiment_storage", "analyze_repeated_native_costs", "analyze_cached_references")
ORIGINAL_PINS = {
    "cached": "2aaed52742c23750d068b16f4ad51e1caaa9427dc1ff39fdd09e3440460bbc1c",
    "repeated": "21bf7fc7562dcf1711a0e3dbad9b740e122a7126304d7f048258f79b32dc773a",
    "storage": "ec5fdb9d9f9001de1080aadb740a87320ccc783f95032c5fdff310dc869a127f",
}


def namespace_name(name):
    return any(name == p or name.startswith(p + ".") for p in PREFIXES)


def put(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def fixture_plan():
    # Explicit fixture of post-authority state. These callbacks do not grant
    # scope and are never used by the no-scope/old-scope boundary tests.
    return types.SimpleNamespace(JOB_NAMES=LABELS, MODES=("cold", "paired"),
        next_job=lambda receipts: None if len(receipts) == 2 else ("cold" if not receipts else "paired"))


def actual_original_numerical_failure(root):
    """Run unchanged original analyze/check_drains/path/numeric functions."""
    trees = {}
    for key, relative in CLI.ANALYZERS.items():
        path = SOURCE / relative
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != ORIGINAL_PINS[key]:
            raise ValueError("original CPU analyzer fixture source differs: " + key)
        trees[key] = (path, ast.parse(raw))

    def functions(key, namespace):
        path, tree = trees[key]
        body = [n for n in tree.body if type(n) is ast.FunctionDef]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec", dont_inherit=True), namespace)

    repeated = dict(Path=Path, json=json, math=math, hashlib=hashlib)
    functions("repeated", repeated)
    runs = root / CLI.RUNS
    storage = dict(Path=Path, ROOT=root, json=json)
    functions("storage", storage)
    # Explicit CPU fixture permission; original authorized_path/details_path
    # code is used, with ordinary Path on Windows and Linux.
    storage["permission"] = lambda project=root: {"approved_experiment_root": str(runs)}
    analyzer = dict(Path=Path, json=json, math=math, hashlib=hashlib,
        require=repeated["require"], check_drains=repeated["check_drains"],
        details_path=storage["details_path"], authorized_path=storage["authorized_path"])
    functions("cached", analyzer)
    paired_storage = str(runs / "explicit-CPU-fixture-existing-cache")
    common = dict(model={"model_id": "CPU_FIXTURE_NOT_GPU"}, model_alias="CPU_FIXTURE_NOT_GPU",
        alias_target="/explicit-CPU-fixture-model", gpu_uuid="CPU_FIXTURE_NO_DEVICE", sizes=[128], reps=3,
        sampling={"temperature": 0.0, "max_tokens": 1, "logprobs": 5}, pythonhashseed="0", acquisition_domain=1024)
    engine = dict(kv_cache_memory_bytes=268435456, max_model_len=1040, max_num_batched_tokens=1040,
        kv_transfer_config=None)
    handler = dict(pinned=True, staging_bytes=917504, staging_budget=134217728,
        io_size=917504, storage_block_bytes=917504,
        aio=dict(accepted=8, completed=8, reaped=8, outstanding=0, pending=0, ready=0, unreaped=0, fatal=None))
    def row(rep, kind):
        token = 500 + rep
        return dict(kind=kind, prefix_tokens=128, rep=rep, warmup=rep == 0,
            prompt_token_ids=list(range(129)), output_token_ids=[token],
            num_cached_tokens=0 if kind == "f" else 128,
            metrics={"is_corrupted": False},
            output_logprobs=[{str(token + i): {"logprob": -float(i + 1), "rank": i + 1} for i in range(5)}],
            trace=dict(foreground_logical_read_bytes=0, preload_actual_read_bytes=128 * 57344 if kind == "g_ssd" else 0,
                transfer_success=True, h2d_events=1, load_transfers=[{"num_bytes": 128 * 57344}]),
            drain={"handlers": [copy.deepcopy(handler)]} if kind in ("g_ssd", "g_mem") else {})
    reports = {}
    for mode in ("cold", "paired"):
        kinds = ("f", "gpu_hot") if mode == "cold" else ("g_ssd", "g_mem")
        reports[mode] = dict(status="PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            native_hot_diagnostic=mode == "cold", cached_reference_logprobs=mode == "paired",
            engine_shutdown="completed", final_drain={},
            rows=[row(rep, kind) for rep in range(3) for kind in kinds])
        config = copy.deepcopy(common)
        config["engine"] = copy.deepcopy(engine)
        if mode == "paired":
            config["engine"]["kv_transfer_config"] = {"kv_connector_extra_config": {
                "shared_storage_path": paired_storage, "staging_mem": 0.125}}
        folder = runs / LABELS[mode] / "details/acquisition"
        folder.mkdir(parents=True)
        if mode == "paired":
            # One real numerical mismatch, including warmup, using the original
            # exact comparator. No new tolerance or verdict synthesis.
            first = reports[mode]["rows"][0]
            first["output_logprobs"][0][str(first["output_token_ids"][0])]["logprob"] -= 1e-6
        put(root, str((folder / "result.json").relative_to(root)), json.dumps(reports[mode]))
        put(root, str((folder / "frozen-config.json").relative_to(root)), json.dumps(config))
    plan = dict(native_label=LABELS["cold"], sizes=[128], reps=3, domain=1024,
        gpu_uuid=common["gpu_uuid"], kv_budget_bytes=268435456, staging_budget_bytes=134217728,
        external_groups=[dict(label=LABELS["paired"], sizes=[128], storage="explicit-CPU-fixture-existing-cache",
            storage_path=paired_storage)],
        run_details={label: str(runs / label / "details/acquisition") for label in LABELS.values()})
    path = put(root, "explicit-CPU-fixture-analyzer-plan.json", json.dumps(plan))
    return analyzer["analyze"](root, path)


class AuthorityTests(unittest.TestCase):
    def test_missing_scope_launch_and_execute_fail_before_sources_modules_or_subprocess(self):
        with tempfile.TemporaryDirectory(prefix="reference-CPU-missing-scope-") as folder:
            root = Path(folder).resolve()
            for action in ("--launch", "--execute"):
                with self.subTest(action=action), mock.patch.object(CLI, "full_sources") as sources, \
                     mock.patch.object(CLI, "load_module") as load, mock.patch.object(CLI.subprocess, "run") as process, \
                     contextlib.redirect_stdout(io.StringIO()) as stdout:
                    self.assertEqual(CLI.main(["--project", str(root), action]), 78)
                    fact = json.loads(stdout.getvalue())
                    self.assertIn("NO_NEW_REFERENCE_HUMAN_SCOPE", fact["reason"])
                    self.assertEqual(fact["actual_gpu_runs"], 0)
                    self.assertIs(fact["production_qualified"], False)
                    sources.assert_not_called(); load.assert_not_called(); process.assert_not_called()
                    self.assertEqual(list(root.iterdir()), [])

    def test_old_G3_purpose_or_unauthorized_status_cannot_read_full_sources(self):
        with tempfile.TemporaryDirectory(prefix="reference-CPU-old-record-") as folder:
            root = Path(folder).resolve()
            for record in (
                {"purpose": "RAW_BASELINE_CALIBRATION_PILOT_ONLY", "status": "USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC"},
                {"purpose": PURPOSE, "status": "NOT_AUTHORIZED_CPU_REFERENCE_TEMPLATE"},
            ):
                put(root, "explicit-CPU-invalid-record.json", json.dumps(record))
                for action in ("--launch", "--execute"):
                    with self.subTest(record=record, action=action), mock.patch.object(CLI, "full_sources") as sources, \
                         mock.patch.object(CLI.subprocess, "run") as process, contextlib.redirect_stdout(io.StringIO()) as stdout:
                        self.assertEqual(CLI.main(["--project", str(root), "--scope-record", "explicit-CPU-invalid-record.json", action]), 78)
                        self.assertIn("OLD_SCOPE_CANNOT_AUTHORIZE_REFERENCE", json.loads(stdout.getvalue())["reason"])
                        sources.assert_not_called(); process.assert_not_called()
                self.assertEqual([p.name for p in root.iterdir()], ["explicit-CPU-invalid-record.json"])

    def test_child_command_binds_both_fixed_modes_exactly(self):
        root = Path("/explicit-CPU-fixture-project")
        for mode in ("cold", "paired"):
            self.assertEqual(CLI.child_command(root, mode, "locked-source.json", "new-reference-scope.json"),
                [".venv/bin/python", "-B", CLI.SCRIPT, "--project", str(root), "--mode", mode,
                 "--source-lock", "locked-source.json", "--scope-record", "new-reference-scope.json", "--execute"])

    def test_guard_labels_budget_and_offline_env_original_nonzero_return_once(self):
        with tempfile.TemporaryDirectory(prefix="reference-CPU-argv-") as folder:
            root = Path(folder).resolve()
            for mode in ("cold", "paired"):
                plan = fixture_plan()
                fixture = (plan, {}, {}, {}, {}, root / "unused-details", root / "unused-storage", None)
                with self.subTest(mode=mode), mock.patch.object(CLI, "gates", return_value=fixture), \
                     mock.patch.object(CLI.subprocess, "run", return_value=types.SimpleNamespace(returncode=23)) as process, \
                     mock.patch.object(CLI, "full_sources") as post, mock.patch.object(CLI, "analyze_pair") as analyze:
                    self.assertEqual(CLI.main(["--project", str(root), "--mode", mode, "--scope-record", "explicit-CPU-not-authority.json", "--launch"]), 23)
                    process.assert_called_once()
                    actual = process.call_args
                    self.assertEqual(actual.args[0], [".venv/bin/python", "-B", CLI.GUARD,
                        "--permissions-path", CLI.PERMISSIONS, "--label", LABELS[mode], "--seconds", "300", "--",
                        *CLI.child_command(root, mode, CLI.LOCK, "explicit-CPU-not-authority.json")])
                    self.assertEqual(actual.kwargs["cwd"], root)
                    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONDONTWRITEBYTECODE"):
                        self.assertEqual(actual.kwargs["env"][name], "1")
                    self.assertEqual(actual.kwargs["env"]["PYTHONHASHSEED"], "0")
                    post.assert_not_called(); analyze.assert_not_called()
                    self.assertEqual(list(root.iterdir()), [])

    def test_execute_missing_original_guard_witness_cannot_load_runtime_or_create_details(self):
        with tempfile.TemporaryDirectory(prefix="reference-CPU-no-guard-") as folder:
            root = Path(folder).resolve()
            fixture = (fixture_plan(), {}, {}, {}, {}, root / "unused-details", root / "unused-storage", None)
            with mock.patch.object(CLI, "gates", return_value=fixture), mock.patch.object(CLI, "load_module") as load, \
                 mock.patch.object(CLI.subprocess, "run") as process, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(CLI.main(["--project", str(root), "--execute"]), 78)
                self.assertIn("original guard witness required", json.loads(stdout.getvalue())["reason"])
                load.assert_not_called(); process.assert_not_called()
                self.assertEqual(list(root.iterdir()), [])

    def test_bootstrap_pins_match_actual_three_cpu_modules(self):
        for row in (CLI.BOOTSTRAP_PLAN_REF, CLI.BOOTSTRAP_RUNTIME_REF, CLI.BOOTSTRAP_RESULT_REF):
            raw = (HERE / Path(row["path"]).name).read_bytes()
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])


class AnalyzerContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="explicit-reference-CPU-analyzer-context-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.refs = {}
        self.yaml = put(self.root, ".venv/lib/python3.12/site-packages/yaml/__init__.py", "CPU_FIXTURE = True\n")
        init = "explicit-fixture-config/__init__.py"
        config = "explicit-fixture-config/config.py"
        put(self.root, init, '"""Explicit CPU fixture namespace, not installed control engine."""\n')
        put(self.root, config, "import json\nfrom pathlib import Path\ndef read_yaml(path):\n    return json.loads(Path(path).read_text())\ndef validate_permissions(value):\n    return dict(value)\n")
        put(self.root, CLI.ANALYZERS["storage"], "from pathlib import Path\nROOT = Path(" + repr(str(self.root)) + ")\ndef permission(project=ROOT):\n    return {'fixture_original_permission': True}\nORIGINAL_PERMISSION = permission\n")
        put(self.root, CLI.ANALYZERS["repeated"], "import experiment_storage\nCPU_FIXTURE = True\n")
        put(self.root, CLI.ANALYZERS["cached"], "import experiment_storage\nimport analyze_repeated_native_costs\nCPU_FIXTURE = True\n")
        self.effective = dict(approved_experiment_root=str(self.root / CLI.RUNS),
            approved_gpu_ids=["CPU_FIXTURE_NO_DEVICE"], approved_auxiliary_storage=None)
        put(self.root, CLI.PERMISSIONS, json.dumps(self.effective))
        self.context = dict(yaml_root=str(self.yaml.parent), yaml_files=[dict(path=str(self.yaml),
            bytes=self.yaml.stat().st_size, sha256=hashlib.sha256(self.yaml.read_bytes()).hexdigest())],
            cpu_init_ref=CLI.ref(self.root, init), cpu_config_ref=CLI.ref(self.root, config),
            original_config_gpu_uuid="CPU_FIXTURE_NO_DEVICE")
        self.refs.update({name: CLI.ref(self.root, name) for name in CLI.ANALYZERS.values()})
        self.saved = {n: m for n, m in list(sys.modules.items()) if namespace_name(n)}
        for name in self.saved:
            sys.modules.pop(name)
        self.markers = {name: types.ModuleType(name) for name in (
            "yaml", "yaml.explicit_existing_submodule", "prefix_io_control", "prefix_io_control.config",
            "experiment_storage", "analyze_repeated_native_costs", "analyze_cached_references")}
        sys.modules.update(self.markers)
        self.addCleanup(self.restore_modules)
        self.patch = mock.patch.object(CLI, "model_context", return_value=self.context)
        self.patch.start(); self.addCleanup(self.patch.stop)

    def restore_modules(self):
        for name in list(sys.modules):
            if namespace_name(name): sys.modules.pop(name)
        sys.modules.update(self.saved)

    def assert_restored(self, loaded=None):
        current = {n: m for n, m in sys.modules.items() if namespace_name(n)}
        self.assertEqual(set(current), set(self.markers))
        for name in current: self.assertIs(current[name], self.markers[name])
        if loaded is not None:
            self.assertIs(loaded.permission, loaded.ORIGINAL_PERMISSION)

    def test_normal_context_restores_modules_and_actual_permission_function(self):
        with CLI.original_analyzer_context(self.root, self.refs) as analyzer:
            self.assertIs(analyzer.CPU_FIXTURE, True)
            storage = analyzer.experiment_storage
            self.assertIsNot(storage.permission, storage.ORIGINAL_PERMISSION)
            self.assertEqual(storage.permission(self.root), self.effective)
            with self.assertRaises(ValueError): storage.permission(self.root.parent)
        self.assert_restored(storage)

    def test_body_exception_retains_original_object_and_restores_every_binding(self):
        error = RuntimeError("explicit CPU original analysis failure")
        storage = None
        try:
            with CLI.original_analyzer_context(self.root, self.refs) as analyzer:
                storage = analyzer.experiment_storage
                raise error
        except RuntimeError as actual:
            self.assertIs(actual, error)
        else:
            self.fail("original analyzer error was swallowed")
        self.assert_restored(storage)

    def test_source_loading_failure_still_restores_modules_and_permission(self):
        path = self.root / CLI.ANALYZERS["cached"]
        path.write_bytes(path.read_bytes() + b"\n# drift\n")
        with self.assertRaisesRegex(ValueError, "source bytes"):
            with CLI.original_analyzer_context(self.root, self.refs): self.fail("drift entered analyzer body")
        self.assert_restored()

    def test_yaml_byte_drift_is_rejected_after_actual_permission_and_modules_restore(self):
        storage = None
        with self.assertRaisesRegex(ValueError, "YAML dependency bytes drift"):
            with CLI.original_analyzer_context(self.root, self.refs) as analyzer:
                storage = analyzer.experiment_storage
                self.yaml.write_bytes(self.yaml.read_bytes() + b"# drift\n")
        self.assert_restored(storage)

    def test_yaml_inventory_drift_is_rejected_and_does_not_leave_fixture_modules(self):
        storage = None
        with self.assertRaisesRegex(ValueError, "YAML dependency inventory drift"):
            with CLI.original_analyzer_context(self.root, self.refs) as analyzer:
                storage = analyzer.experiment_storage
                put(self.root, ".venv/lib/python3.12/site-packages/yaml/extra.py", "VALUE=1\n")
        self.assert_restored(storage)

    def test_invalid_effective_permission_never_enters_analyzer_and_restores_modules(self):
        bad = dict(self.effective, approved_gpu_ids=["OTHER_CPU_FIXTURE"])
        put(self.root, CLI.PERMISSIONS, json.dumps(bad))
        with self.assertRaisesRegex(ValueError, "unchanged PRIMARY"):
            with CLI.original_analyzer_context(self.root, self.refs): self.fail("bad permission yielded")
        self.assert_restored()


class NumericalFailurePreservationTests(unittest.TestCase):
    def test_original_numeric_failure_is_durable_and_returns_one_without_cost_promotion(self):
        with tempfile.TemporaryDirectory(prefix="reference-CPU-original-numeric-failure-") as folder:
            root = Path(folder).resolve()
            original = actual_original_numerical_failure(root)
            self.assertEqual(original["status"], "FAILED_EXACT_CACHED_REFERENCE")
            self.assertEqual(original["failed_comparisons"], 1)
            self.assertEqual(len(original["comparisons"]), 6)
            self.assertTrue(original["comparisons"][0]["warmup"])
            self.assertEqual(original["cached_output_tolerance"], 0)
            for flag in ("latency_fit_allowed", "runtime_curve_exported", "independent_content_validation"):
                self.assertIs(original[flag], False)
            plan = fixture_plan()
            out = root / CLI.RUNS / LABELS["paired"] / "details"
            out.mkdir(parents=True, exist_ok=True)
            source = dict(path="explicit-CPU-fixture-lock.json", bytes=2, sha256="1" * 64)
            scope_ref = dict(path="explicit-CPU-fixture-scope.json", bytes=2, sha256="2" * 64)
            gate = (plan, {}, {"source_lock": source}, scope_ref, {}, out, root / "fixture-storage", None)
            result = dict(status=original["status"], original_analysis=original, origin="CPU_FIXTURE_NOT_GPU",
                cost_qualified=False, production_qualified=False, performance_claim=False, GPU_qualified=False,
                P4_performance_verified=False, runtime_curve_exported=False, latency_fit_allowed=False)
            with mock.patch.object(CLI, "gates", return_value=gate), \
                 mock.patch.object(CLI.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as process, \
                 mock.patch.object(CLI, "full_sources", return_value=(plan, {}, source)), \
                 mock.patch.object(CLI, "read_json", return_value={"active_reservation": None}), \
                 mock.patch.object(CLI, "prior_results", return_value=([{"mode": "cold"}, {"mode": "paired"}], {"cold": {}, "paired": {}})), \
                 mock.patch.object(CLI, "analyze_pair", return_value=result) as analyze:
                self.assertEqual(CLI.main(["--project", str(root), "--mode", "paired", "--launch", "--scope-record", scope_ref["path"]]), 1)
                process.assert_called_once(); analyze.assert_called_once()
            evidence = json.loads((out / "reference-analysis-result.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence, result)
            for flag in ("cost_qualified", "production_qualified", "performance_claim", "GPU_qualified", "P4_performance_verified", "runtime_curve_exported", "latency_fit_allowed"):
                self.assertIs(evidence[flag], False)
            self.assertEqual(json.loads((out / "post-source-verification.json").read_text())["status"], "PASS_ACTUAL_PARENT_FULL_SOURCE_BYTES")
            self.assertFalse(any(p.suffix in (".so", ".bin", ".csv") for p in root.rglob("*")))

    def test_post_guard_source_drift_cannot_write_numeric_pass_or_failure_evidence(self):
        with tempfile.TemporaryDirectory(prefix="reference-CPU-post-source-drift-") as folder:
            root = Path(folder).resolve(); out = root / "fixture-details"; out.mkdir()
            source = dict(path="CPU-fixture-source", bytes=2, sha256="1" * 64)
            changed = dict(source, sha256="9" * 64)
            gate = (fixture_plan(), {}, {"source_lock": source}, {"sha256": "2" * 64}, {}, out, root / "storage", None)
            with mock.patch.object(CLI, "gates", return_value=gate), \
                 mock.patch.object(CLI.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)) as process, \
                 mock.patch.object(CLI, "full_sources", return_value=(fixture_plan(), {}, changed)), \
                 mock.patch.object(CLI, "analyze_pair") as analyze, contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(CLI.main(["--project", str(root), "--launch"]), 78)
                self.assertIn("complete source lock unchanged", json.loads(stdout.getvalue())["reason"])
                process.assert_called_once(); analyze.assert_not_called()
                self.assertEqual(list(out.iterdir()), [])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
