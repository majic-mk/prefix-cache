"""CPU-only permission/ledger regression checks; never launch a subprocess.

Windows lacks the unchanged Unix flock dependency and this private Python has
no PyYAML. On those hosts only, load the same function AST with a JSON-fixture
parser and a flock stub. Linux uses the real installed dependencies. Fixtures
are JSON, a subset of the guard's unchanged YAML input format.
"""
import ast
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_gpu_stage_standing_v1.py"
ORIGINAL = HERE / "original_scripts/run_gpu_stage.py"
source = SOURCE.read_text(encoding="utf-8")
tree = ast.parse(source)
namespace = {"__file__": str(SOURCE), "__name__": "_cpu_standing_guard"}
fixture_dependencies = []
for name in ("fcntl", "yaml"):
    try:
        namespace[name] = importlib.import_module(name)
    except ModuleNotFoundError:
        fixture_dependencies.append(name)
        namespace[name] = (SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
                           if name == "fcntl" else SimpleNamespace(safe_load=json.loads))
nodes = [node for node in tree.body if not (
    isinstance(node, ast.Import) and any(alias.name in ("fcntl", "yaml") for alias in node.names))
    and not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
             and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
guard = SimpleNamespace(**namespace)

GPU = "GPU-cb140ee8-a52e-93ba-38f7-e2fddfbb43b3"
AUTH = "artifacts/prefix_io_v1/server14-dr-long-20261005/DIRECT_USER_GPU_AUTHORIZATION_02.json"
EFFECTIVE = "artifacts/prefix_io_v1/server14-dr-long-20261005/effective.json"


def grant():
    return dict(schema_version=1, authority="direct_user_instruction",
        user_literal="无限续约，没有gpu固定配额，知道完成实验", issued_utc="2026-10-05T12:51:22.607057Z",
        approved_gpu_ids=[GPU], scope="U_VS_F8_PLUS_D_R_ONLY", max_gpu_hours=None,
        completion_condition="narrow_experiment_complete", per_job_seconds_max=300,
        cleanup_reserve_seconds=20, historical_usage_preserved=True, budget_not_reset=True)


class StandingGuard(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = dict(allow_gpu_runs=True, max_gpu_hours=8, approved_gpu_ids=[GPU],
            approved_experiment_root=str(self.root / "experiments/prefix_io_v1/runs"),
            approved_dependency_root=str(self.root), allow_model_downloads=False,
            allow_driver_or_system_changes=False, allow_shared_data_deletion=False,
            allow_payment=False, allow_new_cloud_rental=False)
        self.write(guard.PERMISSION, self.base)
        self.permission = dict(self.base, max_gpu_hours=None)
        self.freeze_grant(grant())

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def freeze_grant(self, value):
        path = self.write(AUTH, value)
        data = path.read_bytes()
        self.permission["standing_gpu_authorization_ref"] = dict(
            path=AUTH, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

    def permission_source(self, permission=None):
        self.write(EFFECTIVE, self.permission if permission is None else permission)
        return guard.permission_source(self.root, EFFECTIVE)

    def test_actual_downloaded_authorization_and_effective_bytes(self):
        actual = HERE / "actual_server14/DIRECT_USER_GPU_AUTHORIZATION_02.json"
        effective = HERE / "actual_server14/EFFECTIVE_UNCAPPED_D_R_GPU_PERMISSION_02.json"
        if actual.is_file() and effective.is_file():
            data = actual.read_bytes()
            self.assertEqual(len(data), 515)
            self.assertEqual(hashlib.sha256(data).hexdigest(),
                "2874da8addb0bf3cea84734ce8bb340597c8bb75dee55251212efb246892b99a")
            self.write(AUTH, json.loads(data))
            # Preserve exact authority bytes, while fixture storage roots remain local.
            (self.root / AUTH).write_bytes(data)
            frozen = json.loads(effective.read_text(encoding="utf-8"))
            self.permission["standing_gpu_authorization_ref"] = frozen["standing_gpu_authorization_ref"]
        loaded, reference = self.permission_source()
        self.assertIsNone(loaded["max_gpu_hours"])
        self.assertEqual(reference["path"], EFFECTIVE)
        self.assertIsNone(guard.gpu_budget_hours(loaded, EFFECTIVE, 300, "server14-dr-method01"))

    def test_scope_authority_gpu_job_cleanup_and_preservation_are_required(self):
        bad = (("authority", "document_instruction"), ("scope", "ALL_GPU_WORK"),
            ("approved_gpu_ids", ["GPU-00000000-0000-0000-0000-000000000000"]),
            ("per_job_seconds_max", 301), ("cleanup_reserve_seconds", 0),
            ("schema_version", True), ("max_gpu_hours", 9999999),
            ("completion_condition", "forever"), ("user_literal", " "),
            ("issued_utc", "not-a-time"), ("historical_usage_preserved", False),
            ("budget_not_reset", False))
        for key, value in bad:
            with self.subTest(key=key):
                changed = grant()
                changed[key] = value
                self.freeze_grant(changed)
                with self.assertRaises(RuntimeError):
                    self.permission_source()

    def test_missing_oversized_traversal_and_drifted_authorization_rejected(self):
        reference = deepcopy(self.permission["standing_gpu_authorization_ref"])
        for changed in (None, dict(reference, path="../grant.json"),
                        dict(reference, path="/outside/grant.json"),
                        dict(reference, bytes=True), dict(reference, bytes=65537),
                        dict(reference, sha256="0" * 64)):
            with self.subTest(reference=changed):
                permission = dict(self.permission)
                if changed is None:
                    permission.pop("standing_gpu_authorization_ref")
                else:
                    permission["standing_gpu_authorization_ref"] = changed
                with self.assertRaises(RuntimeError):
                    self.permission_source(permission)
        (self.root / AUTH).write_text("changed", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.permission_source()

    def test_only_null_may_override_original_finite_base(self):
        for hours in (True, float("inf"), float("nan"), 0, -1, 9):
            with self.subTest(hours=hours), self.assertRaises(RuntimeError):
                self.permission_source(dict(self.base, max_gpu_hours=hours))
        finite, _ = self.permission_source(dict(self.base, max_gpu_hours=4))
        self.assertEqual(guard.gpu_budget_hours(finite, EFFECTIVE, 3600, "ordinary-run"), 4)
        with self.assertRaises(RuntimeError):
            guard.gpu_budget_hours(self.permission, guard.PERMISSION, 300, "server14-dr-u01")

    def test_roots_forbidden_actions_and_primary_only_rules_unchanged(self):
        for key in ("allow_model_downloads", "allow_driver_or_system_changes",
                    "allow_shared_data_deletion", "allow_payment", "allow_new_cloud_rental"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.permission_source(dict(self.permission, **{key: True}))
        for key in ("approved_experiment_root", "approved_dependency_root"):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.permission_source(dict(self.permission, **{key: "/different"}))
        with self.assertRaises(RuntimeError):
            self.permission_source(dict(self.permission, approved_auxiliary_storage={"root": "/other"}))

    def test_jobs_still_have_300_second_limit_and_narrow_label_scope(self):
        for seconds in (0, 301, 3600):
            with self.subTest(seconds=seconds), self.assertRaises(RuntimeError):
                guard.gpu_budget_hours(self.permission, EFFECTIVE, seconds, "server14-dr-u01")
        for label in ("unrelated", "server15-dr-u01", "server14-dr-"):
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                guard.gpu_budget_hours(self.permission, EFFECTIVE, 300, label)
        self.assertIsNone(guard.gpu_budget_hours(self.permission, EFFECTIVE, 300, "server14-dr-cal05"))
        self.assertEqual(guard.RESERVE_SECONDS, 20)

    def test_finite_capacity_unchanged_and_uncapped_keeps_historical_usage(self):
        used = 28045.099937503925
        guard.require_cumulative_capacity(used, 300, None)
        with self.assertRaises(RuntimeError):
            guard.require_cumulative_capacity(28500, 300, 8)
        guard.require_cumulative_capacity(28480, 300, 8)
        self.assertEqual(used, 28045.099937503925)

    def test_active_reservation_bad_ledger_and_finite_exhaustion_stop_before_child(self):
        fake_script = self.root / "artifacts/prefix_io_v1/server14-dr-long-20261005/guard.py"
        runroot = Path(self.base["approved_experiment_root"])
        runroot.mkdir(parents=True)
        ledger_path = "experiments/prefix_io_v1/gpu-budget-ledger.json"
        cases = [dict(gpu_wall_seconds=0, events=[], active_reservation={"unresolved": True}),
                 dict(gpu_wall_seconds=True, events=[], active_reservation=None),
                 dict(gpu_wall_seconds=float("nan"), events=[], active_reservation=None),
                 dict(gpu_wall_seconds=0, events={}, active_reservation=None),
                 dict(gpu_wall_seconds=28500, events=[], active_reservation=None)]
        for budget in cases:
            with self.subTest(budget=budget):
                self.write(ledger_path, budget)
                prior = (self.root / ledger_path).read_bytes()
                argv = [str(fake_script), "--label", "server14-dr-u01", "--seconds", "300", "--", "never-launch"]
                with patch.dict(namespace, __file__=str(fake_script)), patch.object(sys, "argv", argv), \
                        patch.object(namespace["subprocess"], "Popen", side_effect=AssertionError("CPU test must not launch")):
                    with self.assertRaises(RuntimeError):
                        namespace["main"]()
                self.assertEqual((self.root / ledger_path).read_bytes(), prior)
                self.assertFalse((runroot / "server14-dr-u01").exists())

    def test_original_session_accounting_and_cleanup_functions_are_identical(self):
        original = ast.parse(ORIGINAL.read_text(encoding="utf-8"))
        old = {node.name: node for node in original.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        new = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for name in ("Interrupted", "atomic_json", "positive_number", "session_members", "signal_session", "cleanup_session"):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(old[name]), ast.dump(new[name]))
        self.assertIn('budget["gpu_wall_seconds"] = used + elapsed', source)
        self.assertIn('fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)', source)
        self.assertIn('start_new_session=True', source)
        self.assertIn('HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"', source)


if __name__ == "__main__":
    print(json.dumps(dict(CPU_only=True, GPU_jobs=0, fixture_dependencies=fixture_dependencies)))
    unittest.main()
