"""CPU fixtures verify actual original function globals and cleanup, no GPU."""
import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--reference-dir", type=Path)
parser.add_argument("--common-file", type=Path)
ARGS, REST = parser.parse_known_args()
REFDIR = ARGS.reference_dir or HERE.parents[1] / "prefix_io_v1_server09_candidates/g3_calibration_launcher_cpu_v1"
COMMONFILE = ARGS.common_file or HERE.parents[1] / "prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/run_g2_normal_model_lifecycle.py"
COMMON = "artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py"
DELEGATE = "artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/g3_calibration_runtime_metrics_v2.py"
OLD_UUID = "GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


ADAPTER = load(HERE / "server10_reference_runtime.py", "_server10_cpu_test_adapter_constants")


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp).resolve()
        refs = {}
        def write(relative, raw):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            row = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            refs[relative] = row
            return row
        for relative, original in ((ADAPTER.SOURCE, HERE / "server10_reference_runtime.py"),
                                   (ADAPTER.ORIGINAL, REFDIR / "g3_reference_runtime.py"),
                                   (DELEGATE, REFDIR / "g3_calibration_runtime_metrics_v2.py"),
                                   (COMMON, COMMONFILE)):
            write(relative, original.read_bytes())
        config = dict(schema_version=1, gpu_uuid=ADAPTER.GPU_UUID,
                      job_names=dict(ADAPTER.JOB_NAMES), storage=dict(ADAPTER.STORAGE),
                      permissions_ref=write(ADAPTER.PERMISSIONS, b"cpu fixture permission\n"),
                      sdk_inventory_ref=write(ADAPTER.SDK_DIRECTORY + "CUDA13_SOURCE_INVENTORY.json", b"{}\n"),
                      sdk_proof_ref=write(ADAPTER.SDK_DIRECTORY + "CPU_COMPILE_LINK_RESULT.json", b"{}\n"))
        module = load(root / ADAPTER.SOURCE, "_server10_cpu_test_locked_adapter")
        try:
            yield root, refs, config, module
        finally:
            sys.modules.pop(module.__name__, None)


def witness(config, mode="cold"):
    return dict(schema_version=1, purpose="CURRENT_CONTEXT_CACHED_NUMERICAL_REFERENCE_ONLY",
        gpu_uuid=config["gpu_uuid"], label=config["job_names"][mode], source_lock_sha256="a" * 64,
        scope_sha256="b" * 64, permissions_ref=config["permissions_ref"], session_id=88,
        guard_command=["python", "--execute", "--mode", mode], seconds_limit=300, reserved_seconds=320,
        authorized_scope_verified=True, active_reservation_verified=True)


class Server10MigrationTests(unittest.TestCase):
    def test_actual_guard_function_globals_accept_new_uuid_and_reject_old(self):
        with fixture() as (root, refs, config, module):
            runtime = module.create_runtime(root, refs, config)
            with runtime._session() as original:
                self.assertIs(original.validate_guard.__globals__, original.__dict__)
                fake_os = types.SimpleNamespace(name="posix", getsid=lambda _: 88,
                    environ={"CUDA_VISIBLE_DEVICES": module.GPU_UUID})
                with mock.patch.object(original, "os", fake_os):
                    original.validate_guard(witness(config), "cold")
                    with self.assertRaises(ValueError):
                        original.validate_guard(dict(witness(config), gpu_uuid=OLD_UUID), "cold")
                    fake_os.environ["CUDA_VISIBLE_DEVICES"] = OLD_UUID
                    with self.assertRaises(ValueError):
                        original.validate_guard(witness(config), "cold")
            self.assertEqual(original.GPU_UUID, OLD_UUID)
            self.assertEqual(runtime.last_migration_evidence["status"], "RESTORED_PRIVATE_GLOBAL_BINDINGS")

    def test_delegate_and_common_actual_globals_rebound_then_restored(self):
        with fixture() as (root, refs, config, module):
            runtime = module.create_runtime(root, refs, config)
            with runtime._session() as original:
                delegate = original._load_delegate(root, refs)
                common = delegate.load_source(root, refs, delegate.COMMON)
                self.assertEqual(delegate.execute_original_acquisition.__globals__["GPU_UUID"], module.GPU_UUID)
                self.assertEqual(common.load_sdk_assets.__globals__["SDK_INVENTORY"], config["sdk_inventory_ref"]["path"])
                self.assertEqual(common.GPU_UUID, module.GPU_UUID)
                self.assertEqual(common.PERMISSIONS, module.PERMISSIONS)
                # Call the real old common function with fixture leaf I/O only.
                sdk_refs = {p: dict(path=p, bytes=n, sha256=s) for p, n, s in common.SDK_SOURCE_REFS}
                calls = []
                sdk = types.SimpleNamespace(load_audited_assets=lambda inventory, proof: calls.append((inventory, proof)) or "cpu-pin")
                with mock.patch.object(common, "checked_ref", side_effect=lambda *args: None), \
                     mock.patch.object(common, "load_ref", return_value=sdk):
                    self.assertEqual(common.load_sdk_assets(root, sdk_refs), (sdk, "cpu-pin"))
                self.assertEqual(calls[0][0]["path"], str(root / config["sdk_inventory_ref"]["path"]))
                self.assertEqual(calls[0][1]["path"], str(root / config["sdk_proof_ref"]["path"]))
            self.assertEqual(delegate.GPU_UUID, OLD_UUID)
            self.assertEqual(common.GPU_UUID, OLD_UUID)
            self.assertIn("server09", common.SDK_INVENTORY)
            self.assertIn("server09", common.PERMISSIONS)
            for child in (original, delegate, common):
                self.assertNotIn(child.__name__, sys.modules)
            self.assertTrue(runtime.last_migration_evidence["all_bindings_restored"])

    def test_original_error_preserved_and_cleanup_on_exception(self):
        with fixture() as (root, refs, config, module):
            runtime = module.create_runtime(root, refs, config)
            error = KeyboardInterrupt("cpu fixture")
            with self.assertRaises(KeyboardInterrupt) as caught:
                with runtime._session() as original:
                    delegate = original._load_delegate(root, refs)
                    common = delegate.load_source(root, refs, delegate.COMMON)
                    raise error
            self.assertIs(caught.exception, error)
            self.assertEqual(common.GPU_UUID, OLD_UUID)
            self.assertEqual(runtime.last_migration_evidence["status"], "RESTORED_PRIVATE_GLOBAL_BINDINGS")
            with runtime._session() as again:
                self.assertIsNot(again, original)

    def test_exact_paths_use_new_jobs_and_preserve_existing_paired_storage(self):
        with fixture() as (root, refs, config, module):
            runtime = module.create_runtime(root, refs, config)
            with runtime._session() as original:
                for mode in ("cold", "paired"):
                    out = root / "experiments/prefix_io_v1/runs" / module.JOB_NAMES[mode] / "details"
                    out.mkdir(parents=True)
                    storage = root / module.STORAGE[mode]
                    if mode == "paired":
                        storage.mkdir(parents=True)
                    self.assertEqual(original._paths(root, mode, out, storage), (out, storage))
                    with self.assertRaises(ValueError):
                        original._paths(root, mode, out.parent / "old-details", storage)
                self.assertEqual(original.DELEGATE_REF["path"], DELEGATE)

    def test_no_environment_path_file_or_preexisting_module_changes(self):
        with fixture() as (root, refs, config, module):
            before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
            environment, path, argv = dict(os.environ), list(sys.path), sys.argv
            preexisting = types.ModuleType("_g2_lifecycle_cpu_fixture_preserved")
            sys.modules[preexisting.__name__] = preexisting
            try:
                runtime = module.create_runtime(root, refs, config)
                with runtime._session() as original:
                    delegate = original._load_delegate(root, refs)
                    delegate.load_source(root, refs, delegate.COMMON)
                self.assertIs(sys.modules[preexisting.__name__], preexisting)
                self.assertEqual({p: p.read_bytes() for p in root.rglob("*") if p.is_file()}, before)
                self.assertEqual(os.environ, environment)
                self.assertEqual(sys.path, path)
                self.assertIs(sys.argv, argv)
            finally:
                sys.modules.pop(preexisting.__name__)

    def test_drift_and_identity_changes_rejected(self):
        with fixture() as (root, refs, config, module):
            with self.assertRaises(ValueError):
                module.create_runtime(root, refs, dict(config, gpu_uuid=OLD_UUID))
            changed = json.loads(json.dumps(config))
            changed["job_names"]["cold"] = "old-job"
            with self.assertRaises(ValueError):
                module.create_runtime(root, refs, changed)
            (root / config["sdk_inventory_ref"]["path"]).write_bytes(b"changed\n")
            with self.assertRaises(ValueError):
                module.create_runtime(root, refs, config)

    def test_original_source_sha_drift_rejected(self):
        with fixture() as (root, refs, config, module):
            path = root / module.ORIGINAL
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + (b"X" if raw[-1:] != b"X" else b"Y"))
            with self.assertRaisesRegex(ValueError, "SHA drift"):
                module.create_runtime(root, refs, config)

    def test_guard_permissions_rejected_before_original_acquisition(self):
        with fixture() as (root, refs, config, module):
            runtime = module.create_runtime(root, refs, config)
            with mock.patch.object(runtime, "_session", side_effect=AssertionError("must not load acquisition")):
                with self.assertRaisesRegex(ValueError, "server10 permission"):
                    runtime.execute_original_acquisition(root, "cold", root, root, refs,
                        dict(witness(config), permissions_ref=dict(path="old", bytes=1, sha256="a" * 64)))

    def test_config_and_refs_are_copied_at_creation(self):
        with fixture() as (root, refs, config, module):
            runtime = module.create_runtime(root, refs, config)
            config["job_names"]["cold"] = "mutated-after-binding"
            self.assertEqual(runtime.config["job_names"], module.JOB_NAMES)
            refs[module.ORIGINAL]["bytes"] += 1
            self.assertEqual(runtime.refs[module.ORIGINAL], module.ORIGINAL_REF)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST])
