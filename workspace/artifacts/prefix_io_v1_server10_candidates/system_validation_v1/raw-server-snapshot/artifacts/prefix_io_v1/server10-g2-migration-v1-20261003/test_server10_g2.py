"""CPU fixtures for unchanged original G2 gates and machine binding."""
import argparse
import contextlib
import copy
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
parser.add_argument("--original-file", type=Path)
ARGS, REST = parser.parse_known_args()
ORIGINAL = ARGS.original_file or HERE.parents[1] / "prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/run_g2_normal_model_lifecycle.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


ENTRY = load(HERE / "run_server10_g2.py", "_server10_g2_cpu_fixture_entry")


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        refs = {}
        def write(relative, raw):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            row = dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            refs[relative] = row
            return row
        write(ENTRY.ORIGINAL_REF["path"], ORIGINAL.read_bytes())
        inventory = write(ENTRY.INVENTORY_REF["path"], b"{}\n")
        proof = write(ENTRY.PROOF_REF["path"], b"{}\n")
        permit = write(ENTRY.PERMISSIONS, b"CPU fixture permit\n")
        with mock.patch.object(ENTRY, "INVENTORY_REF", inventory), mock.patch.object(ENTRY, "PROOF_REF", proof):
            config = dict(schema_version=1, gpu_uuid=ENTRY.GPU_UUID, job_names=dict(ENTRY.JOBS),
                permissions_ref=permit, sdk_inventory_ref=inventory, sdk_proof_ref=proof,
                source_lock=ENTRY.LOCK, scope_record=ENTRY.SCOPE)
            module, cleanup = ENTRY.bind_original(root, refs, config)
            try:
                yield root, refs, config, module, write
            finally:
                evidence = cleanup()
                assert evidence["status"] == "RESTORED_PRIVATE_G2_MACHINE_BINDINGS", evidence


@contextlib.contextmanager
def gate_fixture(mode="off"):
    with fixture() as (root, refs, config, module, write):
        lock_ref = write(ENTRY.LOCK, b"{}\n")
        human = dict(authorization_origin="direct_human_reply", purpose=module.PURPOSE,
            gpu_uuid=ENTRY.GPU_UUID, allow_gpu_runs=True, allowed_modes=["off", "shadow"],
            maximum_jobs=2, maximum_total_planned_reserve_seconds=640)
        human_rel = ENTRY.DELIVERY + "/human-cpu-fixture.json"
        human_ref = write(human_rel, json.dumps(human).encode())
        scope = dict(module.scope_template(), status="USER_AUTHORIZED_G2_NORMAL_MODEL_LIFECYCLE",
            allow_gpu_initialization=True, allow_gpu_runs=True, source_lock=ENTRY.LOCK,
            source_lock_sha256=lock_ref["sha256"], base_permissions=config["permissions_ref"],
            human_authorization_record=human_ref)
        write(ENTRY.SCOPE, json.dumps(scope).encode())
        ledger = dict(gpu_wall_seconds=100, active_reservation=None, events=[])
        write("experiments/prefix_io_v1/gpu-budget-ledger.json", json.dumps(ledger).encode())
        (root / "experiments/prefix_io_v1/runs").mkdir()
        args = types.SimpleNamespace(project=root, mode=mode, name=ENTRY.JOBS[mode], source_lock=ENTRY.LOCK,
            scope_record=ENTRY.SCOPE, scalar_relative=module.SCALAR, frame_relative=module.FRAME)
        permit = dict(allow_gpu_runs=True, approved_gpu_ids=[ENTRY.GPU_UUID], max_gpu_hours=8,
            approved_experiment_root=str(root / "experiments/prefix_io_v1/runs"))
        prepare = types.SimpleNamespace(permission_fields=lambda text: permit)
        with mock.patch.object(module, "verify_source_lock", return_value=refs) as verifier, \
             mock.patch.object(module, "load_ref", return_value=prepare), \
             mock.patch.object(module.shutil, "disk_usage", return_value=types.SimpleNamespace(free=10**12)):
            refs[module.PREPARE] = dict(path=module.PREPARE, bytes=1, sha256="a" * 64)
            yield root, refs, config, module, write, args, scope, human, ledger, verifier


class Server10G2Contracts(unittest.TestCase):
    def test_original_function_globals_and_full_output_contract_preserved(self):
        with fixture() as (root, refs, config, module, write):
            self.assertIs(module.execute_guarded.__globals__, module.__dict__)
            self.assertEqual(module.execute_guarded.__globals__["GPU_UUID"], ENTRY.GPU_UUID)
            self.assertEqual(Path(module.execute_guarded.__code__.co_filename), root / ENTRY.ORIGINAL_REF["path"])
            self.assertEqual(module.SAMPLING["max_tokens"], 128)
            self.assertEqual(module.SAMPLING["min_tokens"], 128)
            self.assertEqual(module.PROMPT, list(range(1000, 1128)))
            self.assertEqual(module.scope_template()["permitted_run_names"], ENTRY.JOBS)
            self.assertEqual(module.scope_template()["maximum_total_planned_reserve_seconds"], 640)
            self.assertFalse(module.scope_template()["allow_gpu_runs"])

    def test_original_off_path_does_not_access_worker_or_collector(self):
        with fixture() as (_, _, _, module, _):
            class NoTouch:
                def __getattribute__(self, key):
                    raise AssertionError("off touched worker")
            with mock.patch.object(module, "load_ref", side_effect=AssertionError("off loaded collector")):
                self.assertEqual(module.install_worker_observation(NoTouch(), {"mode": "off"})["status"], "OFF_ORIGINAL_PATH")
                self.assertEqual(module.export_worker_observation(NoTouch(), "off")["frames"], [])

    def test_sdk_actual_common_globals_select_new_inventory_and_proof(self):
        with fixture() as (root, refs, config, module, _):
            sdk_refs = {p: dict(path=p, bytes=n, sha256=s) for p, n, s in module.SDK_SOURCE_REFS}
            calls = []
            sdk = types.SimpleNamespace(load_audited_assets=lambda inv, proof: calls.append((inv, proof)) or "pin")
            with mock.patch.object(module, "checked_ref"), mock.patch.object(module, "load_ref", return_value=sdk):
                self.assertEqual(module.load_sdk_assets(root, sdk_refs), (sdk, "pin"))
            self.assertEqual(calls[0][0]["path"], str(root / config["sdk_inventory_ref"]["path"]))
            self.assertEqual(calls[0][1]["path"], str(root / config["sdk_proof_ref"]["path"]))

    def test_original_scope_accepts_new_g2_and_rejects_reference_scope_before_model_scan(self):
        with gate_fixture() as (root, refs, config, module, write, args, scope, human, ledger, verifier):
            self.assertEqual(module.scope_source_gates(root, args)["refs"], refs)
            bad = dict(scope, status="USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC")
            write(ENTRY.SCOPE, json.dumps(bad).encode())
            verifier.reset_mock()
            with self.assertRaises(ValueError):
                module.scope_source_gates(root, args)
            verifier.assert_not_called()

    def test_original_scope_rejects_wrong_uuid_extra_jobs_budget_and_old_label(self):
        with gate_fixture() as (root, refs, config, module, write, args, scope, human, ledger, verifier):
            for field, value in (("gpu_uuid", "GPU-old"), ("maximum_jobs", 3),
                                 ("maximum_total_planned_reserve_seconds", 641)):
                write(ENTRY.SCOPE, json.dumps(dict(scope, **{field: value})).encode())
                with self.assertRaises(ValueError):
                    module.scope_source_gates(root, args)
            write(ENTRY.SCOPE, json.dumps(scope).encode())
            args.name = "server09-g2-normal-off-05"
            with self.assertRaises(ValueError):
                module.scope_source_gates(root, args)

    def test_original_used_job_and_exhausted_budget_rejected(self):
        with gate_fixture() as (root, refs, config, module, write, args, scope, human, ledger, verifier):
            ledger["events"] = [dict(label=args.name)]
            write("experiments/prefix_io_v1/gpu-budget-ledger.json", json.dumps(ledger).encode())
            with self.assertRaisesRegex(ValueError, "job used"):
                module.scope_source_gates(root, args)
            ledger.update(events=[], gpu_wall_seconds=28800-319)
            write("experiments/prefix_io_v1/gpu-budget-ledger.json", json.dumps(ledger).encode())
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                module.scope_source_gates(root, args)

    def test_original_execution_gate_requires_current_sid_new_uuid_and_exact_command(self):
        with gate_fixture() as (root, refs, config, module, write, args, scope, human, ledger, verifier):
            active = dict(label=args.name, gpu_uuid=ENTRY.GPU_UUID, seconds_limit=300, reserved_seconds=320,
                command=module.child_command(args), session_id=88, permissions=scope["base_permissions"])
            ledger["active_reservation"] = active
            write("experiments/prefix_io_v1/gpu-budget-ledger.json", json.dumps(ledger).encode())
            fake_os = types.SimpleNamespace(getsid=lambda _: 88, environ=dict(CUDA_VISIBLE_DEVICES=ENTRY.GPU_UUID,
                HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"))
            with mock.patch.object(module, "os", fake_os):
                module.execution_gates(root, args)
                self.assertEqual(active["command"][2], ENTRY.SCRIPT)
                fake_os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-old"
                with self.assertRaises(ValueError):
                    module.execution_gates(root, args)
                fake_os.environ["CUDA_VISIBLE_DEVICES"] = ENTRY.GPU_UUID
                active["session_id"] = 89
                write("experiments/prefix_io_v1/gpu-budget-ledger.json", json.dumps(ledger).encode())
                with self.assertRaises(ValueError):
                    module.execution_gates(root, args)

    def test_original_shadow_requires_success_shutdown_and_session_drain(self):
        with gate_fixture("shadow") as (root, refs, config, module, write, args, scope, human, ledger, verifier):
            event = dict(label=ENTRY.JOBS["off"], child_exit=0, exit=0, session_drained=True)
            result = dict(status="PASSED_NORMAL_MODEL_FULL_OUTPUT_ONLY", purpose=module.PURPOSE,
                original_engine_shutdown_returned=True, source_lock_sha256=scope["source_lock_sha256"])
            relative = "experiments/prefix_io_v1/runs/" + ENTRY.JOBS["off"] + "/details/normal-model-lifecycle-result.json"
            write(relative, json.dumps(result).encode())
            gates = dict(scope=scope, ledger=dict(events=[event]))
            module.shadow_prerequisite(root, args, gates)
            event["session_drained"] = False
            with self.assertRaises(ValueError):
                module.shadow_prerequisite(root, args, gates)
            event["session_drained"] = True
            write(relative, json.dumps(dict(result, original_engine_shutdown_returned=False)).encode())
            with self.assertRaises(ValueError):
                module.shadow_prerequisite(root, args, gates)

    def test_process_environment_restored_and_original_source_unchanged(self):
        environment, path = dict(os.environ), list(sys.path)
        with fixture() as (root, refs, config, module, _):
            original_bytes = (root / ENTRY.ORIGINAL_REF["path"]).read_bytes()
            os.environ["SERVER10_G2_CPU_TEST_ONLY"] = "fixture"
            sys.path.append("cpu-fixture-path")
            private_name = module.__name__
        self.assertEqual(dict(os.environ), environment)
        self.assertEqual(sys.path, path)
        self.assertNotIn(private_name, sys.modules)
        self.assertEqual(hashlib.sha256(original_bytes).hexdigest(), ENTRY.ORIGINAL_REF["sha256"])

    def test_original_guard_launch_command_retained(self):
        with fixture() as (root, refs, config, module, _):
            args = ["--project", str(root), "--mode", "off", "--name", ENTRY.JOBS["off"],
                    "--scope-record", ENTRY.SCOPE, "--source-lock", ENTRY.LOCK, "--launch"]
            with mock.patch.object(module, "scope_source_gates", return_value=dict(ledger=dict(active_reservation=None))), \
                 mock.patch.object(module, "shadow_prerequisite"), \
                 mock.patch("subprocess.run", return_value=types.SimpleNamespace(returncode=0)) as launch:
                self.assertEqual(module.main(args), 0)
            command = launch.call_args.args[0]
            self.assertEqual(command[:7], [".venv/bin/python", "-B", module.GUARD, "--permissions-path",
                ENTRY.PERMISSIONS, "--label", ENTRY.JOBS["off"]])
            self.assertEqual(command[7:10], ["--seconds", "300", "--"])
            self.assertIn(ENTRY.SCRIPT, command)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST])
