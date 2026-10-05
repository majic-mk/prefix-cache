"""CPU counterexamples; these fixtures never qualify a native GPU receipt."""
import dataclasses
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).parent / "source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py"
spec = importlib.util.spec_from_file_location("cpu_test_single_file_receipt", SOURCE)
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)


def write(root, relative, document):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(document, sort_keys=True).encode()
    path.write_bytes(raw)
    return {"path": relative, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}


class ReceiptTests(unittest.TestCase):
    def test_cannot_construct_from_pass_or_empty(self):
        with self.assertRaises(ValueError):
            R.ExactSingleFileReceipt()
        with self.assertRaises(ValueError):
            R.ExactSingleFileReceipt({"PASS": True})

    def test_only_finite_nonproduction_flags(self):
        self.assertIs(R.ExactSingleFileReceipt.condition_only, True)
        self.assertIs(R.ExactSingleFileReceipt.production_qualified, False)

    def test_reference_is_frozen_and_exact(self):
        value = R.FileRef("a", 1, "0" * 64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.bytes = 2
        with self.assertRaises(ValueError):
            R.FileRef.from_mapping(dict(value.mapping(), origin="native_gpu_recording"))

    def test_file_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            ref = R.FileRef.from_mapping(write(root, "one.json", {"a": 1}))
            self.assertEqual(ref.json(root), {"a": 1})
            (root / "one.json").write_bytes(b'{"a": 2}')
            with self.assertRaises(ValueError):
                ref.read(root)

    def test_paths_cannot_escape_project(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in ("../outside", "/outside", "C:/outside", "a\\b", "a//b", "a/./b"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    R._relative(Path(directory).resolve(), path)

    def test_duplicate_json_does_not_override_origin(self):
        with self.assertRaises(ValueError):
            json.loads('{"origin":"cpu_fixture","origin":"native_gpu_recording"}', object_pairs_hook=R._pairs)

    def test_pass_json_is_not_a_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            write(root, "binding.json", {"PASS": True, "native_execution_verified": True,
                "conditional_cost_cell_qualified": True})
            with self.assertRaisesRegex(ValueError, "exact source-bound"):
                R.load_verified_single_file(root, "binding.json")

    def test_eight_file_cell_cannot_be_relabelled_single_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            plan = write(root, "plan.json", {"job_id": "server11-native-cost-six-window-05",
                "units": 8, "operations": 8, "transfer_quantum_bytes": 917504})
            measurements = write(root, "measurements.json", {"origin": "native_gpu_recording"})
            guard = write(root, "guard.json", {"exit_code": 0})
            write(root, "binding.json", {"schema_version": 1, "scope": "p4_single_file_native_binding_v1",
                "calibration": {"plan_ref": plan, "measurements_ref": measurements, "guard_ref": guard},
                "runtime_common_refs": [], "runtime_overlay_refs": [], "kernel_mode": R._KERNEL_MODE})
            with self.assertRaisesRegex(ValueError, "actual v5 single-file"):
                R.load_verified_single_file(root, "binding.json")

    def test_cost_verifier_source_must_match_frozen_v5(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = root / R._VERIFIER.path
            path.parent.mkdir(parents=True)
            path.write_text("# PASS result cannot substitute for the verifier\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size changed"):
                R._load_frozen_serializer(root)

    @staticmethod
    def budget_fixture():
        plan = {"entries": [{"pair_id": "c0", "split": "calibration"},
            {"pair_id": "c1", "split": "calibration"}, {"pair_id": "h", "split": "validation"}]}
        result = {"calibration_only_cost": {"baseline_ns": 101}, "windows": [
            {"pair_id": "c0", "arm": "baseline", "selected_gpu_elapsed_ns": 99},
            {"pair_id": "c1", "arm": "baseline", "selected_gpu_elapsed_ns": 102},
            {"pair_id": "c0", "arm": "action", "selected_gpu_elapsed_ns": 999999},
            {"pair_id": "h", "arm": "baseline", "selected_gpu_elapsed_ns": 9999999},
            {"pair_id": "h", "arm": "action", "selected_gpu_elapsed_ns": 99999999}]}
        return plan, result

    def test_budget_only_uses_a_calibration(self):
        plan, result = self.budget_fixture()
        self.assertEqual(R._calibration_a_budget(plan, result), 102)
        for row in result["windows"][2:]:
            row["selected_gpu_elapsed_ns"] = 1
        self.assertEqual(R._calibration_a_budget(plan, result), 102)

    def test_budget_rejects_forged_baseline(self):
        plan, result = self.budget_fixture()
        result["calibration_only_cost"]["baseline_ns"] = 9999
        with self.assertRaisesRegex(ValueError, "disagrees"):
            R._calibration_a_budget(plan, result)

    def test_budget_rejects_duplicate_or_missing_a(self):
        plan, result = self.budget_fixture()
        result["windows"].append(result["windows"][0])
        with self.assertRaisesRegex(ValueError, "one measured A"):
            R._calibration_a_budget(plan, result)

    def test_kernel_descriptor_uses_actual_options_and_absolute_storage(self):
        # This helper fixture checks config interpretation only, never issuance.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            windows = []
            for index in range(6):
                storage = "private/window-" + str(index)
                config = dict(dtype="bfloat16", kv_cache_dtype="auto", quantization=None,
                    enforce_eager=True, compilation_config=0, attention_config={"backend": "TRITON_ATTN"},
                    tensor_parallel_size=1, distributed_executor_backend="uni", async_scheduling=False,
                    block_size=16, max_num_seqs=1, kv_cache_memory_bytes=268435456,
                    max_model_len=1040, max_num_batched_tokens=1040, disable_log_stats=True,
                    kv_transfer_config=dict(kv_connector="OffloadingConnector", kv_role="kv_both",
                        kv_connector_extra_config=dict(shared_storage_path=str(root / storage),
                            spec_name="PyKvCacheOffloadingSpec", spec_module_path="py_kvcache.vllm",
                            block_size=16, sync_on_store=False, staging_mem=0.125, iodepth=4, enable_preload=True,
                            preload_lookahead_requests=1, preload_share_staging=True, staging_cache="lru",
                            load_planner="off", io_backend="linux_aio")))
                document = dict(engine_config=config, private_storage=storage)
                windows.append({"raw_child_ref": write(root, "child-" + str(index) + ".json", document)})
            self.assertEqual(R._kernel_mode({"windows": windows}, root), R._KERNEL_MODE)
            document["engine_config"]["attention_config"] = {"backend": "FLASH_ATTN"}
            windows[-1]["raw_child_ref"] = write(root, "child-5.json", document)
            with self.assertRaisesRegex(ValueError, "kernel configuration"):
                R._kernel_mode({"windows": windows}, root)


if __name__ == "__main__":
    unittest.main()
