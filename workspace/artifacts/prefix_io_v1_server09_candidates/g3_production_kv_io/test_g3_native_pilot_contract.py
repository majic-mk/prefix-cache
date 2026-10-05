"""Meaningful CPU negatives; all curve/layout data below are explicit fixtures."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
SOURCE = Path(os.environ.get("PREFIX_G3_CPU_NATIVE_ROOT", str(WORKSPACE / "artifacts/prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source/third_party/work/py-kvcache-p4-02-cpu")))
LOCK = Path(os.environ.get("PREFIX_G3_CPU_LOCK_PATH", str(WORKSPACE / "artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/gpu-source-lock-candidate.json")))
spec = importlib.util.spec_from_file_location("g3_native_pilot_contract", HERE / "g3_native_pilot_contract.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = LOCK.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == "0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b"
        locked = {row["path"]: row for row in json.loads(raw)["files"]}
        cls.refs = {name: locked["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/" + name]
                    for name in C.SOURCE_FILES}
        cls.before = {name: hashlib.sha256((SOURCE / "py_kvcache" / name).read_bytes()).hexdigest()
                      for name in C.SOURCE_FILES}
        cls.probe = C.BoundOriginalCPUProbe(SOURCE, cls.refs)

    @classmethod
    def tearDownClass(cls):
        cls.probe.close()
        assert cls.before == {name: hashlib.sha256((SOURCE / "py_kvcache" / name).read_bytes()).hexdigest()
                              for name in C.SOURCE_FILES}

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="g3_cpu_contract_fixture_")
        self.addCleanup(self.temp.cleanup)
        self.curve_path = Path(self.temp.name) / "CPU_FIXTURE_CURVES.json"
        self.curve = dict(schema_version=2, model_name="CPU_FIXTURE_MODEL", kv_dtype="auto",
            kv_bytes_per_token=57344, break_even_ssd_tokens=64, break_even_mem_tokens=16,
            safety_margin_tokens=0, golden=[], curves={
                "f": dict(floor=0.0, knots={"64": .1, "128": .2, "256": .4, "512": .8}),
                "g_ssd": dict(floor=.005, knots={"64": .01, "128": .015, "256": .02, "512": .03}),
                "g_mem": dict(floor=.002, knots={"64": .005, "128": .006, "256": .008, "512": .01})})
        self.engine = dict(model="CPU_FIXTURE_MODEL", dtype="bfloat16", kv_cache_dtype="auto",
            quantization=None, tensor_parallel_size=1, distributed_executor_backend="uni",
            max_num_seqs=1, max_model_len=256, max_num_batched_tokens=256, block_size=16,
            kv_cache_memory_bytes=64 * 1024**2, enable_prefix_caching=True, enforce_eager=True,
            async_scheduling=False, cpu_offload_gb=0, offload_group_size=0, kv_offloading_size=None)
        self.storage = str(Path(self.temp.name) / "PRIVATE_FIXTURE_SSD")
        self.extra = dict(spec_name="PyKvCacheOffloadingSpec", spec_module_path="py_kvcache.vllm",
            shared_storage_path=self.storage, block_size=16, io_backend="linux_aio", iodepth=1,
            open_lookahead=1, staging_mem=16 / 1024, sync_on_store=False, enable_preload=True,
            preload_share_staging=True, preload_lookahead_requests=1, staging_cache="off",
            load_planner="on", prefix_cache_break_even_path=str(self.curve_path),
            prefix_io_parent_admission=dict(schema_version=1, run_id="cpu-fixture-pair", max_accepted_parents=2),
            prefix_io_p4_policy=dict(mode="off"))
        self.layout = dict(origin="cpu_planned", group_count=1, gpu_block_tokens=16,
            storage_block_tokens=16, canonical_page_bytes=[917504], num_gpu_blocks=16,
            actual_kv_backing_bytes=None)
        self.plan = dict(schema_version=1, mode="off", pair_id="cpu-fixture-pair",
            shared_storage_path=self.storage, engine=self.engine,
            sampling=dict(temperature=0.0, seed=0, max_tokens=128, min_tokens=128, ignore_eos=True, detokenize=False),
            kv_transfer_config=dict(kv_connector="OffloadingConnector", kv_role="kv_both",
                                    kv_connector_extra_config=self.extra),
            seed_token_ids=list(range(1000, 1128)), flush_token_ids=list(range(2000, 2128)),
            consumer_token_ids=list(range(1000, 1128)), layout=self.layout,
            distinct_original_processes=True, producer_shutdown_before_consumer=True, reset_connector=False)
        self.freeze_curve()

    def freeze_curve(self):
        raw = json.dumps(self.curve, allow_nan=False).encode()
        self.curve_path.write_bytes(raw)
        self.ref = dict(path="CPU_FIXTURE_CURVES.json", bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    def check(self, plan=None):
        return C.validate_two_process_plan(plan or self.plan, curve_path=self.curve_path,
                                           curve_ref=self.ref, probe=self.probe)

    def reject(self, substring):
        with self.assertRaisesRegex(ValueError, substring):
            self.check()

    def test_original_off_math_roundtrip_keeps_every_qualification_closed(self):
        result = self.check()
        self.assertEqual(result["cold_admission"]["decision"], "defer")
        self.assertEqual(result["cold_admission"]["planned_preload_blocks"], 8)
        self.assertEqual(result["layout"]["canonical_operand_count"], 1)
        self.assertEqual(result["planned_guarded_jobs_for_both_modes"], 4)
        self.assertFalse(result["gpu_authorized"] or result["GPU_verified"] or result["production_qualified"])
        self.assertFalse(result["cold_admission"]["real_KV_hashes_verified"])

    def test_shadow_has_same_original_decision_no_cost_caps(self):
        self.plan["mode"] = "shadow"
        self.extra["prefix_io_p4_policy"] = dict(schema_version=1, run_id="cpu-fixture-pair", mode="shadow",
            sample_max_age_ns=200000000, max_wait_ns=200000000, internal_step_budget_ns=None,
            candidate_batches=[1, 2, 4, 8], fixed_stage_policy=None, cost_table=None)
        self.assertEqual(self.check()["cold_admission"]["decision"], "defer")
        self.extra["prefix_io_p4_policy"]["cost_table"] = {"status": "PASS"}
        self.reject("cannot activate")

    def test_distinct_28_operand_layout_also_valid_no_tensor_count_assumption(self):
        self.layout["canonical_page_bytes"] = [32768] * 28
        self.assertEqual(self.check()["layout"]["payload_bytes"], 917504)

    def test_actual_scalar_layout_does_not_promote_gpu_qualification(self):
        self.layout.update(origin="native_scalar_candidate", actual_kv_backing_bytes=16 * 917504)
        self.assertFalse(self.check()["layout"]["GPU_verified"])
        self.layout["actual_kv_backing_bytes"] += 1
        self.reject("actual unique")

    def test_same_process_cannot_pass(self):
        self.plan["distinct_original_processes"] = False
        self.reject("fresh consumer")

    def test_producer_shutdown_is_required_not_future_done(self):
        self.plan["producer_shutdown_before_consumer"] = False
        self.reject("fresh consumer")

    def test_connector_reset_cannot_erase_producer_cache(self):
        self.plan["reset_connector"] = True
        self.reject("reset forbidden")

    def test_consumer_full_prompt_must_match(self):
        self.plan["consumer_token_ids"][-1] += 1
        self.reject("exact complete")

    def test_flush_cannot_share_the_seed_prefix(self):
        self.plan["flush_token_ids"][0] = self.plan["seed_token_ids"][0]
        self.reject("prefix-disjoint")

    def test_complete_outputs_not_selected_count(self):
        self.plan["sampling"]["max_tokens"] = self.plan["sampling"]["min_tokens"] = 1
        self.reject("128 outputs")

    def test_full_work_must_fit_context(self):
        self.engine["max_model_len"] = 255
        self.reject("context budget")

    def test_original_admission_cannot_be_disabled(self):
        self.extra["load_planner"] = "off"
        self.reject("original planner")

    def test_shared_staging_is_required(self):
        self.extra["preload_share_staging"] = False
        self.reject("original planner")

    def test_foreground_staging_cannot_masquerade_as_cold_ssd(self):
        self.extra["staging_cache"] = "lru"
        self.reject("original planner")

    def test_different_ssd_roots_rejected(self):
        self.extra["shared_storage_path"] += "_different"
        self.reject("original planner")

    def test_existing_curve_bytes_drift_is_rejected(self):
        self.curve_path.write_bytes(self.curve_path.read_bytes() + b" ")
        self.reject("bytes changed")

    def test_model_label_cannot_be_replaced_by_absolute_local_path(self):
        self.engine["model"] = str(Path(self.temp.name) / "model")
        self.reject("curve binding differs")

    def test_original_worker_ssd_threshold_must_pass(self):
        self.curve["break_even_ssd_tokens"] = 1040
        self.freeze_curve()
        self.reject("declines")

    def test_preserved_preload_mem_gate_must_also_pass(self):
        self.curve["break_even_mem_tokens"] = 1040
        self.freeze_curve()
        self.reject("declines")

    def test_original_planner_decline_is_not_overridden(self):
        self.curve["curves"]["f"]["knots"] = {"64": .001, "128": .002, "256": .003, "512": .004}
        self.freeze_curve()
        self.reject("declines")

    def test_curve_layout_density_must_match(self):
        self.curve["kv_bytes_per_token"] = 1
        self.freeze_curve()
        self.reject("layout bytes")

    def test_original_measured_support_not_extrapolated(self):
        for curve in self.curve["curves"].values():
            curve["knots"] = {"16": .1, "64": .2}
        self.freeze_curve()
        self.reject("measured curve support")

    def test_real_floor_overhead_cannot_exceed_budget(self):
        self.extra["staging_mem"] = 5 * 917504 / (1 << 30)
        self.reject("backing requires")

    def test_gpu_capacity_cannot_be_invented(self):
        self.engine["kv_cache_memory_bytes"] = 917504
        self.reject("exceeds frozen budget")

    def test_group_and_exact_int_geometry(self):
        self.layout["group_count"] = True
        self.reject("one actual KV group")

    def test_speculative_or_async_engine_is_rejected(self):
        self.engine["async_scheduling"] = True
        self.reject("ordinary original")

    def test_off_cannot_carry_mock_cost_table(self):
        self.extra["prefix_io_p4_policy"]["cost_table"] = {"GPU_verified": True}
        self.reject("no strategy state")

    def test_reference_sha_pass_label_is_not_source_binding(self):
        bad = copy.deepcopy(self.refs)
        bad["load_planner.py"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            C.BoundOriginalCPUProbe(SOURCE, bad)

    def test_backend_code_rejected_before_backend_import(self):
        dest = Path(self.temp.name) / "copied_CPU_sources"
        (dest / "py_kvcache").mkdir(parents=True)
        bad = copy.deepcopy(self.refs)
        for name in C.SOURCE_FILES:
            shutil.copyfile(SOURCE / "py_kvcache" / name, dest / "py_kvcache" / name)
        path = dest / "py_kvcache/load_planner.py"
        raw = path.read_bytes() + b"\nimport torch\n"
        path.write_bytes(raw)
        bad["load_planner.py"].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        before = "torch" in sys.modules
        with self.assertRaisesRegex(ValueError, "backend import rejected"):
            C.BoundOriginalCPUProbe(dest, bad)
        self.assertEqual("torch" in sys.modules, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
