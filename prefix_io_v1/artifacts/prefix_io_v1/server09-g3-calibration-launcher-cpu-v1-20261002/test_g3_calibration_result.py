"""Bounded fixture contracts, not actual model execution or native GPU evidence."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root")
options, remaining = parser.parse_known_args()
spec = importlib.util.spec_from_file_location("g3_point_result", HERE / "g3_calibration_result.py")
R = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R
spec.loader.exec_module(R)
STORAGE = "/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/cpu-fixture/storage"
SOURCE_SHA = "1" * 64
SCOPE_SHA = "2" * 64
GPU = "GPU-cpu-fixture-no-device"


def make_config(mode):
    extra = {"spec_name": "PyKvCacheOffloadingSpec", "spec_module_path": "py_kvcache.vllm",
             "shared_storage_path": STORAGE, "block_size": 16, "sync_on_store": False,
             "staging_mem": 0.125, "iodepth": 4, "enable_preload": True,
             "preload_lookahead_requests": 1, "preload_share_staging": True,
             "staging_cache": "lru", "load_planner": "off", "io_backend": "linux_aio"}
    engine = {"max_model_len": 1040, "max_num_batched_tokens": 1040, "max_num_seqs": 1,
              "block_size": 16, "kv_cache_memory_bytes": 268435456, "tensor_parallel_size": 1,
              "compilation_config": 0, "seed": 0, "dtype": "bfloat16", "kv_cache_dtype": "auto",
              "quantization": None, "enable_prefix_caching": True, "enable_chunked_prefill": True,
              "enforce_eager": True, "async_scheduling": False, "prefix_caching_hash_algo": "sha256",
              "distributed_executor_backend": "uni", "attention_config": {"backend": "TRITON_ATTN"},
              "kv_transfer_config": None if mode == "cold" else {"kv_connector": "OffloadingConnector",
                  "kv_role": "kv_both", "kv_connector_extra_config": extra}}
    return {"model": {"model_id": R.MODEL_ALIAS, "revision": "frozen-fixture"},
            "model_alias": R.MODEL_ALIAS, "alias_target": "/cpu-fixture/model", "engine": engine,
            "sampling": {"temperature": 0.0, "seed": 0, "max_tokens": 1, "min_tokens": 1,
                         "ignore_eos": True, "detokenize": False}, "sizes": [128], "reps": 3,
            "acquisition_domain": 1024, "gpu_uuid": GPU, "pythonhashseed": "0",
            "planned_request_batch": 1, "cost_acquisition_only": True, "prompt_manifest": None,
            "native_hot_diagnostic": False, "cached_reference_logprobs": False}


def drain(external, profile=False):
    result = {"profile_flushed": True} if profile else {}
    if external:
        result.update(queued_stores_submitted=0, waited_job_ids=[], handlers=[{
            "staging_bytes": R.FILE_BYTES * (134217728 // R.FILE_BYTES), "staging_budget": 134217728,
            "pinned": True, "io_size": R.FILE_BYTES, "storage_block_bytes": R.FILE_BYTES,
            "aio": {"accepted": 24, "completed": 24, "reaped": 24, "outstanding": 0,
                    "pending": 0, "ready": 0, "unreaped": 0, "fatal": None,
                    "closed": False, "drained": False}}])
    return result


def make_report(mode):
    rows = []
    kinds = {"cold": ("f",), "populate": ("store", "g_mem"), "paired": ("g_ssd", "g_mem")}[mode]
    for rep in range(3):
        for position, kind in enumerate(kinds):
            ordinal = rep * (3 if mode == "populate" else len(kinds)) + (2 if mode == "populate" and position else position)
            request = str(ordinal)
            internal = request + "-1234abcd"
            loaded = kind in ("g_ssd", "g_mem")
            from_cache = loaded and kind == "g_mem"
            transfer = {"job_id": ordinal, "req_id": internal, "direction": "storage_to_gpu",
                        "success": True, "num_bytes": R.READ_BYTES, "num_files": 8, "num_blocks": 8,
                        "src_file": 0, "src_preload": 0 if from_cache else 8, "src_cache": 8 if from_cache else 0}
            trace = {"internal_request_id": internal, "event_count": 100, "load_transfers": [transfer] if loaded else [],
                     "transfer_success": True, "foreground_logical_read_bytes": 0,
                     "preload_actual_read_bytes": R.READ_BYTES if kind == "g_ssd" else 0,
                     "src_cache": 8 if from_cache else 0, "src_file": 0, "src_preload": 8 if kind == "g_ssd" else 0,
                     "h2d_events": 4 if loaded else 0, "planner_events": []}
            metrics = {"num_generation_tokens": 1, "arrival_time": 1790000000.0 + rep,
                       "queued_ts": 1000.0 + rep, "scheduled_ts": 1000.1 + rep,
                       "first_token_ts": 1000.2 + rep, "last_token_ts": 1000.2 + rep,
                       "first_token_latency": (0.1 if kind == "g_mem" else 0.2) + rep / 100,
                       "is_corrupted": False}
            rows.append({"kind": kind, "prefix_tokens": 128, "rep": rep, "warmup": rep == 0,
                         "request_id": request, "prompt_token_ids": R.expected_prompt(rep),
                         "output_token_ids": [500 + rep], "num_cached_tokens": 128 if loaded else 0,
                         "wall_seconds": 99.0, "metrics": metrics, "drain": drain(mode != "cold", True),
                         "trace": trace, "trace_path": "/cpu-fixture/request.trace.json"})
    return {"mode": mode, "status": "PASSED_NATIVE_" + mode.upper() + "_ACQUISITION",
            "performance_claim": False, "full_P1_pass": False, "rows": rows,
            "native_hot_diagnostic": False, "cached_reference_logprobs": False, "model_loaded": True,
            "final_drain": drain(mode != "cold"), "engine_shutdown": "completed"}


def job_fixture(mode):
    number = {"cold": 1, "populate": 2, "paired": 3}[mode]
    report, config = make_report(mode), make_config(mode)
    requests = []
    rows = iter(report["rows"])
    for ordinal in range({"cold": 3, "populate": 9, "paired": 6}[mode]):
        sentinel = mode == "populate" and ordinal % 3 == 1
        if sentinel:
            rep = ordinal // 3
            request = {"ordinal": ordinal, "prompt_token_ids": [30000 + rep],
                       "output_token_ids": [99], "num_cached_tokens": 0, "sentinel": True, "request_id": str(ordinal)}
        else:
            row = next(rows)
            request = {"ordinal": ordinal, "prompt_token_ids": row["prompt_token_ids"],
                       "output_token_ids": row["output_token_ids"], "num_cached_tokens": row["num_cached_tokens"],
                       "sentinel": False, "request_id": row["request_id"]}
        requests.append(request)
    runtime = {"mode": mode, "source_lock_sha256": SOURCE_SHA, "scope_sha256": SCOPE_SHA,
               "process_identity": {"pid": number * 100, "sid": number * 100}, "frozen_config": config,
               "sentinel_observations": [dict(rep=r["prompt_token_ids"][0] - 30000, **r)
                                         for r in requests if r["sentinel"]], "requests": requests,
               "original_exit_code": 0, "observer_fault_types": [],
               "original_shutdown_completed": True, "engine_shutdown_calls": 1,
               "handler_shutdown_calls": 0 if mode == "cold" else 1,
               "source_unchanged": True, "common_helpers_restored": True,
               "native_post_shutdown": None, "GPU_qualified": False, "effect_verified": False}
    command = [".venv/bin/python", "-B", "run_g3_calibration_pilot.py", "--mode", mode, "--execute"]
    binding = {"expected_source_lock_sha256": SOURCE_SHA, "expected_scope_sha256": SCOPE_SHA,
               "expected_gpu_uuid": GPU, "expected_label": "cpu-fixture-" + mode,
               "expected_command": command, "expected_storage": STORAGE,
               "prior_process_identities": [] if mode == "cold" else [{"pid": 100, "sid": 100}]}
    guard = {"exit": 0, "child_exit": 0, "gpu_job_attempted": True, "timed_out": False,
             "session_drained": True, "interrupted_signal": None, "error": None,
             "session_members_before_cleanup": [], "session_members_after_cleanup": [],
             "label": binding["expected_label"], "command": command, "gpu_uuid": GPU,
             "session_id": number * 100, "elapsed_seconds": 80.0}
    return report, runtime, guard, binding


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.report, self.runtime, self.guard, self.binding = job_fixture("paired")

    def validate(self, mode="paired"):
        return R.validate_job_result(mode, self.report, self.runtime, self.guard, self.binding)

    def rejects(self, reason):
        with self.assertRaisesRegex(ValueError, reason):
            self.validate()

    def test_three_finite_job_shapes_and_18_original_requests(self):
        checks, reports, configs = {}, {}, {}
        for mode in ("cold", "populate", "paired"):
            report, runtime, guard, binding = job_fixture(mode)
            checks[mode] = R.validate_job_result(mode, report, runtime, guard, binding)
            self.assertEqual(checks[mode]["status"], "PASS_BOUNDED_ORIGINAL_ACQUISITION_PATH_CHECKS")
            reports[mode], configs[mode] = report, runtime["frozen_config"]
        result = R.verify_point_pilot(reports["cold"], reports["populate"], reports["paired"],
                                      configs=configs, job_checks=checks)
        self.assertEqual(result["request_count"], 18)
        self.assertEqual(result["reported_rows"], 15)
        for values in result["measured_samples_seconds"].values():
            self.assertEqual(len(values), 2)
        for key in R.UNQUALIFIED:
            self.assertIs(result[key], False)

    def test_metric_ttft_preserved_seconds_excludes_host_wall_and_warmup(self):
        result = self.validate()
        self.assertEqual(result["measured_samples_seconds"]["g_ssd"], [0.21000000000000002, 0.22])
        self.assertNotIn(99.0, result["measured_samples_seconds"]["g_ssd"])
        self.assertEqual(result["warmup_reps"], [0])
        self.assertIn("wall-clock", result["timing"])

    def test_original_closed_false_drain_does_not_grant_release(self):
        self.assertFalse(self.report["final_drain"]["handlers"][0]["aio"]["closed"])
        result = self.validate()
        self.assertFalse(result["physical_release_verified"])
        self.assertIn("UNKNOWN", result["native_owner_physical_evidence"])

    def test_cold_hit_external_connector_and_wrong_domain_are_rejected(self):
        report, runtime, guard, binding = job_fixture("cold")
        report["rows"][0]["num_cached_tokens"] = 16
        with self.assertRaisesRegex(ValueError, "zero"):
            R.validate_job_result("cold", report, runtime, guard, binding)
        config = make_config("paired")
        for field, value in (("acquisition_domain", 4096), ("sizes", [64]), ("reps", 6)):
            broken = copy.deepcopy(config); broken[field] = value
            with self.assertRaisesRegex(ValueError, "fixed domain"):
                R.validate_frozen_config("paired", broken)

    def test_wrong_depth_staging_planner_or_raw_curve_rejected(self):
        for key, value in (("iodepth", 8), ("staging_mem", 0.5), ("load_planner", "on"),
                           ("prefix_cache_break_even_path", "/old/curve.json"), ("preload_share_staging", False)):
            with self.subTest(key=key):
                config = make_config("paired")
                config["engine"]["kv_transfer_config"]["kv_connector_extra_config"][key] = value
                with self.assertRaises(ValueError):
                    R.validate_frozen_config("paired", config)

    def test_output_count_prompt_axis_and_warmup_classification_rejected(self):
        for changes in ({"output_token_ids": [1, 2]}, {"prompt_token_ids": [1] * 129}, {"warmup": False}):
            report = copy.deepcopy(self.report); report["rows"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                R.verify_acquisition_report("paired", report, self.runtime["frozen_config"])

    def test_row_missing_duplicate_or_out_of_order_rejected(self):
        for rows in (self.report["rows"][:-1], self.report["rows"] + self.report["rows"][:1],
                     list(reversed(self.report["rows"]))):
            report = copy.deepcopy(self.report); report["rows"] = rows
            with self.assertRaises(ValueError):
                R.verify_acquisition_report("paired", report, self.runtime["frozen_config"])

    def test_invalid_original_metric_corruption_nonfinite_bool_and_clock_order(self):
        for key, value in (("is_corrupted", True), ("first_token_latency", float("nan")),
                           ("first_token_latency", True), ("first_token_latency", -1.0), ("scheduled_ts", 5000.0)):
            report = copy.deepcopy(self.report); report["rows"][0]["metrics"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.verify_acquisition_report("paired", report, self.runtime["frozen_config"])

    def test_foreground_plus_preload_exact_bytes_and_fresh_cache_source(self):
        trace = self.report["rows"][0]["trace"]
        trace["foreground_logical_read_bytes"] = R.FILE_BYTES
        trace["preload_actual_read_bytes"] -= R.FILE_BYTES
        self.validate()
        trace["foreground_logical_read_bytes"] += 1
        self.rejects("SSD foreground")

    def test_actual_h2d_and_transfer_success_required(self):
        for key, value in (("h2d_events", 0), ("transfer_success", False), ("load_transfers", [])):
            trace = self.report["rows"][0]["trace"]
            old = trace[key]; trace[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            trace[key] = old

    def test_gmem_must_use_cache_h2d_and_zero_ssd(self):
        trace = self.report["rows"][1]["trace"]
        trace["preload_actual_read_bytes"] = 1
        self.rejects("zero SSD")

    def test_complete_load_bytes_and_source_sums_cannot_be_spoofed(self):
        transfer = self.report["rows"][0]["trace"]["load_transfers"][0]
        transfer["num_bytes"] -= 1
        self.rejects("byte coverage")
        transfer["num_bytes"] += 1
        transfer["src_preload"] -= 1
        self.rejects("source coverage")

    def test_paired_outputs_match_including_warmup(self):
        self.report["rows"][1]["output_token_ids"] = [999]
        self.rejects("cached pair output")

    def test_populate_cold_store_vs_cached_output_is_not_invented_reference_gate(self):
        report, runtime, guard, binding = job_fixture("populate")
        report["rows"][0]["output_token_ids"] = [999]
        runtime["requests"][0]["output_token_ids"] = [999]
        result = R.validate_job_result("populate", report, runtime, guard, binding)
        self.assertFalse(result["P4_performance_verified"])

    def test_original_aio_settled_not_future_done_substitute(self):
        aio = self.report["final_drain"]["handlers"][0]["aio"]
        aio["pending"] = 1
        self.rejects("unsettled")
        aio["pending"] = 0
        del aio["accepted"]
        aio["Future.done"] = True
        self.rejects("AIO accepted")

    def test_missing_handler_pinning_and_layout_rejected(self):
        for key, value in (("pinned", False), ("io_size", 4096), ("staging_budget", 1073741824)):
            handler = self.report["rows"][0]["drain"]["handlers"][0]
            old = handler[key]; handler[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            handler[key] = old

    def test_original_shutdown_once_and_common_restore_required(self):
        for key, value in (("original_shutdown_completed", False), ("engine_shutdown_calls", 2),
                           ("handler_shutdown_calls", 0), ("source_unchanged", False), ("common_helpers_restored", False)):
            old = self.runtime[key]; self.runtime[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.runtime[key] = old

    def test_os_guard_exit_timeout_and_members_required(self):
        for key, value in (("exit", 1), ("child_exit", 1), ("timed_out", True), ("session_drained", False),
                           ("session_members_before_cleanup", [123]), ("session_members_after_cleanup", [123])):
            old = self.guard[key]; self.guard[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.guard[key] = old

    def test_scope_source_command_label_or_session_drift_rejected(self):
        for key in ("scope_sha256", "source_lock_sha256"):
            old = self.runtime[key]; self.runtime[key] = "3" * 64
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.runtime[key] = old
        self.guard["command"] = ["arbitrary"]
        self.rejects("argv")

    def test_paired_process_must_be_fresh_and_missing_guard_is_unknown(self):
        self.binding["prior_process_identities"] = [{"pid": 300, "sid": 300}]
        self.rejects("fresh")
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            R.validate_job_result("paired", self.report, self.runtime, None, self.binding)

    def test_original_sentinels_and_total_requests_required(self):
        report, runtime, guard, binding = job_fixture("populate")
        runtime["sentinel_observations"] = []
        with self.assertRaisesRegex(ValueError, "sentinel observations UNKNOWN"):
            R.validate_job_result("populate", report, runtime, guard, binding)
        report, runtime, guard, binding = job_fixture("populate")
        runtime["sentinel_observations"][0]["ordinal"] = 2
        with self.assertRaisesRegex(ValueError, "sentinel.*sequence"):
            R.validate_job_result("populate", report, runtime, guard, binding)
        runtime["sentinel_observations"][0]["ordinal"] = 1
        runtime["requests"] = runtime["requests"][:-1]
        with self.assertRaisesRegex(ValueError, "exact actual"):
            R.validate_job_result("populate", report, runtime, guard, binding)

    def test_runtime_sentinel_requests_lack_rep_and_side_receipt_adds_rep_only(self):
        report, runtime, guard, binding = job_fixture("populate")
        self.assertNotIn("rep", runtime["requests"][1])
        self.assertEqual(runtime["sentinel_observations"][0]["rep"], 0)
        R.validate_job_result("populate", report, runtime, guard, binding)
        runtime["sentinel_observations"][0]["output_token_ids"] = [998]
        with self.assertRaisesRegex(ValueError, "same observed"):
            R.validate_job_result("populate", report, runtime, guard, binding)

    def test_plain_fixture_cannot_grant_gpu_production_clock_or_effect(self):
        self.runtime.update(GPU_qualified=True, origin="native", physical_release_verified=True)
        result = self.validate()
        for key in R.UNQUALIFIED:
            self.assertIs(result[key], False)
        class Poison:
            reads = 0
            def __getattribute__(self, name):
                type(self).reads += 1
                raise AssertionError("owner read")
        with self.assertRaises(ValueError):
            R.validate_job_result("paired", self.report, Poison(), self.guard, self.binding)
        self.assertEqual(Poison.reads, 0)

    def test_original_prompt_scalar_AST_exact_source_only(self):
        local = HERE.parent.parent / "prefix_io_v1_server08_primary_qualification/contents"
        root = Path(options.source_root) if options.source_root else local
        path = root / "experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py"
        raw = path.read_bytes()
        self.assertEqual(len(raw), 18995)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), "68b2c045bcc9a7de84771d556b0e01180a592280002a2d4360d1c7500c9c856e")
        source = ast.parse(raw)
        prompt = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "prompt")
        scope = {"__builtins__": {"range": range}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prompt], type_ignores=[])), str(path), "exec"), scope)
        for rep in range(3):
            self.assertEqual(scope["prompt"](128, rep), R.expected_prompt(rep))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="g3-publication-cpu-")
        cls.storage = Path(cls.temp.name).resolve() / "storage"
        cls.base = cls.storage / R.STORAGE_PREFIX
        cls.paths = []
        for index in range(24):
            name = hashlib.sha256(str(index).encode()).hexdigest()
            path = cls.base / name[:3] / name[3:5] / (name + ".bin")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bytes((index,)) * R.FILE_BYTES)
            cls.paths.append(path)
        cls.publication = R.build_storage_publication(cls.storage)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_actual_cpu_file_inventory_rehashes_and_retains_unqualified_flags(self):
        verified = R.verify_storage_publication(self.storage, self.publication)
        self.assertEqual(verified["file_count"], 24)
        self.assertEqual(verified["total_bytes"], 24 * 917504)
        for key in R.UNQUALIFIED:
            self.assertIs(verified[key], False)

    def test_same_size_payload_drift_rejected_and_no_manifest_rewrite(self):
        target = self.paths[0]
        old = target.read_bytes()
        target.write_bytes(b"X" + old[1:])
        original_manifest = copy.deepcopy(self.publication)
        try:
            with self.assertRaisesRegex(ValueError, "content changed"):
                R.verify_storage_publication(self.storage, self.publication)
            self.assertEqual(original_manifest, self.publication)
        finally:
            target.write_bytes(old)

    def test_extra_file_and_wrong_size_rejected(self):
        extra = self.storage / "foreign.txt"
        extra.write_bytes(b"foreign")
        try:
            with self.assertRaises(ValueError):
                R.build_storage_publication(self.storage)
        finally:
            extra.unlink()
        target = self.paths[1]
        old = target.read_bytes(); target.write_bytes(old[:-1])
        try:
            with self.assertRaisesRegex(ValueError, "file size"):
                R.build_storage_publication(self.storage)
        finally:
            target.write_bytes(old)

    def test_missing_file_and_wrong_hash_shards_rejected(self):
        target = self.paths[2]
        old = target.read_bytes(); target.unlink()
        try:
            with self.assertRaisesRegex(ValueError, "exact24"):
                R.build_storage_publication(self.storage)
        finally:
            target.write_bytes(old)
        wrong = target.with_name("0" * 64 + ".bin")
        target.rename(wrong)
        try:
            with self.assertRaisesRegex(ValueError, "hash shards"):
                R.build_storage_publication(self.storage)
        finally:
            wrong.rename(target)

    def test_tampered_manifest_count_hash_order_target_and_qualification_rejected(self):
        for update in ({"file_count": 23}, {"rows_sha256": "0" * 64}, {"GPU_qualified": True},
                       {"storage": "/another-storage"}, {"files": list(reversed(self.publication["files"]))}):
            publication = copy.deepcopy(self.publication); publication.update(update)
            with self.subTest(update=list(update)), self.assertRaises(ValueError):
                R.verify_storage_publication(self.storage, publication)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
