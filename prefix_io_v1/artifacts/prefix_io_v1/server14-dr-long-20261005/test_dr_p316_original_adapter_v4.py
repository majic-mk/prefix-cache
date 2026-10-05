"""Targeted CPU checks for the two-arm adapter contract; no model imports."""
import ast
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("_cpu_dr_p316_adapter", HERE / "dr_p316_original_adapter_v4.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
CONTROL_CANDIDATES = ((Path(os.environ["DR_CPU_ADAPTER_CONTROL_ROOT"]),)
    if os.environ.get("DR_CPU_ADAPTER_CONTROL_ROOT") else ()) + (HERE / "runtime_source_v4/control", HERE / "runtime_source_v3/control", HERE / "runtime_source_v2/control", HERE / "runtime_source/control",
    HERE / "upload_source/runtime_source/control",
    HERE.parent.parent / "prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src")
CONTROL_SOURCE = next(path for path in CONTROL_CANDIDATES if (path / "prefix_io_control/p4_options.py").is_file())
sys.path.insert(0, str(CONTROL_SOURCE))
from prefix_io_control.p4_options import parse_p4_options


def parent():
    return dict(schema_version=1, run_id="actual-run", max_accepted_parents=64)


def fixed():
    # Unit-input fixture for the real startup parser; never used for a GPU run.
    amounts = {name:dict(ops=8, bytes=8*917504) for name in ("ssd_read", "ssd_write", "h2d", "d2h")}
    return dict(schema_version=1, run_id="actual-run", mode="fixed", epoch_ns=10000000,
        byte_quantum=917504, cumulative=deepcopy(amounts), inflight=deepcopy(amounts),
        shared_ssd_cumulative=dict(ops=16, bytes=16*917504),
        shared_ssd_inflight=dict(ops=16, bytes=16*917504),
        shared_copy_cumulative_bytes=16*917504, shared_copy_inflight_bytes=16*917504,
        reserve_staging_bytes=0, max_accepted_parents=64,
        sample_max_age_ns=20000000, max_wait_ns=200000000)


def method_controls():
    return dict(prefix_io_parent_admission=parent(),
        prefix_io_p4_policy=dict(schema_version=1, run_id="actual-run",
            mode="dependency_only", sample_max_age_ns=20000000, max_wait_ns=200000000,
            internal_step_budget_ns=None, candidate_batches=[1,2,4,8],
            fixed_stage_policy=fixed(), cost_table=None))


def original_stats_functions():
    # Compile the actual author's pure producer/reducer function bodies and its
    # exact operation dataclass. No model, cache or GPU modules are mocked.
    relative = "third_party/work/vllm-author-p4-02-cpu/vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py"
    candidates = [HERE / "original_scripts/author_offloading_metrics.py"]
    candidates.extend(ancestor / relative for ancestor in HERE.parents)
    path = next(candidate for candidate in candidates if candidate.is_file())
    tree = ast.parse(path.read_text(encoding="utf-8"))
    operation = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OffloadingOperationMetrics")
    stats_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OffloadingConnectorStats")
    functions = [node for node in stats_class.body if isinstance(node, ast.FunctionDef) and node.name in ("record_transfer", "reduce")]
    namespace = dict(dataclass=dataclass, TransferType=tuple)
    exec(compile(ast.Module(body=[operation] + functions, type_ignores=[]), str(path), "exec"), namespace)
    class OriginalStats:
        def __init__(self):
            self.data = {}
    OriginalStats.record_transfer = namespace["record_transfer"]
    OriginalStats.reduce = namespace["reduce"]
    return OriginalStats, namespace["OffloadingOperationMetrics"]


def engine():
    return dict(dtype="bfloat16", kv_cache_memory_bytes=2147483648,
        max_num_seqs=2, max_model_len=16400, max_num_batched_tokens=16400,
        enable_prefix_caching=True, prefix_caching_hash_algo="sha256",
        kv_transfer_config=dict(kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config=dict(iodepth=8, load_planner="on", staging_mem=1,
                enable_preload=True, preload_share_staging=True,
                shared_storage_path="/unchanged/private/storage",
                prefix_cache_break_even_path="/unchanged/original/curve.json")))


class AdapterContract(unittest.TestCase):
    def test_U_keeps_original_config_and_disables_research(self):
        before = engine()
        controls = dict(prefix_io_parent_admission=parent(), prefix_io_p4_policy=dict(mode="off"))
        actual = adapter.control_engine(before, controls, "U")
        self.assertEqual(before, engine())
        extra = actual["kv_transfer_config"]["kv_connector_extra_config"]
        for key, value in before["kv_transfer_config"]["kv_connector_extra_config"].items():
            self.assertEqual(extra[key], value)
        self.assertEqual({k:v for k,v in actual.items() if k != "kv_transfer_config"},
                         {k:v for k,v in before.items() if k != "kv_transfer_config"})
        with self.assertRaises(ValueError):
            adapter.control_engine(before, dict(prefix_io_stage_policy=dict(mode="fixed")), "U")

    def test_method_is_truthfully_fixed_plus_restore_only(self):
        controls = method_controls()
        method = adapter.control_engine(engine(), controls, "F8+D_R")
        self.assertEqual(method["kv_transfer_config"]["kv_connector_extra_config"]["prefix_io_p4_policy"],
                         controls["prefix_io_p4_policy"])
        self.assertNotIn("prefix_io_stage_policy", method["kv_transfer_config"]["kv_connector_extra_config"])
        self.assertEqual(adapter.control_engine(engine(), controls, "CAL"), method)
        for changed in (dict(prefix_io_stage_policy=dict(mode="off"), prefix_io_p4_policy=dict(mode="dependency_only")),
                        dict(prefix_io_stage_policy=dict(mode="fixed"), prefix_io_p4_policy=dict(mode="budget"))):
            with self.assertRaises(ValueError):
                adapter.control_engine(engine(), changed, "F8+D_R")
        # Exercise the real unchanged native startup parser. This detects the
        # actual mutual-exclusion rule that the old mirror-only tests missed.
        parsed = parse_p4_options(method["kv_transfer_config"]["kv_connector_extra_config"])
        self.assertEqual(parsed.config.mode, "dependency_only")
        self.assertEqual(parsed.fixed.policy.mode, "fixed")
        self.assertEqual(parsed.fixed.parent_cap, 64)
        self.assertEqual([amount.ops for amount in parsed.fixed.policy.cumulative], [8]*4)
        actual_U = adapter.control_engine(engine(), dict(prefix_io_parent_admission=parent(),
            prefix_io_p4_policy=dict(mode="off")), "U")
        Uparsed = parse_p4_options(actual_U["kv_transfer_config"]["kv_connector_extra_config"])
        self.assertIsNone(Uparsed.config)
        self.assertIsNone(Uparsed.fixed)
        for bad_extra in (dict(controls, prefix_io_stage_policy=fixed()),
                          dict(prefix_io_p4_policy=dict(mode="off"), prefix_io_stage_policy=dict(mode="off"))):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                parse_p4_options(bad_extra)
            with self.assertRaises(ValueError):
                adapter.control_engine(engine(), bad_extra, "F8+D_R")

    def test_no_executor_capacity_or_admission_edits(self):
        for key, value in (("max_num_seqs", 1), ("max_model_len", 16384),
                           ("kv_cache_memory_bytes", 1), ("max_num_batched_tokens", 8192)):
            bad = engine()
            bad[key] = value
            with self.assertRaises(ValueError):
                adapter.control_engine(bad, {}, "U")
        for key, value in (("load_planner", "off"), ("iodepth", 4), ("staging_mem", .5),
                           ("enable_preload", False), ("preload_share_staging", False)):
            bad = engine()
            bad["kv_transfer_config"]["kv_connector_extra_config"][key] = value
            with self.assertRaises(ValueError):
                adapter.control_engine(bad, {}, "U")
        with self.assertRaises(ValueError):
            adapter.control_engine(engine(), dict(force_loading=True), "U")

    def test_explicit_device_migration_keeps_historical_validator(self):
        manifest = dict(gpu_uuid="old")
        candidate = dict(provenance=dict(gpu_uuid="old"))
        migration = dict(historical_gpu_uuid="old", current_gpu_uuid="current",
                         scope="P316_CURRENT_DEVICE_EXPLORATORY_REPLAY")
        calls = []
        def original(m, c, uuid):
            calls.append((deepcopy(m), deepcopy(c), uuid))
            return dict(original_contract=True)
        value = adapter.validate_original_identity(original, manifest, candidate, "current", "old", migration)
        self.assertEqual(value, dict(original_contract=True))
        self.assertEqual(calls, [(manifest, candidate, "old")])
        with self.assertRaises(ValueError):
            adapter.validate_original_identity(original, manifest, candidate, "other", "old", migration)
        self.assertEqual(manifest["gpu_uuid"], "old")

    def test_original_request_loop_and_token_contract_remain_in_original_module(self):
        source = (HERE / "dr_p316_original_adapter_v4.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertNotIn("engine.step(", source)
        self.assertNotIn("engine.add_request(", source)
        self.assertIn("code = main_module.main()", source)
        self.assertIn("snapshot = reactor.inspect_snapshot(timeout=2)", source)
        self.assertNotIn("_prefix_p4_bridge.snapshot(", source)
        self.assertIn("value[\"probe\"].update(deepcopy(actual_metadata))", source)
        top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        imports = {name.name.split(".")[0] for node in top_imports if isinstance(node, ast.Import) for name in node.names}
        self.assertFalse(imports & {"torch", "vllm", "py_kvcache"})

    def test_actual_submit_receipt_records_only_successful_prefix(self):
        calls = []
        kernel = SimpleNamespace(submit=lambda batch: calls.append(len(batch)) or 1)
        old = kernel.submit
        tap = adapter.KernelSubmissionTap(limit=2)
        tap.install(kernel)
        batch = [SimpleNamespace(user_data=n, fd=7, nbytes=917504, kind="read") for n in (21, 22)]
        self.assertEqual(kernel.submit(batch), 1)
        self.assertEqual(calls, [2])
        receipt = tap.snapshot()
        self.assertTrue(receipt["valid"])
        self.assertEqual([r["user_data"] for r in receipt["records"]], [21])
        self.assertFalse(tap.restore_after_shutdown(joined=False))
        self.assertTrue(tap.restore_after_shutdown(joined=True))
        self.assertIs(kernel.submit, old)

    def test_submit_exception_zero_and_overflow_never_change_native_result(self):
        marker = OSError("original kernel failure")
        def failed(batch):
            raise marker
        kernel = SimpleNamespace(submit=failed)
        tap = adapter.KernelSubmissionTap(limit=1)
        tap.install(kernel)
        with self.assertRaises(OSError) as caught:
            kernel.submit([])
        self.assertIs(caught.exception, marker)
        self.assertEqual(tap.snapshot()["successful_operation_count"], 0)
        self.assertTrue(tap.restore_after_shutdown(joined=True))
        kernel = SimpleNamespace(submit=lambda batch: len(batch))
        tap = adapter.KernelSubmissionTap(limit=1)
        tap.install(kernel)
        batch = [SimpleNamespace(user_data=n, fd=7, nbytes=917504, kind="write") for n in (1, 2)]
        self.assertEqual(kernel.submit([]), 0)
        self.assertEqual(kernel.submit(batch), 2)
        self.assertTrue(tap.snapshot()["overflow"])
        self.assertFalse(tap.snapshot()["valid"])

    def test_sdk_owned_details_replays_only_the_one_original_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            details = Path(folder) / "actual-run" / "details"
            checks = []
            replay = adapter.DetailsPathReplay(details, run_id="actual-run", session_id=123,
                revalidate_guard=lambda: checks.append("same-session-unit-check"))
            self.assertTrue(details.is_dir())
            (details / "runtime-cache").mkdir()
            original_path = replay.path_type()
            out = original_path(details).resolve()
            self.assertIsInstance(out, original_path)
            out.mkdir(parents=True, exist_ok=False)
            self.assertTrue(replay.consumed)
            self.assertEqual(len(checks), 2)
            with self.assertRaises(FileExistsError):
                out.mkdir(parents=True, exist_ok=False)
            child = original_path(details / "storage")
            child.mkdir(parents=True, exist_ok=False)
            self.assertTrue(child.is_dir())
            with self.assertRaises(FileExistsError):
                child.mkdir(parents=True, exist_ok=False)
            with self.assertRaises(FileExistsError):
                adapter.DetailsPathReplay(details, run_id="actual-run", session_id=123,
                    revalidate_guard=lambda: None)

    def test_owned_creation_refuses_changed_guard_marker_and_existing_result(self):
        for bad in ("marker", "result", "guard"):
            with tempfile.TemporaryDirectory() as folder:
                state = dict(allowed=True)
                def actual_check():
                    if not state["allowed"]:
                        raise ValueError("session no longer owns this unit directory")
                details = Path(folder) / "details"
                replay = adapter.DetailsPathReplay(details, run_id="actual-run", session_id=123,
                    revalidate_guard=actual_check)
                if bad == "marker":
                    replay.marker.write_text("changed ownership", encoding="utf-8")
                elif bad == "result":
                    (details / "result.json").write_text("old result", encoding="utf-8")
                else:
                    state["allowed"] = False
                with self.assertRaises(ValueError):
                    replay.path_type()(details).mkdir(parents=True, exist_ok=False)
                self.assertFalse(replay.consumed)

    def test_actual_author_stats_producer_and_reducer_keep_measured_values(self):
        stats_type, operation_type = original_stats_functions()
        original = stats_type.record_transfer
        before = stats_type()
        before.record_transfer(917504, .125, ("cpu", "gpu"))
        self.assertIs(type(before.data["cpu_to_gpu"][-1]), operation_type)
        with self.assertRaises(AssertionError):
            before.reduce()
        serialization = adapter.InprocessOffloadStatsSerialization(stats_type, operation_type)
        try:
            owner = stats_type()
            self.assertIsNone(owner.record_transfer(num_bytes=917504, time=.125, transfer_type=("cpu", "gpu")))
            first = owner.data["cpu_to_gpu"][0]
            self.assertEqual(first, dict(op_size=917504, op_time=.125))
            self.assertIsNone(owner.record_transfer(1835008, .25, ("cpu", "gpu")))
            self.assertIs(owner.data["cpu_to_gpu"][0], first)
            self.assertEqual(owner.reduce(), dict(cpu_to_gpu_total_bytes=2752512,
                                                cpu_to_gpu_total_time=.375))
            self.assertEqual(serialization.snapshot()["conversions"], 2)
            self.assertEqual(serialization.snapshot()["original_producer_calls"], 2)
            self.assertFalse(serialization.restore_after_shutdown(ended=False))
        finally:
            self.assertTrue(serialization.restore_after_shutdown(ended=True))
        self.assertIs(stats_type.record_transfer, original)

    def test_stats_adapter_rejects_unexpected_entry_and_preserves_original_exception(self):
        @dataclass
        class Operation:
            op_size: int
            op_time: float
        class Unexpected:
            def __init__(self):
                self.data = {}
            def record_transfer(self, num_bytes, time, transfer_type):
                src, dst = transfer_type
                self.data[src + "_to_" + dst] = [Operation(num_bytes, time), Operation(num_bytes, time)]
        serialization = adapter.InprocessOffloadStatsSerialization(Unexpected, Operation)
        try:
            with self.assertRaises(ValueError):
                Unexpected().record_transfer(1, .1, ("cpu", "gpu"))
            self.assertEqual(serialization.snapshot()["conversions"], 0)
        finally:
            self.assertTrue(serialization.restore_after_shutdown(ended=True))
        marker = OSError("original stats failure")
        class Failed:
            def __init__(self):
                self.data = {}
            def record_transfer(self, num_bytes, time, transfer_type):
                raise marker
        serialization = adapter.InprocessOffloadStatsSerialization(Failed, Operation)
        try:
            with self.assertRaises(OSError) as caught:
                Failed().record_transfer(1, .1, ("cpu", "gpu"))
            self.assertIs(caught.exception, marker)
        finally:
            self.assertTrue(serialization.restore_after_shutdown(ended=True))


if __name__ == "__main__":
    unittest.main()
