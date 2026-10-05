"""Stdlib AST replay of pinned original metric methods; no model/GPU fixtures."""

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


HERE = Path(__file__).resolve().parent
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--source-audit-root", type=Path, default=HERE.parent / "source-audit")
ARGS, REST = PARSER.parse_known_args()
SOURCE = ARGS.source_audit_root.resolve() / "vllm_offloading_metrics.py"
SPEC = importlib.util.spec_from_file_location("cpu_stats_dict_compat", HERE / "compat_offloading_stats_dict.py")
COMPAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPAT)


@dataclasses.dataclass
class BaseStats:
    # Stub only the original imported metric container. No backend/executor.
    data: dict = dataclasses.field(default_factory=dict)


class BaseProm:
    pass


class ScalarMetric:
    def __init__(self):
        self.observed = []
        self.incremented = []

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value):
        self.incremented.append(value)


class ProducerFault(Exception):
    pass


class RaisesDuringOriginalUnpack:
    def __init__(self, error):
        self.error = error

    def __iter__(self):
        raise self.error


def original_fixture():
    raw = SOURCE.read_bytes()
    assert len(raw) == COMPAT.SOURCE_BYTES
    assert hashlib.sha256(raw).hexdigest() == COMPAT.SOURCE_SHA256
    module = types.ModuleType(COMPAT.MODULE_NAME)
    module.__file__ = str(SOURCE)
    namespace = vars(module)
    namespace.update(dataclass=dataclasses.dataclass, KVConnectorStats=BaseStats,
                     KVConnectorPromMetrics=BaseProm, Any=object, TransferType=tuple,
                     VllmConfig=object, PromMetric=object, PromMetricT=object)
    assert COMPAT.MODULE_NAME not in sys.modules
    sys.modules[COMPAT.MODULE_NAME] = module
    classes = [node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
               and node.name in ("OffloadingOperationMetrics", "OffloadingConnectorStats", "OffloadPromMetrics")]
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec", dont_inherit=True), namespace)
    return module


def original_prom(module, directions):
    prom = object.__new__(module.OffloadPromMetrics)
    prom.histogram_transfer_size = {}
    prom.counter_kv_bytes = {}
    prom.counter_kv_transfer_time = {}
    metrics = {}
    for direction in directions:
        key = (0, direction)
        histogram, byte_counter, time_counter = ScalarMetric(), ScalarMetric(), ScalarMetric()
        prom.histogram_transfer_size[key] = histogram
        prom.counter_kv_bytes[key] = byte_counter
        prom.counter_kv_transfer_time[key] = time_counter
        metrics[direction] = (histogram, byte_counter, time_counter)
    return prom, metrics


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.module = original_fixture()
        self.stats_cls = self.module.OffloadingConnectorStats
        self.original = self.stats_cls.record_transfer
        self.other_methods = {name: getattr(self.stats_cls, name)
                              for name in ("reset", "aggregate", "reduce", "is_empty")}
        self.prom_method = self.module.OffloadPromMetrics.observe
        self.handles = []

    def tearDown(self):
        for handle in self.handles:
            if handle._active:
                if self.stats_cls.record_transfer is not handle._wrapper:
                    # Fixture-only cleanup after the intentional foreign override.
                    self.stats_cls.record_transfer = handle._wrapper
                handle.detach()
        sys.modules.pop(COMPAT.MODULE_NAME, None)

    def install(self):
        handle = COMPAT.install_stats_dict_compat(self.module, SOURCE)
        self.handles.append(handle)
        return handle

    def calls(self, action):
        count = 0
        original_code = self.original.__code__
        def profile(frame, event, arg):
            nonlocal count
            if event == "call" and frame.f_code is original_code:
                count += 1
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            action()
        finally:
            sys.setprofile(previous)
        return count

    def test_original_failure_reproduces_both_consumers(self):
        stats = self.stats_cls()
        stats.record_transfer(917504, 0.125, ("GPU", "CPU"))
        self.assertIs(type(stats.data["GPU_to_CPU"][0]), self.module.OffloadingOperationMetrics)
        with self.assertRaises(AssertionError):
            stats.reduce()
        prom, _ = original_prom(self.module, ["GPU_to_CPU"])
        with self.assertRaises(AssertionError):
            prom.observe(stats.data)

    def test_once_producer_and_both_original_consumers_preserve_values(self):
        handle = self.install()
        stats = self.stats_cls()
        returns = []
        self.assertEqual(self.calls(lambda: returns.append(stats.record_transfer(917504, 0.125, ("GPU", "CPU")))), 1)
        self.assertEqual(returns, [None])
        self.assertEqual(stats.data, {"GPU_to_CPU": [{"op_size": 917504, "op_time": 0.125}]})
        self.assertEqual(stats.reduce(), {"GPU_to_CPU_total_bytes": 917504, "GPU_to_CPU_total_time": 0.125})
        prom, metrics = original_prom(self.module, ["GPU_to_CPU"])
        prom.observe(stats.data)
        histogram, byte_counter, time_counter = metrics["GPU_to_CPU"]
        self.assertEqual(histogram.observed, [917504])
        self.assertEqual(byte_counter.incremented, [917504])
        self.assertEqual(time_counter.incremented, [0.125])
        for name, original in self.other_methods.items():
            self.assertIs(getattr(self.stats_cls, name), original)
        self.assertIs(self.module.OffloadPromMetrics.observe, self.prom_method)
        self.assertFalse(handle.witness()["production_qualified"])
        self.assertFalse(handle.witness()["GPU_qualified"])

    def test_multiple_transfers_aggregate_and_reset_original_methods(self):
        self.install()
        left, right = self.stats_cls(), self.stats_cls()
        left.record_transfer(10, 0.125, ("GPU", "CPU"))
        left.record_transfer(20, 0.25, ("GPU", "CPU"))
        right.record_transfer(30, 0.5, ("CPU", "GPU"))
        right.record_transfer(40, 0.125, ("GPU", "CPU"))
        self.assertIs(left.aggregate(right), left)
        self.assertEqual(left.reduce(), {"GPU_to_CPU_total_bytes": 70, "GPU_to_CPU_total_time": 0.5,
                                        "CPU_to_GPU_total_bytes": 30, "CPU_to_GPU_total_time": 0.5})
        prom, metrics = original_prom(self.module, ["GPU_to_CPU", "CPU_to_GPU"])
        prom.observe(left.data)
        self.assertEqual(metrics["GPU_to_CPU"][1].incremented, [10, 20, 40])
        self.assertEqual(metrics["CPU_to_GPU"][2].incremented, [0.5])
        left.reset()
        self.assertTrue(left.is_empty())
        self.assertEqual(left.reduce(), {})

    def test_existing_serialized_dict_objects_and_lists_preserved(self):
        self.install()
        existing = {"op_size": 9, "op_time": 0.125}
        other = {"op_size": 11, "op_time": 0.25}
        original_list = [existing]
        other_list = [other]
        stats = self.stats_cls(data={"GPU_to_CPU": original_list, "CPU_to_GPU": other_list})
        stats.record_transfer(17, 0.5, ("GPU", "CPU"))
        self.assertIs(stats.data["GPU_to_CPU"], original_list)
        self.assertIs(stats.data["GPU_to_CPU"][0], existing)
        self.assertIs(stats.data["CPU_to_GPU"], other_list)
        self.assertIs(stats.data["CPU_to_GPU"][0], other)
        self.assertEqual(stats.reduce()["GPU_to_CPU_total_bytes"], 26)

    def test_clean_detach_restores_exact_original_and_can_reinstall(self):
        handle = self.install()
        self.assertFalse(handle.detach()["installed"])
        self.assertIs(self.stats_cls.record_transfer, self.original)
        self.assertFalse(handle.detach()["installed"])
        second = self.install()
        second.detach()
        self.assertIs(self.stats_cls.record_transfer, self.original)
        stats = self.stats_cls()
        stats.record_transfer(1, 0.25, ("GPU", "CPU"))
        with self.assertRaises(AssertionError):
            stats.reduce()

    def test_duplicate_install_refuses_without_replacing_first_wrapper(self):
        handle = self.install()
        with self.assertRaises(COMPAT.StatsCompatibilityError):
            COMPAT.install_stats_dict_compat(self.module, SOURCE)
        self.assertIs(self.stats_cls.record_transfer, handle._wrapper)

    def test_source_drift_rejected_before_method_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            drift = Path(temporary) / "metrics.py"
            original_bytes = SOURCE.read_bytes()
            drift.write_bytes(original_bytes[:-1] + bytes([original_bytes[-1] ^ 1]))
            with self.assertRaises(COMPAT.StatsCompatibilityError):
                COMPAT.install_stats_dict_compat(self.module, drift)
        self.assertIs(self.stats_cls.record_transfer, self.original)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), COMPAT.SOURCE_SHA256)

    def test_method_or_globals_drift_rejected(self):
        def foreign(instance, num_bytes, time, transfer_type):
            return None
        self.stats_cls.record_transfer = foreign
        with self.assertRaises(COMPAT.StatsCompatibilityError):
            COMPAT.install_stats_dict_compat(self.module, SOURCE)
        self.assertIs(self.stats_cls.record_transfer, foreign)
        self.stats_cls.record_transfer = self.original
        old_name = self.module.__name__
        self.module.__name__ = "unbound_metrics"
        with self.assertRaises(COMPAT.StatsCompatibilityError):
            COMPAT.install_stats_dict_compat(self.module, SOURCE)
        self.module.__name__ = old_name

    def test_unexpected_schema_rejected_after_original_once_without_dropping_record(self):
        self.install()
        stats = self.stats_cls(data={"GPU_to_CPU": [{"op_size": 7}]})
        def action():
            with self.assertRaises(COMPAT.StatsCompatibilityError):
                stats.record_transfer(8, 0.125, ("GPU", "CPU"))
        self.assertEqual(self.calls(action), 1)
        self.assertEqual(stats.data["GPU_to_CPU"][0], {"op_size": 7})
        self.assertIs(type(stats.data["GPU_to_CPU"][1]), self.module.OffloadingOperationMetrics)

    def test_original_exception_object_and_once_call_preserved(self):
        self.install()
        stats = self.stats_cls()
        error = ProducerFault("original unpack failure")
        def action():
            with self.assertRaises(ProducerFault) as caught:
                stats.record_transfer(8, 0.125, RaisesDuringOriginalUnpack(error))
            self.assertIs(caught.exception, error)
        self.assertEqual(self.calls(action), 1)
        self.assertEqual(stats.data, {})

    def test_non_scalar_original_record_rejected_without_fabrication(self):
        self.install()
        stats = self.stats_cls()
        with self.assertRaises(COMPAT.StatsCompatibilityError):
            stats.record_transfer(8, float("nan"), ("GPU", "CPU"))
        op = stats.data["GPU_to_CPU"][0]
        self.assertIs(type(op), self.module.OffloadingOperationMetrics)
        self.assertNotEqual(op.op_time, op.op_time)

    def test_foreign_override_preserved_and_detach_refuses(self):
        handle = self.install()
        def foreign(instance, num_bytes, time, transfer_type):
            return "foreign"
        self.stats_cls.record_transfer = foreign
        with self.assertRaises(COMPAT.StatsCompatibilityError):
            handle.detach()
        self.assertIs(self.stats_cls.record_transfer, foreign)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *REST], verbosity=2)
