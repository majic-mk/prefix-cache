"""Finite resource/parser/source rejection checks, without benchmark work."""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_combined_cpu_resource_preflight", HERE / "resource_preflight.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
STAT = "usage_usec 12\nuser_usec 10\nsystem_usec 2\nnr_periods 8\nnr_throttled 3\nthrottled_usec 7\n"


class ResourceParserTests(unittest.TestCase):
    def decide(self, quota="50000 100000", stat=STAT, affinity=(0, 1), active=False):
        return R.resource_decision(R.parse_quota(quota), R.parse_cpu_stat(stat), affinity, ledger_active=active)

    def test_actual_half_core_fact_cannot_enter_measurement(self):
        result = self.decide()
        self.assertEqual(result["status"], "RESOURCE_LIMITED")
        self.assertFalse(result["resource_ready"])
        self.assertFalse(result["formal_benchmark_started"])
        self.assertEqual(result["measurement_iterations"], 0)

    def test_quota_exactly_one_core_is_readiness_only(self):
        result = self.decide("100000 100000")
        self.assertEqual(result["status"], "CPU_RESOURCE_PREREQUISITE_MET")
        self.assertTrue(result["resource_ready"])
        self.assertFalse(result["cpu_qualified"])
        self.assertFalse(result["old_paired_qualification_changed"])
        self.assertEqual((result["old_total_trials"], result["old_scored_trials"]), (132, 84))
        self.assertEqual(result["old_protocol_sha256"], R.OLD_PROTOCOL_SHA256)
        for key in R.FLAGS: self.assertIs(result[key], False)
        for key in ("valid_native_receipt", "effective_cost_upper_ns", "effective_step_budget_ns", "gpu_uuid"):
            self.assertIsNone(result[key])

    def test_positive_historical_throttle_counters_do_not_become_new_pair_results(self):
        result = self.decide("200000 100000")
        self.assertTrue(result["resource_ready"])
        self.assertFalse(result["cpu_qualified"])
        self.assertEqual(result["performance_comparisons"], 0)

    def test_known_unlimited_quota_is_readiness_only(self):
        result = self.decide("max 100000")
        self.assertEqual(result["status"], "CPU_RESOURCE_PREREQUISITE_MET")
        self.assertTrue(result["resource_ready"])
        self.assertFalse(result["cpu_qualified"])
        self.assertFalse(result["formal_benchmark_started"])
        for key in R.FLAGS: self.assertIs(result[key], False)

    def test_known_unlimited_quota_still_needs_counters_and_affinity(self):
        quota = R.parse_quota("max 100000"); stat = R.parse_cpu_stat(STAT)
        for counters, affinity in ((None, [0]), ({}, [0]), (stat, None), (stat, []),
            (dict(stat, nr_throttled=True), [0])):
            with self.subTest(counters=counters, affinity=affinity):
                result = R.resource_decision(quota, counters, affinity)
                self.assertEqual(result["status"], "RESOURCE_UNKNOWN")
                self.assertFalse(result["resource_ready"])
                self.assertFalse(result["formal_benchmark_started"])

    def test_bad_quota_values_fail_closed(self):
        for value in (None, "", "50000", "50000 0", "0 100000", "-1 100000", "max 0",
                      "50000 100000 ignored", "0.5 100000", "True 100000", str(2**64) + " 100000",
                      "maxx 100000", "MAX 100000", "max -1", "max unknown", "max 100000 trailing"):
            with self.subTest(value=value), self.assertRaises(ValueError): R.parse_quota(value)

    def test_missing_duplicate_or_bad_stat_is_unknown(self):
        values = (None, "", "usage_usec 3", STAT + "nr_throttled 4\n", STAT.replace("nr_periods 8", "nr_periods -1"),
                  STAT.replace("nr_periods 8", "nr_periods 1.5"), STAT + "unknown trailing words\n")
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError): R.parse_cpu_stat(value)

    def test_unknown_stat_never_becomes_readiness(self):
        quota = R.parse_quota("100000 100000")
        for counters in (None, {}, dict(usage_usec=1, nr_periods=1, nr_throttled=True, throttled_usec=0)):
            self.assertEqual(R.resource_decision(quota, counters, [0])["status"], "RESOURCE_UNKNOWN")

    def test_affinity_missing_bool_duplicate_and_empty_refused(self):
        for affinity in (None, [], [True], [0, 0], [-1], "0"):
            with self.subTest(affinity=affinity):
                self.assertEqual(self.decide("100000 100000", affinity=affinity)["status"], "RESOURCE_UNKNOWN")

    def test_active_ledger_refuses_readiness_without_changing_it(self):
        result = self.decide("100000 100000", active=True)
        self.assertEqual(result["status"], "RESOURCE_BUSY_LEDGER_ACTIVE")
        self.assertFalse(result["resource_ready"])

    def test_nonfinite_and_duplicate_metadata_keys_refused(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): R.read_json(raw)


class SourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve(); self.source = self.root / "code.py"
        self.source.write_bytes(b"value = 1\n")
        self.row = dict(path="code.py", bytes=10, sha256=sha256(self.source.read_bytes()).hexdigest())
        self.lock = dict(schema=R.SCHEMA, gpu_uuid=None, files=[self.row], **{name:False for name in R.FLAGS})
        self.lockpath = self.root / "lock.json"

    def check(self, lock=None):
        self.lockpath.write_text(json.dumps(self.lock if lock is None else lock), encoding="utf-8")
        return R.verify_lock(self.root, self.lockpath, [self.source])

    def test_actual_source_hash_and_count_stable(self):
        before = self.check(); after = R.verify_lock(self.root, self.lockpath, [self.source])
        self.assertEqual(before, after); self.assertEqual(before["source_count"], 1)

    def test_modified_source_refused(self):
        self.check(); self.source.write_bytes(b"value = 2\n")
        with self.assertRaises(ValueError): R.verify_lock(self.root, self.lockpath, [self.source])

    def test_duplicate_source_refused(self):
        self.lock["files"].append(deepcopy(self.row))
        with self.assertRaises(ValueError): self.check()

    def test_source_lock_cannot_promote_qualification_or_bind_gpu(self):
        for field, value in (("gpu_uuid", "GPU-OLD"), *((name, True) for name in R.FLAGS)):
            with self.subTest(field=field):
                modified = deepcopy(self.lock); modified[field] = value
                with self.assertRaises(ValueError): self.check(modified)

    def test_stray_and_missing_source_paths_refused(self):
        for path in ("../code.py", "/code.py", "code.py:stream", "C:\\code.py", "absent.py", ""):
            with self.subTest(path=path):
                modified = deepcopy(self.lock); modified["files"][0]["path"] = path
                with self.assertRaises(ValueError): self.check(modified)

    def test_own_source_missing_from_lock_refused(self):
        other = self.root / "other.py"; other.write_text("x=1", encoding="utf-8")
        self.check()
        with self.assertRaises(ValueError): R.verify_lock(self.root, self.lockpath, [other])

    def test_bool_size_and_bad_hash_refused(self):
        for field, value in (("bytes", True), ("bytes", -1), ("sha256", "bad"), ("sha256", "A" * 64)):
            modified = deepcopy(self.lock); modified["files"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError): self.check(modified)

    def test_frozen_protocol_substitution_and_outside_project_refused(self):
        for path in (self.source, self.root.parent / "unrelated-protocol.json"):
            with self.subTest(path=str(path)), self.assertRaises(ValueError): R.closed_protocol(self.root, path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
