import copy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pacekv.audit import verify_audit, write_new_json
from pacekv.exact_key import ExactBlockKey, ExactContext, digest, prefix_keys
from pacekv.ownership import OwnershipLedger, Phase
from pacekv.evidence import dual_slo_goodput, token_metrics, quantile
from pacekv.plan import arm_order, pilot_plan, verify_plan
from pacekv.analysis import analyze, paired_interval


class ExactKeyTests(unittest.TestCase):
    def setUp(self):
        self.context = ExactContext("revision", "tokenizer", "attention", "position", "layout", "tenant")

    def test_equal_exact_prefix_hits(self):
        self.assertEqual(prefix_keys(self.context, [1, 2, 3, 4], 2),
                         prefix_keys(self.context, [1, 2, 3, 4], 2))

    def test_same_target_different_preceding_tokens_misses(self):
        a = prefix_keys(self.context, [1, 2, 3, 4], 2)
        b = prefix_keys(self.context, [7, 8, 3, 4], 2)
        self.assertEqual(a[-1].token_ids, b[-1].token_ids)
        self.assertNotEqual(a[-1].sha256, b[-1].sha256)

    def test_all_signatures_isolate_cache(self):
        for field in self.context.__dataclass_fields__:
            self.assertNotEqual(prefix_keys(self.context, [1, 2], 2),
                                prefix_keys(replace(self.context, **{field: "other"}), [1, 2], 2))

    def test_incomplete_block_not_published(self):
        self.assertEqual(len(prefix_keys(self.context, [1, 2, 3], 2)), 1)
        self.assertEqual(prefix_keys(self.context, [1], 2), ())

    def test_invalid_keys(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                prefix_keys(self.context, [1], value)
        for token in (True, -1, 0.5):
            with self.assertRaises(ValueError):
                prefix_keys(self.context, [token], 1)
        with self.assertRaises(ValueError):
            replace(self.context, namespace="")
        with self.assertRaises(ValueError):
            ExactBlockKey("x", "y", (1,), 0, 1)


class OwnershipTests(unittest.TestCase):
    def test_dma_probe_labels_map_to_ledger_directions(self):
        from pacekv.gpu_pilot import ledger_direction
        for label in ("h2d", "d2h"):
            ledger = OwnershipLedger(16, 16)
            self.assertTrue(ledger.reserve("copy", ledger_direction(label), 0,
                                           8, 8, True, 1, ledger.epoch))
            ledger.issue("copy", 0, 2)
            ledger.dma_complete("copy", 0, 3)
            self.assertEqual(ledger.snapshot(3)["outstanding"], 0)
        with self.assertRaises(ValueError):
            ledger_direction("unknown")

    def setUp(self):
        self.x = OwnershipLedger(100, 80)

    def reserve(self, name="a", direction="D2H", optional=True, gpu=50, pin=40, now=0):
        return self.x.reserve(name, direction, 2, gpu, pin, optional, now, self.x.epoch)

    def test_atomic_failure(self):
        self.assertTrue(self.reserve())
        self.assertFalse(self.reserve("b", gpu=60))
        self.assertEqual(self.x.snapshot(0)["gpu_bytes"], 50)
        self.assertNotIn("b", self.x.tickets)

    def test_two_reservations_fill_capacity(self):
        self.reserve(); self.reserve("b")
        self.assertFalse(self.reserve("c", gpu=1, pin=1))

    def test_unissued_optional_cancel(self):
        self.reserve(); self.x.cancel("a", 2, 10)
        self.assertEqual(self.x.snapshot(10)["outstanding"], 0)
        self.assertEqual(self.x.gpu_byte_ns, 500)

    def test_cancel_cannot_release_inflight(self):
        self.reserve(); self.x.issue("a", 2, 1); self.x.cancel("a", 2, 5)
        self.assertEqual(self.x.tickets["a"].phase, Phase.CANCEL_PENDING)
        self.assertEqual(self.x.gpu_bytes, 50)
        self.x.dma_complete("a", 2, 10)
        self.assertEqual(self.x.gpu_byte_ns, 500)
        self.assertEqual(self.x.pinned_bytes, 0)

    def test_gpu_and_storage_lifetime_differ(self):
        self.reserve(); self.x.issue("a", 2, 1)
        self.x.dma_complete("a", 2, 10, storage_pending=True)
        self.assertEqual((self.x.gpu_bytes, self.x.pinned_bytes), (0, 40))
        self.x.storage_complete("a", 2, 20)
        self.assertEqual((self.x.gpu_byte_ns, self.x.pinned_byte_ns), (500, 800))

    def test_required_inference_never_dropped(self):
        self.reserve(optional=False)
        with self.assertRaises(ValueError): self.x.cancel("a", 2, 1)

    def test_failure_keeps_owners_until_fence(self):
        self.reserve(); self.x.issue("a", 2, 1); self.x.fail("a", 2, 2)
        with self.assertRaises(ValueError):
            self.x.release_quarantined("a", 2, 3, completion_fenced=False)
        self.assertEqual(self.x.gpu_bytes, 50)
        self.x.release_quarantined("a", 2, 4, completion_fenced=True)
        self.assertEqual(self.x.snapshot(4)["outstanding"], 0)

    def test_stale_generation_and_epoch(self):
        self.reserve()
        with self.assertRaises(ValueError): self.x.issue("a", 1, 1)
        with self.assertRaises(ValueError):
            self.x.reserve("b", "H2D", 2, 10, 10, True, 1, 0)

    def test_no_premature_completion(self):
        self.reserve()
        with self.assertRaises(ValueError): self.x.dma_complete("a", 2, 1)

    def test_h2d_cannot_have_writeback_storage(self):
        self.reserve(direction="H2D"); self.x.issue("a", 2, 1)
        with self.assertRaises(ValueError): self.x.dma_complete("a", 2, 2, storage_pending=True)

    def test_backwards_clock(self):
        self.reserve(now=10)
        with self.assertRaises(ValueError): self.x.snapshot(9)

    def test_duplicate_id(self):
        self.reserve()
        with self.assertRaises(ValueError): self.reserve()


class EvidenceTests(unittest.TestCase):
    def test_ttft_includes_queue_and_itl(self):
        r = token_metrics(0, [5_000_000, 8_000_000, 12_000_000], completed=True,
                          ttft_slo_ms=6, itl_slo_ms=5)
        self.assertEqual(r["ttft_ms"], 5)
        self.assertEqual(r["itl_ms"], [3, 4])
        self.assertTrue(r["dual_slo_met"])

    def test_missing_itl_is_not_success(self):
        r = token_metrics(0, [1], completed=True, ttft_slo_ms=6, itl_slo_ms=5)
        self.assertIsNone(r["p95_itl_ms"])
        self.assertFalse(r["dual_slo_met"])

    def test_failed_requests_in_goodput_denominator(self):
        a = token_metrics(0, [1, 2], completed=True, ttft_slo_ms=6, itl_slo_ms=5)
        b = token_metrics(0, [], completed=False, ttft_slo_ms=6, itl_slo_ms=5)
        self.assertEqual(dual_slo_goodput([a, b], 1_000_000_000), 1)

    def test_bad_metrics_rejected(self):
        with self.assertRaises(ValueError): quantile([float("nan")], .95)
        with self.assertRaises(ValueError):
            token_metrics(10, [9], completed=True, ttft_slo_ms=6, itl_slo_ms=5)
        with self.assertRaises(ValueError): dual_slo_goodput([{}], 1)

    def test_bootstrap_not_independent_tokens(self):
        ci = paired_interval([1, 1, 1])
        self.assertEqual((ci["lower"], ci["upper"]), (1, 1))
        self.assertEqual(ci["resampling_unit"], "complete_round")

    def test_failed_directory_refused(self):
        with tempfile.TemporaryDirectory() as d:
            write_new_json(Path(d)/"failed.json", {"reason": "OOM"})
            with self.assertRaises(ValueError): analyze(d)

    def test_evidence_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"a.json"
            write_new_json(p, {"a": 1})
            with self.assertRaises(FileExistsError): write_new_json(p, {"a": 2})


class PlanTests(unittest.TestCase):
    def test_5090_stack_is_explicit_and_legacy_a800_is_not_reused(self):
        plan = pilot_plan()
        self.assertEqual(plan["gpu_model"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(plan["compute_capability"], [12, 0])
        self.assertEqual((plan["torch_version"], plan["torch_cuda_version"]), ("2.7.1", "12.8"))
        self.assertGreaterEqual(plan["minimum_free_gpu_bytes"], 24 * 1024**3)
        self.assertGreaterEqual(plan["minimum_host_headroom_bytes"], 32 * 1024**3)
        self.assertIn("5090", plan["protocol"])

    def test_torch28_is_separate_frozen_protocol(self):
        from pacekv.plan import plan_for_protocol, plan_for_sha
        previous = pilot_plan()
        current = pilot_plan("torch28")
        self.assertEqual((current["torch_version"], current["torch_cuda_version"]), ("2.8.0", "12.8"))
        self.assertNotEqual(current["protocol"], previous["protocol"])
        self.assertNotEqual(current["plan_sha256"], previous["plan_sha256"])
        self.assertEqual(plan_for_protocol(current["protocol"]), current)
        self.assertEqual(plan_for_sha(current["plan_sha256"]), current)
        verify_plan(current)
        current["output_tokens"] = 1
        with self.assertRaises(ValueError): verify_plan(current)

    def test_old_hardware_audit_cannot_unlock_new_pilot(self):
        from pacekv.audit import code_files
        root = Path(__file__).resolve().parents[1]
        body = dict(protocol="pacekv-r1a-20260923", code_files=code_files(root),
                    plan_sha256=pilot_plan()["plan_sha256"],
                    cpu_asset_preflight_passed=True, assets={}, dependencies={})
        body["audit_sha256"] = digest(body)
        with self.assertRaisesRegex(ValueError, "another hardware/protocol"):
            verify_audit(body, root)

    def test_manifest_immutable(self):
        plan = pilot_plan(); verify_plan(plan)
        plan["output_tokens"] = 1
        with self.assertRaises(ValueError): verify_plan(plan)

    def test_no_claims_without_native_integration(self):
        for field in ("native_serving_integration", "formal_profile", "research_gain_claim_allowed", "gpu_execution_authorized"):
            self.assertIs(pilot_plan()[field], False)

    def test_order_balanced(self):
        arms = pilot_plan()["arms"]
        for index in range(4):
            self.assertEqual(set(arm_order(i, arms)[index] for i in range(4)), set(arms))

    def test_cpu_import_does_not_import_cuda_model_libraries(self):
        root = Path(__file__).resolve().parents[1]
        code = "import pacekv.gpu_pilot, sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gpu_cli_requires_explicit_opt_in(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, str(root/"scripts/pacekv/run_pilot.py"),
                                 "--preflight", "unused", "--output", "unused", "--max-seconds", "1"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--execute-gpu", result.stderr)


if __name__ == "__main__":
    unittest.main()
