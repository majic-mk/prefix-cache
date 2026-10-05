"""Real CPU tensors check Source ownership, not model/GPU qualification."""
import gc
import unittest
import weakref
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from probekv.v8_schema10_canonical import (
    capture_exact_dense_source,
    export_original_full_prefill,
    publish_exact_prefix_shadow,
)
from probekv.v8_schema10_prefix_shadow import PrefixShadowStore


class DecoupledCaptureTests(unittest.TestCase):
    def fixture(self, *, length=40, begin=20, count=4):
        tensors = tuple((torch.full((length, 2, 3), i + 1, dtype=torch.bfloat16),
                         torch.full((length, 2, 3), i + 11, dtype=torch.bfloat16))
                        for i in range(3))
        adapter = NS(
            inner=NS(layers=[NS(self_attn=NS(hack_kv=p, num_kv_heads=2, head_dim=3))
                             for p in tensors]),
            spec=NS(num_layers=3, checkpoints=(1, 2)),
            provenance={"tokenizer_hash": "tokenizer", "runtime_compatibility": "runtime"},
            shadows=PrefixShadowStore(model_signature="model", num_layers=3,
                                      kv_heads=2, head_dim=3, capacity_bytes=100000),
            outer=Mock(side_effect=AssertionError("export cannot execute forward")),
        )
        request = {"token_ids": list(range(length)), "segments": [{
            "segment_id": "C", "positions": list(range(begin, begin + count)),
            "token_ids": list(range(begin, begin + count)), "content_key": "content"}]}
        return adapter, request, tensors

    def test_source_export_does_not_call_shadow_or_model(self):
        adapter, request, _ = self.fixture()
        with patch.object(adapter.shadows, "publish", side_effect=AssertionError("implicit shadow")):
            source = export_original_full_prefill(adapter, request)["C"]
        adapter.outer.assert_not_called()
        self.assertEqual(adapter.shadows.resident_bytes, 0)
        self.assertFalse(source["capture_audit"]["shadow_published"])
        self.assertEqual(source["capture_audit"]["shadow_bytes"], 0)
        self.assertEqual(source["capture_audit"]["extra_full_prefill_count"], 0)
        self.assertEqual(source["capture_audit"]["origin"], "exact_dense_full_prefill")

    def test_export_works_without_a_shadow_subsystem(self):
        adapter, request, _ = self.fixture()
        del adapter.shadows
        source = export_original_full_prefill(adapter, request)["C"]
        self.assertEqual(source["source_metadata"]["token_ids"], [20, 21, 22, 23])

    def test_target_and_selection_allocations_do_not_own_parent_storage(self):
        adapter, request, tensors = self.fixture()
        source = export_original_full_prefill(adapter, request)["C"]
        parents = [weakref.ref(t) for pair in tensors for t in pair]
        for original, exported in zip(tensors, source["layers"]):
            for parent, target in zip(original, exported):
                self.assertIsNone(target._base)
                self.assertEqual(target.untyped_storage().nbytes(), target.numel() * target.element_size())
                self.assertNotEqual(parent.data_ptr(), target.data_ptr())
        for depth, state in source["selection_states"].items():
            self.assertIsNone(state._base)
            self.assertNotEqual(state.data_ptr(), source["layers"][depth][0].data_ptr())
        del parent, target, original, exported, tensors, adapter
        gc.collect()
        self.assertTrue(all(ref() is None for ref in parents))
        self.assertEqual(source["layers"][0][0].sum().item(), 24.0)
        self.assertEqual(source["capture_audit"]["target_owned_kv_bytes"], 2 * 3 * 4 * 2 * 3 * 2)
        self.assertEqual(source["capture_audit"]["selection_state_bytes"], 2 * 4 * 2 * 3 * 2)
        self.assertEqual(source["capture_audit"]["parent_owned_kv_bytes"], 0)

    def test_target_bytes_do_not_scale_with_prefix_length(self):
        a, q, _ = self.fixture(length=40, begin=20)
        b, r, _ = self.fixture(length=200, begin=180)
        x = export_original_full_prefill(a, q)["C"]
        y = export_original_full_prefill(b, r)["C"]
        self.assertEqual(x["capture_audit"]["target_owned_kv_bytes"], y["capture_audit"]["target_owned_kv_bytes"])
        self.assertEqual(x["capture_audit"]["selection_state_bytes"], y["capture_audit"]["selection_state_bytes"])

    def test_explicit_exact_shadow_has_separate_owned_bytes(self):
        adapter, request, tensors = self.fixture()
        source = export_original_full_prefill(adapter, request)["C"]
        self.assertTrue(publish_exact_prefix_shadow(adapter, request, origin="exact_dense_full_prefill"))
        self.assertEqual(adapter.shadows.resident_bytes, 2 * 3 * 40 * 2 * 3 * 2)
        self.assertFalse(source["capture_audit"]["shadow_published"])
        tensors[0][0].zero_()
        self.assertEqual(source["layers"][0][0].sum().item(), 24)
        row = next(iter(adapter.shadows.entries.values()))
        self.assertEqual(row["layers"][0][0].sum().item(), 240)

    def test_mixed_and_unknown_cannot_publish_exact_shadow(self):
        adapter, request, _ = self.fixture()
        for origin in ("MIXED_CONTEXT_FULL_SEGMENT", "UNKNOWN_PROVENANCE", "native_prefix_dense_remaining"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                publish_exact_prefix_shadow(adapter, request, origin=origin)
        self.assertEqual(adapter.shadows.resident_bytes, 0)

    def test_incomplete_and_nonfinite_target_capture_fails_closed(self):
        for fault in ("missing_layer", "missing_pair", "nan"):
            with self.subTest(fault=fault):
                adapter, request, tensors = self.fixture()
                if fault == "missing_layer":
                    adapter.inner.layers.pop()
                elif fault == "missing_pair":
                    adapter.inner.layers[1].self_attn.hack_kv = []
                else:
                    tensors[1][0][20, 0, 0] = float("nan")
                with self.assertRaises(ValueError):
                    export_original_full_prefill(adapter, request)
                self.assertEqual(adapter.shadows.resident_bytes, 0)

    def test_shadow_rejects_empty_pair_and_nonfinite_before_publication(self):
        adapter, request, tensors = self.fixture()
        with self.assertRaises(ValueError):
            adapter.shadows.publish(request["token_ids"], ((), tensors[1], tensors[2]),
                                    origin="exact_dense_full_prefill")
        tensors[0][0][0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            publish_exact_prefix_shadow(adapter, request, origin="exact_dense_full_prefill")
        self.assertFalse(adapter.shadows.entries)

    def test_explicit_dense_capture_uses_one_forward_and_no_shadow(self):
        # This tests Python ownership/control flow only. CUDA timings and the
        # native model operation are mocked; it is not a GPU numerical test.
        adapter, request, tensors = self.fixture()
        request["request_id"] = "diagnostic-source-build"
        adapter.inner.cache_fuse_metadata = {"original_setting": "unchanged"}
        adapter.hbm = NS(reserve_batch=Mock(return_value=[NS(reservation_id="capture")]), release=Mock())
        adapter.prepare = Mock(return_value=(torch.tensor(request["token_ids"]),
            torch.arange(40), NS(slot_mapping=torch.full((40,), -1))))
        adapter.outer = Mock()
        vllm, sequence = ModuleType("vllm"), ModuleType("vllm.sequence")
        vllm.SamplingParams = lambda **kwargs: NS(**kwargs)
        sequence.SequenceData = lambda tokens: tokens
        sequence.SequenceGroupMetadata = lambda **kwargs: NS(**kwargs)
        event = NS(record=Mock(), synchronize=Mock(), elapsed_time=Mock(return_value=0.0))
        with patch.dict("sys.modules", {"vllm": vllm, "vllm.sequence": sequence}), \
                patch.object(torch.cuda, "Event", return_value=event), \
                patch.object(torch.cuda, "synchronize"), \
                patch.object(adapter.shadows, "publish", side_effect=AssertionError("implicit shadow")):
            source = capture_exact_dense_source(adapter, request, "C")
        adapter.outer.assert_called_once()
        self.assertIsNone(adapter.outer.call_args.kwargs["kv_caches"][0])
        self.assertEqual(adapter.inner.cache_fuse_metadata, {"original_setting": "unchanged"})
        self.assertTrue(all(block.self_attn.hack_kv == [] for block in adapter.inner.layers))
        adapter.hbm.release.assert_called_once_with("capture")
        self.assertEqual(source["capture_audit"]["parent_owned_kv_bytes"], 0)
        self.assertFalse(source["capture_audit"]["shadow_published"])
        self.assertFalse(source["capture_audit"]["cfo"]["collected"])
        self.assertEqual(source["layers"][0][0].untyped_storage().nbytes(), 4 * 2 * 3 * 2)


if __name__ == "__main__":
    unittest.main()
