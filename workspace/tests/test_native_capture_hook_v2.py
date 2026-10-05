"""CPU native-module hooks: no model, no CUDA/performance qualification."""
import gc
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from probekv.native_capture_hook_v2 import CurrentLayerCapture, ProjectedKVSlice


class Projection(torch.nn.Module):
    def __init__(self, *, dtype=torch.bfloat16, width=16):
        super().__init__()
        self.dtype, self.width, self.calls, self.output_ref = dtype, width, 0, None

    def forward(self, hidden):
        self.calls += 1
        out = torch.arange(hidden.shape[0] * self.width, dtype=torch.float32).reshape(
            hidden.shape[0], self.width).to(self.dtype)
        self.output_ref = weakref.ref(out)
        return out, None


class NativeCaptureHookTests(unittest.TestCase):
    def fixture(self, **projection_kwargs):
        attention = NS(qkv_proj=Projection(**projection_kwargs), q_size=8, kv_size=4,
                       num_heads=4, num_kv_heads=2, head_dim=2)
        return attention, torch.zeros((5, 8), dtype=torch.bfloat16)

    def capture(self, attention, **kwargs):
        return CurrentLayerCapture(attention, layer_1based=3,
            projection_positions=(10, 11, 12, 15, 17), target_positions=(10, 12, 17),
            max_target_bytes=48, **kwargs)

    def test_actual_projection_once_pre_rope_owned_slices_and_no_cuda(self):
        attention, hidden = self.fixture()
        with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("sync")), \
                patch.object(torch.cuda, "Event", side_effect=AssertionError("event")):
            with self.capture(attention) as tap:
                output, _ = attention.qkv_proj(hidden)
                expected_k = output[[0, 2, 4], 8:12].clone().reshape(3, 2, 2)
                expected_v = output[[0, 2, 4], 12:16].clone().reshape(3, 2, 2)
                # The actual forward is free to rotate projected K in place.
                output[:, 8:12].zero_()
            packet = tap.packet_after_block()
        self.assertIsInstance(packet, ProjectedKVSlice)
        self.assertEqual(attention.qkv_proj.calls, 1)
        self.assertEqual(tap.invocation_count, 1)
        self.assertEqual(packet.layer_1based, 3)
        self.assertEqual(packet.full_projection_positions, (10, 11, 12, 15, 17))
        self.assertEqual(packet.captured_positions, (10, 12, 17))
        self.assertTrue(torch.equal(packet.key, expected_k))
        self.assertTrue(torch.equal(packet.value, expected_v))
        self.assertEqual(packet.owned_tensor_bytes, 48)
        self.assertEqual(sum(t.untyped_storage().nbytes() for t in (packet.key, packet.value)), 48)
        self.assertIsNone(packet.key._base)
        self.assertIsNone(packet.value._base)
        self.assertFalse(attention.qkv_proj._forward_hooks)
        ref = attention.qkv_proj.output_ref
        del output
        gc.collect()
        self.assertIsNone(ref())

    def test_contiguous_and_reordered_layout_keeps_absolute_identity(self):
        attention, hidden = self.fixture()
        with CurrentLayerCapture(attention, layer_1based=1,
                projection_positions=(15, 10, 11, 17, 12),
                target_positions=(10, 11, 12), max_target_bytes=48) as tap:
            output, _ = attention.qkv_proj(hidden)
        packet = tap.packet_after_block()
        self.assertTrue(torch.equal(packet.key, output[[1, 2, 4], 8:12].reshape(3, 2, 2)))

    def test_budget_failure_does_not_change_or_abort_projection(self):
        attention, hidden = self.fixture()
        with CurrentLayerCapture(attention, layer_1based=3,
                projection_positions=(10, 11, 12, 15, 17),
                target_positions=(10, 12, 17), max_target_bytes=47) as tap:
            output, _ = attention.qkv_proj(hidden)
        self.assertEqual(tuple(output.shape), (5, 16))
        self.assertIn("MemoryError", tap.failure)
        self.assertIsNone(tap.packet_after_block())
        self.assertFalse(attention.qkv_proj._forward_hooks)

    def test_invalid_projection_geometry_rejects_capture_only(self):
        for kwargs in ({"dtype": torch.float32}, {"width": 15}):
            with self.subTest(kwargs=kwargs):
                attention, hidden = self.fixture(**kwargs)
                with self.capture(attention) as tap:
                    attention.qkv_proj(hidden)
                self.assertIsNone(tap.packet_after_block())
                self.assertIn("ValueError", tap.failure)
        attention, hidden = self.fixture()
        attention.num_heads = 3
        with self.capture(attention) as tap:
            attention.qkv_proj(hidden)
        self.assertIn("GQA", tap.failure)

    def test_missing_rows_no_call_and_multiple_calls_are_not_full_capture(self):
        attention, hidden = self.fixture()
        with CurrentLayerCapture(attention, layer_1based=3,
                projection_positions=(10, 11, 12, 15, 17),
                target_positions=(10, 99), max_target_bytes=48) as missing:
            attention.qkv_proj(hidden)
        self.assertIsNone(missing.packet_after_block())
        self.assertIn("not fully projected", missing.failure)
        with self.capture(attention) as absent:
            pass
        self.assertIsNone(absent.packet_after_block())
        self.assertIn("one actual", absent.failure)
        with self.capture(attention) as repeated:
            attention.qkv_proj(hidden)
            attention.qkv_proj(hidden)
        self.assertIsNone(repeated.packet_after_block())
        self.assertIn("exactly one", repeated.failure)

    def test_runtime_fault_is_not_swallowed_and_handle_is_removed(self):
        attention, hidden = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "device fault"):
            with self.capture(attention) as tap:
                with patch.object(torch, "empty", side_effect=RuntimeError("device fault")):
                    attention.qkv_proj(hidden)
        self.assertEqual(tap.failure, "BLOCK_EXECUTION_FAILED: RuntimeError")
        self.assertIsNone(tap.packet_after_block())
        self.assertFalse(attention.qkv_proj._forward_hooks)

    def test_python_memory_failure_skips_only_capture(self):
        attention, hidden = self.fixture()
        with self.capture(attention) as tap:
            with patch.object(torch, "empty", side_effect=MemoryError("bounded staging unavailable")):
                output, _ = attention.qkv_proj(hidden)
        self.assertEqual(tuple(output.shape), (5, 16))
        self.assertIn("MemoryError", tap.failure)
        self.assertIsNone(tap.packet_after_block())

    def test_later_block_failure_invalidates_successful_projection(self):
        attention, hidden = self.fixture()
        with self.assertRaisesRegex(ValueError, "attention failed"):
            with self.capture(attention) as tap:
                attention.qkv_proj(hidden)
                raise ValueError("attention failed")
        self.assertIsNone(tap.packet_after_block())
        self.assertFalse(attention.qkv_proj._forward_hooks)

    def test_packet_requires_completed_scope_and_scope_is_single_use(self):
        attention, hidden = self.fixture()
        with self.capture(attention) as tap:
            attention.qkv_proj(hidden)
            with self.assertRaisesRegex(RuntimeError, "scope exits"):
                tap.packet_after_block()
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            with tap:
                pass


if __name__ == "__main__":
    unittest.main()
