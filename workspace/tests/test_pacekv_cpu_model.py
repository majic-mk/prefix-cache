"""Tiny random CPU model: API/cache semantics only, NOT real-model evidence."""
import importlib.util
import unittest

from pacekv.gpu_pilot import cache_digest, compare_logits, kv_bytes, legacy_cache


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                     "CPU cache API test requires existing torch and transformers")
class CpuModelContractTests(unittest.TestCase):
    def test_legacy_cache_restore_continuation_and_identity(self):
        import torch
        from transformers import MistralConfig, MistralForCausalLM
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(19)
            config = MistralConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, max_position_embeddings=128,
                                   sliding_window=None)
            model = MistralForCausalLM(config).eval()
            with torch.inference_mode():
                output = model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=True)
                cache = legacy_cache(output.past_key_values)
                sha = cache_digest(cache)
                copied = tuple(tuple(t.clone() for t in layer) for layer in cache)
                self.assertGreater(kv_bytes(cache), 0)
                def continue_from(initial):
                    past, logits = initial, output.logits[:, -1]
                    ids, rows = [], []
                    for _ in range(32):
                        token = logits.argmax(-1)
                        ids.append(token.item()); rows.append(logits.detach().float())
                        result = model(token.reshape(1, 1), past_key_values=past, use_cache=True)
                        past = legacy_cache(result.past_key_values)
                        logits = result.logits[:, -1]
                    return ids, rows
                baseline, a = continue_from(cache)
                restored, b = continue_from(copied)
                self.assertEqual(baseline, restored)
                self.assertEqual(max(compare_logits(torch, a, b)), 0)
                self.assertEqual(cache_digest(cache), sha)
                self.assertEqual(cache_digest(copied), sha)
                copied[0][0].flatten()[0] += 1
                self.assertNotEqual(cache_digest(copied), sha)
                self.assertEqual(cache_digest(cache), sha)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
