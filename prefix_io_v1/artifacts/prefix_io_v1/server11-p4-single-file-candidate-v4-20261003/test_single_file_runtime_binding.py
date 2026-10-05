"""CPU metadata counterexamples only; no device or model qualification."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import uuid


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("single_file_runtime_binding", HERE / "single_file_runtime_binding.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


class Runner:
    pass


def fixture():
    torch = NS(bfloat16="bf16", int8="int8")
    device = NS(type="cuda", index=0)
    hf = NS(num_hidden_layers=28, num_key_value_heads=4, hidden_size=3584,
            num_attention_heads=28, model_type="qwen2")
    model = NS(dtype="bf16", quantization=None, enforce_eager=True, hf_config=hf, max_model_len=1040)
    cache = NS(block_size=16, enable_prefix_caching=True, cache_dtype="auto", kv_cache_memory_bytes=268435456)
    parallel = NS(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                  distributed_executor_backend="uni")
    config = NS(model_config=model, cache_config=cache, parallel_config=parallel,
        speculative_config=None, scheduler_config=NS(async_scheduling=False, max_num_seqs=1,
        max_num_batched_tokens=1040), compilation_config=NS(mode=0, cudagraph_mode=0))
    runner = Runner()
    runner.vllm_config, runner.model_config, runner.device = config, model, device
    worker = NS(model_runner=runner, vllm_config=config)
    tensor = NS(is_cuda=True, dtype="int8", ndim=2, device=device,
                element_size=lambda: 1, stride=lambda index: 917504)
    layout = NS(storage_block_size_factor=1, storage_block_bytes=917504,
                bytes_per_kernel_block=[917504], gpu_tensors=[tensor])
    original = NS(iodepth=4, staging_mem=0.125, io_backend="linux_aio", sync_on_store=False,
                  enable_preload=True, preload_share_staging=True, staging_cache="lru", load_planner="off")
    reactor = NS(layout=layout, config=original, iodepth=4, open_lookahead=4,
                 file_store=NS(io_size=917504), staging_budget_bytes=134217728, actual_staging_bytes=133038080)
    handler = NS(coordinator=NS(reactor=reactor, layout=layout))
    return worker, handler, torch


class RuntimeBindingMetadataTests(unittest.TestCase):
    def test_import_does_not_import_torch_or_vllm(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("vllm", sys.modules)

    def test_direct_runtime_identity_construction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "actual native verification"):
            M.RuntimeSingleFileIdentity(("a", "b", "c", "d"), (), "model", {}, Runner())

    def test_original_canonical_int8_view_matches_bf16_model_geometry(self):
        worker, handler, torch = fixture()
        actual = M._geometry(worker, handler, "a" * 64, torch)
        self.assertEqual(actual, dict(model_config_sha256="a" * 64, num_hidden_layers=28,
            num_key_value_heads=4, head_dim=128, dtype="bfloat16", dtype_bytes=2,
            tokens_per_block=16, tensor_parallel_size=1, physical_block_bytes=917504))

    def test_original_metadata_changes_are_rejected(self):
        changes = (("async", lambda w, h: setattr(w.vllm_config.scheduler_config, "async_scheduling", True)),
            ("device_count", lambda w, h: setattr(w.vllm_config.parallel_config, "tensor_parallel_size", 2)),
            ("compile", lambda w, h: setattr(w.vllm_config.compilation_config, "mode", 3)),
            ("graph", lambda w, h: setattr(w.vllm_config.compilation_config, "cudagraph_mode", 1)),
            ("dtype", lambda w, h: setattr(w.model_runner.model_config, "dtype", "fp16")),
            ("block", lambda w, h: setattr(w.vllm_config.cache_config, "block_size", 32)),
            ("kv_budget", lambda w, h: setattr(w.vllm_config.cache_config, "kv_cache_memory_bytes", 67108864)),
            ("iodepth", lambda w, h: setattr(h.coordinator.reactor, "iodepth", 8)),
            ("lookahead", lambda w, h: setattr(h.coordinator.reactor, "open_lookahead", 8)),
            ("staging", lambda w, h: setattr(h.coordinator.reactor, "actual_staging_bytes", 134217729)),
            ("backend", lambda w, h: setattr(h.coordinator.reactor.config, "io_backend", "io_uring")),
            ("physical", lambda w, h: setattr(h.coordinator.reactor.file_store, "io_size", 1835008)),
            ("stride", lambda w, h: setattr(h.coordinator.layout.gpu_tensors[0], "stride", lambda i: 1835008)))
        for name, mutate in changes:
            with self.subTest(name=name):
                worker, handler, torch = fixture()
                mutate(worker, handler)
                with self.assertRaises(ValueError):
                    M._geometry(worker, handler, "a" * 64, torch)

    def test_uuid_formats_bind_to_same_physical_device_string(self):
        value = "38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9"
        for item in (value, "GPU-" + value, value.upper(), uuid.UUID(value).bytes):
            self.assertEqual(M.normalized_uuid(item), "GPU-" + value)
        with self.assertRaises(ValueError):
            M.normalized_uuid(b"wrong")


if __name__ == "__main__":
    unittest.main(verbosity=2)
