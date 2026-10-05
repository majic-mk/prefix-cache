"""CPU synthetic configuration guard tests; no torch/vLLM or GPU import."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_prefix_dtype_test",
    ROOT / "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
BF16 = "synthetic-bf16-dtype"


def config():
    return SimpleNamespace(kv_transfer_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None, enable_prefix_caching=True,
            block_size=16, cache_dtype="auto", kv_cache_memory_bytes=67108864),
        model_config=SimpleNamespace(dtype=BF16, quantization=None))


def test_auto_preserves_fixed_bf16_configuration():
    result = smoke.validate_effective_config(config(), BF16)
    assert smoke.ENGINE["kv_cache_dtype"] == "auto"
    assert smoke.ENGINE["dtype"] == "bfloat16" and smoke.ENGINE["quantization"] is None
    assert result["effective_kv_cache_dtype"] == "auto"
    assert result["effective_model_dtype"] == BF16
    assert result["effective_model_quantization"] is None
    assert result["effective_kv_cache_memory_bytes"] == 67108864
    assert smoke.EXPECTED_HOT_TOKENS == 112


@pytest.mark.parametrize("field,value", [
    ("cache_dtype", "bfloat16"), ("cache_dtype", "fp8"),
    ("enable_prefix_caching", False), ("block_size", 32),
    ("kv_cache_memory_bytes", 67108863), ("kv_offloading_size", 1),
])
def test_cache_change_rejected(field, value):
    cfg = config()
    setattr(cfg.cache_config, field, value)
    with pytest.raises(RuntimeError):
        smoke.validate_effective_config(cfg, BF16)


@pytest.mark.parametrize("field,value", [("dtype", "synthetic-fp16"), ("quantization", "fp8")])
def test_model_dtype_or_quantization_change_rejected(field, value):
    cfg = config()
    setattr(cfg.model_config, field, value)
    with pytest.raises(RuntimeError, match="unquantized torch.bfloat16"):
        smoke.validate_effective_config(cfg, BF16)


def test_connector_rejected():
    cfg = config()
    cfg.kv_transfer_config = object()
    with pytest.raises(RuntimeError, match="connector"):
        smoke.validate_effective_config(cfg, BF16)
