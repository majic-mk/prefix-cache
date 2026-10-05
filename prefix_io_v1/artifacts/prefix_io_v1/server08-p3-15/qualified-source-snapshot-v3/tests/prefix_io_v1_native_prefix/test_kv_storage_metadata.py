"""CPU synthetic metadata tests: no torch/vLLM imports and no GPU allocation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_prefix_metadata_test",
    ROOT / "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)
BF16 = object()


class Storage:
    def __init__(self, pointer, size):
        self.pointer, self.size = pointer, size
    def nbytes(self):
        return self.size
    def data_ptr(self):
        return self.pointer


class Tensor:
    def __init__(self, storage, shape=(2, 16, 4, 128), device="cuda", dtype=BF16):
        self.storage = storage
        self.shape = shape
        self.device = SimpleNamespace(type=device, index=0)
        self.dtype = dtype
    def untyped_storage(self):
        return self.storage


@pytest.fixture(autouse=True)
def synthetic_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Tensor=Tensor, bfloat16=BF16))


def worker(shared=False):
    size = 4096
    storages = [Storage(100 + (0 if shared else i), size) for i in range(28)]
    count = 1 if shared else 28
    config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=[f"layer{i}" for i in range(28)])],
        kv_cache_tensors=[SimpleNamespace(size=size) for _ in range(count)],
        num_blocks=2)
    return SimpleNamespace(model_runner=SimpleNamespace(
        kv_caches=[Tensor(storage) for storage in storages], kv_cache_config=config))


def test_separate_storages():
    result = smoke.worker_kv_storage_metadata(worker())
    assert result["tensor_count"] == result["unique_storage_count"] == 28
    assert result["actual_kv_tensor_allocation_bytes"] == 28 * 4096
    assert result["cuda_allocator_reserved_bytes"] is None
    assert result["physical_release_witness"] is None
    assert "pointer" not in str(result)


def test_distinct_views_of_same_storage_deduplicate():
    # Distinct Python storage wrappers share the same CUDA storage pointer.
    obj = worker(shared=True)
    obj.model_runner.kv_caches[1].shape = (16, 2, 4, 128)
    result = smoke.worker_kv_storage_metadata(obj)
    assert result["unique_storage_count"] == 1
    assert result["actual_kv_tensor_allocation_bytes"] == 4096


@pytest.mark.parametrize("mutate", [
    lambda r: setattr(r, "kv_caches", []),
    lambda r: r.kv_caches.append(r.kv_caches[0]),
    lambda r: setattr(r.kv_cache_config, "num_blocks", 0),
    lambda r: setattr(r.kv_cache_config, "num_blocks", True),
    lambda r: setattr(r.kv_cache_config, "kv_cache_tensors", []),
    lambda r: setattr(r.kv_cache_config.kv_cache_tensors[0], "size", 0),
    lambda r: setattr(r.kv_cache_config.kv_cache_tensors[0], "size", 67108865),
    lambda r: setattr(r.kv_cache_config.kv_cache_tensors[0], "size", 4000),
    lambda r: setattr(r.kv_caches[0].storage, "size", 0),
    lambda r: setattr(r.kv_caches[0].storage, "size", 67108865),
    lambda r: setattr(r.kv_caches[0].storage, "pointer", 0),
    lambda r: setattr(r.kv_caches[0], "device", SimpleNamespace(type="cpu", index=None)),
    lambda r: setattr(r.kv_caches[0], "dtype", object()),
    lambda r: setattr(r.kv_caches[0], "shape", (0, 16)),
    lambda r: setattr(r.kv_caches[0], "shape", (1,) * 9),
    lambda r: setattr(r.kv_cache_config, "kv_cache_groups", []),
    lambda r: setattr(r.kv_cache_config.kv_cache_groups[0], "layer_names", ["same"] * 28),
])
def test_invalid_or_uninitialized_rejected(mutate):
    obj = worker()
    mutate(obj.model_runner)
    with pytest.raises((RuntimeError, AttributeError)):
        smoke.worker_kv_storage_metadata(obj)


def test_conflicting_size_of_shared_storage_rejected():
    obj = worker(shared=True)
    obj.model_runner.kv_caches[1].storage.size += 1
    with pytest.raises(RuntimeError, match="inconsistent"):
        smoke.worker_kv_storage_metadata(obj)


def test_missing_runner_rejected():
    with pytest.raises(AttributeError):
        smoke.worker_kv_storage_metadata(SimpleNamespace())


def test_metadata_exception_propagates():
    obj = worker()
    def fail():
        raise RuntimeError("metadata unavailable")
    obj.model_runner.kv_caches[0].untyped_storage = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        smoke.worker_kv_storage_metadata(obj)


@pytest.mark.parametrize("args", [(27, 67108864), (28, 67108863)])
def test_only_frozen_bounds(args):
    with pytest.raises(RuntimeError):
        smoke.worker_kv_storage_metadata(worker(), *args)


def test_callable_serialization_is_process_environment_only(monkeypatch, tmp_path):
    import ast
    import os
    source = Path(smoke.__file__).read_text()
    tree = ast.parse(source)
    flag = "VLLM_ALLOW_INSECURE_SERIALIZATION"
    assert sum(isinstance(n, ast.Constant) and n.value == flag
               for n in ast.walk(tree)) == 2  # one setter and one frozen field
    with monkeypatch.context() as local:
        local.setattr(smoke, "ROOT", tmp_path)
        local.setenv(flag, "0")
        for key in (*smoke.RUNTIME_PATH_ENV_KEYS, "VLLM_USE_MODELSCOPE",
                    "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK"):
            local.setenv(key, "cpu-test-previous-value")
        smoke.configure_runtime_environment()
        assert os.environ[flag] == "1"
        assert source.index("configure_runtime_environment()",
                            source.index("def main():")) < source.index("        import vllm")
    assert not (tmp_path / "etc").exists()
