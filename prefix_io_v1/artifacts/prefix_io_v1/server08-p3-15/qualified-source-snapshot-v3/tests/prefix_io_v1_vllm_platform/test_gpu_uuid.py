"""CPU-only tests of exact AST-extracted author methods; NVML is a test double.

Importing vllm.platforms.cuda itself probes NVML, so the real source methods are
compiled without importing that module or Torch. This is unit evidence only.
"""
import ast
from functools import wraps
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / "third_party/work/vllm-author-build"
UUID_A = "GPU-11111111-1111-1111-1111-111111111111"
UUID_B = "GPU-22222222-2222-2222-2222-222222222222"


class UnknownUUID(Exception):
    pass


class NvmlDouble:
    def __init__(self):
        self.calls = []
        self.depth = 0
        self.fail_init = False
        self.fail_index = False
        self.handles = {UUID_A: "handle-a", UUID_B: "handle-b"}
        self.indices = {"handle-a": 6, "handle-b": 2}

    def nvmlInit(self):
        self.calls.append("init")
        if self.fail_init:
            raise RuntimeError("NVML init failed")
        self.depth += 1

    def nvmlShutdown(self):
        self.calls.append("shutdown")
        assert self.depth > 0
        self.depth -= 1

    def nvmlDeviceGetHandleByUUID(self, value):
        assert self.depth > 0
        self.calls.append(("uuid_handle", value))
        if value not in self.handles:
            raise UnknownUUID(value)
        return self.handles[value]

    def nvmlDeviceGetIndex(self, handle):
        assert self.depth > 0
        self.calls.append(("index", handle))
        if self.fail_index:
            raise LookupError("index query failed")
        return self.indices[handle]

    def nvmlDeviceGetHandleByIndex(self, index):
        assert self.depth > 0
        self.calls.append(("index_handle", index))
        return {value: key for key, value in self.indices.items()}[index]

    def nvmlDeviceGetUUID(self, handle):
        assert self.depth > 0
        self.calls.append(("handle_uuid", handle))
        return {value: key for key, value in self.handles.items()}[handle]


def source_node(tree, kind, name):
    return next(node for node in tree.body if isinstance(node, kind) and node.name == name)


@pytest.fixture
def platform():
    interface_tree = ast.parse((AUTHOR / "vllm/platforms/interface.py").read_text())
    cuda_tree = ast.parse((AUTHOR / "vllm/platforms/cuda.py").read_text())
    base = source_node(interface_tree, ast.ClassDef, "Platform")
    native = source_node(cuda_tree, ast.ClassDef, "NvmlCudaPlatform")
    base_method = source_node(base, ast.FunctionDef, "device_id_to_physical_device_id")
    nvml_context = source_node(cuda_tree, ast.FunctionDef, "with_nvml_context")
    methods = [node for node in native.body if isinstance(node, ast.FunctionDef) and node.name in
               {"device_id_to_physical_device_id", "_device_uuid_to_physical_device_id", "get_device_uuid"}]
    skeleton = ast.parse("""
from __future__ import annotations
class CudaPlatformBase:
    device_control_env_var = "CUDA_VISIBLE_DEVICES"
class NvmlCudaPlatform(CudaPlatformBase):
    pass
""")
    skeleton.body[1].body.append(base_method)
    skeleton.body[2].body = methods
    skeleton.body.insert(1, nvml_context)
    namespace = {"os": os, "wraps": wraps, "pynvml": NvmlDouble()}
    exec(compile(ast.fix_missing_locations(skeleton), str(AUTHOR / "vllm/platforms/cuda.py"), "exec"), namespace)
    return namespace["NvmlCudaPlatform"], namespace["pynvml"]


@pytest.mark.parametrize("visible,index,expected", [
    ("6", 0, 6), ("6,2", 1, 2), (" 6 ", 0, 6), ("", 3, 3), (None, 4, 4),
])
def test_original_numeric_empty_and_unset_paths_do_not_touch_nvml(platform, monkeypatch, visible, index, expected):
    cls, nvml = platform
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert cls.device_id_to_physical_device_id(index) == expected
    assert nvml.calls == []


@pytest.mark.parametrize("visible,index,expected,uuid,handle", [
    (UUID_A, 0, 6, UUID_A, "handle-a"),
    (f"{UUID_A},{UUID_B}", 1, 2, UUID_B, "handle-b"),
    (f"3,{UUID_A}", 1, 6, UUID_A, "handle-a"),
])
def test_full_uuid_is_resolved_with_balanced_nvml_context(platform, monkeypatch, visible, index, expected, uuid, handle):
    cls, nvml = platform
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert cls.device_id_to_physical_device_id(index) == expected
    assert nvml.calls == ["init", ("uuid_handle", uuid), ("index", handle), "shutdown"]
    assert nvml.depth == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


def test_unknown_uuid_propagates_without_fallback_and_shutdown_runs(platform, monkeypatch):
    cls, nvml = platform
    missing = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", missing)
    with pytest.raises(UnknownUUID, match=missing):
        cls.device_id_to_physical_device_id(0)
    assert nvml.calls == ["init", ("uuid_handle", missing), "shutdown"]
    assert nvml.depth == 0


def test_index_lookup_failure_still_shuts_down(platform, monkeypatch):
    cls, nvml = platform
    nvml.fail_index = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID_A)
    with pytest.raises(LookupError, match="index query failed"):
        cls.device_id_to_physical_device_id(0)
    assert nvml.calls[-1] == "shutdown"
    assert nvml.depth == 0


def test_initialization_failure_is_not_replaced_with_numeric_fallback(platform, monkeypatch):
    cls, nvml = platform
    nvml.fail_init = True
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID_A)
    with pytest.raises(RuntimeError, match="NVML init failed"):
        cls.device_id_to_physical_device_id(0)
    assert nvml.calls == ["init"]
    assert nvml.depth == 0


@pytest.mark.parametrize("visible", [UUID_A, "6"])
def test_out_of_range_logical_id_raises_without_nvml(platform, monkeypatch, visible):
    cls, nvml = platform
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(IndexError):
        cls.device_id_to_physical_device_id(1)
    assert nvml.calls == []


def test_other_invalid_device_tokens_keep_original_failure(platform, monkeypatch):
    cls, nvml = platform
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-a-device")
    with pytest.raises(ValueError):
        cls.device_id_to_physical_device_id(0)
    assert nvml.calls == []


def test_real_nvml_decorated_caller_has_balanced_nested_lifetime(platform, monkeypatch):
    cls, nvml = platform
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID_A)
    assert cls.get_device_uuid(0) == UUID_A
    assert nvml.calls == [
        "init", "init", ("uuid_handle", UUID_A), ("index", "handle-a"), "shutdown",
        ("index_handle", 6), ("handle_uuid", "handle-a"), "shutdown",
    ]
    assert nvml.depth == 0


def test_nested_failure_releases_both_nvml_references(platform, monkeypatch):
    cls, nvml = platform
    missing = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", missing)
    with pytest.raises(UnknownUUID):
        cls.get_device_uuid(0)
    assert nvml.calls == ["init", "init", ("uuid_handle", missing), "shutdown", "shutdown"]
    assert nvml.depth == 0
