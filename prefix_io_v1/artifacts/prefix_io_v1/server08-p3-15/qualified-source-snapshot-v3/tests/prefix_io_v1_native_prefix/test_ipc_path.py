"""CPU-only ZMQ regression for the real project's native UUID IPC paths."""
import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import pytest
import zmq

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("native_prefix_ipc_test",
    ROOT / "experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def set_env(monkeypatch):
    for key in (*smoke.RUNTIME_PATH_ENV_KEYS, "VLLM_USE_MODELSCOPE",
                "VLLM_NO_USAGE_STATS", "VLLM_DO_NOT_TRACK",
                "VLLM_ALLOW_INSECURE_SERIALIZATION"):
        monkeypatch.setenv(key, "cpu-test")


def test_real_project_uuid_ipc_bind_and_close(monkeypatch):
    set_env(monkeypatch)
    smoke.configure_runtime_environment()
    directory = Path(os.environ["TMPDIR"])
    assert directory == ROOT / ".p1tmp"
    assert os.environ["VLLM_RPC_BASE_PATH"] == str(directory)
    assert directory.is_relative_to(ROOT) and not directory.is_symlink()
    filename = str(uuid4())
    path = directory / filename
    limit = getattr(zmq, "IPC_PATH_MAX_LEN", 107)
    assert len(os.fsencode(str(path))) <= limit
    old = ROOT / "experiments/prefix_io_v1/runtime-tmp" / filename
    assert len(os.fsencode(str(old))) > limit
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    try:
        socket.bind("ipc://" + str(path))
        assert socket.getsockopt_string(zmq.LAST_ENDPOINT) == "ipc://" + str(path)
    finally:
        socket.close(linger=0)
        context.term()
        if path.exists():
            assert path.parent == ROOT / ".p1tmp"
            path.unlink()


def test_tmp_symlink_rejected(monkeypatch, tmp_path):
    set_env(monkeypatch)
    monkeypatch.setattr(smoke, "ROOT", tmp_path)
    target = tmp_path / "other"
    target.mkdir()
    (tmp_path / ".p1tmp").symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="must not be a symlink"):
        smoke.configure_runtime_environment()
