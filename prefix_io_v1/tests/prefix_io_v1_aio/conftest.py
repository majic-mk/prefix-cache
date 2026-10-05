import json
import os
from pathlib import Path
import pytest
import torch

@pytest.fixture(scope="session", autouse=True)
def forbid_cuda_initialization():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    original = torch.cuda._lazy_init
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU qualification must not initialize CUDA")
    torch.cuda._lazy_init = forbidden
    try:
        yield
    finally:
        torch.cuda._lazy_init = original
        assert not torch.cuda.is_initialized()
        path = os.environ.get("AIO_CPU_GPU_GUARD_PATH")
        if path:
            Path(path).write_text(json.dumps({"cuda_visible_devices":"",
                "cuda_initialized":False,"gpu_workloads_run":0},indent=2))
