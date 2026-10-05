"""Run selected tests with CUDA initialization forbidden; no GPU qualification."""
import json
import os
import sys
from pathlib import Path
import torch
import pytest
assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
assert not torch.cuda.is_initialized()
def forbidden(*args,**kwargs):
    raise AssertionError("CPU-only run attempted CUDA initialization")
torch.cuda._lazy_init=forbidden
code=pytest.main(sys.argv[1:])
assert not torch.cuda.is_initialized()
path=os.environ.get("AIO_CPU_GPU_GUARD_PATH")
if path:Path(path).write_text(json.dumps({"cuda_initialized":False,"gpu_workloads_run":0,"pytest_exit":int(code)},indent=2))
raise SystemExit(code)
