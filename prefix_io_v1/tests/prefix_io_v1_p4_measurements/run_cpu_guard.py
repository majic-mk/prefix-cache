"""No CUDA import/initialization during this CPU qualification."""
from pathlib import Path
import builtins
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
original_import=builtins.__import__
def cpu_import(name,*args,**kwargs):
    if name.split(".")[0] in ("torch","cuda","cupy","pycuda","vllm","py_kvcache"):
        raise AssertionError("GPU/backend import denied by CPU measurement guard: "+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=cpu_import
import pytest
dest=Path(sys.argv[1])
dest.mkdir(parents=True,exist_ok=True)
exit_code=pytest.main(["-q","tests/prefix_io_v1_p4_measurements",
                       "--junitxml="+str(dest/"tests.xml")])
guard={"cuda_initialized":False,"gpu_workloads_run":0,"backend_imports_denied":True,
       "test_process_exit":exit_code}
(dest/"cpu-guard.json").write_text(json.dumps(guard,sort_keys=True,indent=2)+"\n")
raise SystemExit(exit_code)
