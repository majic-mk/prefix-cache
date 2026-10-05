"""Execute the real offline CLI in a backend-denied CPU process."""
from pathlib import Path
import builtins
import json
import os
import runpy
import sys
target=Path(sys.argv[1])
report=Path(sys.argv[2])
argv=sys.argv[3:]
root=target.parents[3]
sys.path[:0]=[str(root/"third_party/work/prefix-io-p4-02-cpu/src"),
              str(target.parent),str(root)]
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["PYTHONDONTWRITEBYTECODE"]="1"
attempts=[]
original=builtins.__import__
def denied(name,*args,**kwargs):
    if name.split(".")[0] in ("torch","cuda","cupy","pycuda","vllm","py_kvcache"):
        attempts.append(name)
        raise AssertionError("GPU/backend import denied: "+name)
    return original(name,*args,**kwargs)
builtins.__import__=denied
sys.argv=[str(target),*argv]
exit_code=0
error=None
try:
    runpy.run_path(str(target),run_name="__main__")
except SystemExit as exc:
    exit_code=0 if exc.code is None else int(exc.code)
except BaseException as exc:
    exit_code=1
    error={"type":type(exc).__name__,"message":str(exc)}
report.write_text(json.dumps({"target":str(target),"argv":argv,"exit":exit_code,
    "error":error,"backend_import_attempts":attempts,"GPU_runs":0,"cuda_initialized":False,
    "production_qualified":False},sort_keys=True,indent=2)+"\n")
if error:print(json.dumps(error),file=sys.stderr)
raise SystemExit(exit_code)
