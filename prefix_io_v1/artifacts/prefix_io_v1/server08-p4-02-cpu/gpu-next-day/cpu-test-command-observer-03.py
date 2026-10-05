import builtins,sys,pathlib,json,hashlib,os
root=pathlib.Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
sys.path.insert(0,str(root/"experiments/prefix_io_v1/scripts"))
os.chdir(root)
ledger=root/"experiments/prefix_io_v1/gpu-budget-ledger.json"
before=hashlib.sha256(ledger.read_bytes()).hexdigest()
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split(".")[0] in {"torch","vllm","py_kvcache","cupy","pycuda","cuda"}:raise AssertionError("CPU guard rejects backend import "+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
import pytest
exitcode=pytest.main(["-q","--import-mode=importlib","tests/prefix_io_v1_p4_observation","-k","boundary_tie or original_invalidate or host_clock_backwards","--junitxml=artifacts/prefix_io_v1/server08-p4-02-cpu/observation-cpu/run-03-review-fixes/tests.xml"])
after=hashlib.sha256(ledger.read_bytes()).hexdigest()
result=dict(GPU_modules_imported=sorted(set(sys.modules)&{"torch","vllm","py_kvcache","cupy"}),gpu_runs=0,gpu_initialized=False,CPU_import_guard=True,ledger_before=before,ledger_after=after,ledger_unchanged=before==after,exit=exitcode)
(root/"artifacts/prefix_io_v1/server08-p4-02-cpu/observation-cpu/run-03-review-fixes/CPU_GUARD.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result));raise SystemExit(exitcode)
