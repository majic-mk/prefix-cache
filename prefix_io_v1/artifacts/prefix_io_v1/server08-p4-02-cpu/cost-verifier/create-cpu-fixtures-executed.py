from pathlib import Path
import sys, json, builtins, hashlib
root=Path.cwd()
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split(".")[0] in ("torch","cuda","cupy","pycuda","vllm","py_kvcache"):
        raise RuntimeError("GPU/backend import forbidden for CPU fixture creation")
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
sys.path.insert(0,str(root/"tests/prefix_io_v1_p4_measurements"))
from test_semantic_verifier import Fixture
from prefix_io_control.p4_paired_measurement_verifier import write_semantic_verification_manifest
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/cost-verifier"
results=[]
for name,origin,basis in (
    ("cpu_fixture","cpu_fixture","existing_io_plus_delta"),
    ("forged_native_origin","native_gpu_recording","existing_io_plus_delta"),
    ("no_io_plus_joint","cpu_fixture","no_io_plus_joint")):
    dest=out/"fixtures"/name
    dest.mkdir(parents=True,exist_ok=False)
    f=Fixture(dest,origin=origin,basis=basis)
    p=f.load()
    ref=write_semantic_verification_manifest(dest,p.verification,"semantic-verification.json")
    c=p.verification.cells[0].cost
    results.append({"fixture_root":str(dest.relative_to(root)), "declared_origin":origin,
       "actual_execution":"CPU_SYNTHETIC_RAW_MEASUREMENT_FIXTURE",
       "status":p.status,"gpu_verified":p.gpu_verified,"production_qualified":p.production_qualified,
       "production_lookup":p.lookup(c.load_signature,c.existing_io,c.stage,c.physical_bytes),
       "total_fixture_ns":c.total_ns,"semantic_receipt_ref":dict(path=ref.path,bytes=ref.bytes,sha256=ref.sha256)})
(out/"fixtures-index.json").write_text(json.dumps(results,indent=2)+"\n")
print(json.dumps({"fixture_count":len(results),"gpu_runs":0,"production_lookup":[r["production_lookup"] for r in results]}))
