from pathlib import Path
import hashlib,sys,json,builtins
root=Path(".").resolve();tool=root/"experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py"
assert hashlib.sha256(tool.read_bytes()).hexdigest()=="e41f5c2300d1b4dd4cf0724521cd95c049dc54dcdc49055442fe4f7383c43c30"
out=root/"artifacts/prefix_io_v1/server08-p3-16";audit=out/"private-cache-merge-manifest.json";journal=out/"private-cache-merge-apply-journal.jsonl"
assert hashlib.sha256(audit.read_bytes()).hexdigest()=="e692ad4495548d9729423d956ceea405bbcac4202d5b9e50bee2db4ac424e062" and not journal.exists()
auth=root/"experiments/prefix_io_v1/configs/authorizations/server08_p316_archive_dedup.json"
assert hashlib.sha256(auth.read_bytes()).hexdigest()=="8b7d58b0379203e0d7e7e663d1c1808eb81a9d03a3c7885b5be3bc931599ecab"
assert json.loads(auth.read_text())["approved"] is True
sys.path[:0]=[str(root/"experiments/prefix_io_v1/scripts"),str(root/"src"),str(root/".venv/lib/python3.12/site-packages")]
original_import=builtins.__import__
def guarded_import(name,*a,**k):
 if name.split(".")[0] in {"torch","vllm","py_kvcache"}:raise AssertionError("GPU libraries forbidden in cache dryrun")
 return original_import(name,*a,**k)
builtins.__import__=guarded_import
import subprocess
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
import consolidate_private_cache_copies as m
assert m.__file__==str(tool)
m.AUX=root/"experiments/prefix_io_v1"
apply="--apply" in sys.argv[1:]
sys.argv=[str(tool),"--audit",str(audit),"--journal",str(journal),"--authorization",str(auth)]+(["--apply"] if apply else [])
result=m.main()
assert result==0 and (journal.exists() if apply else not journal.exists())
assert all(name not in sys.modules for name in ["torch","vllm","py_kvcache"])
