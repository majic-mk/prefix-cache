"""Record source and installed package metadata without loading any GPU backend."""
from pathlib import Path
import sys,json,hashlib,datetime,subprocess,importlib.metadata
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu'
def ref(p):
 return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def head(p):
 return subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
expected=['torch','vllm','py_kvcache','cupy']
assert not any(x in sys.modules for x in expected)
result=dict(schema_version=1,status='SOURCE_AND_PACKAGE_METADATA_LOCK_GPU_ABI_UNQUALIFIED',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),project=str(ROOT),server='connect.westd.seetacloud.com:20739',
 project_HEAD=head(ROOT),native_author_HEAD=head(ROOT/'third_party/work/py-kvcache-p4-02-cpu'),
 author_vllm_HEAD=head(ROOT/'third_party/work/vllm-author-p4-02-cpu'),python=sys.version,python_executable=sys.executable,
 package_metadata={x:importlib.metadata.version(x) for x in ('torch','pytest','numpy')},
 source_locks=[ref(OUT/n) for n in ('test-input-lock.json','gpu-source-lock.json')],
 P3_protected_lock=ref(ROOT/'artifacts/prefix_io_v1/server08-p3-16/execution-lock-12-final-p3.json'),
 permissions=ref(ROOT/'experiments/prefix_io_v1/configs/permissions.yaml'),
 GPU_ledger=ref(ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'),
 native='third_party/work/py-kvcache-p4-02-cpu',control='third_party/work/prefix-io-p4-02-cpu/src',
 author='third_party/work/vllm-author-p4-02-cpu',binary_fallback='third_party/work/vllm-author-build',
 runtime_supplement=ref(OUT/'gpu-next-day/AUTHOR_PYTHON_OVERLAY_COPY_92.json'),
 GPU_or_backend_imported=False,new_GPU_runs=0,binary_ABI_qualified=False,
 driver_or_system_changes=0,downloads=0,build_or_install=0)
assert not any(x in sys.modules for x in expected)
with (OUT/'P4_02_VERSION_LOCK.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
