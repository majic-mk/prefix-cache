"""Read-only approved storage usage/preflight snapshot for P4-02."""
from pathlib import Path
import sys,json,time,datetime,shutil,hashlib
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/prefix_io_v1/scripts'))
sys.path.insert(0,str(ROOT/'third_party/work/prefix-io-p4-02-cpu/src'))
from experiment_storage import permission,usage,budget_values,preflight
OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu'
start=time.monotonic();p=permission(ROOT);aux=p['approved_auxiliary_storage']
root=Path(aux['root']);consumed=usage(root);free=shutil.disk_usage(root).free
floor=8*1024**3;reserve=3*1024**3;primary_free=shutil.disk_usage(ROOT/'experiments/prefix_io_v1/runs').free
def fit(n):
 try:return dict(allowed=True,preflight=budget_values(consumed['used_bytes'],free,n,aux['max_bytes'],aux['minimum_free_bytes']))
 except ValueError as e:return dict(allowed=False,error=str(e),deficit_cap=max(0,consumed['used_bytes']+n-aux['max_bytes']),
     deficit_floor=max(0,aux['minimum_free_bytes']+n-free))
result=dict(schema_version=1,status='CPU_READ_ONLY_REAL_METADATA_USAGE_SNAPSHOT',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,
 PRIMARY=dict(free_bytes=primary_free,minimum_free_bytes=floor,native_128MiB_fit=primary_free-128*1024**2>=floor,
    model_3GiB_fit=primary_free-reserve>=floor,model_3GiB_deficit=max(0,floor+reserve-primary_free)),
 AUX=dict(root=str(root),**consumed,free_bytes=free,max_bytes=aux['max_bytes'],minimum_free_bytes=aux['minimum_free_bytes'],
    current=fit(0),native_128MiB=fit(128*1024**2),model_3GiB=fit(reserve)),
 path_walk_scope='Existing approved auxiliary root metadata only; no symlinks followed, no payload bytes read',
 GPU_initialized=False,GPU_runs=0,cache_links_or_writes=0,cache_deletion_or_move=0,model_payload_read=False)
dest=OUT/'storage-current-final.json'
with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
