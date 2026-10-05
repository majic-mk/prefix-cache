"""Reuse the pinned memory-only SSH worker for the device compatibility revision."""
import importlib.util, json
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent
p=BASE/'notification_v5_gpu_entry_delivery/rpc_local.py'
s=importlib.util.spec_from_file_location('sealed_entry_rpc',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.DIRS['candidate']=m.BASE+'server11-c5-gpu-entry-path-revision-20261004'
m.DIRS['delivery']=m.BASE+'server11-c5-gpu-entry-path-delivery-20261004'
if __name__=='__main__':
 import sys
 if sys.argv[1]=='upload':
  scope=sys.argv[2];local=Path(sys.argv[3])
  names=[p.relative_to(local).as_posix() for p in local.rglob('*') if p.is_file()
         and '__pycache__' not in p.parts and (p.suffix in ('.py','.md')
         or p.name in ('BASE_PACKAGE_COPY.json','CANDIDATE_MANIFEST.json','SOURCE_INHERITANCE.json','NATIVE_SOURCE_INHERITANCE.json'))]
  result=m.upload(local,scope,sorted(names))
 elif sys.argv[1]=='run':
  result=m.run_logged(sys.argv[2],sys.argv[3],sys.argv[4:],timeout=120)
 elif sys.argv[1]=='download':
  result=m.rpc(dict(op='download',remote=sys.argv[2],local=sys.argv[3],timeout=60))
 else:raise ValueError('only explicit bounded helper actions')
 print(json.dumps(result,sort_keys=True))

