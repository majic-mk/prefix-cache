"""Reuse authenticated append-only transfer code; never retain credentials."""
import importlib.util
import json
from pathlib import Path
import sys

PARENT=Path(__file__).resolve().parent.parent/'notification_v5_native_cost_preparation_delivery/rpc_local.py'
spec=importlib.util.spec_from_file_location('_c5_sealed_transfer',PARENT)
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
ROOT=T.ROOT;PYTHON=T.PYTHON;BASE=T.BASE
T.DIRS={k:BASE+'server11-c5-combined-runtime'+('' if k=='candidate' else '-'+k)+'-cpu-20261004'
        for k in ('candidate','review','resource','delivery')}
DIRS=T.DIRS;rpc=T.rpc;remote_python=T.remote_python;upload=T.upload;run_logged=T.run_logged

if __name__=='__main__':
    action=sys.argv[1]
    if action=='upload':
        local,scope=sys.argv[2:4];p=Path(local)
        names=[str(f.relative_to(p)).replace('\\','/') for f in p.rglob('*')
               if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.md')]
        if (p/'COMBINED_SOURCE_INHERITANCE.json').is_file():names.append('COMBINED_SOURCE_INHERITANCE.json')
        result=upload(p,scope,sorted(names))
    elif action=='run':result=run_logged(sys.argv[2],sys.argv[3],sys.argv[4:])
    elif action=='download':result=rpc(dict(op='download',remote=sys.argv[2],local=sys.argv[3],timeout=60))
    else:raise ValueError('action')
    print(json.dumps(result,sort_keys=True))
