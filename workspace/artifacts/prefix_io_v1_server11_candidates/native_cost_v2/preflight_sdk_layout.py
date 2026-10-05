"""Actual installed SDK/optional-source preflight in the exact child layout.

Uses the unmodified frozen G2 helpers, existing source assets and original SDK
path guard. No framework import, model construction, GPU initialization,
compiler, download, installed-package change or library loading is performed.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
RUN='experiments/prefix_io_v1/runs/server11-native-cost-v2-cpu-preflight'
COMMON='artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py'
COMMON_PIN=dict(path=COMMON,bytes=46746,sha256='82cea015522596a89a89eb1ea0245f9ea111c942b74456cd87288507aec54422')


def require(ok,message):
    if not ok:raise ValueError(message)


def source(root,ref):
    path=root/ref['path']
    require(path.resolve().is_relative_to(root) and not path.is_symlink() and path.is_file(), 'canonical source')
    raw=path.read_bytes()
    require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'exact source bytes')
    return path,raw


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-lock',required=True)
    parser.add_argument('--source-lock-sha256',required=True)
    args=parser.parse_args(argv)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU preflight must hide all GPUs')
    require(os.name=='posix' and ROOT.resolve(strict=True)==ROOT,'exact server project')
    lock_path=ROOT/args.source_lock
    require(lock_path.resolve().is_relative_to(ROOT) and lock_path.is_file() and not lock_path.is_symlink(), 'real lock')
    raw=lock_path.read_bytes();require(hashlib.sha256(raw).hexdigest()==args.source_lock_sha256,'actual lock SHA')
    document=json.loads(raw);refs={row['path']:row for row in document['files']}
    require(len(refs)==len(document['files']) and refs.get(COMMON)==COMMON_PIN,'original helper in frozen lock')
    path,raw=source(ROOT,COMMON_PIN)
    spec=importlib.util.spec_from_file_location('_server11_actual_cpu_sdk_common',path)
    common=importlib.util.module_from_spec(spec);sys.modules[spec.name]=common
    exec(compile(raw,str(path),'exec',dont_inherit=True),common.__dict__)
    base=common.load_ref(ROOT,refs[common.SMOKE])
    run=ROOT/RUN;require(run.parent.is_dir() and not run.exists(),'new approved CPU evidence directory')
    run.mkdir(exist_ok=False);out=run/'details';out.mkdir(exist_ok=False)
    before=set(sys.modules);old_environment=dict(os.environ)
    result=dict(status='FAILED_ACTUAL_CPU_SDK_CHILD_LAYOUT',source_lock=args.source_lock,
        source_lock_sha256=args.source_lock_sha256,out=str(out),GPU_operations=0,
        compiler_executions=0,framework_imports=0,installed_packages_modified=False,
        system_or_driver_modified=False,downloads=False,library_loaded=False)
    try:
        result['cache_environment']=common.configure_process_caches(out,base)
        result['sdk_environment']=common.prepare_process_sdk(ROOT,refs,out)
        result['ninja_environment']=common.prepare_process_ninja(ROOT,refs,out,result['sdk_environment'])
        optional,absence=common.load_optional_probe(ROOT,refs)
        result['optional_absence']=absence
        require(not any(name=='torch' or name.startswith('torch.') or name=='vllm' or name.startswith('vllm.')
                for name in set(sys.modules)-before),'no backend imported by actual helpers')
        require(Path(result['sdk_environment']['overlay'])==out/'runtime-cache/cuda13-sdk','actual original SDK layout')
        result['status']='PASS_ACTUAL_CPU_ORIGINAL_SDK_CHILD_LAYOUT_AND_HELPERS'
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error_message=str(exc),traceback=traceback.format_exc(limit=12))
    finally:
        os.environ.clear();os.environ.update(old_environment)
        result['environment_restored']=dict(os.environ)==old_environment
        write(out/'CPU_ACTUAL_HELPER_RESULT.json',result)
    print(json.dumps({key:result[key] for key in ('status','out','GPU_operations','compiler_executions','framework_imports','environment_restored')}))
    return 0 if result['status'].startswith('PASS_') else 1


if __name__=='__main__':raise SystemExit(main())
