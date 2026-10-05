"""Explicit-root, source-locked independent CPU contract review. No native run."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

HERE=Path(__file__).resolve().parent

def sha(data):return hashlib.sha256(data).hexdigest()
def ref(path):
 data=path.read_bytes();return dict(path=str(path.resolve()),bytes=len(data),sha256=sha(data))
def require(value,reason):
 if not value:raise ValueError(reason)
def unique(pairs):
 result={}
 for key,value in pairs:
  require(key not in result,'duplicate JSON key');result[key]=value
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('project-root','author-root','factory-root','native-root','candidate-root','preparation-root','source-lock','output-dir'):
  parser.add_argument('--'+name,type=Path,required=True)
 parser.add_argument('--source-lock-sha256',required=True)
 parser.add_argument('--location',choices=('local_windows','server_cpu'),required=True)
 args=parser.parse_args()
 project=args.project_root.resolve(strict=True);author=args.author_root.resolve(strict=True)
 factory=args.factory_root.resolve(strict=True);native=args.native_root.resolve(strict=True)
 candidate=args.candidate_root.resolve(strict=True);preparation=args.preparation_root.resolve(strict=True)
 lockpath=args.source_lock.resolve(strict=True);lockbytes=lockpath.read_bytes()
 require(sha(lockbytes)==args.source_lock_sha256,'independently supplied source lock SHA')
 lock=json.loads(lockbytes,object_pairs_hook=unique)
 require(lock['schema']=='c5_cost_binding_cpu_source_lock_v1','CPU cost-binding closure schema')
 require(lock.get('gpu_launch_allowed') is False and lock.get('native_execution_verified') is False,'CPU-only source lock')
 paths={lockpath,Path(__file__).resolve()};seen=set()
 for row in lock['files']:
  name=row['path'];require(type(name) is str and not any(c in name for c in ('\\',':','\0')) and not name.startswith('/')
   and all(p not in ('','.','..') for p in name.split('/')),'relative source path')
  require(name not in seen,'duplicate source path');seen.add(name)
  path=project/name;require(path.resolve().is_relative_to(project),'source escaped project')
  actual=ref(path);require(type(row['bytes']) is int and actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'source drift: '+name)
  paths.add(path)
 for root in (factory,native,HERE):paths.update(root.glob('*.py'))
 paths.update(candidate.rglob('*.py'))
 paths.update(preparation.glob('*.py'))
 original=author/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_paired_measurement_verifier.py'
 paths.add(original)
 before=[ref(path) for path in sorted(paths)]
 os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
  SERVER11_NATIVE_V6_ROOT=str(native),SERVER11_AUTHOR_SOURCE_ROOT=str(author),SERVER11_C5_CANDIDATE_ROOT=str(candidate),
  SERVER11_C5_COST_FACTORY_ROOT=str(factory),SERVER11_C5_COST_NATIVE_ROOT=str(native),
  SERVER11_C5_COST_CANDIDATE_ROOT=str(candidate),SERVER11_C5_COST_PREPARATION_ROOT=str(preparation),
  NATIVE_ORIGINAL_ESTIMATOR=str(original))
 suite=unittest.TestSuite()
 for name in ('test_original_cost_boundaries','test_cost_binding_attacks'):
  spec=importlib.util.spec_from_file_location('_independent_cost_review_'+name,HERE/(name+'.py'))
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
 stream=io.StringIO();started=time.monotonic_ns()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);elapsed=time.monotonic_ns()-started
 after=[ref(path) for path in sorted(paths)]
 forbidden=[n for n in sys.modules if n=='torch' or n.startswith('torch.') or n=='vllm' or n.startswith('vllm.')]
 okay=result.wasSuccessful() and before==after and not forbidden
 document=dict(status='PASS' if okay else 'FAIL',tests=result.testsRun,
  passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
  failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),gpu_runs=0,
  source_before=before,source_after=after,source_count=len(before),source_lock_sha256=sha(lockbytes),
  source_lock_files=len(lock['files']),sources_unchanged=before==after,
  location=args.location,elapsed_ns=elapsed,python=sys.version,platform=platform.platform(),
  command=[sys.executable,'-B','-I','-S']+sys.argv,forbidden_modules_imported=forbidden,
  origin='synthetic_cpu_contract',actual_vllm_collector_install=False,native_execution_verified=False,
  native_cost_qualified=False,full_runtime_cost_qualified=False,on_observation_cost_measured=False,
  gpu_launch_allowed=False,performance_claim=False,effective_cost_upper_ns=None,effective_step_budget_ns=None)
 args.output_dir.mkdir(parents=True,exist_ok=False)
 (args.output_dir/'TEST_STDERR.log').write_text(stream.getvalue(),encoding='utf-8')
 (args.output_dir/'TEST_RESULT.json').write_text(json.dumps(document,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in document.items() if k not in ('source_before','source_after')},sort_keys=True))
 return 0 if okay else 1

if __name__=='__main__':raise SystemExit(main())
