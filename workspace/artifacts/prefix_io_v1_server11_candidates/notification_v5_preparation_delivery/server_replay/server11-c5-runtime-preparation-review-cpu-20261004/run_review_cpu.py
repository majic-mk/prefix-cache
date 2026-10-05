"""Independent append-only CPU semantic review; never starts native/GPU work."""
from __future__ import annotations
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

def ref(path):
 data=path.read_bytes()
 return dict(path=str(path.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--preparation-root',type=Path,required=True)
 parser.add_argument('--preparation-lock-sha256',required=True)
 parser.add_argument('--candidate-root',type=Path,required=True)
 parser.add_argument('--native-root',type=Path,required=True)
 parser.add_argument('--race-review-root',type=Path,required=True)
 parser.add_argument('--author-root',type=Path,required=True)
 parser.add_argument('--output-dir',type=Path,required=True)
 parser.add_argument('--location',choices=('local_windows','server_cpu'),required=True)
 args=parser.parse_args()
 prep,candidate,native,race,author=(x.resolve(strict=True) for x in
  (args.preparation_root,args.candidate_root,args.native_root,args.race_review_root,args.author_root))
 lockpath=prep/'PREPARATION_SOURCE_LOCK.json';lockbytes=lockpath.read_bytes()
 if hashlib.sha256(lockbytes).hexdigest()!=args.preparation_lock_sha256:raise ValueError('preparation lock hash mismatch')
 lock=json.loads(lockbytes)
 if lock['gpu_launch_allowed'] is not False or lock['gpu_qualified'] is not False:raise ValueError('CPU-only lock required')
 paths={lockpath,Path(__file__).resolve()}
 for row in lock['files']:
  rel=Path(row['path'])
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source reference')
  path={'preparation':prep,'candidate':candidate}[row['scope']]/rel
  actual=ref(path)
  if actual['sha256']!=row['sha256'] or actual['bytes']!=row['bytes']:raise ValueError('preparation source mismatch: '+str(path))
  paths.add(path)
 for name in ('test_frozen_entry_constraints.py','test_runtime_preparation_adversarial.py'):
  paths.add(HERE/name)
 paths.update(native.glob('*.py'))
 paths.update((race/'frozen_observer_sources').glob('*.py'))
 paths.update((author/'third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control').rglob('*.py'))
 before=[ref(p) for p in sorted(paths)]
 os.environ.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
  SERVER11_C5_PREPARATION=str(prep),SERVER11_C5_CANDIDATE=str(candidate),
  SERVER11_C5_CANDIDATE_ROOT=str(candidate),SERVER11_NATIVE_V6_ROOT=str(native),
  SERVER11_NOTIFICATION_RACE_REVIEW=str(race),SERVER11_AUTHOR_SOURCE_ROOT=str(author))
 suite=unittest.TestSuite()
 for name in ('test_frozen_entry_constraints','test_runtime_preparation_adversarial'):
  spec=importlib.util.spec_from_file_location('_independent_review_'+name,HERE/(name+'.py'))
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
 stream=io.StringIO();started=time.monotonic_ns()
 result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
 elapsed=time.monotonic_ns()-started
 after=[ref(p) for p in sorted(paths)]
 forbidden=[n for n in sys.modules if n=='torch' or n.startswith('torch.') or n=='vllm' or n.startswith('vllm.')]
 status='PASS' if result.wasSuccessful() and before==after and not forbidden else 'FAIL'
 doc=dict(status=status,passed=result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
  failed=len(result.failures)+len(result.errors),skipped=len(result.skipped),tests=result.testsRun,
  elapsed_ns=elapsed,location=args.location,command=[sys.executable]+sys.argv,
  python=sys.version,platform=platform.platform(),source_before=before,source_after=after,
  sources_unchanged=before==after,preparation_source_lock_sha256=args.preparation_lock_sha256,
  candidate_reactor_sha256='a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47',
  candidate_collector_sha256='bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf',
  forbidden_modules_imported=forbidden,gpu_workloads_run=0,ssh_connections=0,
  gpu_qualified=False,native_execution_verified=False,performance_claim=False,
  timing_qualification=False,actual_vllm_collector_install=False,
  fixture_scope='Actual frozen AST/control/collector and original Queue; synthetic events/native backend metadata',
  old_cpu_score_covers_new_runtime=False)
 args.output_dir.mkdir(parents=True,exist_ok=False)
 (args.output_dir/'TEST_STDERR.log').write_text(stream.getvalue(),encoding='utf-8')
 (args.output_dir/'TEST_RESULT.json').write_text(json.dumps(doc,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps({k:v for k,v in doc.items() if k not in ('source_before','source_after')},sort_keys=True))
 return 0 if status=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
