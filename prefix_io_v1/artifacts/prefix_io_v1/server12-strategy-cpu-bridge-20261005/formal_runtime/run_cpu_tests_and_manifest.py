"""Record one local CPU run and exact source hashes. No GPU evidence issued."""
import ast
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PREP_BASE = HERE.parent.parent
PREP = PREP_BASE / 'gpu_prerental_preparation_20261004'
if not PREP.is_dir():
    PREP = PREP_BASE / 'server12-gpu-prerental-preparation-20261004'


def closed(path):
    raw = path.read_bytes()
    return dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def append(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        if isinstance(value, dict):
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        else:
            stream.write(value)


parser = argparse.ArgumentParser()
parser.add_argument('--revision-tag', default='')
args = parser.parse_args()
assert args.revision_tag == '' or re.fullmatch('[0-9]{2}', args.revision_tag)
tag = '_' + args.revision_tag if args.revision_tag else ''
def evidence(name):
    item = Path(name)
    return HERE / (item.stem + tag + item.suffix)

command = [sys.executable, '-B', '-I', '-S', str(HERE / 'test_formal_runtime_cpu.py')]
run = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='strict')
append(evidence('LOCAL_CPU_TEST_STDOUT.log'), run.stdout)
append(evidence('LOCAL_CPU_TEST_STDERR.log'), run.stderr)
count = re.search(r'Ran ([0-9]+) tests', run.stderr)
assert count is not None
result = dict(schema='formal_runtime_v3_CPU_branch_tests_v1', command=command, exit_code=run.returncode,
              status='PASS_CPU_BRANCH_SOURCE_RESTORE_ONLY' if run.returncode == 0 else 'FAILED_CPU_TESTS',
              CPU_test_count=int(count.group(1)), actual_GPU_operations=0, actual_native_model_executions=0,
              actual_private_table_or_GPU_frames_generated_by_tests=False,
              formal_effect_qualified=False, gpu_eligible=False)
append(evidence('LOCAL_CPU_TEST_RESULT.json'), result)
old = PREP / 'runner/native_runtime_v2.py'
assert closed(old)['sha256'] == '26b6b8b2e704c09d066963013d4124edf19098dfe1a7085d9de3535a9cddd1c5'
old_tree, new_tree = ast.parse(old.read_bytes()), ast.parse((HERE / 'native_runtime_v3.py').read_bytes())
old_functions = {node.name: node for node in old_tree.body if isinstance(node, ast.FunctionDef)}
new_functions = {node.name: node for node in new_tree.body if isinstance(node, ast.FunctionDef)}
old_finally = next(node for node in old_functions['execute'].body if isinstance(node, ast.Try)).finalbody
new_finally = next(node for node in new_functions['execute'].body if isinstance(node, ast.Try)).finalbody
assert [ast.dump(row) for row in old_finally] == [ast.dump(row) for row in new_finally]
review = dict(schema='formal_runtime_v3_incremental_CPU_source_review_v1', status='PASS_SOURCE_AND_BRANCH_REVIEW_ONLY',
              ancestor_ref=closed(old), old_frozen_bytes_unchanged=True,
              same_original_shutdown_and_restore_finally_AST=True,
              changed_original_functions=['execute'], added_helpers=sorted(set(new_functions) - set(old_functions)),
              formal_roles=[['development', 'off', 'U'], ['development', 'shadow', 'I'],
                            ['effect', 'off', 'U'], ['effect', 'on', 'I']],
              U_constructs_I_controller=False, original_request_driver_calls=1,
              complete_CUDA_and_output_verifier='unchanged original reserve_join._capture',
              actual_GPU_operations=0, formal_effect_qualified=False, gpu_eligible=False,
              remaining_real_prerequisites=['actual formal natural input and CPU tokenizer provenance',
                                           'independent prospective deadline and service SLO',
                                           'guard-closed same-formal-input development U raw evidence',
                                           'actual qualified I development host reserve',
                                           'all current frozen/runtime/live-device original guard gates'])
append(evidence('FORMAL_RUNTIME_CPU_REVIEW.json'), review)
manifest = dict(schema='formal_runtime_v3_local_source_manifest_v1', actual_GPU_operations=0,
                gpu_eligible=False, files=[closed(path) for path in sorted(HERE.iterdir())
                                          if path.is_file() and path != evidence('SOURCE_MANIFEST.json')])
append(evidence('SOURCE_MANIFEST.json'), manifest)
print(json.dumps(dict(result=result, runtime_ref=closed(HERE / 'native_runtime_v3.py'),
                     manifest_ref=closed(evidence('SOURCE_MANIFEST.json'))), ensure_ascii=True))
raise SystemExit(run.returncode)
