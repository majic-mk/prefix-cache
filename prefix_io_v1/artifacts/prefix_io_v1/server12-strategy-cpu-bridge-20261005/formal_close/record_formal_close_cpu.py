"""Append local CPU test and CLI evidence; not an actual parent proof."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def append(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        if isinstance(value, dict):
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        else:
            stream.write(value)


def source(path):
    raw = path.read_bytes()
    return dict(path=path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(tag, tail, expected):
    command = [sys.executable, '-B', '-I', '-S', *tail]
    value = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='strict')
    append(HERE / (tag + '_STDOUT.log'), value.stdout)
    append(HERE / (tag + '_STDERR.log'), value.stderr)
    count = re.search(r'Ran ([0-9]+) tests', value.stderr)
    result = dict(command=command, exit_code=value.returncode, expected_exit_code=expected,
                  status='PASS_CPU_ONLY_EXPECTED_RESULT' if value.returncode == expected else 'FAILED_CPU_CHECK',
                  CPU_test_count=None if count is None else int(count.group(1)),
                  actual_GPU_operations=0, actual_model_executions=0,
                  actual_formal_parent_deliveries=0, test_checker_mock_only=True,
                  gpu_eligible=False, formal_effect_qualified=False)
    append(HERE / (tag + '_RESULT.json'), result)
    assert value.returncode == expected
    return result


test = run('LOCAL_CPU_TEST', [str(HERE / 'test_close_formal_peer_cpu.py')], 0)
help_result = run('LOCAL_CLI_HELP', [str(HERE / 'close_formal_peer_cpu.py'), '--help'], 0)
blocked = run('LOCAL_CLI_UNBOUND', [str(HERE / 'close_formal_peer_cpu.py'), '--project-root', str(HERE),
    '--source-lock', 'CPU_INPUTS_UNBOUND_DO_NOT_EXIST.json', '--child', 'CPU_INPUTS_UNBOUND_CHILD.json',
    '--guard', 'CPU_INPUTS_UNBOUND_GUARD.json', '--config', 'CPU_INPUTS_UNBOUND_CONFIG.json',
    '--output-relative', 'artifacts/prefix_io_v1/CPU_NEVER_CREATE/FORMAL_PARENT.json'], 78)
assert not (HERE / 'artifacts/prefix_io_v1/CPU_NEVER_CREATE').exists()
manifest = dict(schema='formal_close_CPU_source_manifest_v1', actual_GPU_operations=0,
                actual_formal_parent_deliveries=0, mock_unit_test_receipts_are_not_native_evidence=True,
                gpu_eligible=False, formal_effect_qualified=False,
                files=[source(path) for path in sorted(HERE.iterdir())
                       if path.is_file() and path.name != 'SOURCE_MANIFEST.json'])
append(HERE / 'SOURCE_MANIFEST.json', manifest)
print(json.dumps(dict(test=test, help=help_result, UNBOUND=blocked,
                     producer_ref=source(HERE / 'close_formal_peer_cpu.py'),
                     manifest_ref=source(HERE / 'SOURCE_MANIFEST.json')), ensure_ascii=True))
