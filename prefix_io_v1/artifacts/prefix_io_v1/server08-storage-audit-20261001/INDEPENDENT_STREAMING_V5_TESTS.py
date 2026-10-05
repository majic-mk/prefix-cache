"""Isolated local review tests: never runs cleanup.run or any deletion API.

All created fixtures remain under this audit directory. st_blocks is supplied
for Windows metadata tests; the actual inventory and merge functions are used.
"""
from pathlib import Path
import ast
import collections
import hashlib
import json
import os
import random
import stat
import types

AUDIT = Path(__file__).resolve().parent
SCRIPT = AUDIT / 'EXECUTE_APPROVED_CPU_TEMP_CLEANUP_V5.py'
EXPECTED_SHA = 'a05dd423461899c271663c7ed7823cfc8809cbaf5f5f19e4b160e9f5b5fcbf4f'
TEST_ROOT = AUDIT / 'independent_streaming_v5_isolated_fixtures'
TEST_ROOT.mkdir(exist_ok=False)
source = SCRIPT.read_text(encoding='utf-8')
assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == EXPECTED_SHA
compile(ast.parse(source), str(SCRIPT), 'exec')
ns = {'__name__': 'independent_streaming_v5_review', '__file__': str(SCRIPT)}
exec(compile(source, str(SCRIPT), 'exec'), ns)
results = []


def actual_sha(path, expected=None):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, (str(path), digest, expected)
    return digest


def windows_stat_meta(s):
    # Only supplement missing Windows st_blocks. Preserve other real fields.
    supplemented = types.SimpleNamespace(
        st_dev=s.st_dev, st_ino=s.st_ino, st_mode=s.st_mode,
        st_nlink=s.st_nlink, st_size=s.st_size,
        st_blocks=getattr(s, 'st_blocks', (s.st_size + 511) // 512),
        st_mtime_ns=s.st_mtime_ns)
    return original_meta(supplemented)


original_meta = ns['meta']
ns['meta'] = windows_stat_meta
ns['stable_sha'] = actual_sha
original_path = Path
project = TEST_ROOT / 'metadata_project'
aux = TEST_ROOT / 'metadata_AUX'
audit_out = project / '.current_audit'
candidate = project / 'experiments' / 'approved_candidate'
for directory in (audit_out, candidate, aux, project / 'prefix', project / 'prefix0'):
    directory.mkdir(parents=True, exist_ok=True)
for relative in ('prefix/child', 'prefix-other', 'prefix.file', 'prefix0/x',
                 'source.py', 'model.mock', '中文.txt'):
    (project / relative).write_bytes(('fixture:' + relative).encode('utf-8'))
(candidate / 'excluded_temporary.bin').write_bytes(b'excluded')
(audit_out / 'excluded_receipt.json').write_bytes(b'{}')
(aux / 'protected_aux_result.json').write_bytes(b'{"result":"fixture"}')
symlink_created = False
try:
    os.symlink(str(project / 'source.py'), str(project / 'fixture_link'))
    symlink_created = True
except (OSError, NotImplementedError):
    pass


def test_path(value):
    if str(value) == '/root/prefix-io-v1-validation':
        return aux
    return original_path(value)


ns['Path'] = test_path
ns['ROOT'] = project
ns['OUT'] = audit_out
old = ns['inventory']([candidate], 'legacy_metadata_snapshot.jsonl.gz')
stream = list(ns['streaming_metadata_rows']([candidate]))
actual = {r['path']: {k: v for k, v in r.items() if k != 'path'} for r in stream}
assert len(actual) == len(stream) and actual == old
assert not any('excluded_temporary' in p or 'excluded_receipt' in p for p in actual)
assert '@AUX/protected_aux_result.json' in actual
assert 'mtime_ns' not in actual['.']
results.append(dict(test='legacy_vs_streaming_scope_and_normalization', status='PASS',
                    metadata_objects=len(actual), symlink_fixture_created=symlink_created,
                    Windows_st_blocks_supplemented=True))
ns['Path'] = original_path
ns['meta'] = original_meta


def row(key, index):
    return dict(path=key, device=1, inode=index, mode=stat.S_IFREG | 0o600,
                nlink=1, bytes=index % 19, allocated_bytes=4096, mtime_ns=1234)


metadata_generator = ns['streaming_metadata_rows']


def comparison_case(name, before, after, expected):
    out = TEST_ROOT / name
    out.mkdir()
    ns['OUT'] = out
    baseline = out / 'PREFLIGHT_V3_NONCANDIDATE_BEFORE.jsonl.gz'
    ns['write_chunk'](baseline, before)
    ns['PREFLIGHT_INVENTORY_SHA'] = actual_sha(baseline)
    shuffled = list(after)
    random.Random(714).shuffle(shuffled)
    ns['streaming_metadata_rows'] = lambda roots: iter(shuffled)
    result = ns['streaming_post_inventory_comparison']([], len(before))
    for kind in ('removed', 'added', 'changed'):
        assert result[kind + '_count'] == len(expected[kind]), result
        assert result[kind] == sorted(expected[kind])[:1000], (kind, result)
    merged = list(ns['gzip_rows'](out / 'EXECUTE_V5_NONCANDIDATE_AFTER.jsonl.gz'))
    assert merged == sorted(after, key=lambda r: r['path'])
    assert result['after_objects'] == len(after) and result['before_objects'] == len(before)
    results.append(dict(test=name, status='PASS', before=len(before), after=len(after),
                        chunks=result['streaming_chunk_files'],
                        removed_count=result['removed_count'], added_count=result['added_count'],
                        changed_count=result['changed_count']))
    return result


keys = ['.', '@AUX/.', '@AUX/result.json', 'prefix', 'prefix-', 'prefix-other',
        'prefix.file', 'prefix/child', 'prefix0', '中文/结果.json']
keys += ['records/%05d' % i for i in range(12017)]
before = [row(k, i + 1) for i, k in enumerate(keys)]
result = comparison_case('unchanged_multichunk_punctuation', before, before,
                         dict(removed=[], added=[], changed=[]))
assert result['streaming_chunk_files'] == 2

remove = {'.', 'prefix/child', 'records/12016'}
change = {'prefix-', 'records/00001'}
after = []
for item in before:
    if item['path'] in remove:
        continue
    copied = dict(item)
    if copied['path'] in change:
        copied['mtime_ns'] += 1
    after.append(copied)
added = ['!new-before', 'prefix!between', 'zz-after']
after += [row(k, 30000 + i) for i, k in enumerate(added)]
comparison_case('added_removed_changed_boundaries', before, after,
                dict(removed=remove, added=added, changed=change))

cap_before = [row('old/%05d' % i, i + 1) for i in range(3500)]
cap_removed = {'old/%05d' % i for i in range(1205)}
cap_changed = {'old/%05d' % i for i in range(1205, 2308)}
cap_after = []
for item in cap_before:
    if item['path'] in cap_removed:
        continue
    copied = dict(item)
    if copied['path'] in cap_changed:
        copied['bytes'] += 1
    cap_after.append(copied)
cap_added = ['new/%05d' % i for i in range(1210)]
cap_after += [row(k, 50000 + i) for i, k in enumerate(cap_added)]
comparison_case('uncapped_counts_capped_samples', cap_before, cap_after,
                dict(removed=cap_removed, added=cap_added, changed=cap_changed))
comparison_case('empty_after_inventory', before[:4], [],
                dict(removed=[r['path'] for r in before[:4]], added=[], changed=[]))
comparison_case('empty_before_inventory', [], before[:4],
                dict(removed=[], added=[r['path'] for r in before[:4]], changed=[]))


def invalid_baseline_case(name, baseline_rows, message):
    out = TEST_ROOT / name
    out.mkdir()
    ns['OUT'] = out
    # Preserve intentional bad ordering rather than write_chunk's sorting.
    with ns['gzip'].open(out / 'PREFLIGHT_V3_NONCANDIDATE_BEFORE.jsonl.gz', 'wt',
                         encoding='utf-8') as f:
        for item in baseline_rows:
            f.write(json.dumps(item) + '\n')
    ns['PREFLIGHT_INVENTORY_SHA'] = actual_sha(out / 'PREFLIGHT_V3_NONCANDIDATE_BEFORE.jsonl.gz')
    (out / 'PREFLIGHT_V3_PREFLIGHT.json').write_text(json.dumps(dict(
        status='PASS_CLEANUP_PREFLIGHT_NO_DELETION', frozen_manifest_sha256=ns['MANIFEST_SHA'],
        restricted_plan_sha256=ns['PLAN_SHA'], deletion_performed=False,
        noncandidate_metadata_objects=1020725)), encoding='utf-8')
    (out / 'PREFLIGHT_COMMAND_03_V3.json').write_text('{"exit":0}', encoding='utf-8')
    try:
        ns['verified_preflight_inventory']()
    except RuntimeError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError('invalid baseline was accepted')
    results.append(dict(test=name, status='PASS', expected_rejection=message))


invalid_baseline_case('reject_unsorted_baseline', [row('b', 1), row('a', 2)],
                      'unsorted or duplicate baseline path')
invalid_baseline_case('reject_duplicate_baseline', [row('a', 1), row('a', 2)],
                      'unsorted or duplicate baseline path')
invalid_baseline_case('reject_wrong_baseline_count', [row('a', 1)],
                      'baseline object count differs')

ns['streaming_metadata_rows'] = metadata_generator
receipt = dict(status='PASS_INDEPENDENT_V5_STREAMING_TESTS', script_sha256=EXPECTED_SHA,
               tests=results, test_count=len(results), production_cleanup_run_calls=0,
               existing_file_deletions=0, GPU_runs=0,
               all_created_test_files_retained_under=str(TEST_ROOT))
output = AUDIT / 'INDEPENDENT_STREAMING_V5_TEST_RESULT.json'
with output.open('x', encoding='utf-8') as f:
    json.dump(receipt, f, ensure_ascii=False, indent=2)
    f.write('\n')
print(json.dumps(receipt, ensure_ascii=True))
