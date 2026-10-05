"""Verify every tar member by SHA before extracting into a new local directory."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--extract', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--server-receipt', type=Path, required=True)
    args = parser.parse_args()
    destination = args.extract.resolve()
    assert not destination.exists()
    server_receipt_raw = args.server_receipt.read_bytes()
    trusted = json.loads(server_receipt_raw)
    archive_raw = args.archive.read_bytes()
    assert len(archive_raw) == trusted['archive']['bytes']
    assert sha(archive_raw) == trusted['archive']['sha256']
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:gz') as tar:
        members = tar.getmembers()
        assert len({member.name for member in members}) == len(members)
        windows_names = set()
        reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {'COM' + str(i) for i in range(1, 10)} | {'LPT' + str(i) for i in range(1, 10)}
        for member in members:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute()
            pieces = member.name.split('/')
            assert relative.as_posix() == member.name
            assert not any(part in ('', '.', '..') or part.endswith((' ', '.'))
                or any(c in part for c in ':\\<>"|?*')
                or any(ord(c) < 32 for c in part)
                or part.split('.')[0].upper() in reserved
                or PureWindowsPath(part).is_reserved() for part in pieces)
            folded = member.name.casefold()
            assert folded not in windows_names
            windows_names.add(folded)
            (destination / member.name).resolve().relative_to(destination)
        for name in windows_names:
            parts = name.split('/')
            assert not any('/'.join(parts[:i]) in windows_names for i in range(1, len(parts)))
        manifest_bytes = tar.extractfile('EVIDENCE_MANIFEST.json').read()
        assert len(manifest_bytes) == trusted['manifest']['bytes']
        assert sha(manifest_bytes) == trusted['manifest']['sha256']
        manifest = json.loads(manifest_bytes)
        expected = {row['archive_path']: row for row in manifest['files']}
        assert len(expected) == manifest['file_count']
        assert set(member.name for member in members) == set(expected) | {'EVIDENCE_MANIFEST.json'}
        data = {}
        for member in members:
            raw = tar.extractfile(member).read()
            if member.name != 'EVIDENCE_MANIFEST.json':
                row = expected[member.name]
                assert len(raw) == row['bytes'] and sha(raw) == row['sha256'], member.name
            data[member.name] = raw
        assert sum(row['bytes'] for row in expected.values()) == manifest['raw_bytes']
    after = json.loads(data['delivery/SESSION_AFTER_CPU.json'])
    assert after['original_source_refs_unchanged'] == 101
    assert after['unchanged_C4_source_inheritance_verified'] == 50
    before = json.loads(data['review/SESSION_BEFORE_CPU.json'])
    assert before['ledger_sha256'] == after['ledger_sha256']
    assert before['gpu_wall_seconds'] == after['gpu_wall_seconds']
    assert before['source_refs'] == after['source_refs']
    assert before['GPU_nodes'] == after['GPU_nodes'] == []
    assert before['tracked_git_status'] == after['tracked_git_status'] == ''
    assert after['GPU_runs'] == 0 and after['active'] is None
    analysis_path = after['analysis_ref']['path']
    selected = next(row for row in expected.values() if row['path'] == analysis_path)
    analysis_raw = data[selected['archive_path']]
    assert sha(analysis_raw) == after['analysis_ref']['sha256']
    analysis = json.loads(analysis_raw)
    assert analysis['GPU_effect_verified'] is False and analysis['GPU_runs_allowed'] == 0
    assert after['CPU_candidate_qualified'] == analysis['CPU_candidate_qualified']
    semantic = json.loads(data['candidate/SERVER_CPU_QUALIFICATION_EXPLICIT_PROVENANCE.json'])
    assert semantic['passed'] and semantic['tests'] == 106 and len(semantic['groups']) == 8
    assert sum(row['tests'] for row in semantic['groups']) == 106
    for row in semantic['groups']:
        assert row['passed'] and row['exit_code'] == 0 and row['tests'] == row['expected_tests']
    candidate_root = semantic['candidate_root'].rstrip('/')
    for item in semantic['source_refs']:
        relative = item['path'].removeprefix(candidate_root + '/')
        assert relative != item['path']
        raw = data['candidate/' + relative]
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    policy = json.loads(data['candidate/ORIGINAL_POLICY_ABI_CPU_STDOUT.log'])
    assert policy['passed'] and policy['tests'] == 101
    assert policy['errors'] == policy['failures'] == policy['skipped'] == 0
    assert all(path.startswith(candidate_root + '/') for path in policy['actual_overlay_modules'])
    race = json.loads(data['review/SERVER_REVIEW/TEST_RESULT.json'])
    assert race['success'] and race['source_stable'] and race['tests_run'] == 32 and race['returncode'] == 0
    assert race['source_before'] == race['source_after']
    for key, relative in (('reactor', 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'),
                          ('collector', 'native_full_step_collector.py')):
        raw = data['candidate/' + relative]
        item = race['source_before'][key]
        assert len(raw) == item['bytes'] and sha(raw) == item['sha256']
    inheritance = json.loads(data['candidate/UNCHANGED_C4_SOURCE_INHERITANCE.json'])
    assert len(inheritance['files']) == len({row['path'] for row in inheritance['files']}) == 50
    for row in inheritance['files']:
        raw = data['candidate/' + row['path']]
        assert len(raw) == row['bytes'] and sha(raw) == row['sha256']
    for group in analysis['scenarios'].values():
        for pair in group['pairs']:
            for arm in pair['arms'].values():
                assert arm['worker_plus_producer_cpu_ns'] == arm['worker_cpu_ns'] + arm['producer_cpu_ns']
    destination.mkdir(parents=True)
    for name, raw in data.items():
        target = (destination / name).resolve()
        target.relative_to(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as output:
            output.write(raw)
    receipt = dict(status='PASS_LOCAL_COMPLETE_BYTE_VERIFICATION', data_files=len(expected),
        archive_sha256=sha(archive_raw), manifest_sha256=sha(manifest_bytes),
        trusted_server_receipt_sha256=sha(server_receipt_raw),
        raw_bytes=manifest['raw_bytes'], old_source_refs_unchanged=101,
        inherited_source_files_verified=50, GPU_runs=0, budget_ledger_unchanged=True,
        CPU_candidate_qualified=analysis['CPU_candidate_qualified'], decision=analysis['decision'])
    with args.receipt.open('x', encoding='utf-8') as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
