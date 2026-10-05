"""Verify every tar member by SHA before extracting into a new local directory."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--extract', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    destination = args.extract.resolve()
    assert not destination.exists()
    with tarfile.open(args.archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len({member.name for member in members}) == len(members)
        for member in members:
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute()
            assert not any(part in ('', '.', '..') or ':' in part or '\\' in part
                           for part in relative.parts)
        manifest_bytes = tar.extractfile('EVIDENCE_MANIFEST.json').read()
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
        archive_sha256=sha(args.archive.read_bytes()), manifest_sha256=sha(manifest_bytes),
        raw_bytes=manifest['raw_bytes'], old_source_refs_unchanged=101,
        inherited_source_files_verified=50, GPU_runs=0, budget_ledger_unchanged=True,
        CPU_candidate_qualified=analysis['CPU_candidate_qualified'], decision=analysis['decision'])
    with args.receipt.open('x', encoding='utf-8') as output:
        json.dump(receipt, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
