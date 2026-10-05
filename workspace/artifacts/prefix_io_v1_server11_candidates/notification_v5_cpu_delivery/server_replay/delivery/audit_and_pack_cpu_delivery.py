"""Append-only audit and small evidence backup; CPU-only, no asset cleanup."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=digest(raw))


def audit(args):
    root = args.project.resolve()
    cand, review, bench, delivery = [p.resolve() for p in
        (args.candidate, args.review, args.benchmark, args.delivery)]
    for path in (cand, review, bench, delivery):
        path.relative_to(root / 'artifacts/prefix_io_v1')
        assert path.is_dir()
    before = json.loads((review / 'SESSION_BEFORE_CPU.json').read_text())
    refs = []
    for old in before['source_refs']:
        path = root / old['path']
        now = ref(path)
        assert now['sha256'] == old['sha256'] and now['bytes'] == old['bytes'], path
        refs.append(dict(path=old['path'], sha256=now['sha256'], bytes=now['bytes']))
    assert len(refs) == 101
    ledger_path = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    assert digest(ledger_raw) == before['ledger_sha256']
    assert ledger['gpu_wall_seconds'] == before['gpu_wall_seconds']
    assert ledger.get('active_reservation') is None
    nodes = sorted(str(p) for p in Path('/dev').glob('nvidia*'))
    assert not nodes, 'unexpected GPU mode; this delivery claims CPU-only'
    git = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain',
                                  '--untracked-files=no'], text=True)
    assert git == before['tracked_git_status'] == ''
    assert shutil.disk_usage(root).free >= 8 * 1024**3
    inherited = json.loads((cand / 'UNCHANGED_C4_SOURCE_INHERITANCE.json').read_text())
    assert len(inherited['files']) == 50
    for item in inherited['files']:
        raw = (cand / item['path']).read_bytes()
        assert digest(raw) == item['sha256'] and len(raw) == item['bytes'], item['path']
    manifest = json.loads((cand / 'CANDIDATE_MANIFEST.json').read_text())
    for item in manifest['files']:
        raw = (cand / item['path']).read_bytes()
        assert digest(raw) == item['sha256'] and len(raw) == item['bytes'], item['path']
    source_base = root / 'artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003'
    candidate_source = sorted(p.relative_to(cand / 'source').as_posix()
        for p in (cand / 'source').rglob('*.py'))
    original_source = sorted(p.relative_to(source_base / 'source').as_posix()
        for p in (source_base / 'source').rglob('*.py'))
    assert candidate_source == original_source and len(candidate_source) == 51
    changed = []
    patches = []
    for relative in ('source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
                     'native_full_step_collector.py'):
        old, new = source_base / relative, cand / relative
        assert old.read_bytes() != new.read_bytes()
        changed.append(dict(relative_path=relative, baseline=ref(old), candidate=ref(new)))
        patches.extend(difflib.unified_diff(old.read_text().splitlines(True),
            new.read_text().splitlines(True), fromfile=str(old), tofile=str(new)))
    with (delivery / 'C4_TO_C5_CPU_CANDIDATE.patch').open('x', encoding='utf-8') as stream:
        stream.writelines(patches)
    semantic = json.loads((cand / 'SERVER_CPU_QUALIFICATION_EXPLICIT_PROVENANCE.json').read_text())
    policy = json.loads((cand / 'ORIGINAL_POLICY_ABI_CPU_STDOUT.log').read_text())
    idle = (cand / 'FROZEN_C4_IDLE_CPU_STDERR.log').read_text()
    race = json.loads((review / 'SERVER_REVIEW/TEST_RESULT.json').read_text())
    assert semantic['passed'] and semantic['tests'] == 106
    assert policy['passed'] and policy['tests'] == 101
    assert 'Ran 15 tests' in idle and 'OK' in idle
    assert race['success'] and race['source_stable'] and race['tests_run'] == 32
    analysis = json.loads(args.analysis.read_text())
    after = dict(utc=datetime.now(timezone.utc).isoformat(), source_refs=refs,
        original_source_refs_unchanged=101, ledger_sha256=digest(ledger_raw),
        gpu_wall_seconds=ledger['gpu_wall_seconds'], active=None,
        GPU_runs=0, GPU_nodes=nodes, tracked_git_status=git,
        disk_free_bytes=shutil.disk_usage(root).free,
        cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
        memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        source_changes=changed, source_manifest_verified=True,
        unchanged_C4_source_inheritance_verified=50,
        tests=dict(candidate_semantic=106, original_policy_ABI=101,
                   independent_race=32, separate_frozen_C4_idle=15),
        GPU_qualified=False, GPU_effect_verified=False,
        CPU_candidate_qualified=analysis['CPU_candidate_qualified'],
        benchmark_decision=analysis['decision'], analysis_ref=ref(args.analysis))
    write(delivery / 'SESSION_AFTER_CPU.json', after)
    return after


def pack(args):
    directory = args.delivery.resolve()
    archive = directory / 'C5_CPU_EVIDENCE.tar.gz'
    manifest_path = directory / 'EVIDENCE_MANIFEST.json'
    entries = []
    for label, parent in (('candidate', args.candidate), ('review', args.review),
                          ('benchmark', args.benchmark), ('delivery', args.delivery)):
        for path in sorted(parent.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path in (archive, manifest_path):
                continue
            relative = label + '/' + path.relative_to(parent).as_posix()
            item = ref(path)
            assert item['bytes'] <= 10 * 1024**2, 'not a bounded CPU evidence file'
            entries.append((relative, path, item))
    assert sum(item['bytes'] for _, _, item in entries) <= 30 * 1024**2
    result = dict(schema=1, scope='new_C5_CPU_evidence_only', GPU_runs=0,
        files=[dict(archive_path=name, **item) for name, _, item in entries],
        raw_bytes=sum(item['bytes'] for _, _, item in entries), file_count=len(entries))
    write(manifest_path, result)
    with archive.open('xb') as output:
        with tarfile.open(fileobj=output, mode='w:gz') as tar:
            for name, path, _ in entries:
                tar.add(path, arcname=name, recursive=False)
            tar.add(manifest_path, arcname='EVIDENCE_MANIFEST.json', recursive=False)
    receipt = dict(status='PACKED_NOT_YET_LOCAL_VERIFIED', archive=ref(archive),
        manifest=ref(manifest_path), data_files=len(entries), raw_bytes=result['raw_bytes'],
        GPU_runs=0, source_assets_deleted=0)
    write(directory / 'SERVER_BACKUP_RECEIPT.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('audit', 'pack'))
    for name in ('project', 'candidate', 'review', 'benchmark', 'delivery'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--analysis', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args) if args.mode == 'audit' else pack(args), ensure_ascii=False))


if __name__ == '__main__':
    main()
