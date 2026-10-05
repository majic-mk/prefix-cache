"""Package finite server10 source/configuration and actual results; no GPU calls."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
DEST = ROOT / 'artifacts/prefix_io_v1/server10-system-validation-v1-20261003'
ARTIFACTS = (
    'server10-reference-migration-v1-20261003',
    'server10-reference-postprocess-v1-20261003',
    'server10-native-qualification-v1-20261003',
    'server10-g2-migration-v1-20261003',
    'server10-sdk-rebind-v1-20261003',
    'server10-cuda13-cpu-20261003',
    'server10-system-validation-v1-20261003',
)
JOBS = (
    'server10-g3-reference-native-01', 'server10-g3-reference-paired-01',
    'server10-p4-native-off-01', 'server10-p4-native-shadow-01',
    'server10-g2-normal-off-01', 'server10-g2-normal-shadow-01',
)

def main():
    assert ROOT.resolve(strict=True) == ROOT and not DEST.is_symlink()
    rows = {}
    def add(path):
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
        assert path.stat().st_size <= 8 * 1024**2, str(path)
        raw = path.read_bytes()
        rel = path.relative_to(ROOT).as_posix()
        rows[rel] = dict(path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    for name in ARTIFACTS:
        folder = ROOT / 'artifacts/prefix_io_v1' / name
        for path in folder.iterdir():
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.py', '.log', '.md', '.yaml'):
                if path.name not in ('EVIDENCE_MANIFEST.json', 'EVIDENCE_ARCHIVE_RECEIPT.json'):
                    add(path)
    for name in JOBS:
        folder = ROOT / 'experiments/prefix_io_v1/runs' / name
        assert folder.is_dir() and not folder.is_symlink(), name
        for path in folder.rglob('*'):
            if 'runtime-cache' in path.parts:
                continue
            if path.is_file() and not path.is_symlink() and path.suffix in ('.json', '.jsonl', '.log', '.txt'):
                add(path)
    add(ROOT / 'experiments/prefix_io_v1/configs/permissions.server10.reference.yaml')
    ledger_path = ROOT / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    ledger_raw = ledger_path.read_bytes()
    ledger = json.loads(ledger_raw)
    assert ledger['active_reservation'] is None
    snapshot = DEST / 'BUDGET_LEDGER_AFTER_SIX_JOBS.json'
    with snapshot.open('xb') as stream:
        stream.write(ledger_raw)
    add(snapshot)
    manifest = dict(schema_version=1, status='EXACT_SERVER10_SIX_JOB_DELIVERY',
                    files=[rows[name] for name in sorted(rows)],
                    total_bytes=sum(row['bytes'] for row in rows.values()),
                    GPU_operations_in_packaging=0, credentials_included=False,
                    model_weights_and_private_compiler_caches_included=False)
    manifest_path = DEST / 'EVIDENCE_MANIFEST.json'
    with manifest_path.open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    archive = DEST / 'SERVER10_SIX_JOB_EVIDENCE.tar.gz'
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as bundle:
            for name in sorted(rows):
                path = ROOT / name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == rows[name]['sha256']
                bundle.add(path, arcname=name, recursive=False)
            bundle.add(manifest_path, arcname=manifest_path.relative_to(ROOT).as_posix(), recursive=False)
    receipt = dict(path=str(archive), bytes=archive.stat().st_size,
                   sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                   manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   file_count=len(rows), original_ledger_unchanged=ledger_path.read_bytes() == ledger_raw)
    assert receipt['original_ledger_unchanged']
    with (DEST / 'EVIDENCE_ARCHIVE_RECEIPT.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
