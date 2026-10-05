"""Archive closed server11 runs, raw payloads and immutable source receipts.

Runtime SDK/model aliases and reproducible compiler caches are excluded.
Regular cache payload .bin files and all measurement/failed-run logs are retained.
No deletion or GPU operation.
"""
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
OUT = ROOT/'artifacts/prefix_io_v1/server11-delivery-v1-20261003'
EXCLUDED = {'runtime-cache', '__pycache__'}


def reference(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=h.hexdigest())


def write(name, value):
    with (OUT/name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')


def main():
    ledger_path=ROOT/'experiments/prefix_io_v1/gpu-budget-ledger.json'
    ledger=json.loads(ledger_path.read_bytes())
    assert ledger['active_reservation'] is None
    closed=[]
    for index in range(1,5):
        base=ROOT/('experiments/prefix_io_v1/runs/server11-native-cost-six-window-%02d'%index)
        record=json.loads((base/'result.json').read_bytes())
        assert record['session_drained'] and not record['session_members_after_cleanup']
        closed.append(record)
    assert closed[-1]['exit']==0
    OUT.mkdir(exist_ok=True)
    roots=sorted(path for path in (ROOT/'artifacts/prefix_io_v1').glob('server11-*')
                 if path.is_dir() and path != OUT)
    roots+=sorted(path for path in (ROOT/'experiments/prefix_io_v1/runs').glob('server11-*')
                  if path.is_dir())
    files={}; skipped=[]
    for base in roots:
        for directory,names,fnames in os.walk(base,followlinks=False):
            keep=[]
            for name in names:
                p=Path(directory)/name
                if name in EXCLUDED or p.is_symlink(): skipped.append(p.relative_to(ROOT).as_posix())
                else: keep.append(name)
            names[:]=keep
            for name in fnames:
                p=Path(directory)/name
                if p.is_symlink(): skipped.append(p.relative_to(ROOT).as_posix()); continue
                assert p.is_file() and ROOT in p.resolve().parents
                files[p.relative_to(ROOT).as_posix()]=p
    for path in [ledger_path,ROOT/'experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml']:
        files[path.relative_to(ROOT).as_posix()]=path
    # Snapshot the ledger as raw evidence in the archive, without changing it.
    rows=[reference(files[name]) for name in sorted(files)]
    total=sum(row['bytes'] for row in rows)
    assert os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize > 8*1024**3+total+64*1024**2
    manifest=dict(status='CLOSED_REAL_RUNS_AND_RAW_KV_PAYLOADS', files=rows,
        file_count=len(rows), uncompressed_bytes=total, skipped_reproducible_cache_or_aliases=skipped,
        all_four_guard_results=closed, total_new_guard_seconds=sum(r['elapsed_seconds'] for r in closed),
        ledger_gpu_wall_seconds=ledger['gpu_wall_seconds'], remaining_gpu_seconds=28800-ledger['gpu_wall_seconds'],
        new_GPU_operations=0, deletions=0)
    write('SERVER11_EVIDENCE_MANIFEST.json',manifest)
    archive=OUT/'SERVER11_RAW_EVIDENCE.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        tar.add(OUT/'SERVER11_EVIDENCE_MANIFEST.json',arcname='SERVER11_EVIDENCE_MANIFEST.json',recursive=False)
        for row in rows:
            p=files[row['path']]
            assert reference(p)==row, 'evidence changed during archive'
            tar.add(p,arcname=row['path'],recursive=False)
    receipt=dict(status='ARCHIVED',archive=reference(archive),manifest=reference(OUT/'SERVER11_EVIDENCE_MANIFEST.json'),
                 file_count=len(rows),uncompressed_bytes=total, GPU_operations=0)
    write('ARCHIVE_RECEIPT.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__': main()
