"""Append-only backup of closed new server11 experiments; never deletes data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
SKIP = {'runtime-cache', '__pycache__'}


def reference(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=digest.hexdigest())


def safe(relative):
    path = ROOT / relative
    if path.is_symlink() or ROOT not in path.resolve(strict=True).parents:
        raise ValueError('existing regular project descendant required: ' + relative)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--run', action='append', required=True)
    parser.add_argument('--source', action='append', required=True)
    args = parser.parse_args()
    if not args.output.startswith('artifacts/prefix_io_v1/server11-delivery-'):
        raise ValueError('new server11 delivery directory required')
    out = ROOT / args.output
    if out.exists() or ROOT not in out.resolve().parents:
        raise ValueError('output must be new and within project')
    ledger_path = safe('experiments/prefix_io_v1/gpu-budget-ledger.json')
    ledger = json.loads(ledger_path.read_bytes())
    if ledger['active_reservation'] is not None:
        raise ValueError('archive only after the original GPU guard closes')
    closed, roots = [], []
    for relative in args.run:
        if not relative.startswith('experiments/prefix_io_v1/runs/server11-'):
            raise ValueError('explicit server11 run required')
        base = safe(relative)
        record = json.loads((base/'result.json').read_bytes())
        if not record['session_drained'] or record['session_members_after_cleanup']:
            raise ValueError('all sessions must have drained')
        closed.append(record)
        roots.append(base)
    for relative in args.source:
        if not relative.startswith('artifacts/prefix_io_v1/server11-'):
            raise ValueError('explicit server11 source required')
        base = safe(relative)
        if 'delivery-' in base.name:
            raise ValueError('never recursively package prior archives')
        roots.append(base)
    files, skipped = {}, []
    for base in roots:
        for directory, names, fnames in os.walk(base, followlinks=False):
            keep = []
            for name in names:
                path = Path(directory)/name
                if name in SKIP or path.is_symlink():
                    skipped.append(path.relative_to(ROOT).as_posix())
                else:
                    keep.append(name)
            names[:] = keep
            for name in fnames:
                path = Path(directory)/name
                if path.is_symlink():
                    skipped.append(path.relative_to(ROOT).as_posix())
                    continue
                if not path.is_file() or ROOT not in path.resolve().parents:
                    raise ValueError('nonregular/outside evidence')
                files[path.relative_to(ROOT).as_posix()] = path
    for path in [ledger_path, safe('experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml')]:
        files[path.relative_to(ROOT).as_posix()] = path
    rows = [reference(files[name]) for name in sorted(files)]
    size = sum(row['bytes'] for row in rows)
    stat = os.statvfs(ROOT)
    if stat.f_bavail*stat.f_frsize < 8*1024**3 + size + 64*1024**2:
        raise ValueError('preserve 8 GiB free floor')
    out.mkdir()
    manifest = dict(status='CLOSED_NATIVE_RUN_DELTA', files=rows, file_count=len(rows),
                    uncompressed_bytes=size, excluded_reproducible_cache_or_aliases=skipped,
                    closed_guard_results=closed, gpu_wall_seconds=ledger['gpu_wall_seconds'],
                    remaining_gpu_seconds=28800-ledger['gpu_wall_seconds'],
                    archived_runs=args.run, archived_sources=args.source,
                    new_GPU_operations=0, deletions=0)
    manifest_path = out/'SERVER11_DELTA_MANIFEST.json'
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    archive = out/'SERVER11_RAW_DELTA.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        tar.add(manifest_path, arcname=manifest_path.name, recursive=False)
        for row in rows:
            path = files[row['path']]
            if reference(path) != row:
                raise ValueError('evidence changed while archiving')
            tar.add(path, arcname=row['path'], recursive=False)
    receipt = dict(status='ARCHIVED_DELTA', archive=reference(archive),
                   manifest=reference(manifest_path), file_count=len(rows),
                   uncompressed_bytes=size, new_GPU_operations=0, deletions=0)
    with (out/'ARCHIVE_RECEIPT.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
