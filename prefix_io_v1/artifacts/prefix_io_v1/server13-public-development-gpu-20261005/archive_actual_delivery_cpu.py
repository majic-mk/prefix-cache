"""Append-only archive of selected actual sources and closed experiment evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    root = Path(args.project).resolve(strict=True)
    def safe(name):
        assert name and not name.startswith('/') and ':' not in name and '\\' not in name
        assert all(part not in ('', '.', '..') for part in name.split('/'))
        path = root / name
        assert path.resolve().is_relative_to(root)
        assert not any(p.is_symlink() for p in (path, *path.parents))
        return path
    ledger = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    before = ledger.read_bytes()
    assert json.loads(before)['active_reservation'] is None
    names = json.loads(safe(args.selection).read_bytes())['files']
    assert len(names) == len(set(names)) and 0 < len(names) <= 256
    rows = []
    for name in names:
        path = safe(name)
        assert path.is_file() and path.stat().st_size <= 8 * 1024**2
        data = path.read_bytes()
        rows.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(schema='actual_server13_selected_delivery_bytes_v1', files=rows,
        archive_scope='new sources, actual raw results and inherited source lock metadata only',
        model_weights_or_private_KV_caches_archived=False, GPU_operations=0)
    manifest_path = safe(args.manifest)
    with manifest_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    archive_path = safe(args.archive)
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for row in rows:
            data = safe(row['path']).read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
            archive.writestr(row['path'], data)
        archive.writestr(args.manifest, manifest_path.read_bytes())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None and len(archive.namelist()) == len(rows) + 1
        for row in rows:
            data = archive.read(row['path'])
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
            assert safe(row['path']).read_bytes() == data
    assert ledger.read_bytes() == before
    data = archive_path.read_bytes()
    print(json.dumps(dict(status='PASS_ACTUAL_SELECTED_SERVER_ZIP_BYTES', files=len(rows),
        archive_bytes=len(data), archive_sha256=hashlib.sha256(data).hexdigest(),
        archive=args.archive, manifest=args.manifest, GPU_operations=0)))


if __name__ == '__main__':
    main()
