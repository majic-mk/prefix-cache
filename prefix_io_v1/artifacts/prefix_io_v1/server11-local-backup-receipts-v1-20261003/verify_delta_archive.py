"""Verify closed-run archive bytes and every member; never extract or delete."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    directory = Path(args.directory)
    receipt = json.loads((directory / 'ARCHIVE_RECEIPT.json').read_bytes())
    manifest_path = directory / 'SERVER11_DELTA_MANIFEST.json'
    raw = manifest_path.read_bytes()
    assert len(raw) == receipt['manifest']['bytes']
    assert hashlib.sha256(raw).hexdigest() == receipt['manifest']['sha256']
    manifest = json.loads(raw)
    archive = directory / 'SERVER11_RAW_DELTA.tar.gz'
    assert archive.stat().st_size == receipt['archive']['bytes']
    with archive.open('rb') as stream:
        assert digest(stream) == receipt['archive']['sha256']
    expected = {row['path']: row for row in manifest['files']}
    assert len(expected) == len(manifest['files'])
    expected['SERVER11_DELTA_MANIFEST.json'] = receipt['manifest']
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            assert member.isfile() and member.name not in seen
            row = expected[member.name]
            assert member.size == row['bytes']
            with tar.extractfile(member) as stream:
                assert digest(stream) == row['sha256'], member.name
            seen.add(member.name)
    assert seen == set(expected)
    result = dict(status='PASS_LOCAL_ARCHIVE_ALL_MEMBERS', file_count=len(manifest['files']),
                  archive_sha256=receipt['archive']['sha256'], archive_bytes=archive.stat().st_size,
                  manifest_sha256=receipt['manifest']['sha256'], extracted=False, GPU_operations=0)
    with (directory / 'LOCAL_ARCHIVE_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))

if __name__ == '__main__':
    main()
