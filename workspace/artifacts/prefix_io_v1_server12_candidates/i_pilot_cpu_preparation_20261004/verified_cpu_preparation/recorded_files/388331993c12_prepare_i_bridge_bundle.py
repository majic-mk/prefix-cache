"""Copy only byte-verified historical candidate sources into a new CPU bundle.

Never connects to the server, imports candidate code or executes GPU operations.
The historical original and its received archive payload must both match the
already recorded GPU delivery manifest before any new source copy is written.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


REMOTE_BASE = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/'
LOCAL_BASE = 'artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate'
CONTROL = 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control'
REACTOR = 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', type=Path, required=True)
    a = p.parse_args()
    root = a.project_root.resolve()
    here = Path(__file__).resolve().parent
    manifest_path = root / 'artifacts/prefix_io_v1_server12_candidates/native_gpu_delivery/LOCAL_GPU_EVIDENCE_BYTE_VERIFICATION.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['status'] != 'PASS_SERVER_GPU_EVIDENCE_LOCAL_BYTE_VERIFICATION':
        raise ValueError('historical local payload verification did not pass')
    by_path = {r['path']: r for r in manifest['files']}
    original = root / LOCAL_BASE
    relative = [str(p.relative_to(original)).replace('\\', '/')
                for p in sorted((original / CONTROL).glob('*.py'))]
    relative += [REACTOR, 'native_full_step_collector.py']
    sources = []
    for rel in relative:
        row = by_path[REMOTE_BASE + rel]
        data = (original / rel).read_bytes()
        archived = Path(row['local_path']).read_bytes()
        if sha(data) != row['sha256'] or len(data) != row['bytes'] or data != archived:
            raise ValueError('candidate or actual received archive bytes differ: ' + rel)
        dst = 'frozen/control/' + Path(rel).name if rel.startswith(CONTROL + '/') else (
            'frozen/reactor.py' if rel == REACTOR else 'frozen/native_full_step_collector.py')
        sources.append((dst, data, dict(path=dst, original_project_path=row['path'],
            original_local_path=LOCAL_BASE + '/' + rel, bytes=len(data), sha256=sha(data))))
    for rel, data, _row in sources:
        path = here / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as out:
            out.write(data)
    result = dict(schema='i-only-cpu-source-bundle-v1',
        status='PASS_MATCHES_ACTUAL_GPU_DELIVERY_PAYLOAD',
        cpu_fixture_only=True, gpu_operations=0, production_source_changes=0,
        input_manifest=dict(project_path=str(manifest_path.relative_to(root)).replace('\\', '/'),
            bytes=manifest_path.stat().st_size, sha256=sha(manifest_path.read_bytes())),
        files=[r for _dst, _data, r in sources])
    with (here / 'FROZEN_CPU_SOURCE_MANIFEST.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(dict(status=result['status'], sources=len(sources), gpu_operations=0)))


if __name__ == '__main__':
    main()
