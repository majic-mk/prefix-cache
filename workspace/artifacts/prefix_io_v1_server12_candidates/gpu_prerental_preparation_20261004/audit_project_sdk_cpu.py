"""Verify project SDK assets in no-card mode; never qualify a host driver.

Original receipts and the original tree verifier are retained unchanged.
No compiler, shared-library loader, GPU query, or framework import is used.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PREVIOUS = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2'
REL = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'


def require(ok, why):
    if not ok:
        raise ValueError('CPU_SDK_ASSET_REJECTED: ' + why)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment')
    root = args.project_root.resolve(strict=True)
    closure = json.loads((root/PREVIOUS/'REVISION_SOURCE_LOCK.json').read_bytes())
    refs = {r['path']: r for r in closure['files']}
    source = PREVIOUS + '/site_sdk_binding.py'
    raw = (root/source).read_bytes()
    require(refs[source] == dict(path=source, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
            'unchanged original site adapter')
    spec = importlib.util.spec_from_file_location('_actual_cpu_sdk_original', root/source)
    sdk = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sdk
    spec.loader.exec_module(sdk)
    rows = {name:sdk.checked(root, refs, name) for name in sdk.PINS}
    inventory, proof, result = (sdk.read(root, name) for name in sdk.SDK_EVIDENCE)
    sdk.validate_receipts(root, rows, inventory, proof, result)
    helper = sdk.original_helper(root, rows[sdk.HELPER])
    count = 0
    size = 0
    for name in ('bin', 'include', 'nvvm'):
        original = inventory['directories'][name]
        files = [dict(path=r['relative'], bytes=r['bytes'], sha256=r['sha256']) for r in original['files']]
        files.sort(key=lambda r:r['path'])
        tree = dict(path=original['root'], bytes=original['total_bytes'],
                    sha256=helper.tree_manifest_digest(files), files=files)
        require(original['file_count'] == len(files) and
                all(r['path'] == str(Path(original['root'])/r['relative']) for r in original['files']),
                'original complete absolute/relative tree identity')
        require(helper.verify_tree(tree) == root/'.venv/lib/python3.12/site-packages/nvidia/cu13'/name,
                'original coherent private project toolkit')
        count += len(files)
        size += tree['bytes']
    require(proof['source_inventory_counts'] == {n:inventory['directories'][n]['file_count']
                for n in ('bin', 'include', 'nvvm')} and proof['source_inventory_bytes'] == size,
            'historical compiler receipt full tree accounting')
    for row in [inventory['cudart'], *proof['output_refs']]:
        path = Path(row['path'])
        require(path.is_relative_to(root), 'project asset only')
        helper.digest_file(path, row['bytes'], row['sha256'])
    driver_path = Path(sdk.DRIVER['path'])
    if driver_path.is_file():
        with driver_path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        observed = dict(path=str(driver_path), bytes=driver_path.stat().st_size, sha256=digest)
    else:
        observed = dict(path=str(driver_path), exists=False)
    report = dict(schema='actual_no_card_project_SDK_asset_audit_v1',
                  status='PASS_PROJECT_ASSET_BYTES_DRIVER_DEFERRED',
                  project_tree_files=count, project_tree_bytes=size,
                  actual_project_probe_and_cudart_bytes_verified=True,
                  historical_CPU_compile_receipt_revalidated=True,
                  compiler_executions_this_turn=0, shared_objects_loaded=False,
                  GPU_operations=0, original_source_ref=refs[source],
                  host_driver_expected=sdk.DRIVER, host_driver_observed=observed,
                  host_driver_status='DEFERRED_UNTIL_LIVE_GPU_MODE',
                  live_original_load_site_assets_required_before_guard=True,
                  full_SDK_runtime_qualified=False, production_qualified=False)
    target = root/REL/'PRERENT_PROJECT_SDK_CPU_ASSET_RESULT.json'
    with target.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
