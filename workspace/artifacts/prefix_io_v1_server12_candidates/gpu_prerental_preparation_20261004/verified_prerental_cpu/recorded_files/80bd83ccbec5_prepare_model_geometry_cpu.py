"""Freeze model-derived geometry only; actual running GPU geometry stays unknown."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REL = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
ANCESTOR = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004/calibration_v2/REVISION_SOURCE_LOCK.json'
ANCESTOR_SHA = '20fbabe41c5fbef7a3e9a99ed48cea11b36afe5dd43e6c7861637ea7032f51e5'
MODEL = 'models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444/config.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CPU-only environment required')
    ancestor = (root/ANCESTOR).read_bytes()
    if hashlib.sha256(ancestor).hexdigest() != ANCESTOR_SHA:
        raise ValueError('original model source closure drift')
    row = next(r for r in json.loads(ancestor)['files'] if r['path'] == MODEL)
    path = root/MODEL
    if path.is_symlink():
        raise ValueError('model config symlink refused')
    raw = path.read_bytes()
    if len(raw) != row['bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('actual model configuration bytes differ')
    model = json.loads(raw)
    expected = dict(num_hidden_layers=28, num_key_value_heads=4, hidden_size=3584, num_attention_heads=28)
    if any(type(model.get(k)) is not int or model[k] != v for k, v in expected.items()):
        raise ValueError('fixed Qwen geometry differs')
    geometry = dict(model_config_sha256=row['sha256'], num_hidden_layers=28,
        num_key_value_heads=4, head_dim=128, dtype='bfloat16', dtype_bytes=2,
        tokens_per_block=16, tensor_parallel_size=1, physical_block_bytes=917504)
    encoded = json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    target = root/REL/'runner/kv_geometry.json'
    with target.open('xb') as stream:
        stream.write(encoded)
    proof = dict(status='PASS_CPU_MODEL_CONFIG_DERIVED_CANONICAL_GEOMETRY_ONLY',
        model_config_ref=row, geometry_ref=dict(path=target.relative_to(root).as_posix(),
            bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest()),
        actual_GPU_runs=0, actual_GPU_layout_read=False, actual_model_or_kernel_loaded=False,
        actual_dtype_tensor_or_allocator_qualified=False, production_qualified=False,
        original_running_tensor_layout_match_required_before_strategy=True)
    with (root/REL/'CPU_MODEL_GEOMETRY_SOURCE_ONLY_RESULT.json').open('x') as stream:
        json.dump(proof, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(proof, sort_keys=True))


if __name__ == '__main__':
    main()
