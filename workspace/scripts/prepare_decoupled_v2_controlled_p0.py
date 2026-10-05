"""Prepare isolated P0 storage or freeze a bounded batch; never run a model.

init-pool creates only empty target/registry catalogs in a fresh directory.
freeze-batch requires current explicit authorization and actual model/pool
bindings; no sample tokens, tolerance, device, price, or authority defaults.
The later server runner rechecks the hardware and --execute is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from probekv.p0_evidence_v2 import write_new_json
from probekv.source_manifest_v2 import RequestManifestRegistry
from probekv.source_store_v2 import TargetSourceStoreV2, parse_target_catalog_v2
from probekv.v8_schema10_native_factory import validate_native_attachment


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('explicit bounded non-symlink metadata file required')
    raw = path.read_bytes()
    value = RequestManifestRegistry._parse(raw)
    if not isinstance(value, dict):
        raise ValueError('metadata must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def _native(path, expected):
    native, actual = read_json(path)
    if actual != expected:
        raise ValueError('native attachment file digest mismatch')
    return validate_native_attachment(native, allow_unmeasured=True)


def _new_output(path, protected=()):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError('use a fresh output directory; prior results are immutable')
    output = path.resolve()
    for ancestor in output.parents:
        if ((ancestor / 'catalog.json').exists() and (ancestor / 'writer.lock').exists()
                or (ancestor / 'registry.json').exists() and (ancestor / 'manifests').is_dir()):
            raise ValueError('output cannot be nested inside an existing Source store or registry')
    for root in protected:
        root = Path(root).resolve()
        if output == root or root in output.parents or output in root.parents:
            raise ValueError('output cannot overlap the existing pool or registry')
    return output


def _registry_config(config):
    if (not isinstance(config, dict) or set(config) != {'max_bytes', 'max_manifest_bytes'}
            or any(type(n) is not int or n <= 0 for n in config.values())
            or config['max_manifest_bytes'] > config['max_bytes']):
        raise ValueError('explicit positive shared registry budgets required')
    return config


def initialize_pool(args):
    output = _new_output(args.output)
    runtime = _native(args.native_manifest, args.native_manifest_sha256)
    spec, spec_sha = read_json(args.pool_spec)
    if set(spec) != {'store', 'registry_budget'}:
        raise ValueError('pool specification must contain only store and registry_budget')
    budget = _registry_config(spec['registry_budget'])
    store_options = spec['store']
    required = {'authorization_domain', 'policy', 'max_bytes', 'staging_bytes',
                'max_variants', 'probation_opportunities'}
    if not isinstance(store_options, dict) or set(store_options) != required:
        raise ValueError('all isolated store options must be explicit, without identity overrides')
    if (not isinstance(store_options['authorization_domain'], str)
            or not store_options['authorization_domain'].strip()
            or store_options['policy'] not in ('EXACT_ONLY', 'ALLOW_MIXED_G1')
            or type(store_options['max_variants']) is not int or store_options['max_variants'] not in (1, 2, 4)
            or any(type(store_options[k]) is not int or store_options[k] <= 0
                   for k in ('max_bytes', 'staging_bytes'))
            or type(store_options['probation_opportunities']) is not int
            or store_options['probation_opportunities'] < 0):
        raise ValueError('invalid isolated store policy, domain, capacity or grace')
    source = runtime['source_provenance']
    output.mkdir(parents=True, exist_ok=False)
    store = None
    try:
        registry = RequestManifestRegistry(persistent_root=output / 'registry', **budget)
        store = TargetSourceStoreV2(output / 'store', registry=registry,
            model_signature=source['model_signature'], tokenizer_hash=source['tokenizer_hash'], **store_options)
        report = dict(kind='empty_isolated_p0_pool_v2', status='EMPTY_POOL_CREATED',
            native_manifest_path=str(args.native_manifest.resolve()),
            native_manifest_sha256=args.native_manifest_sha256,
            pool_spec_path=str(args.pool_spec.resolve()), pool_spec_sha256=spec_sha,
            initial_pool=dict(path=str(output / 'store' / 'catalog.json'),
                              sha256=read_json(output / 'store' / 'catalog.json')[1]),
            initial_registry=dict(path=str(output / 'registry' / 'registry.json'),
                                  sha256=read_json(output / 'registry' / 'registry.json')[1]),
            stored_sources=0, model_loaded=False, gpu_actions_executed=0,
            gpu_execution_allowed=False, P1_execution_allowed=False,
            automatic_rental_allowed=False, locked_test_accessed=False)
        write_new_json(output / 'pool_preparation.json', report)
    except BaseException as exc:
        write_new_json(output / 'failed.json', dict(status='FAILED', error_type=type(exc).__name__,
                       detail=str(exc), gpu_execution_allowed=False))
        raise
    finally:
        if store is not None:
            store.close()
    return report


def freeze_batch(args):
    from probekv.p0_launch_recipe_v2 import build_controlled_p0_manifest
    output = _new_output(args.output, (args.store_root, args.registry_root))
    # Require both durable identities before opening a constructor which could
    # otherwise initialize missing state. No inferred/rebuilt missing catalog.
    envelope, _ = read_json(args.store_root / 'catalog.json')
    catalog = parse_target_catalog_v2(envelope)
    read_json(args.registry_root / 'registry.json')
    if not (args.registry_root / 'manifests').is_dir():
        raise ValueError('existing registry manifest directory required')
    if args.store_root.resolve() == args.registry_root.resolve():
        raise ValueError('Source store and registry require distinct roots')
    spec, spec_sha = read_json(args.recipe)
    natural = getattr(args, 'natural_controls', None)
    required = {'authority', 'limits', 'registry_budget', 'numerical_policy'}
    if natural is None:
        required |= {'exact_controls', 'mixed_controls', 'birth_controls'}
    if set(spec) != required:
        raise ValueError('explicit complete controlled recipe required')
    registry = RequestManifestRegistry(persistent_root=args.registry_root,
                                      **_registry_config(spec['registry_budget']))
    config = catalog.get('config', {})
    if config.get('purpose') != 'isolated_P0_diagnostic':
        raise ValueError('existing isolated P0 catalog required')
    store = TargetSourceStoreV2(args.store_root, registry=registry,
                               **{k: v for k, v in config.items() if k != 'purpose'})
    try:
        common = dict(
            native_manifest_path=args.native_manifest,
            native_manifest_sha256=args.native_manifest_sha256,
            instance_id=args.instance_id, store=store, now_unix=time.time(), **spec)
        if natural is None:
            result = build_controlled_p0_manifest(**common)
        else:
            from probekv.p0_natural_input_v2 import bind_natural_p0_manifest
            controls, controls_sha = read_json(natural)
            result = bind_natural_p0_manifest(controls, **common)
            result['preflight']['natural_controls'] = dict(path=str(natural.resolve()), sha256=controls_sha)
    finally:
        store.close()
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / 'manifest.json', result['manifest'])
    write_new_json(output / 'recipe_preflight.json', dict(result['preflight'],
        recipe_path=str(args.recipe.resolve()), recipe_sha256=spec_sha,
        model_loaded=False, gpu_actions_executed=0, gpu_execution_allowed=False,
        automatic_rental_allowed=False, P1_execution_allowed=False))
    return dict(status='BATCH_FROZEN_RUNTIME_RECHECK_REQUIRED', output=str(output),
                action_count=len(result['manifest']['jobs']), gpu_actions_executed=0,
                gpu_execution_allowed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    initialize = commands.add_parser('init-pool', help='new empty isolated CPU store; no model execution')
    freeze = commands.add_parser('freeze-batch', help='freeze exact actions under explicit current authorization')
    for command in (initialize, freeze):
        command.add_argument('--native-manifest', type=Path, required=True)
        command.add_argument('--native-manifest-sha256', required=True)
        command.add_argument('--output', type=Path, required=True)
    initialize.add_argument('--pool-spec', type=Path, required=True)
    freeze.add_argument('--recipe', type=Path, required=True)
    freeze.add_argument('--natural-controls', type=Path,
                        help='compiled frozen natural P0 controls; recipe then contains only authority/limits/registry/numerics')
    freeze.add_argument('--store-root', type=Path, required=True)
    freeze.add_argument('--registry-root', type=Path, required=True)
    freeze.add_argument('--instance-id', required=True)
    args = parser.parse_args(argv)
    result = initialize_pool(args) if args.operation == 'init-pool' else freeze_batch(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
