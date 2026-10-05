"""CPU-only CLI/persistence checks, not native launch qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_decoupled_v2_controlled_p0 as cli


class ControlledLaunchCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.native = self.root / 'native.json'
        self.native.write_text('{}', encoding='utf-8')
        self.native_sha = hashlib.sha256(self.native.read_bytes()).hexdigest()
        self.pool_spec = self.root / 'pool.json'
        self.spec = dict(store=dict(authorization_domain='diagnostic-unit-test',
            policy='ALLOW_MIXED_G1', max_bytes=1000000, staging_bytes=1000000,
            max_variants=4, probation_opportunities=2),
            registry_budget=dict(max_bytes=100000, max_manifest_bytes=20000))
        self.pool_spec.write_text(json.dumps(self.spec), encoding='utf-8')
        self.output = self.root / 'isolated'
        self.args = argparse.Namespace(output=self.output, native_manifest=self.native,
            native_manifest_sha256=self.native_sha, pool_spec=self.pool_spec)
        self.runtime = dict(source_provenance=dict(model_signature='unit-model', tokenizer_hash='unit-tokenizer'))

    def initialize(self):
        with patch.object(cli, 'validate_native_attachment', return_value=self.runtime):
            return cli.initialize_pool(self.args)

    def frozen_args(self):
        self.initialize()
        recipe = self.root / 'recipe.json'
        recipe.write_text(json.dumps(dict(authority={}, limits={}, numerical_policy={},
            registry_budget=self.spec['registry_budget'], exact_controls=[], mixed_controls=[], birth_controls=[])))
        return argparse.Namespace(output=self.root / 'frozen', store_root=self.output / 'store',
            registry_root=self.output / 'registry', recipe=recipe, native_manifest=self.native,
            native_manifest_sha256=self.native_sha, instance_id='explicit-unit-instance')

    def test_init_creates_only_empty_isolated_catalogs_and_no_gpu_claim(self):
        result = self.initialize()
        envelope, sha = cli.read_json(self.output / 'store' / 'catalog.json')
        catalog = cli.parse_target_catalog_v2(envelope)
        self.assertEqual(catalog['rows'], {})
        self.assertEqual(catalog['config']['model_signature'], 'unit-model')
        self.assertEqual(result['initial_pool']['sha256'], sha)
        self.assertFalse(result['gpu_execution_allowed'])
        self.assertFalse(result['model_loaded'])
        self.assertEqual(result['gpu_actions_executed'], 0)

    def test_missing_or_stale_native_digest_writes_nothing(self):
        self.args.native_manifest_sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.initialize()
        self.assertFalse(self.output.exists())

    def test_existing_output_is_never_overwritten(self):
        self.initialize()
        before = (self.output / 'pool_preparation.json').read_bytes()
        with self.assertRaises(FileExistsError):
            self.initialize()
        self.assertEqual((self.output / 'pool_preparation.json').read_bytes(), before)

    def test_init_cannot_nest_in_existing_store_or_registry(self):
        self.initialize()
        for name in ('store', 'registry'):
            self.args.output = self.output / name / 'new-nested-pool'
            with self.assertRaisesRegex(ValueError, 'nested'):
                self.initialize()
            self.assertFalse(self.args.output.exists())

    def test_bad_capacity_and_identity_override_rejected_before_creating_pool(self):
        for changes in ({'max_variants': True}, {'model_signature': 'override'}, {'staging_bytes': 0}):
            bad = dict(self.spec, store=dict(self.spec['store'], **changes))
            self.pool_spec.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                self.initialize()
            self.assertFalse(self.output.exists())

    def test_duplicate_json_and_nonfinite_values_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '[]'):
            self.pool_spec.write_text(raw)
            with self.assertRaises(ValueError):
                cli.read_json(self.pool_spec)

    def test_freeze_requires_existing_registry_does_not_recreate_it(self):
        args = self.frozen_args()
        path = args.registry_root / 'registry.json'
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'metadata'):
            cli.freeze_batch(args)
        self.assertFalse(path.exists())
        self.assertFalse(args.output.exists())

    def test_freeze_no_default_missing_authority_or_policy(self):
        args = self.frozen_args()
        spec, _ = cli.read_json(args.recipe)
        del spec['authority']
        args.recipe.write_text(json.dumps(spec))
        with self.assertRaisesRegex(ValueError, 'complete controlled'):
            cli.freeze_batch(args)
        self.assertFalse(args.output.exists())

    def test_freeze_cannot_write_into_store(self):
        args = self.frozen_args()
        args.output = args.store_root / 'frozen'
        with self.assertRaisesRegex(ValueError, 'overlap|nested'):
            cli.freeze_batch(args)
        self.assertFalse(args.output.exists())

    def test_success_only_freezes_manifest_never_executes(self):
        args = self.frozen_args()
        catalog = (args.store_root / 'catalog.json').read_bytes()
        registry = (args.registry_root / 'registry.json').read_bytes()
        returned = dict(manifest=dict(jobs=[{'action_id': 'unit'}]),
                        preflight=dict(status='UNIT_TEST_ONLY'))
        with patch('probekv.p0_launch_recipe_v2.build_controlled_p0_manifest', return_value=returned) as builder:
            result = cli.freeze_batch(args)
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(builder.call_args.kwargs['instance_id'], 'explicit-unit-instance')
        self.assertEqual(result['action_count'], 1)
        self.assertFalse(result['gpu_execution_allowed'])
        self.assertEqual((args.store_root / 'catalog.json').read_bytes(), catalog)
        self.assertEqual((args.registry_root / 'registry.json').read_bytes(), registry)
        saved, _ = cli.read_json(args.output / 'recipe_preflight.json')
        self.assertFalse(saved['model_loaded'])
        self.assertFalse(saved['P1_execution_allowed'])

    def test_builder_failure_retains_no_frozen_manifest_and_releases_writer(self):
        args = self.frozen_args()
        with patch('probekv.p0_launch_recipe_v2.build_controlled_p0_manifest', side_effect=ValueError('missing authority')):
            with self.assertRaisesRegex(ValueError, 'missing authority'):
                cli.freeze_batch(args)
        self.assertFalse(args.output.exists())
        # A second opener succeeds, proving the failed freeze did not keep a lease.
        with patch('probekv.p0_launch_recipe_v2.build_controlled_p0_manifest', return_value=dict(
                manifest=dict(jobs=[]), preflight={})):
            cli.freeze_batch(args)


class ControlledRecipeIntegrationTests(unittest.TestCase):
    """Real builder + file store + CLI, using explicitly CPU-only tiny assets."""
    def setUp(self):
        from tests.test_p0_launch_recipe_v2 import LaunchRecipeTests
        self.fixture = LaunchRecipeTests(methodName='test_T20_pair_is_signed_no_mutation_and_not_gpu_permission')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def freeze(self):
        fixture = self.fixture
        keys = ('authority', 'limits', 'registry_budget', 'numerical_policy',
                'exact_controls', 'mixed_controls')
        recipe = dict((key, fixture.args[key]) for key in keys)
        recipe['birth_controls'] = fixture.args.get('birth_controls', [])
        recipe_path = fixture.root / 'recipe.json'
        recipe_path.write_text(json.dumps(recipe), encoding='utf-8')
        fixture.store.close()
        args = argparse.Namespace(output=fixture.root / 'frozen', store_root=fixture.root / 'store',
            registry_root=fixture.root / 'registry', recipe=recipe_path,
            native_manifest=fixture.args['native_manifest_path'],
            native_manifest_sha256=fixture.args['native_manifest_sha256'],
            instance_id=fixture.args['instance_id'])
        with patch.object(cli.time, 'time', return_value=1000.):
            return cli.freeze_batch(args), args

    def test_real_exact_builder_consumes_actual_catalog_envelope(self):
        result, args = self.freeze()
        saved, _ = cli.read_json(args.output / 'manifest.json')
        self.assertEqual(len(saved['exact_capture_pairs']), 1)
        self.assertEqual(result['action_count'], 2)
        self.assertFalse(result['gpu_execution_allowed'])

    def test_real_mixed_builder_rechecks_published_source_files(self):
        self.fixture.mixed(r1=True)
        result, args = self.freeze()
        saved, _ = cli.read_json(args.output / 'manifest.json')
        self.assertEqual(result['action_count'], 4)
        self.assertEqual(len(saved['mixed_reference_pairs']), 2)
        preflight, _ = cli.read_json(args.output / 'recipe_preflight.json')
        self.assertEqual(len(preflight['source_files_verified']), 2)
        self.assertFalse(preflight['model_loaded'])


if __name__ == '__main__':
    unittest.main()
