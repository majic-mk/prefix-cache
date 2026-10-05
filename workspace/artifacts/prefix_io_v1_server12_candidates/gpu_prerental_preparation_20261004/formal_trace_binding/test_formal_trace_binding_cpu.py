"""In-memory CPU metadata fixtures. No trace files or GPU capability objects."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import formal_trace_binding as F


def sealed(document):
    result = deepcopy(document)
    result['workload_sha256'] = F.digest({key: value for key, value in result.items() if key != 'workload_sha256'})
    return result


def metadata_fixture():
    # This only reproduces schema metadata in memory. It is never frozen as an
    # actual natural input, executed by a model or passed to a GPU table issuer.
    records = []
    for index in range(6):
        records.append(dict(request_id=index, prompt_sha256=F.digest(['CPU_METADATA_FIXTURE', index]),
                            prompt_token_ids=[index + 17], scheduled_ns=index * 100, max_tokens=128,
                            min_tokens=128, seed=3, split=F.PARTITIONS[index // 2], prefix_family='fixture-family-' + str(index)))
    return sealed(dict(schema='natural_trace_workload_v1', origin='frozen_actual_existing_trace_no_new_gpu_outcomes',
                       dataset_sha256='a' * 64, ordered_prompt_digest='b' * 64,
                       author_trace_sha256=F.AUTHOR_TRACE_SHA, author_common_sha256=F.AUTHOR_COMMON_SHA,
                       tokenizer_receipt_digest='c' * 64, declaration_digest='d' * 64,
                       model_manifest_sha256='e' * 64, records=records,
                       partition_counts=dict(calibration=2, development=2, evaluation=2),
                       max_concurrency=1, arrival_rate=None, schedule_seed=3,
                       schedule_origin='unchanged_author_build_global_specs', recorded_natural_arrival_claim=False,
                       no_prefix_injection=True, no_per_request_cache_reset=True, no_request_drops_or_reorder=True,
                       initial_cache_state=F.INITIAL, cost_domain_covered=False, gpu_effect_qualified=False,
                       gpu_operations=0))


def pair_fixture(root):
    def arm(name):
        return dict(engine=dict(skip_tokenizer_init=True, speculative_config=None,
            kv_transfer_config=dict(kv_connector_extra_config=dict(
                shared_storage_path=(root / 'experiments/prefix_io_v1/runs' / ('fixture-' + name) / 'storage').as_posix(),
                prefix_io_p4_policy=dict(mode='off' if name == 'U' else 'interference'),
                prefix_io_parent_admission=dict(run_id='fixture-' + name, max_accepted_parents=8)))), sampling={})
    return dict(U=arm('U'), I=arm('I'))


class FormalMetadataTests(unittest.TestCase):
    def test_original_formal_schema_is_metadata_only(self):
        result = F.validate_manifest_shape(metadata_fixture())
        self.assertEqual(result['complete_records'], 6)
        self.assertFalse(result['gpu_eligible'])

    def test_qualification_schema_cannot_be_promoted(self):
        for name in ('controlled_original_p3_qualification_workload_v1', 'natural_off_qualification_workload_v1'):
            document = metadata_fixture()
            document['schema'] = name
            with self.assertRaisesRegex(ValueError, 'qualification relabeling'):
                F.validate_manifest_shape(sealed(document))

    def test_qualification_row_tags_cannot_be_reused(self):
        document = metadata_fixture()
        document['records'][4]['split'] = 'qualification'
        document['records'][4]['prefix_family'] = None
        with self.assertRaisesRegex(ValueError, 'contiguous split'):
            F.validate_manifest_shape(sealed(document))

    def test_family_or_exact_prompt_leak_rejected_after_valid_digest(self):
        for field in ('prefix_family', 'prompt_sha256'):
            document = metadata_fixture()
            document['records'][5][field] = document['records'][0][field]
            with self.assertRaisesRegex(ValueError, 'leaks across partitions'):
                F.validate_manifest_shape(sealed(document))

    def test_full_accepted_input_order_and_denominator_preserved(self):
        for kind in ('drop', 'reorder', 'relabel'):
            document = metadata_fixture()
            if kind == 'drop':
                document['records'].pop()
            elif kind == 'reorder':
                document['records'][4], document['records'][5] = document['records'][5], document['records'][4]
            else:
                document['records'][4]['request_id'] = 0
            with self.assertRaises(ValueError):
                F.validate_manifest_shape(sealed(document))

    def test_chunk_or_non_native_token_lane_rejected(self):
        for value in (None, [], [False], [152064], [-1]):
            document = metadata_fixture()
            document['records'][3]['prompt_token_ids'] = value
            with self.assertRaises(ValueError):
                F.validate_manifest_shape(sealed(document))

    def test_original_schedule_and_seed_cannot_be_refit(self):
        for kind in ('time', 'seed', 'schedule_origin'):
            document = metadata_fixture()
            if kind == 'time':
                document['records'][3]['scheduled_ns'] = 1
            elif kind == 'seed':
                document['records'][3]['seed'] = 4
            else:
                document['schedule_origin'] = 'fit_to_GPU_outcomes'
            with self.assertRaises(ValueError):
                F.validate_manifest_shape(sealed(document))

    def test_positive_bounded_concurrency_and_counts_have_exact_types(self):
        for name, value in (('max_concurrency', True), ('gpu_operations', False), ('max_concurrency', 9)):
            document = metadata_fixture()
            document[name] = value
            with self.assertRaises(ValueError):
                F.validate_manifest_shape(sealed(document))

    def test_data_contract_never_becomes_GPU_qualified(self):
        for field in ('cost_domain_covered', 'gpu_effect_qualified'):
            document = metadata_fixture()
            document[field] = True
            with self.assertRaisesRegex(ValueError, 'never GPU qualification'):
                F.validate_manifest_shape(sealed(document))

    def test_data_safety_flags_cannot_manufacture_an_opportunity(self):
        for field in ('no_prefix_injection', 'no_per_request_cache_reset', 'no_request_drops_or_reorder'):
            document = metadata_fixture()
            document[field] = False
            with self.assertRaises(ValueError):
                F.validate_manifest_shape(sealed(document))

    def test_complete_digest_and_unknown_schema_fields_rejected(self):
        document = metadata_fixture()
        document['records'][0]['prompt_token_ids'] = [77]
        with self.assertRaisesRegex(ValueError, 'manifest digest'):
            F.validate_manifest_shape(document)
        document = metadata_fixture()
        document['fit_or_evaluation_input_allowed'] = True
        with self.assertRaises(ValueError):
            F.validate_manifest_shape(sealed(document))

    def test_missing_independent_service_targets_are_explicit_failure(self):
        for value in (None, {}, {'TTFT_ns': 1, 'request_ITL_P95_ns': 1}):
            with self.assertRaisesRegex(ValueError, 'SLO missing'):
                F.require_independent_service_SLO(value)

    def test_posthoc_or_Amax_service_requirement_rejected(self):
        value = dict(schema='independent_service_SLO_v1', origin='independent_requirement_before_development',
                     TTFT_ns=100, request_ITL_P95_ns=10)
        self.assertEqual(F.require_independent_service_SLO(value), value)
        for origin in ('A_max', 'after_evaluation', 'after_on'):
            with self.assertRaises(ValueError):
                F.require_independent_service_SLO(dict(value, origin=origin))
        with self.assertRaises(ValueError):
            F.require_independent_service_SLO(dict(value, TTFT_ns=True))

    def test_missing_actual_natural_source_ref_fails_CPU(self):
        with self.assertRaisesRegex(ValueError, 'frozen file ref'):
            F.validate_formal_workload(metadata_fixture(), root=Path(__file__).parent,
                workload_ref=None, binding_ref=None, refs={}, pair={}, partition='evaluation')

    def test_foreign_source_path_or_nonhex_digest_rejected(self):
        root = Path(__file__).parent.resolve()
        for row in (dict(path='../dataset.json', bytes=1, sha256='a' * 64),
                    dict(path='metadata.json', bytes=True, sha256='a' * 64),
                    dict(path='metadata.json', bytes=1, sha256='x' * 64)):
            with self.assertRaises(ValueError):
                F.normalize_ref(root, row)

    def test_actual_source_bytes_and_complete_closure_are_required(self):
        root = Path(__file__).parent.resolve()
        row = F.file_ref(root, 'formal_trace_binding.py')
        with self.assertRaisesRegex(ValueError, 'outside frozen closure'):
            F.closed(root, row, {})
        changed = dict(row, sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'actual source bytes changed'):
            F.closed(root, changed, {changed['path']: changed})
        actual, path = F.closed(root, row, {row['path']: row})
        self.assertEqual(actual, row)
        self.assertEqual(path, root / 'formal_trace_binding.py')

    def test_duplicate_nonfinite_JSON_rejected(self):
        for raw in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                F.parse(raw)

    def test_equal_namespace_is_whole_partition_and_model_bound(self):
        root = Path(__file__).parent.resolve()
        manifest = metadata_fixture()
        pair = pair_fixture(root)
        spaces = {}
        for name in F.PARTITIONS:
            spaces[name] = {arm: pair[arm]['engine']['kv_transfer_config']['kv_connector_extra_config']['shared_storage_path']
                            if name == 'evaluation' else (root / 'experiments/prefix_io_v1/runs' / (name + '-' + arm) / 'storage').as_posix()
                            for arm in ('U', 'I')}
        contract = dict(schema='formal_trace_namespace_contract_v1', workload_sha256=manifest['workload_sha256'],
                        model_manifest_sha256=manifest['model_manifest_sha256'],
                        tokenizer_receipt_digest=manifest['tokenizer_receipt_digest'],
                        common_runtime_domain_sha256=F.common_domain_sha(pair), initial_cache_state=F.INITIAL,
                        no_per_request_reset=True, partition_namespaces=spaces)
        self.assertEqual(F.validate_namespace(contract, root=root, manifest=manifest, pair=pair, partition='evaluation'), spaces['evaluation'])
        for kind in ('alias', 'reset', 'model', 'domain'):
            changed = deepcopy(contract)
            if kind == 'alias':
                changed['partition_namespaces']['development']['U'] = spaces['calibration']['U']
            elif kind == 'reset':
                changed['no_per_request_reset'] = False
            elif kind == 'model':
                changed['model_manifest_sha256'] = 'f' * 64
            else:
                changed['common_runtime_domain_sha256'] = 'f' * 64
            with self.assertRaises(ValueError):
                F.validate_namespace(changed, root=root, manifest=manifest, pair=pair, partition='evaluation')

    def test_both_effect_arms_require_actual_development_not_boolean_flags(self):
        root = Path(__file__).parent.resolve()
        pair = pair_fixture(root)
        metadata = dict(schema='formal_natural_token_id_partition_CPU_binding_v1', partition='evaluation', gpu_eligible=False)
        # We mock only the CPU byte-input replay so this test can reach the
        # actual-development missing gate. No native/GPU object is constructed.
        for arm, mode in (('U', 'off'), ('I', 'on')):
            config = dict(phase='effect', mode=mode, arm=arm, run_id='fixture-' + arm,
                          workload_ref=None, formal_trace_binding_ref=None)
            with mock.patch.object(F, 'json_leaf', return_value=({}, {})), \
                 mock.patch.object(F, 'validate_formal_workload', return_value=metadata):
                with self.assertRaisesRegex(ValueError, 'both effect Uoff and Ion'):
                    F.validate_formal_phase(config, root=root, refs={}, pair=pair, formal_workload=metadata)

    def test_old_qualification_route_must_stay_on_original_validator(self):
        with self.assertRaisesRegex(ValueError, 'qualification must use original validator'):
            F.validate_formal_phase(dict(phase='qualification', mode='off', arm='U'),
                                   root=Path(__file__).parent, refs={}, pair={}, formal_workload={})

    def test_effect_cannot_use_development_or_calibration_families(self):
        for partition in ('calibration', 'development'):
            with self.assertRaisesRegex(ValueError, 'no qualification promotion'):
                F.validate_formal_phase(dict(phase='effect', mode='off', arm='U'), root=Path(__file__).parent,
                    refs={}, pair={}, formal_workload=dict(schema='formal_natural_token_id_partition_CPU_binding_v1',
                                                         partition=partition, gpu_eligible=False))


if __name__ == '__main__':
    unittest.main()
