"""Synthetic CPU arithmetic/validation only; no model or GPU evidence."""
import copy
import hashlib
import json
import unittest

from probekv.source_selection_identifiability_audit_v2 import analyze_source_selection_identifiability_v2


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def ref(name):
    return dict(path='mock-only/' + name + '.json', sha256=digest(name))


def seal_observation(row):
    row['observation_sha256'] = digest({k: v for k, v in row.items() if k != 'observation_sha256'})


def fixture():
    contract = dict(model_signature='mock-bf16-model', code_commit='1'*40,
        runtime_worktree_sha256='9'*64, patch_sha256='2'*64, execution_policy_sha256='3'*64,
        repair_policy_sha256='4'*64, first_reuse_layer=9,
        timing_scope='executor_prefill_start_not_request_arrival',
        content_group_id='synthetic-group', content_key='exact-token-content',
        comparison_depths=[1, 2], max_answer_f1_drop=.02)
    binding = digest(contract)
    sources = [dict(source_id=s, content_key=contract['content_key'], artifact_sha256=digest(s+'-full'),
        generation=i, publication_epoch=i+1, birth_request_id='birth-'+s,
        selection_k_sha256_by_depth={str(d): digest(s+'-k'+str(d)) for d in (1, 2)},
        evidence_ref=ref('source-'+s)) for i, s in enumerate(('A', 'B'))]
    requests = []
    for index, f1 in enumerate(({'A': 1., 'B': .5}, {'A': .5, 'B': 1.})):
        rid = 'q'+str(index)
        request = dict(request_id=rid, request_epoch=10+index, input_sha256=digest(rid),
            visible_source_ids=['A', 'B'],
            expected_repair_mask_sha256_by_source={s: digest(rid+s+'mask') for s in ('A', 'B')},
            dense=dict(contract_sha256=binding, request_id=rid, input_sha256=digest(rid),
                repair_mask_sha256=None, f1=1., duration_ms=100., evidence_ref=ref(rid+'dense')),
            source_outcomes={s: dict(contract_sha256=binding, request_id=rid, input_sha256=digest(rid),
                repair_mask_sha256=digest(rid+s+'mask'), f1=f1[s], duration_ms=60.,
                artifact_sha256=sources[i]['artifact_sha256'], evidence_ref=ref(rid+s))
                for i, s in enumerate(('A', 'B'))}, observations=[])
        for depth in (1, 2):
            # d1 chooses A for both requests; d2 correctly changes winner.
            scores = {'A': .1, 'B': .2} if depth == 1 or index == 0 else {'A': .2, 'B': .1}
            obs = dict(contract_sha256=binding, request_id=rid, input_sha256=request['input_sha256'],
                completed_depth=depth, current_k_sha256=digest(rid+'k'+str(depth)),
                source_k_sha256_by_id={s['source_id']: s['selection_k_sha256_by_depth'][str(depth)] for s in sources},
                scores=scores, selected_source_id=min(scores, key=scores.get), evidence_ref=ref(rid+'d'+str(depth)))
            seal_observation(obs)
            request['observations'].append(obs)
        requests.append(request)
    return dict(kind='source_selection_identifiability_input_v2', evidence_origin='mock_cpu_test',
                cohort_kind='historical_source_cohort',
                contract=contract, contract_sha256=binding, sources=sources, requests=requests)


class SourceSelectionIdentifiabilityAuditTests(unittest.TestCase):
    def analyze(self, value=None):
        return analyze_source_selection_identifiability_v2(fixture() if value is None else value)

    def test_actual_qa_oracle_headroom_and_safe_coverage(self):
        report = self.analyze()
        summary = report['summary']
        self.assertEqual(summary['status'], 'EVALUATED')
        self.assertEqual(summary['oracle_mean_f1'], 1.)
        self.assertEqual(summary['posthoc_best_fixed_mean_f1'], .75)
        self.assertEqual(summary['headroom_f1'], .25)
        self.assertEqual(summary['oracle_safe_coverage'], 1.)
        self.assertEqual(summary['best_fixed_safe_coverage'], .5)
        self.assertEqual(summary['safe_coverage_headroom'], .5)

    def test_selected_source_qa_regret_is_not_residual_regret(self):
        result = self.analyze()['summary']['selection_by_depth']
        self.assertEqual(result['1']['mean_qa_regret'], .25)
        self.assertEqual(result['2']['mean_qa_regret'], 0.)
        self.assertEqual(result['1']['selected_quality_safe_coverage'], .5)
        self.assertEqual(result['2']['selected_quality_safe_coverage'], 1.)

    def test_definitions_match_existing_dominance_utility(self):
        from scripts.server.audit_source_dominance import summarize_group
        value = fixture()
        old = summarize_group([dict(dense_f1=r['dense']['f1'],
            f1_by_source={s: v['f1'] for s, v in r['source_outcomes'].items()}) for r in value['requests']], .02)
        new = self.analyze(value)['summary']
        for key in ('oracle_mean_f1', 'fixed_mean_f1', 'fixed_safe_coverage', 'oracle_safe_coverage'):
            self.assertEqual(old[key], new[key])

    def test_same_qa_all_sources_not_false_complementarity(self):
        value = fixture()
        for r in value['requests']:
            for cell in r['source_outcomes'].values():
                cell['f1'] = 1.
        result = self.analyze(value)
        self.assertEqual(result['summary']['headroom_f1'], 0.)
        self.assertEqual(result['summary']['safe_coverage_headroom'], 0.)
        self.assertEqual(result['summary']['selection_by_depth']['1']['oracle_tie_hit_count'], 2)

    def test_qa_tie_is_not_wrong_lock(self):
        value = fixture()
        value['requests'][0]['source_outcomes']['B']['f1'] = 1.
        obs = value['requests'][0]['observations'][0]
        obs['scores'] = {'A': .1, 'B': .1}
        obs['selected_source_id'] = 'B'
        seal_observation(obs)
        row = self.analyze(value)['requests'][0]['selection_by_depth']['1']
        self.assertTrue(row['qa_oracle_tie_hit'])
        self.assertEqual(row['qa_regret'], 0.)

    def test_missing_source_qa_does_not_impute_zero(self):
        value = fixture()
        value['requests'][0]['source_outcomes']['B']['f1'] = None
        result = self.analyze(value)
        self.assertEqual(result['summary']['status'], 'NOT_EVALUATED')
        self.assertIsNone(result['summary']['headroom_f1'])
        self.assertIsNone(result['requests'][0]['oracle_max_f1'])
        self.assertIsNone(result['requests'][0]['selection_by_depth']['1']['qa_regret'])
        self.assertEqual(result['summary']['qa_complete_request_count'], 1)

    def test_missing_dense_qa_prevents_safe_coverage(self):
        value = fixture()
        value['requests'][0]['dense']['f1'] = None
        result = self.analyze(value)
        self.assertIsNone(result['summary']['oracle_safe_coverage'])
        self.assertIsNone(result['requests'][0]['quality_safe_source_ids'])

    def test_missing_duration_is_not_zero_or_time_oracle(self):
        value = fixture()
        for r in value['requests']:
            r['dense'].pop('duration_ms')
            for cell in r['source_outcomes'].values():
                cell['duration_ms'] = None
        result = self.analyze(value)
        self.assertEqual(result['summary']['status'], 'EVALUATED')
        self.assertIsNone(result['requests'][0]['dense']['duration_ms'])
        self.assertFalse(result['timing_oracle_evaluated'])

    def test_same_shallow_k_different_full_artifacts_is_reported(self):
        value = fixture()
        value['sources'][1]['selection_k_sha256_by_depth']['1'] = value['sources'][0]['selection_k_sha256_by_depth']['1']
        for r in value['requests']:
            obs = r['observations'][0]
            obs['source_k_sha256_by_id']['B'] = obs['source_k_sha256_by_id']['A']
            obs['scores']['B'] = obs['scores']['A']
            seal_observation(obs)
        result = self.analyze(value)
        collisions = result['candidate_shallow_collisions']
        self.assertEqual(len(collisions), 2)
        self.assertTrue(all(c['artifact_differs'] for c in collisions))
        self.assertTrue(all(c['absolute_f1_difference'] == .5 for c in collisions))
        self.assertFalse(any(c['scores_disagree'] for c in collisions))

    def test_identical_observations_but_disjoint_qa_winners(self):
        value = fixture()
        second = value['requests'][1]['observations'][0]
        second['current_k_sha256'] = value['requests'][0]['observations'][0]['current_k_sha256']
        seal_observation(second)
        result = self.analyze(value)['cross_request_shallow_equivalence']
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]['no_common_qa_optimal_source'])
        self.assertTrue(result[0]['qa_vectors_differ'])
        self.assertEqual(result[0]['common_qa_oracle_source_ids'], [])

    def test_collision_with_missing_qa_does_not_claim_risk(self):
        value = fixture()
        second = value['requests'][1]['observations'][0]
        second['current_k_sha256'] = value['requests'][0]['observations'][0]['current_k_sha256']
        seal_observation(second)
        value['requests'][0]['source_outcomes']['A']['f1'] = None
        result = self.analyze(value)['cross_request_shallow_equivalence'][0]
        self.assertEqual(result['status'], 'NOT_EVALUATED')
        self.assertIsNone(result['no_common_qa_optimal_source'])

    def test_variable_causal_pool_does_not_use_future_best_fixed(self):
        value = fixture()
        value['sources'][1]['publication_epoch'] = 10
        first = value['requests'][0]
        first['visible_source_ids'] = ['A']
        first['source_outcomes'].pop('B')
        first['expected_repair_mask_sha256_by_source'].pop('B')
        for obs in first['observations']:
            obs['scores'].pop('B')
            obs['source_k_sha256_by_id'].pop('B')
            seal_observation(obs)
        result = self.analyze(value)
        self.assertEqual(result['summary']['status'], 'UNSUPPORTED_FIXED_COHORT')
        self.assertIsNone(result['summary']['posthoc_best_fixed_mean_f1'])
        self.assertIsNone(result['summary']['headroom_f1'])
        self.assertEqual(result['requests'][0]['oracle_max_f1'], 1.)

    def test_future_or_current_epoch_source_rejected(self):
        for epoch in (10, 100):
            value = fixture()
            value['sources'][0]['publication_epoch'] = epoch
            with self.subTest(epoch=epoch), self.assertRaisesRegex(ValueError, 'future/current'):
                self.analyze(value)

    def test_request_cannot_read_own_source_even_if_epoch_forged(self):
        value = fixture()
        value['sources'][0]['birth_request_id'] = 'q0'
        with self.assertRaisesRegex(ValueError, 'future/current'):
            self.analyze(value)

    def test_candidate_coverage_mismatch_rejected(self):
        for field in ('source_outcomes', 'scores', 'source_k_sha256_by_id'):
            value = fixture()
            if field == 'source_outcomes':
                value['requests'][0][field].pop('B')
            else:
                obs = value['requests'][0]['observations'][0]
                obs[field].pop('B')
                seal_observation(obs)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.analyze(value)

    def test_wrong_artifact_or_execution_contract_rejected(self):
        for field in ('artifact_sha256', 'contract_sha256'):
            value = fixture()
            value['requests'][0]['source_outcomes']['A'][field] = 'a'*64
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.analyze(value)

    def test_qa_from_another_request_cannot_enter_current_matrix(self):
        value = fixture()
        value['requests'][0]['source_outcomes']['A'] = copy.deepcopy(value['requests'][1]['source_outcomes']['A'])
        with self.assertRaisesRegex(ValueError, 'QA request/input'):
            self.analyze(value)

    def test_actual_mask_cannot_be_changed_under_same_repair_policy(self):
        value = fixture()
        value['requests'][0]['source_outcomes']['A']['repair_mask_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'actual repair mask'):
            self.analyze(value)
        value = fixture()
        value['requests'][0]['dense']['repair_mask_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'actual repair mask'):
            self.analyze(value)

    def test_dirty_worktree_digest_required_not_only_head(self):
        value = fixture()
        value['contract'].pop('runtime_worktree_sha256')
        with self.assertRaisesRegex(ValueError, 'complete frozen'):
            self.analyze(value)

    def test_identical_signature_score_conflict_is_not_information_limit(self):
        value = fixture()
        second = value['requests'][1]['observations'][0]
        second['current_k_sha256'] = value['requests'][0]['observations'][0]['current_k_sha256']
        second['scores']['A'] = .05
        seal_observation(second)
        result = self.analyze(value)['cross_request_shallow_equivalence'][0]
        self.assertEqual(result['status'], 'INCONSISTENT_SCORE_OR_SELECTION_BINDING')
        self.assertIsNone(result['no_common_qa_optimal_source'])
        self.assertFalse(result['shallow_information_insufficiency_evaluable'])

    def test_identical_signature_choice_conflict_is_not_information_limit(self):
        value = fixture()
        for request in value['requests']:
            obs = request['observations'][0]
            obs['current_k_sha256'] = 'c'*64
            obs['scores'] = {'A': .1, 'B': .1}
            obs['selected_source_id'] = 'A' if request['request_id'] == 'q0' else 'B'
            seal_observation(obs)
        result = self.analyze(value)['cross_request_shallow_equivalence'][0]
        self.assertFalse(result['identical_signature_choices_consistent'])
        self.assertIsNone(result['no_common_qa_optimal_source'])

    def test_controlled_origin_pair_cannot_claim_natural_headroom(self):
        value = fixture()
        value.update(cohort_kind='controlled_origin_pair', evidence_origin='controlled_provenance_diagnostic')
        result = self.analyze(value)
        self.assertEqual(result['summary']['status'], 'NOT_APPLICABLE_CONTROLLED_ORIGIN_PAIR')
        self.assertIsNone(result['summary']['headroom_f1'])
        self.assertFalse(result['historical_headroom_applicable'])
        self.assertIsNone(result['natural_multisource_go_decision'])
        self.assertEqual(result['requests'][0]['qa_status'], 'EVALUATED')
        value['cohort_kind'] = 'historical_source_cohort'
        with self.assertRaisesRegex(ValueError, 'controlled provenance'):
            self.analyze(value)

    def test_bad_observation_digest_rejected(self):
        value = fixture()
        value['requests'][0]['observations'][0]['scores']['A'] = .05
        with self.assertRaisesRegex(ValueError, 'content digest'):
            self.analyze(value)

    def test_wrong_request_or_tensor_identity_rejected(self):
        for field in ('request_id', 'input_sha256', 'contract_sha256', 'current_k_sha256'):
            value = fixture()
            obs = value['requests'][0]['observations'][0]
            obs[field] = 'wrong'
            seal_observation(obs)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.analyze(value)

    def test_missing_or_invalid_evidence_digest_rejected(self):
        for target in ('source', 'dense', 'observation'):
            value = fixture()
            cell = (value['sources'][0] if target == 'source' else value['requests'][0]['dense']
                    if target == 'dense' else value['requests'][0]['observations'][0])
            cell['evidence_ref']['sha256'] = None
            if target == 'observation':
                seal_observation(cell)
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.analyze(value)

    def test_no_external_file_verification_claim_or_filesystem_access(self):
        from unittest.mock import patch
        with patch('builtins.open', side_effect=AssertionError('must not read external evidence')):
            report = self.analyze()
        self.assertFalse(report['references_verified'])
        self.assertIn('not independently verified', report['evidence_authenticity'])

    def test_qualification_flags_cannot_be_injected(self):
        for name in ('paper_evidence', 'gpu_execution_allowed', 'P1_execution_allowed'):
            value = fixture()
            value['requests'][0][name] = True
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'authority'):
                self.analyze(value)
        report = self.analyze()
        for name in ('paper_evidence', 'gpu_execution_allowed', 'P1_execution_allowed', 'gpu_runtime_qualified'):
            self.assertIs(report[name], False)

    def test_nonfinite_invalid_f1_boolean_and_negative_time_rejected(self):
        for value in (-.1, 1.1, True, float('nan'), float('inf')):
            payload = fixture()
            payload['requests'][0]['dense']['f1'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.analyze(payload)
        for value in (-1, 0, True):
            payload = fixture()
            payload['requests'][0]['dense']['duration_ms'] = value
            with self.subTest(duration=value), self.assertRaises(ValueError):
                self.analyze(payload)

    def test_unknown_g2_or_duplicate_source_rejected(self):
        for generation in (None, 2, True):
            value = fixture()
            value['sources'][0]['generation'] = generation
            with self.subTest(generation=generation), self.assertRaises(ValueError):
                self.analyze(value)
        value = fixture()
        value['sources'][1]['source_id'] = 'A'
        with self.assertRaises(ValueError):
            self.analyze(value)

    def test_nonminimum_selected_source_rejected(self):
        value = fixture()
        obs = value['requests'][0]['observations'][0]
        obs['selected_source_id'] = 'B'
        seal_observation(obs)
        with self.assertRaisesRegex(ValueError, 'residual minimum'):
            self.analyze(value)

    def test_abstention_has_null_regret_not_zero_and_stays_in_coverage_denominator(self):
        value = fixture()
        obs = value['requests'][0]['observations'][0]
        obs['selected_source_id'] = None
        seal_observation(obs)
        result = self.analyze(value)
        self.assertIsNone(result['requests'][0]['selection_by_depth']['1']['qa_regret'])
        summary = result['summary']['selection_by_depth']['1']
        self.assertEqual(summary['abstention_count'], 1)
        self.assertEqual(summary['selected_quality_safe_coverage'], 0.)

    def test_request_order_duplicates_and_wrong_content_rejected(self):
        for mutate in (lambda x: x['requests'].reverse(),
                       lambda x: x['requests'][1].update(request_id='q0'),
                       lambda x: x['sources'][0].update(content_key='wrong')):
            value = fixture()
            mutate(value)
            with self.assertRaises(ValueError):
                self.analyze(value)

    def test_all_input_immutable_report_deterministic_json_roundtrip(self):
        value = fixture()
        before = copy.deepcopy(value)
        report = self.analyze(value)
        self.assertEqual(value, before)
        self.assertEqual(report, self.analyze(json.loads(json.dumps(value))))
        self.assertEqual(report['report_sha256'], digest({k:v for k,v in report.items() if k != 'report_sha256'}))
        report['contract']['model_signature'] = 'mutated-result'
        self.assertEqual(value, before)

    def test_threshold_must_be_explicit_and_bound_before_analysis(self):
        value = fixture()
        value['contract'].pop('max_answer_f1_drop')
        with self.assertRaises(ValueError):
            self.analyze(value)
        value = fixture()
        value['contract']['max_answer_f1_drop'] = .1
        with self.assertRaisesRegex(ValueError, 'contract digest'):
            self.analyze(value)


if __name__ == '__main__':
    unittest.main()
