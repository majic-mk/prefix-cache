"""Validate this design package only; never authorize model/GPU execution."""
from __future__ import annotations
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent


def validate_contract(c: dict) -> list[str]:
    passed = []
    def require(condition: bool, name: str):
        if not condition:
            raise ValueError(name)
        passed.append(name)

    require(c['document_kind']=='design_contract_draft', 'draft_not_runtime_config')
    require(c['directly_executable_runtime_config'] is False, 'not_CLI_config')
    require(c['frozen'] is False, 'not_pre_frozen')
    require(c['scope']['mainline']=='multi_source_decoupled', 'multisource_preserved')
    require(c['scope']['source_selection_uses_query'] is False, 'source_query_decoupled')
    require(c['scope']['repair_uses_query_and_drift'] is True, 'joint_repair_preserved')
    require(c['provenance']['allow_approximate_upstream'] is True, 'mixed_upstream_allowed')
    require(c['provenance']['require_exact_whole_request_for_target_capture'] is False, 'no_upstream_dense_requirement')
    require(c['provenance']['target_full_every_token_every_layer'] is True, 'target_full_required')
    require(c['provenance']['max_stored_generation']==1, 'generation_one_main_cap')
    require(c['provenance']['within_request_full_propagation']=='max_input_rank_without_increment', 'no_generation_washing_or_per_layer_growth')
    require(c['provenance']['mixed_allowed_in_exact_prefix_cache'] is False, 'mixed_not_exact_prefix')
    require(c['provenance']['unknown_origin_action']=='reject_publication', 'unknown_rejected')
    require(c['storage']['artifact_owns_target_KV_only'] is True, 'target_only_artifact')
    for key in ('store_prefix_KV_per_source','own_parent_KV_through_views_or_leases',
                'source_capture_creates_prefix_shadow','audit_reference_pins_parent_KV',
                'implicit_extra_capture_forward'):
        require(c['storage'][key] is False, f'forbidden_storage_{key}')
    require(c['storage']['max_variants_total_across_origin_types']==4, 'shared_total_K_cap')
    require(c['growth']['birth_request_can_consume_own_new_source'] is False, 'no_self_history')
    require(c['growth']['mismatch_requires_stored_eligible_compared_equal'] is True, 'honest_full_pool_scope')
    require(c['data']['locked_test_accessed'] is False, 'locked_test_preserved')
    require(c['data']['reference_dense_can_populate_runtime_pool'] is False, 'no_oracle_pool_leakage')
    require(c['data']['ten_segment_case_uses_new_geometry'] is True, 'new_geometry_explicit')
    require(10*c['data']['legacy_target_length'] > c['data']['legacy_prompt_limit'], 'legacy_10x512_conflict_detected')
    require(c['data']['new_ten_segment_prompt_limit_proposal'] >= 10*c['data']['legacy_target_length'], 'proposal_has_target_token_capacity')
    require(c['repair']['primary']=='QxD_global' and c['repair']['ratio']==0.15, 'primary_formula_and_ratio')
    require(c['repair']['full_causal_global_softmax'] is True, 'global_causal_denominator')
    require(c['stages']['P1M']['must_execute_for_v2_claims'] is True, 'mixed_origin_trial_required')
    for stage_id in ('P1E','P1M'):
        s=c['stages'][stage_id]
        require(s['main_target_action_cap']==s['target_cap']*s['max_actions_per_target'], f'{stage_id}_bounded_action_count')
        require(s['source_creation_actions_bounded_separately'] is True, f'{stage_id}_construction_not_free')
    require(c['rescue']['max_batches']==1, 'bounded_rescue')
    require(c['rescue']['main_target_action_cap']==c['rescue']['target_cap']*c['rescue']['max_actions_per_target'], 'rescue_action_count')
    require(c['stages']['P5']['same_total_K'] and c['stages']['P5']['same_byte_budget'], 'fair_pool_budget')
    require(c['authority']['gpu_execution_enabled'] is False and c['authority']['automatic_GPU_rental_or_payment'] is False, 'no_gpu_authorization_in_package')
    for key in ('paper_evidence','gpu_runtime_qualified','real_model_executed_in_package','remote_repository_modified'):
        require(c['evidence'][key] is False, f'evidence_boundary_{key}')
    visiting, done = set(), set()
    def visit(k):
        require(k in c['stages'], f'known_stage_{k}')
        if k in done:
            return
        if k in visiting:
            raise ValueError('Cyclic experiment stages')
        visiting.add(k)
        for parent in c['stages'][k]['requires']:
            visit(parent)
        visiting.remove(k); done.add(k)
    for k in c['stages']:
        visit(k)
    passed.append('stage_dependency_DAG_acyclic')
    return passed


def run():
    c=json.loads((ROOT/'04_实验契约草案.json').read_text(encoding='utf-8'))
    checks=validate_contract(c)
    for filename in ('00_阅读入口.md','01_系统方案与实验任务书_v2.md','02_发给Codex的执行指令_v2.md',
                     '03_逻辑问题与修订清单.md','05_测试与验收矩阵.md','06_依据与查新登记.md',
                     'reference/README.md','reference/provenance_reference.py','reference/test_provenance_reference.py'):
        text=(ROOT/filename).read_text(encoding='utf-8')
        if not text.strip():
            raise ValueError(f'Empty file: {filename}')
        if filename.endswith('.md'):
            fences=sum(1 for l in text.splitlines() if l.lstrip().startswith('```'))
            if fences%2:
                raise ValueError(f'Unbalanced fence: {filename}')
        elif filename.endswith('.py'):
            ast.parse(text, filename)
    checks.append('required_files_nonempty_fences_and_python_syntax')
    main=(ROOT/c['source_of_truth']).read_text(encoding='utf-8')
    for item in ('P1-M','TARGET_FULL_RECOMPUTE','ALLOW_MIXED_G1','MIXED_CONTEXT_FULL_SEGMENT',
                 '浅层签名碰撞','5120','8192','Prefix shadow','G=2'):
        if item not in main:
            raise ValueError(f'Missing key contract phrase: {item}')
    checks.append('critical_revisions_in_main_document')
    mutations=[('provenance','allow_approximate_upstream',False),
               ('provenance','max_stored_generation',2),
               ('storage','store_prefix_KV_per_source',True),
               ('storage','source_capture_creates_prefix_shadow',True),
               ('scope','source_selection_uses_query',True),
               ('scope','repair_uses_query_and_drift',False),
               ('data','reference_dense_can_populate_runtime_pool',True),
               ('evidence','paper_evidence',True)]
    negative=[]
    for section,key,bad in mutations:
        changed=copy.deepcopy(c); changed[section][key]=bad
        try:
            validate_contract(changed)
        except ValueError:
            negative.append(f'{section}.{key}')
        else:
            raise RuntimeError(f'Invalid mutation accepted: {section}.{key}')
    output=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'reference'), pattern='test_*.py')
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    (ROOT/'reference/policy_test_log.txt').write_text(output.getvalue(),encoding='utf-8')
    report={
        'artifact_kind':'design_package_validation_only','plan_version':'2.0',
        'python_version':platform.python_version(),
        'contract_checks_passed':checks,
        'invalid_contract_mutations_rejected':negative,
        'policy_reference_tests':{'run':result.testsRun,'failures':len(result.failures),
                                  'errors':len(result.errors),'skipped':len(result.skipped),
                                  'successful':result.wasSuccessful()},
        'real_model_executed':False,'repository_tests_executed':False,
        'gpu_experiment_executed':False,'remote_repository_modified':False,
        'performance_claim':None,'paper_evidence':False,
        'contract_sha256':hashlib.sha256((ROOT/'04_实验契约草案.json').read_bytes()).hexdigest(),
        'note':'No model quality, GPU numerical behavior, storage integration, or runtime provenance has been certified.'}
    (ROOT/'07_本包校验记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'contract_checks':len(checks),'invalid_mutations_rejected':len(negative),
                      'policy_tests':result.testsRun,'passed':result.wasSuccessful()},ensure_ascii=False))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__=='__main__':
    run()
