"""Prepare append-only real development inputs, then a frozen U/off run config.

Only stdlib/source-only CPU consumers are used. No model, CUDA, cache engine,
namespace directory, external deadline, preview budget or GPU authority is made.
The existing original guarded runner alone may execute the later U/off job.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

A = 'artifacts/prefix_io_v1/server12-natural-source-audit-20261005'
P = 'artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004'
G = 'artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005'
PAIR = A + '/ACTUAL_PUBLIC_DEVELOPMENT_U_I_CONFIG_01.json'
NAMESPACE = A + '/ACTUAL_PUBLIC_NAMESPACE_CONTRACT_01.json'
INPUT = A + '/ACTUAL_PUBLIC_DEVELOPMENT_BINDING_01.json'
ACTIVATION = A + '/ACTUAL_PUBLIC_DEVELOPMENT_ACTIVATION_01.json'
CONFIG = A + '/ACTUAL_PUBLIC_DEVELOPMENT_UOFF_RUN_CONFIG_01.json'
U = 'server12-public-development-uoff01'
I = 'server12-public-development-ishadow01'
UUID = 'GPU-b2de2c25-cdc7-a350-267f-56e7763a287f'
RUNNER_SHA = '0519543ab496d54a11ce2b7bff08e69d77d8127976121dddfea2563bef337091'
BRIDGE_SHA = 'e7a10755f42f9edf24c04cbc1461bed6f5e15c7ec3cac3c53cdb43431bfec8f7'
V13_SHA = '1d1aef9591a8c319668260d56b1cbe55b8775f0f18b39ea89a36ca83952070b1'
LEDGER_SHA = '774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b'


def require(ok, reason):
    if not ok:
        raise ValueError('ACTUAL_DEVELOPMENT_PREPARATION_REJECTED: ' + reason)


def source_module(root, relative, expected, name):
    path = root / relative
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'stable production source: ' + relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    require(name not in sys.modules, 'fresh source-only private module')
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
    require(path.read_bytes() == raw, 'source changed while loading')
    return module


def write_new(root, relative, document):
    path = root / relative
    require(path.parent.is_dir() and not path.exists() and not path.is_symlink(), 'fresh existing-parent output')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def run(root, action):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only invocation')
    root = root.resolve(strict=True)
    ledger_path = root / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    before = ledger_path.read_bytes()
    require(hashlib.sha256(before).hexdigest() == LEDGER_SHA and
            json.loads(before)['active_reservation'] is None, 'unchanged idle original GPU ledger')
    driver = source_module(root, P + '/runner/strong_trace_runner_v4.py', RUNNER_SHA, '_actual_development_cpu_driver')
    bridge = source_module(root, P + '/development_runtime_bridge/development_diagnostic_bridge.py',
                           BRIDGE_SHA, '_actual_development_cpu_bridge')
    if action == 'prepare':
        lock_ref = driver.ref(root, A + '/PRERENT_SOURCE_LOCK_V13.json')
        require(lock_ref['sha256'] == V13_SHA, 'actual V13 ancestor')
        rows = driver.read(driver.check_ref(root, lock_ref))['files']
        require(len(rows) == 4990, 'full actual ancestor row count')
        refs = {row['path']: row for row in rows}

        def leaf(relative):
            row = driver.ref(root, relative)
            if relative in refs:
                require(refs[relative] == row, 'existing source byte drift')
            else:
                refs[relative] = row
            return row

        original_pair_ref = leaf(P + '/runner/PRIVATE_U_I_CONFIG.json')
        original = driver.read(driver.check_ref(root, original_pair_ref))
        pair_doc = deepcopy(original)
        configurations = pair_doc['configurations']
        for arm, label in [('U', U), ('I', I)]:
            extra = configurations[arm]['engine']['kv_transfer_config']['kv_connector_extra_config']
            extra['shared_storage_path'] = str(root / 'experiments/prefix_io_v1/runs' / label / 'storage')
            extra['prefix_io_parent_admission']['run_id'] = label
            extra['prefix_io_observation_run_id'] = label
            if arm == 'I':
                extra['prefix_io_p4_policy']['run_id'] = label
                require(extra['prefix_io_p4_policy']['internal_step_budget_ns'] is None,
                        'no engineering threshold or SLO invented')
            require(not Path(extra['shared_storage_path']).exists(), 'prospective fresh arm namespace')
        require(driver.common_domain_sha(configurations) == driver.common_domain_sha(original['configurations']),
                'only original run IDs and storage namespace differ; common runtime retained')
        pair_doc.update(qualification_workload_kind='prospectively_fixed_public_document_QA_development_diagnostic',
                        natural_trace_bound=True, GPU_operations=0, gpu_effect_qualified=False,
                        source_pair_ref=original_pair_ref,
                        development_diagnostic_only=True,
                        shadow_engineering_budget_status='UNBOUND_REQUIRES_ACTUAL_U_COSTS_AND_A_FROZEN_ENGINEERING_SETTING',
                        production_request_arrivals_claim=False)
        write_new(root, PAIR, pair_doc)
        pair_ref = leaf(PAIR)
        manifest_ref = leaf(A + '/ACTUAL_PUBLIC_ORIGINAL_MANIFEST_01.json')
        manifest = driver.read(driver.check_ref(root, manifest_ref))
        namespaces = {}
        for part in ('calibration', 'development', 'evaluation'):
            namespaces[part] = {}
            for arm in ('U', 'I'):
                label = U if arm == 'U' else I
                if part != 'development':
                    label = 'server12-public-' + part + '-' + arm.lower() + '-future01'
                absolute = str(root / 'experiments/prefix_io_v1/runs' / label / 'storage')
                require(not Path(absolute).exists(), 'each partition/arm namespace initially absent')
                namespaces[part][arm] = absolute
        namespace = dict(schema='formal_trace_namespace_contract_v1',
            workload_sha256=manifest['workload_sha256'], model_manifest_sha256=manifest['model_manifest_sha256'],
            tokenizer_receipt_digest=manifest['tokenizer_receipt_digest'],
            common_runtime_domain_sha256=driver.common_domain_sha(configurations),
            initial_cache_state='fresh_equal_namespace_preserved_through_whole_partition',
            no_per_request_reset=True, partition_namespaces=namespaces)
        write_new(root, NAMESPACE, namespace)
        real_inputs = dict(manifest_ref=manifest_ref,
            dataset_ref=leaf(A + '/ACTUAL_SQUAD_AUTHOR_TRACE_02.json'),
            author_trace_ref=leaf('third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py'),
            author_common_ref=leaf('third_party/upstream/kvcache-experiments/common/prefix_cache_common.py'),
            protocol_source_ref=leaf(P + '/protocol/prerental_protocol.py'),
            declaration_ref=leaf(A + '/ACTUAL_PUBLIC_REPLAY_DECLARATION_01.json'),
            tokenizer_receipt_ref=leaf(A + '/ACTUAL_CPU_TOKENIZER_FAMILY_RECEIPT_01.json'),
            model_manifest_ref=leaf(driver.MODEL_PLAN), namespace_contract_ref=leaf(NAMESPACE),
            strong_pair_validator_ref=leaf(driver.STRONG))
        role = dict(phase='development', mode='off', arm='U', pair_config_ref=pair_ref)
        binding = bridge.prepare_binding_descriptor(root, real_inputs, refs=refs, config=role, driver=driver)
        write_new(root, INPUT, binding['descriptor'])
        plan_ref = leaf(G + '/raw05/BOUND_PLAN.json')
        plan = driver.read(driver.check_ref(root, plan_ref))
        cost_report = driver.read(driver.check_ref(root, leaf(G + '/raw05/ACTUAL_COST_ISSUER_REPORT.json')))
        require(cost_report['plan_ref'] == plan_ref and cost_report['status'] == 'PASS_ACTUAL_EXACT_GPU_COST_CELL',
                'actual original calibration, not a fabricated cost receipt')
        require(plan['gpu_uuid'] == UUID and plan['common_runtime_domain_sha256'] == driver.common_domain_sha(configurations),
                'existing actual calibration device/common runtime domain')
        real_activation = dict(plan_ref=plan_ref, measurements_ref=cost_report['measurements_ref'],
            completed_guard_ref=cost_report['guard_ref'], intent_ref=cost_report['intent_ref'],
            calibration_source_lock_ref=plan['source_lock_ref'], collector_ref=plan['collector_source_ref'],
            protocol_source_ref=real_inputs['protocol_source_ref'], issuer_source_ref=plan['issuer_source_ref'],
            cost_table_source_ref=plan['cost_table_source_ref'],
            finite_binding_ref=leaf(P + '/runner/finite_current_binding.py'),
            startup_ref=leaf(P + '/runner/finite_startup.py'),
            reserve_join_source_ref=leaf(P + '/activation/control_observation/reserve_join.py'))
        for key in bridge.POST_DEVELOPMENT:
            real_activation[key] = None
        for row in real_activation.values():
            if row is not None:
                leaf(row['path'])
                require(refs[row['path']] == row, 'actual original evidence refs inherited unchanged')
        activation = bridge.prepare_activation_descriptor(root, real_activation, refs=refs, config=role, driver=driver)
        write_new(root, ACTIVATION, activation['descriptor'])
        result = dict(schema='actual_CPU_development_descriptor_preparation_v1',
            status='PASS_REAL_INPUT_DESCRIPTORS_PREPARED_REQUIRES_V14_AND_FULL_PRODUCTION_CPU_PREFLIGHT',
            pair_config_ref=pair_ref, namespace_ref=leaf(NAMESPACE), binding_ref=leaf(INPUT), activation_ref=leaf(ACTIVATION),
            common_runtime_domain_sha256=namespace['common_runtime_domain_sha256'],
            all_source_questions_retained=True, development_requests=manifest['partition_counts']['development'],
            calibration_and_evaluation_not_executed=True, service_SLO=None,
            independent_deadline_ref=None, shadow_engineering_budget=None,
            ordinary_I_authorized=False, ready_for_GPU=False, actual_GPU_operations=0,
            source_lock_inputs_not_yet_frozen=True)
        write_new(root, A + '/ACTUAL_PUBLIC_DEVELOPMENT_PREPARATION_01.json', result)
    else:
        lock_ref = driver.ref(root, A + '/PRERENT_SOURCE_LOCK_V14.json')
        proof_ref = driver.ref(root, A + '/PRERENT_SOURCE_PROOF_V14.json')
        proof = driver.read(driver.check_ref(root, proof_ref))
        require(proof['source_lock_ref'] == lock_ref and proof['status'] == 'PASS_FULL_CPU_SOURCE_BYTES',
                'completed actual whole-byte V14 source freeze')
        rows = driver.read(driver.check_ref(root, lock_ref))['files']
        refs = {row['path']: row for row in rows}

        def leaf(relative):
            row = driver.ref(root, relative)
            require(refs.get(relative) == row, 'actual config dependency frozen in V14: ' + relative)
            return row

        config = dict(schema=driver.SCHEMA, phase='development', arm='U', mode='off', run_id=U,
            source_lock_ref=lock_ref, source_proof_ref=proof_ref, pair_config_ref=leaf(PAIR),
            workload_ref=leaf(A + '/ACTUAL_PUBLIC_ORIGINAL_MANIFEST_01.json'),
            permissions_ref=leaf(P + '/live-off01/EFFECTIVE_GPU_PERMISSION.json'), gpu_uuid=UUID,
            seconds_limit=300, storage_reserve_bytes=128 * 1024**2, storage_floor_bytes=8 * 1024**3,
            output_relative='experiments/prefix_io_v1/runs/' + U + '/details',
            runner_ref=leaf(P + '/runner/strong_trace_runner_v4.py'),
            runtime_ref=leaf(P + '/runner/native_runtime_v4.py'), off_qualification_ref=None,
            activation_ref=leaf(ACTIVATION), formal_trace_binding_ref=leaf(INPUT), formal_peer_closed_ref=None)
        write_new(root, CONFIG, config)
        result = dict(schema='actual_CPU_development_run_configuration_preparation_v1',
            status='REAL_UOFF_CONFIG_CREATED_REQUIRES_ACTUAL_RUNNER_CPU_PREFLIGHT',
            config_ref=driver.ref(root, CONFIG), config_outside_its_source_lock_avoids_hash_cycle=True,
            guarded_future_command=driver.guard_command(root, CONFIG, config),
            no_command_launched=True, actual_GPU_operations=0,
            service_SLO=None, formal_effect_qualified=False)
        write_new(root, A + '/ACTUAL_PUBLIC_UOFF_CONFIG_PREPARATION_01.json', result)
    require(ledger_path.read_bytes() == before, 'original GPU ledger unchanged')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--action', choices=('prepare', 'config'), required=True)
    args = parser.parse_args()
    run(args.project, args.action)


if __name__ == '__main__':
    main()
