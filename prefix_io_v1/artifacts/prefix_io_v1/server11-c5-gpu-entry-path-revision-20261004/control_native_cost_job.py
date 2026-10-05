"""Prepare a reviewable UNBOUND template or bind one explicitly granted live job.

No GPU imports, downloads, system edits, cache deletion, retry or budget reset.
The unchanged original guard exclusively launches and accounts the native job.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_entry_binding as B

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def put(root, relative, document):
    path = B.safe(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return B.ref(root, relative)


def prepare(root, output_relative):
    # This action creates no valid target, grant, permission, config or scope.
    if (not isinstance(output_relative, str)
            or not Path(output_relative).name.startswith('UNBOUND_REVIEW_TEMPLATE')
            or not output_relative.endswith('.json')):
        B.reject('prepare output must be an explicitly UNBOUND review template')
    commands = {
        'context_after_gpu_available': ['.venv/bin/python', '-B', B.DIR + '/control_native_cost_job.py', 'context'],
        'bind_after_explicit_human_gpu_grant': ['.venv/bin/python', '-B', B.DIR + '/control_native_cost_job.py', 'bind'],
        'plan': ['.venv/bin/python', '-B', B.DIR + '/prepare_and_verify_native_cost.py', '--prepare',
                 '--project', str(Path(root).resolve()), '--source-lock', B.SITE_LOCK,
                 '--config', B.CONFIG, '--plan', B.DIR + '/NATIVE_COST_PLAN.json',
                 '--plan-ref', B.DIR + '/PLAN_REFERENCE.json'],
        'scope': ['.venv/bin/python', '-B', B.DIR + '/control_native_cost_job.py', 'scope', '--plan-relative', B.DIR + '/NATIVE_COST_PLAN.json'],
        'launch': ['.venv/bin/python', '-B', B.DIR + '/control_native_cost_job.py', 'launch'],
        'after_original_guard_completed': ['.venv/bin/python', '-B', B.DIR + '/control_native_cost_job.py', 'after'],
        'verify_after_full_after_audit': ['.venv/bin/python', '-B', B.DIR + '/prepare_and_verify_native_cost.py', '--verify',
                 '--project', str(Path(root).resolve()), '--plan', B.DIR + '/NATIVE_COST_PLAN.json',
                 '--plan-ref', B.DIR + '/PLAN_REFERENCE.json',
                 '--runtime', B.RUN + '/details/native-cost-runtime-result.json',
                 '--guard', B.RUN + '/result.json',
                 '--before-source', B.DIR + '/SOURCE_BEFORE_VERIFICATION.json',
                 '--after-source', B.DIR + '/SOURCE_AFTER_VERIFICATION.json',
                 '--measurements', B.DIR + '/NATIVE_PAIRED_MEASUREMENTS.json',
                 '--result', B.DIR + '/NATIVE_COST_VERIFICATION.json'],
    }
    value = dict(schema='c5_gpu_entry_unbound_review_template_v1', status='UNBOUND',
        gpu_uuid=None, effective_permission=None, valid_config=None, site_binding=None,
        human_grant=None, effective_cost_upper_ns=None, effective_step_budget_ns=None,
        native_receipt=None, gpu_launch_allowed=False, actual_gpu_runs=0,
        native_execution_verified=False, native_cost_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        proposed_label=B.LABEL, seconds_limit=1200, reserved_seconds=1220,
        max_attempts=1, max_processes=6, original_budget_limit_seconds=28800,
        required_cpu_cores=1, required_available_memory_bytes=32 * 1024**3,
        required_free_gpu_mib=28000, primary_reserve_bytes=2 * 1024**3,
        primary_free_floor_bytes=8 * 1024**3, no_retry=True,
        human_gpu_grant_required=True, commands=commands,
        original_guard_relative=B.GUARD, original_guard_sha256=B.GUARD_SHA,
        source_ancestry_is_not_authority=True)
    result = put(root, output_relative, value)
    return dict(status='UNBOUND_REVIEW_TEMPLATE_WRITTEN', template_ref=result,
                gpu_started=False, actual_gpu_runs=0, gpu_launch_allowed=False)


def freeze_revision(root):
    ancestor_ref = B.ref(root, B.ANCESTOR)
    if ancestor_ref['sha256'] != B.ANCESTOR_SHA:
        B.reject('original source ancestry drift')
    original = B.read(root, B.ANCESTOR)
    if len(original.get('files', [])) != 4655:
        B.reject('original source ancestry incomplete')
    rows = B.checked_rows(root, original['files'])
    required = [B.GUARD, B.BASE_PERMISSION, B.ANCESTOR, B.MANIFEST]
    for path in sorted(B.safe(root, B.DIR).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            if path.suffix in ('.py', '.md') or path.name in (
                    'BASE_PACKAGE_COPY.json', 'CANDIDATE_MANIFEST.json', 'SOURCE_INHERITANCE.json',
                    'NATIVE_SOURCE_INHERITANCE.json'):
                required.append(path.relative_to(Path(root).resolve()).as_posix())
    manifest = B.read(root, B.MANIFEST)
    if manifest.get('file_count') != 24:
        B.reject('production input manifest count')
    for group in manifest.get('groups', []):
        required += [B.STORAGE + '/' + item['path'] for item in group['files']]
    required += [item['path'] for item in manifest.get('source_refs', [])]
    for relative in required:
        actual = B.ref(root, relative)
        if relative in rows and rows[relative] != actual:
            B.reject('old source cannot be rewritten')
        rows[relative] = actual
    document = dict(schema='c5_gpu_entry_revision_asset_lock_v1',
                    baseline_source_lock=ancestor_ref, files=[rows[key] for key in sorted(rows)],
                    source_ancestry_is_not_authority=True, gpu_launch_allowed=False,
                    production_qualified=False, strategy_effect_verified=False)
    lock_ref = put(root, B.REVISION_LOCK, document)
    B._revision(root, lock_ref, full=True)
    return lock_ref


def context(root):
    resources = B.probe_resources(root)
    B._resource_shape(resources)  # Fail before writing any live site binding.
    ledger = B.read(root, B.LEDGER)
    if ledger.get('active_reservation') is not None or B.number(ledger.get('gpu_wall_seconds'), 'budget') + 1220 > 28800:
        B.reject('original budget/reservation unavailable')
    if B.safe(root, B.RUN).exists():
        B.reject('label already used')
    revision_ref = freeze_revision(root)
    value = dict(schema='c5_gpu_entry_live_context_v1', origin='live_site_readonly_context',
        root=str(Path(root).resolve()), label=B.LABEL, revision_lock_ref=revision_ref,
        resources=resources, seconds_limit=1200, reserved_seconds=1220,
        max_attempts=1, max_processes=6, ledger_before_ref=B.ref(root, B.LEDGER),
        base_permission_ref=B.ref(root, B.BASE_PERMISSION), created_utc=utc(),
        GPU_runs_this_action=0, GPU_execution_authorized=False,
        human_grant_required=True)
    return dict(status='LIVE_CONTEXT_FROZEN_NEEDS_EXPLICIT_HUMAN_GPU_GRANT',
                context_ref=put(root, B.CONTEXT, value), gpu_launch_allowed=False, actual_gpu_runs=0)


def full_source_proof(root, phase, config, binding, refs):
    # The large-asset audit happens only outside native timed frames.
    actual = B.checked_rows(root, list(refs.values()))
    proof = dict(schema='c5_gpu_entry_full_source_verification_v1',
        status='FULL_BYTE_SOURCE_CLOSURE_VERIFIED', phase=phase, label=B.LABEL,
        gpu_uuid=binding['gpu_uuid'], source_lock_ref=B.ref(root, B.SITE_LOCK),
        revision_lock_ref=binding['revision_lock_ref'], config_ref=B.ref(root, B.CONFIG),
        binding_ref=B.ref(root, B.BINDING), context_ref=binding['context_ref'],
        human_grant_ref=binding['human_grant_ref'], permission_ref=binding['effective_permission_ref'],
        source_count=len(actual), files_verified=len(actual), source_rows_sha256=B.closure_digest(actual),
        failed=[], actual_verification_utc=utc(), GPU_operations=0)
    return put(root, B.DIR + '/SOURCE_' + phase.upper() + '_VERIFICATION.json', proof)


def bind(root):
    context_document = B.read(root, B.CONTEXT)
    actual_resource = B.probe_resources(root)
    uuid = B._resource_shape(actual_resource)
    if (context_document.get('origin') != 'live_site_readonly_context'
            or context_document.get('root') != str(Path(root).resolve())
            or context_document.get('label') != B.LABEL
            or B._resource_shape(context_document.get('resources')) != uuid):
        B.reject('live context target mismatch')
    revision_ref = context_document.get('revision_lock_ref')
    refs = B._revision(root, revision_ref, full=True)
    context_ref, grant_ref = B.ref(root, B.CONTEXT), B.ref(root, B.GRANT)
    grant = B.read(root, B.GRANT)
    expected = dict(schema='c5_gpu_entry_explicit_human_grant_v1', issuer='human_user',
                    authorization=True, context_ref=context_ref, gpu_uuid=uuid,
                    label=B.LABEL, revision_lock_ref=revision_ref, max_attempts=1,
                    max_processes=6, seconds_limit=1200, reserved_seconds=1220,
                    instruction_is_gpu_execution_authorization=True)
    if any(not B.exact(grant.get(key), value) for key, value in expected.items()):
        B.reject('explicit user grant source/UUID/job mismatch')
    if (not isinstance(grant.get('human_instruction'), str) or not grant['human_instruction'].strip()
            or grant['human_instruction'] in ('继续', '继续直至可以上gpu验证', '继续直至可以上GPU验证')):
        B.reject('preparation request cannot authorize execution')
    permission = dict(B.permission_document(root, B.BASE_PERMISSION))
    permission.update(allow_gpu_runs=True, approved_gpu_ids=[uuid], approved_auxiliary_storage=None)
    for key in B.DENIED: permission[key] = False
    permission['gpu_entry_binding'] = dict(label=B.LABEL, revision_lock_ref=revision_ref,
        context_ref=context_ref, human_grant_ref=grant_ref, seconds_limit=1200,
        reserved_seconds=1220, max_attempts=1, max_processes=6)
    permission_ref = put(root, B.PERMISSION, permission)
    binding = dict(schema='c5_gpu_entry_site_binding_v1', status='LIVE_HUMAN_BOUND',
        origin='live_site_binding', synthetic=False, root=str(Path(root).resolve()),
        label=B.LABEL, gpu_uuid=uuid, revision_lock_ref=revision_ref,
        context_ref=context_ref, human_grant_ref=grant_ref, effective_permission_ref=permission_ref,
        site_lock_relative=B.SITE_LOCK, config_relative=B.CONFIG)
    binding_ref = put(root, B.BINDING, binding)
    config = dict(root=str(Path(root).resolve()), label=B.LABEL, gpu_uuid=uuid,
        purpose=B.PURPOSE, seconds_limit=1200, reserved_seconds=1220,
        active_ledger=B.LEDGER, storage=B.STORAGE, out=B.RUN + '/details',
        source_lock=B.SITE_LOCK, input_manifest=B.MANIFEST,
        collector_relative=B.DIR + '/common_candidate/native_full_step_collector.py',
        gpu_entry_binding_ref=binding_ref)
    put(root, B.CONFIG, config)
    for path in (B.REVISION_LOCK, B.CONFIG, B.BINDING, B.CONTEXT, B.GRANT, B.PERMISSION):
        refs[path] = B.ref(root, path)
    put(root, B.SITE_LOCK, dict(schema='c5_gpu_entry_site_source_lock_v1',
        revision_lock_ref=revision_ref, files=[refs[key] for key in sorted(refs)],
        production_qualified=False, strategy_effect_verified=False))
    full_source_proof(root, 'before', config, binding, refs)
    B.verify_configuration(root, B.CONFIG)
    return dict(status='EXPLICIT_LIVE_SITE_BOUND_NOT_RUN', config_ref=B.ref(root, B.CONFIG),
                site_lock_ref=B.ref(root, B.SITE_LOCK), actual_gpu_runs=0)


def scope(root, plan_relative):
    config, refs, binding = B.verify_configuration(root, B.CONFIG)
    if plan_relative != B.DIR + '/NATIVE_COST_PLAN.json':
        B.reject('only current native common plan allowed')
    plan = B.read(root, plan_relative)
    if (plan.get('site_binding_ref') != config['gpu_entry_binding_ref']
            or plan.get('config_ref') != B.ref(root, B.CONFIG)
            or plan.get('source_lock_ref') != B.ref(root, B.SITE_LOCK)):
        B.reject('plan source/config/site binding mismatch')
    document = dict(schema='c5_gpu_entry_six_process_scope_v1', label=B.LABEL,
        gpu_uuid=binding['gpu_uuid'], permission_ref=binding['effective_permission_ref'],
        source_lock_ref=B.ref(root, B.SITE_LOCK), plan_ref=B.ref(root, plan_relative),
        binding_ref=config['gpu_entry_binding_ref'], human_grant_ref=binding['human_grant_ref'],
        seconds_limit=1200, reserved_seconds=1220, max_attempts=1, max_processes=6,
        primary_reserve_bytes=2 * 1024**3, primary_free_floor_bytes=8 * 1024**3,
        no_retry=True, valid_native_receipt=None, native_cost_qualified=False)
    return dict(status='SIX_PROCESS_SCOPE_FROZEN_NOT_RUN',
                scope_ref=put(root, B.DIR + '/SIX_PROCESS_AUTHORIZED_SCOPE.json', document), actual_gpu_runs=0)


def launch(root):
    config, refs, binding = B.verify_configuration(root, B.CONFIG)
    scope_document = B.read(root, B.DIR + '/SIX_PROCESS_AUTHORIZED_SCOPE.json')
    for key in ('permission_ref', 'source_lock_ref', 'plan_ref', 'binding_ref', 'human_grant_ref'):
        B.checked_ref(root, scope_document.get(key))
    if (scope_document.get('gpu_uuid') != binding['gpu_uuid'] or scope_document.get('label') != B.LABEL
            or scope_document.get('max_attempts') != 1 or scope_document.get('max_processes') != 6
            or scope_document.get('permission_ref') != binding['effective_permission_ref']
            or scope_document.get('source_lock_ref') != B.ref(root, B.SITE_LOCK)
            or scope_document.get('binding_ref') != config['gpu_entry_binding_ref']
            or scope_document.get('human_grant_ref') != binding['human_grant_ref']
            or scope_document.get('seconds_limit') != 1200 or scope_document.get('reserved_seconds') != 1220):
        B.reject('scope is foreign or enlarged')
    resources, ledger = B.check_launch(root, config, binding)
    before_ref = full_source_proof(root, 'launch', config, binding, refs)
    command = B.guard_command()
    overrides = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                     PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    put(root, B.DIR + '/GPU_LAUNCH_INTENT.json', dict(command=command,
        environment_overrides=overrides, launch_source_ref=before_ref,
        site_lock_ref=B.ref(root, B.SITE_LOCK), binding_ref=config['gpu_entry_binding_ref'],
        human_grant_ref=binding['human_grant_ref'], scope_ref=B.ref(root, B.DIR + '/SIX_PROCESS_AUTHORIZED_SCOPE.json'),
        plan_ref=scope_document['plan_ref'], plan_ref_receipt_ref=B.ref(root, B.DIR + '/PLAN_REFERENCE.json'),
        ledger_before_ref=B.ref(root, B.LEDGER), ledger_gpu_wall_seconds_before=ledger['gpu_wall_seconds'], resources=resources, utc=utc()))
    with B.safe(root, B.DIR + '/GPU_GUARD_STDOUT.log').open('xb') as out, B.safe(root, B.DIR + '/GPU_GUARD_STDERR.log').open('xb') as err:
        env = dict(os.environ); env.update(overrides)
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, start_new_session=True, close_fds=True)
    return dict(status='ORIGINAL_GUARD_LAUNCHED_NOT_QUALIFIED',
        receipt_ref=put(root, B.DIR + '/GPU_LAUNCH_RECEIPT.json',
                        dict(guard_pid=proc.pid, command=command, utc=utc(), launch_is_not_success=True)))


def after(root):
    config, refs, binding = B.verify_configuration(root, B.CONFIG)
    if B.read(root, B.LEDGER).get('active_reservation') is not None or not B.safe(root, B.RUN + '/result.json').exists():
        B.reject('wait for original guard to complete and drain')
    proof_ref = full_source_proof(root, 'after', config, binding, refs)
    event = B.verify_completed_guard(root, config, binding)
    return dict(status='SOURCE_AND_ORIGINAL_GUARD_COMPLETE_NOT_COST_QUALIFIED',
                proof_ref=proof_ref, guard_ref=B.ref(root, B.RUN + '/result.json'),
                actual_gpu_elapsed_seconds=event['elapsed_seconds'], native_cost_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'context', 'bind', 'scope', 'launch', 'after'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output-relative', default=B.DIR + '/UNBOUND_REVIEW_TEMPLATE.json')
    parser.add_argument('--plan-relative')
    args = parser.parse_args()
    if args.root.resolve(strict=True) != ROOT and args.action != 'prepare':
        B.reject('live binding/launch requires designated project root')
    try:
        if args.action == 'prepare': value = prepare(args.root, args.output_relative)
        elif args.action == 'context': value = context(args.root)
        elif args.action == 'bind': value = bind(args.root)
        elif args.action == 'scope': value = scope(args.root, args.plan_relative)
        elif args.action == 'launch': value = launch(args.root)
        else: value = after(args.root)
    except Exception as exc:
        print(json.dumps(dict(status='GPU_ENTRY_BLOCKED', reason=str(exc),
                              gpu_started=False, actual_gpu_runs=0, gpu_launch_allowed=False)))
        return 2
    print(json.dumps(value, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
