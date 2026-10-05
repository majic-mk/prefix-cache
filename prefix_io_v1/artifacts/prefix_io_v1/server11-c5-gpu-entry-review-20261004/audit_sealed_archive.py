"""Postseal independent byte/closure audit. This does not execute candidate code or tests."""
from pathlib import Path, PurePosixPath
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
import tarfile

WORK = Path(__file__).resolve().parents[3]
EXTRACTED = WORK / 'artifacts/c5g5'
DOWNLOAD = WORK / 'artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_delivery'
OUTPUT = Path(__file__).with_name('INDEPENDENT_ARCHIVE_REVIEW.json')
REV = 'server11-c5-gpu-entry-revision-20261004'
REVIEW = 'server11-c5-gpu-entry-review-20261004'
DELIVERY = 'server11-c5-gpu-entry-delivery-20261004'
SERVER = '/root/autodl-tmp/prefix-io-v1-handoff/project/'
ARCHIVE_SHA = 'bba9dd8eaf23d346f25503b57cb415897591a6ee5ef4c34d1312c0a85d969ccf'
RECEIPT_SHA = '81793300066cdbdc262622717ee7b2625669504566d907d46f2dc4c9872d500b'
MANIFEST_SHA = '9d7d6babe8154541357b366700a283c20f81da92a84a2802bc2194556e3c08fb'
LOCK_SHA = 'e91df96e4203ae4b4925e90613a277632ad2ae3a3a337e9e947ad29c7f7431f5'
LEDGER_SHA = 'bef6d78e43058eaad811ab5ecb08937356c911784f9000e76eea9de0276c0b72'
checked = []


def require(condition, label):
    if not condition:
        raise ValueError(label)
    checked.append(label)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(raw):
    def reject_constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject_constant)


def safe(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name == p.as_posix() and not p.is_absolute()
            and '\\' not in name and ':' not in name and '..' not in p.parts,
            'safe archive reference ' + name)


def refindex(rows):
    result = {}
    for row in rows:
        require(set(row) == {'path', 'bytes', 'sha256'} and type(row['bytes']) is int
                and row['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
                'typed source reference ' + row['path'])
        safe(row['path'])
        require(row['path'] not in result, 'unique source reference ' + row['path'])
        result[row['path']] = row
    return result


def flags(document, label):
    for key in ('gpu_launch_allowed', 'native_execution_verified', 'native_cost_qualified',
                'full_runtime_cost_qualified', 'on_observation_cost_measured'):
        require(document.get(key) is False, label + ' ' + key + '=false')
    for key in ('valid_native_receipt', 'effective_cost_upper_ns', 'effective_step_budget_ns', 'gpu_uuid'):
        require(key in document and document[key] is None, label + ' ' + key + '=null')


proof = dict(schema='c5_gpu_entry_independent_postseal_archive_review_v1',
             utc=datetime.now(timezone.utc).isoformat(), status='FAIL', findings=[],
             audit_kind='stdlib read-only archive and recorded-server-evidence audit',
             candidate_tests_rerun=0, server_connections=0, model_loads=0,
             actual_gpu_runs=0, actual_gpu_seconds=0, gpu_launch_allowed=False,
             native_execution_verified=False, native_cost_qualified=False,
             full_runtime_cost_qualified=False, on_observation_cost_measured=False,
             valid_native_receipt=None, effective_cost_upper_ns=None, effective_step_budget_ns=None,
             limits=['The raw GPU ledger is not archived; its unchanged SHA and zero reservation are '
                     'corroborated across the saved server baseline, resource observation, final audit and receipt.',
                     'The 1,153 historical references were verified by the source-pinned server sealer; '
                     'this local audit rehashes all 318 archived source/asset snapshots, not all historical files.',
                     'The historical 4,655-row model inventory is source-pinned, but the full model '
                     'assets were not reverified in this CPU preparation phase.',
                     'CPU tests and source preparation establish rejection boundaries only; '
                     'they do not qualify native GPU cost, actual on installation, P4, or a performance gain.'])
try:
    archive_path = DOWNLOAD / 'C5_GPU_ENTRY_PREPARATION_EVIDENCE.tar.gz'
    archive = archive_path.read_bytes()
    require(len(archive) == 18499021 and sha(archive) == ARCHIVE_SHA, 'archive pinned size and SHA')
    receipt_raw = (DOWNLOAD / 'SERVER_BACKUP_RECEIPT.json').read_bytes()
    require(len(receipt_raw) == 842 and sha(receipt_raw) == RECEIPT_SHA, 'receipt pinned size and SHA')
    receipt = parse(receipt_raw)
    require(receipt['archive'] == dict(bytes=18499021, file=archive_path.name, sha256=ARCHIVE_SHA),
            'receipt archive binding')
    require(receipt['manifest'] == dict(bytes=58415, file='ARCHIVE_CONTENTS_MANIFEST.json', sha256=MANIFEST_SHA),
            'receipt manifest binding')
    require(receipt['status'] == 'PASS_SEALED_CPU_GATED_NATIVE_ENTRY' and receipt['GPU_runs'] == 0
            and receipt['formal_benchmark_runs'] == 0 and receipt['gpu_launch_allowed'] is False
            and receipt['native_cost_qualified'] is False and receipt['gpu_ledger_sha256'] == LEDGER_SHA,
            'receipt remains non-authority and zero-GPU')
    members = {}
    with tarfile.open(archive_path, 'r:gz') as handle:
        for item in handle:
            safe(item.name)
            require(item.isfile() and item.name not in members, 'unique regular tar member ' + item.name)
            require(item.name == 'ARCHIVE_CONTENTS_MANIFEST.json'
                    or PurePosixPath(item.name).parts[0] in {REV, REVIEW, DELIVERY},
                    'bounded archive root ' + item.name)
            raw = handle.extractfile(item).read()
            require(len(raw) == item.size, 'tar member length ' + item.name)
            members[item.name] = raw
    require(len(members) == 234, '234 archive members')
    manifest_raw = members['ARCHIVE_CONTENTS_MANIFEST.json']
    require(len(manifest_raw) == 58415 and sha(manifest_raw) == MANIFEST_SHA, 'manifest pinned size and SHA')
    manifest = parse(manifest_raw)
    files = refindex(manifest['files'])
    require(set(files) | {'ARCHIVE_CONTENTS_MANIFEST.json'} == set(members), 'exact archive manifest closure')
    require(len(files) == manifest['data_file_count'] == receipt['data_files'] == 233,
            '233 declared data files')
    require(sum(len(members[p]) for p in files) == manifest['raw_bytes'] == receipt['raw_bytes'] == 25695932,
            'raw byte total 25695932')
    require(manifest['GPU_runs'] == manifest['formal_benchmark_runs'] == 0, 'manifest zero GPU and benchmark')
    for path, row in files.items():
        require(len(members[path]) == row['bytes'] and sha(members[path]) == row['sha256'],
                'manifest member bytes ' + path)
    for path, raw in members.items():
        extracted = EXTRACTED / path
        require(extracted.is_file() and not extracted.is_symlink() and extracted.read_bytes() == raw,
                'extracted copy identical ' + path)

    def doc(path):
        return parse(members[path])

    lock_path = DELIVERY + '/SOURCE_LOCK_GPU_ENTRY_CPU.json'
    lock_raw = members[lock_path]
    require(len(lock_raw) == 88170 and sha(lock_raw) == LOCK_SHA, 'CPU lock pinned size and SHA')
    lock = parse(lock_raw)
    sources = refindex(lock['files'])
    require(len(sources) == lock['source_files'] == 318 and lock['source_only'] is True
            and lock['gpu_entry_code_prepared'] is True, '318 source-only prepared closure')
    flags(lock, 'CPU lock')
    mapping = doc(DELIVERY + '/SOURCE_SNAPSHOT_MAP.json')
    map_rows = mapping['files']
    map_sources = refindex([{k: r[k] for k in ('path', 'bytes', 'sha256')} for r in map_rows])
    require(map_sources == sources and mapping['source_rows'] == 318 and mapping['gpu_runs'] == 0,
            'source map exact 318-reference lock closure')
    for row in map_rows:
        safe(row['snapshot_path'])
        raw = members[row['snapshot_path']]
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'],
                'source snapshot actual bytes ' + row['path'])
    external = {p for p in members if p.startswith(DELIVERY + '/SOURCE_DEPENDENCY_SNAPSHOTS/')}
    require(len(external) == mapping['unique_external_snapshots'] == receipt['unique_external_snapshots'] == 104,
            '104 external byte snapshots')

    baseline_path = DELIVERY + '/SESSION_BEFORE_GPU_ENTRY_PREPARATION.json'
    baseline = doc(baseline_path)
    require(lock['baseline_ref'] == dict(path='artifacts/prefix_io_v1/' + baseline_path,
            bytes=len(members[baseline_path]), sha256=sha(members[baseline_path])), 'lock baseline binding')
    require(len(baseline['refs']) == 1153, 'historical baseline 1153 references')
    require(baseline['active_reservation'] is None and baseline['GPU_nodes'] == [] and baseline['GPU_runs'] == 0,
            'baseline no reservation or GPU')
    final = doc(DELIVERY + '/SERVER_FINAL_AUDIT.json')
    readiness = doc(DELIVERY + '/GPU_ENTRY_READINESS.json')
    freeze = doc(DELIVERY + '/SOURCE_FREEZE_RECEIPT_CPU.json')
    resource = doc(DELIVERY + '/SERVER_PREPARATION_RESOURCE_ASSETS.json')
    for label, value in [('final audit', final), ('readiness', readiness)]:
        flags(value, label)
        require(value['gpu_runs'] == value['gpu_seconds'] == 0
                and value['gpu_entry_code_prepared'] is True and value['source_rows'] == 318
                and value['source_lock_sha256'] == LOCK_SHA and value['cpu_targeted_tests_passed'] == 88
                and value['ledger_sha256'] == LEDGER_SHA, label + ' code only, 88 tests, unchanged ledger')
    require(final['status'] == 'PASS_CPU_GATED_NATIVE_ENTRY_CODE_ONLY'
            and readiness['status'] == 'CODE_PREPARED_RESOURCES_AND_AUTHORIZATION_PENDING'
            and final['prior_evidence_unchanged'] is True and final['prior_refs_verified'] == 1153
            and final['gpu_nodes'] == [] and final['native_tests'] == 23
            and final['binding_tests'] == 35 and final['review_tests'] == 30, 'final result scope')
    require(freeze['status'] == 'PASS_CPU_SOURCE_FREEZE_NO_GPU_GRANT'
            and freeze['prior_references_verified'] == 1153 and freeze['source_rows'] == 318
            and freeze['gpu_launch_allowed'] is False and freeze['gpu_runs'] == 0 and freeze['gpu_uuid'] is None
            and freeze['source_lock_ref'] == dict(path='artifacts/prefix_io_v1/' + lock_path,
                                                 bytes=88170, sha256=LOCK_SHA)
            and receipt['source_lock_sha256'] == LOCK_SHA and receipt['source_snapshots'] == 318
            and receipt['prior_references_verified'] == 1153, 'freeze/receipt source closure and historical count')
    require(baseline['ledger_sha256'] == resource['ledger_sha256'] == LEDGER_SHA
            and resource['active_reservation'] is None and resource['GPU_nodes'] == []
            and resource['gpu_runs'] == 0 and resource['effective_gpu_config'] is None
            and final['remaining_gpu_seconds'] == readiness['remaining_gpu_seconds'] == resource['gpu_remaining_seconds']
            and abs(28800-baseline['gpu_wall_seconds']-final['remaining_gpu_seconds']) < 1e-8,
            'saved server evidence ledger unchanged and budget arithmetic')
    require(all(x['cpu_max'] == '50000 100000' and x['memory_max'] == '2147483648'
                for x in (baseline, resource, final)), 'saved resource observation confirms no-card quota')
    require(resource['native_parent_full_rows_reverified'] is False and resource['native_parent_inventory_rows'] == 4655,
            'full model asset recheck not claimed')
    for row in resource['actual_input_and_provenance_refs_verified']:
        require(sources[row['path']] == row, 'production KV/provenance pinned ' + row['path'])
    require(len(resource['actual_input_and_provenance_refs_verified']) == 27, '27 actual KV/provenance references')

    test_proofs = []
    for prefix, group, resultpath, count in [
            (REV, 'NATIVE', 'SERVER_NATIVE_CPU_01/CPU_NATIVE_ENTRY_RESULT.json', 23),
            (REV, 'BINDING', 'SERVER_BINDING_CPU_01/CPU_BINDING_RESULT.json', 35),
            (REVIEW, 'REVIEW', 'SERVER_REVIEW_CPU_01/TEST_RESULT.json', 30)]:
        value = doc(prefix + '/' + resultpath)
        command = doc(prefix + '/SERVER_CPU_' + group + '_COMMAND.json')
        outer = doc(prefix + '/SERVER_CPU_' + group + '_RESULT.json')
        require(value['status'] == 'PASS' and value['location'] == 'server_cpu'
                and value['tests'] == value['passed'] == count
                and value['failed'] == value['errors'] == value['skipped'] == 0,
                group + ' actual recorded test result')
        flags(value, group)
        require(value['gpu_runs'] == 0 and value['forbidden_imports'] == []
                and value['source_lock_sha256'] == LOCK_SHA and value['source_before'] == value['source_after'],
                group + ' zero GPU and unchanged frozen source')
        require(outer['exit'] == 0 and outer['GPU_runs'] == 0 and outer['stderr_bytes'] == 0,
                group + ' actual outer exit0')
        require(command['CUDA_VISIBLE_DEVICES'] == '' and command['argv'][1:4] == ['-B', '-I', '-S']
                and command['cwd'] == SERVER.rstrip('/')
                and command['argv'][command['argv'].index('--source-lock')+1] == SERVER + 'artifacts/prefix_io_v1/' + lock_path,
                group + ' CPU isolated actual command and lock')
        outer_stdout = members[prefix + '/SERVER_CPU_' + group + '_STDOUT.log']
        outer_stderr = members[prefix + '/SERVER_CPU_' + group + '_STDERR.log']
        require(len(outer_stdout) == outer['stdout_bytes'] and len(outer_stderr) == outer['stderr_bytes'] == 0,
                group + ' outer log byte counters')
        logpath = prefix + '/' + resultpath.rsplit('/', 1)[0] + '/TEST_STDERR.log'
        log = members[logpath].decode('utf-8')
        require(re.search(r'Ran ' + str(count) + r' tests in [0-9.]+s\s+OK\s*$', log) is not None
                and len(re.findall(r' \.\.\. ok\n', log)) == count, group + ' actual unittest success log')
        test_proofs.append(dict(group=group, tests=count, exit=0, result_sha256=sha(members[prefix+'/'+resultpath]),
                                command_sha256=sha(members[prefix+'/SERVER_CPU_'+group+'_COMMAND.json']),
                                test_log_sha256=sha(members[logpath])))
        if group == 'NATIVE':
            require(value['source_before'] == dict(files_verified=318, source_lock_sha256=LOCK_SHA, failed=[]),
                    'native full source rehash report')
            require(value['import_audit']['status'] == 'PASS'
                    and value['import_audit']['forbidden_import_attempts'] == []
                    and value['import_audit']['loaded_forbidden_modules'] == [], 'native isolated import audit')
        elif group == 'BINDING':
            digest = sha(json.dumps([sources[key] for key in sorted(sources)], sort_keys=True, separators=(',', ':')).encode())
            require(value['source_before'] == digest and value['tests_are_CPU_boundary_checks'] is True,
                    'binding digest independently reproduced from 318 frozen rows')
        else:
            actual = {}
            for row in value['source_before']:
                require(row['path'].startswith(SERVER), 'review real server path')
                rel = row['path'][len(SERVER):]
                require(rel not in actual, 'review unique before source ' + rel)
                actual[rel] = dict(path=rel, bytes=row['bytes'], sha256=row['sha256'])
            expected = dict(sources)
            key = 'artifacts/prefix_io_v1/' + lock_path
            expected[key] = dict(path=key, bytes=88170, sha256=LOCK_SHA)
            require(actual == expected and value['source_count'] == 319 and value['source_lock_files'] == 318
                    and value['sources_unchanged'] is True, 'review before/after exact 319-reference closure')
            require(value['forbidden_import_attempts'] == value['forbidden_modules_imported'] == []
                    and value['model_loads'] == value['gpu_jobs_created'] == value['permission_scopes_created']
                    == value['formal_benchmark_runs'] == 0 and value['actual_on_installed'] is False
                    and value['performance_claim'] is False, 'review remains negative CPU evidence only')

    for mode, status, exitcode in [('UNBOUND_PREPARE', 'UNBOUND_REVIEW_TEMPLATE_WRITTEN', 0),
                                  ('LIVE_CONTEXT_GATE', 'GPU_ENTRY_BLOCKED', 2)]:
        outer = doc(REV + '/SERVER_' + mode + '_RESULT.json')
        stdout = doc(REV + '/SERVER_' + mode + '_STDOUT.log')
        require(outer['exit'] == exitcode and outer['GPU_runs'] == 0
                and stdout['status'] == status and stdout['gpu_started'] is False
                and stdout['actual_gpu_runs'] == 0 and stdout['gpu_launch_allowed'] is False
                and members[REV+'/SERVER_'+mode+'_STDERR.log'] == b'', mode + ' actual inert/blocked execution')
        if mode == 'UNBOUND_PREPARE':
            raw = members[REV+'/UNBOUND_REVIEW_TEMPLATE.json']
            require(stdout['template_ref'] == dict(path='artifacts/prefix_io_v1/'+REV+'/UNBOUND_REVIEW_TEMPLATE.json',
                                                   bytes=len(raw), sha256=sha(raw)), 'prepare template actual-byte binding')
        else:
            require(stdout['reason'] == 'GPU_ENTRY_REJECTED: actual resource observation unavailable',
                    'context stopped at actual resource boundary')
    template = doc(REV+'/UNBOUND_REVIEW_TEMPLATE.json')
    require(template['schema'] == 'c5_gpu_entry_unbound_review_template_v1' and template['status'] == 'UNBOUND'
            and template['source_ancestry_is_not_authority'] is True
            and template['human_gpu_grant_required'] is True and template['actual_gpu_runs'] == 0,
            'prepared template explicitly lacks authority')
    require(all(template[k] is None for k in ('gpu_uuid','effective_permission','valid_config','site_binding',
                                            'human_grant','native_receipt','effective_cost_upper_ns','effective_step_budget_ns'))
            and template['gpu_launch_allowed'] is False
            and template['native_cost_qualified'] is False, 'UNBOUND template has no grant/config/receipt')
    forbidden_names = {'GPU_ENTRY_BINDING.json', 'LIVE_BINDING_CONTEXT.json', 'HUMAN_GPU_GRANT.json',
                       'EFFECTIVE_GPU_PERMISSION.json', 'REVISION_SOURCE_LOCK.json', 'SITE_SOURCE_LOCK.json',
                       'NATIVE_COST_CONFIG.json', 'SIX_PROCESS_AUTHORIZED_SCOPE.json', 'GPU_LAUNCH_INTENT.json',
                       'GPU_LAUNCH_RECEIPT.json', 'NATIVE_COST_PLAN.json', 'PLAN_REFERENCE.json',
                       'NATIVE_PAIRED_MEASUREMENTS.json', 'NATIVE_COST_VERIFICATION.json'}
    require(not any(PurePosixPath(p).name in forbidden_names for p in members),
            'no live authority/config/plan/scope/launch/native qualification file in sealed three directories')
    proof.update(status='PASS_INDEPENDENT_BYTE_CLOSURE_REVIEW', findings=[],
                 archive_ref=dict(bytes=len(archive), sha256=ARCHIVE_SHA),
                 receipt_ref=dict(bytes=len(receipt_raw), sha256=RECEIPT_SHA),
                 manifest_ref=dict(bytes=len(manifest_raw), sha256=MANIFEST_SHA),
                 source_lock_ref=dict(bytes=len(lock_raw), sha256=LOCK_SHA),
                 regular_archive_members_verified=234, data_files_verified=233,
                 extracted_file_bytes_independently_verified=234, raw_data_bytes=25695932,
                 source_rows_independently_verified=318, external_snapshots_verified=104,
                 reviewer_before_after_rows_independently_verified=319, recorded_server_tests_verified=88,
                 recorded_server_test_proofs=test_proofs,
                 gpu_entry_code_prepared=True, preparation_execution_exit=0, context_execution_exit=2,
                 live_authority_files_absent=True, no_gpu_or_model_run_in_this_review=True,
                 ledger_sha_unchanged_in_saved_server_evidence=LEDGER_SHA,
                 remaining_gpu_seconds_from_server_evidence=final['remaining_gpu_seconds'],
                 source_pinned_server_historical_refs_verified=1153,
                 prepared_only_conditional=True)
except Exception as exc:
    proof['findings'].append(type(exc).__name__ + ': ' + str(exc))
finally:
    proof['assertions_passed'] = len(checked)
    proof['auditor_source_ref'] = dict(file=Path(__file__).name, bytes=Path(__file__).stat().st_size,
                                      sha256=sha(Path(__file__).read_bytes()), frozen_package_member=False)
    proof['command'] = [sys.executable, '-B', '-I', '-S', str(Path(__file__).resolve())]
    with OUTPUT.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], findings=proof['findings'], assertions_passed=len(checked),
                         proof_path=str(OUTPUT), proof_sha256=sha(OUTPUT.read_bytes()))))
sys.exit(0 if not proof['findings'] else 1)
