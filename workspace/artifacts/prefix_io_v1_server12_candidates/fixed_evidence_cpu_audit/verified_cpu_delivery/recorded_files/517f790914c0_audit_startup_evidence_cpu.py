"""Read actual fixed diagnostics and audit host startup observations using CPU.

No experiment is launched or qualified. Standard-library parsing and exact
source hashes suffice; model/framework imports and shared-library loads are
forbidden. A local byte-verified flat archive map may replace server paths.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
OLD = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
A = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004'
OUT = 'artifacts/prefix_io_v1/server12-c5-fixed-evidence-cpu-audit-20261004'
LABELS = tuple('server12-c5-native-repeat-off' + str(i+1).zfill(2) for i in range(3))
MAX_FILE = 32 * 1024**2
FORBIDDEN = ('torch', 'vllm', 'py_kvcache', 'ctypes', 'cffi')
IMPORT_ATTEMPTS = []


def require(value, reason):
    if not value:
        raise ValueError('STARTUP_CPU_AUDIT_REJECTED: ' + reason)


class CPUOnlyImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name+'.') for name in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise ImportError('startup audit is CPU source/data parsing only: ' + fullname)
        return None


def clean_imports():
    actual = sorted(name for name in sys.modules
        if any(name == prefix or name.startswith(prefix+'.') for prefix in FORBIDDEN))
    require(not actual and not IMPORT_ATTEMPTS, 'no GPU/framework/shared-library import')
    return actual


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def name(value):
    require(type(value) is str and 0 < len(value) <= 240 and '\\' not in value and ':' not in value and
        not value.startswith('/') and all(part not in ('', '.', '..') for part in value.split('/')),
        'plain project-relative evidence path')
    return value


def no_links(path):
    path = Path(path).absolute()
    require(not any(item.is_symlink() for item in (path, *path.parents)), 'evidence symlink refused')
    return path.resolve(strict=True)


def bytes_ref(path, relative):
    path = no_links(path)
    before = path.stat()
    require(path.is_file() and 0 <= before.st_size <= MAX_FILE, 'bounded regular metadata file')
    raw = path.read_bytes()
    after = path.stat()
    require(len(raw) == before.st_size == after.st_size and
        (before.st_dev, before.st_ino, before.st_mtime_ns) ==
        (after.st_dev, after.st_ino, after.st_mtime_ns), 'evidence changed while reading')
    return raw, dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def ref_shape(row):
    require(type(row) is dict and set(row) == {'path','bytes','sha256'}, 'exact evidence reference')
    name(row['path'])
    require(type(row['bytes']) is int and 0 <= row['bytes'] <= MAX_FILE and
        type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']) is not None,
        'typed evidence reference')
    return row


class Evidence:
    def __init__(self, root, mapping, prior=False):
        self.root = no_links(root)
        self.checked = {}
        self.local = {}
        if mapping is not None:
            self.mapping = no_links(mapping)
            raw, map_ref = bytes_ref(self.mapping, self.mapping.name)
            value = parse(raw)
            expected = 'PASS_SERVER12_NORMAL_DELIVERY_LOCAL_BYTE_VERIFICATION' if prior else \
                'PASS_SERVER12_REPEAT_DELIVERY_LOCAL_BYTE_VERIFICATION'
            require(value.get('status') == expected and value.get('experiment_qualification_issued') is False and
                value.get('GPU_operations') == 0 and type(value.get('files')) is list, 'actual byte-verified local map')
            self.mapping_ref = map_ref
            for row in value['files']:
                require(type(row) is dict and set(row) == {'path','bytes','sha256','local_path'}, 'exact flat mapping row')
                ref = ref_shape({key:row[key] for key in ('path','bytes','sha256')})
                path = no_links(row['local_path'])
                require(path.is_relative_to(self.mapping.parent/'recorded_files') and ref['path'] not in self.local,
                    'unique existing flat file within recorded_files')
                self.local[ref['path']] = (path, ref)
            manifest_ref = ref_shape(value['manifest'])
            manifest_path = no_links(value['manifest_local_path'])
            require(manifest_path.is_relative_to(self.mapping.parent/'recorded_files') and
                manifest_ref['path'] not in self.local, 'separate self-excluded manifest within flat directory')
            self.local[manifest_ref['path']] = (manifest_path,manifest_ref)
            manifest = self.read(manifest_ref['path'], expected=manifest_ref)
        else:
            self.mapping_ref = None
            result_path = A + ('/NORMAL_NORMAL_DELIVERY_off01-final_RESULT.json' if prior else
                '/REPEAT_REPEAT_DELIVERY_fixed3-final_RESULT.json')
            result = self.read_unlocked(result_path)
            expected = 'PASS_NORMAL_NORMAL_DELIVERY_BYTE_ARCHIVE' if prior else 'PASS_FIXED_REPEAT_DELIVERY_BYTE_ARCHIVE'
            require(result.get('status') == expected and result.get('gpu_ledger_byte_unchanged') is True,
                'actual completed server delivery result')
            manifest_ref = ref_shape(result['manifest'])
            manifest = self.read_unlocked(manifest_ref['path'], expected=manifest_ref)
        require(type(manifest.get('files')) is list and 1 <= len(manifest['files']) <= 2048,
            'bounded actual manifest files')
        self.refs = {}
        for row in manifest['files']:
            ref_shape(row)
            require(row['path'] not in self.refs, 'unique actual manifest rows')
            self.refs[row['path']] = row
        self.manifest_ref = manifest_ref
        self.manifest = manifest

    def path(self, relative):
        name(relative)
        if self.local:
            require(relative in self.local, 'actual evidence is absent from flat map: '+relative)
            return self.local[relative][0]
        result = no_links(self.root/relative)
        require(result.is_relative_to(self.root), 'server evidence outside project')
        return result

    def raw(self, relative, expected=None):
        raw, ref = bytes_ref(self.path(relative), relative)
        if expected is not None:
            require(ref == ref_shape(expected), 'actual evidence SHA/byte mismatch: '+relative)
        if self.local:
            require(ref == self.local[relative][1], 'flat map byte mismatch: '+relative)
        self.checked[relative] = ref
        return raw

    def read_unlocked(self, relative, expected=None):
        return parse(self.raw(relative, expected))

    def read(self, relative, expected=None):
        if hasattr(self,'refs'):
            require(relative in self.refs, 'manifest evidence required: '+relative)
            require(expected is None or expected == self.refs[relative], 'evidence provenance reference disagreement')
            expected = self.refs[relative]
        return self.read_unlocked(relative, expected)

    def text(self, relative):
        require(relative in self.refs, 'manifest text evidence required')
        return self.raw(relative,self.refs[relative]).decode('utf-8',errors='replace')

    def check_after(self):
        for relative, row in list(self.checked.items()):
            require(bytes_ref(self.path(relative),relative)[1] == row, 'evidence drift after audit')
        if self.mapping_ref is not None:
            require(bytes_ref(self.mapping,self.mapping.name)[1] == self.mapping_ref, 'local map drift after audit')


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, 'typed integer: '+label)
    return value


def seconds(value, label):
    require(type(value) in (int,float) and math.isfinite(value) and value >= 0, 'finite seconds: '+label)
    return value


def span(value, label):
    require(type(value) is dict, 'startup interval: '+label)
    start = integer(value['start_ns'],label+' start',1)
    end = integer(value['end_ns'],label+' end',start)
    return dict(start_ns=start,end_ns=end,duration_ns=end-start,duration_seconds=(end-start)/1e9)


def log_observations(text):
    jit = []; selected = []; durations = {}
    for index,line in enumerate(text.splitlines(),1):
        if 'Triton kernel JIT compilation during inference:' in line:
            kernel = line.split('Triton kernel JIT compilation during inference:',1)[1].split('.',1)[0].strip()
            jit.append(dict(line=index,kernel=kernel,message=line[:500],request_id=None,native_step_ordinal=None))
        if any(token in line for token in ('Autotuning process', 'init engine (', 'Loading weights took',
            'Model loading took','disabling torch.compile and CUDAGraphs','Cudagraph is disabled')):
            selected.append(dict(line=index,message=line[:600]))
        for key,pattern in (('weight_loading_seconds',r'Loading weights took ([0-9.]+) seconds'),
            ('model_loading_seconds',r'Model loading took .*? ([0-9.]+) seconds'),
            ('engine_profile_cache_warmup_seconds',r'init engine \(.*\) took ([0-9.]+) s')):
            found = re.search(pattern,line)
            if found:
                require(key not in durations, 'ambiguous repeated initialization log duration')
                durations[key] = float(found.group(1))
    return dict(inference_JIT_warnings=jit,inference_JIT_warning_count=len(jit),
        initialization_and_autotune_log_observations=selected,logged_durations_seconds=durations,
        JIT_warning_to_selected_step_binding_available=False,
        wall_log_to_host_monotonic_mapping_available=False)


def audit_slot(reader,index):
    token = 'off'+str(index+1).zfill(2); label=LABELS[index]
    run='experiments/prefix_io_v1/runs/'+label
    report=reader.read(D+'/DIAGNOSTIC_RESULT_'+token+'.json')
    raw=reader.read(run+'/details/p4-single-file-runtime-result.json',report['evidence_refs']['raw_result'])
    guard=reader.read(run+'/result.json',report['evidence_refs']['actual_guard'])
    log=log_observations(reader.text(run+'/process.log'))
    require(raw.get('origin') == 'native_gpu_recording' and raw.get('diagnostic_index') == index and
        raw.get('label') == label and raw.get('mode') == 'off' and raw.get('original_engine_shutdown_returned') is True,
        'actual completed off raw origin/index/lifecycle')
    require(guard.get('label') == label and guard.get('exit') == guard.get('child_exit') == 0 and
        guard.get('session_drained') is True and guard.get('session_members_after_cleanup') == [],
        'actual original successful drained guard')
    context=raw['process_entry_context_metadata']
    require(type(context.get('public_loader_calls')) is int and context['public_loader_calls'] == 1 and
        type(context.get('internal_metadata_revalidations')) is int and context['internal_metadata_revalidations'] == 7 and
        context.get('private_receipt_issuer_used') is False and context.get('cached_PASS_or_qualification') is False and
        context.get('cost_gate_changed') is False and context.get('whole_source_before_after_required') is True,
        'actual public-loader and metadata observation contract')
    require(context['actual_guard_identity']['id'] == guard['reservation_id'] and
        context['actual_identity']['sid'] == guard['session_id'] and context['actual_identity']['pid'] == raw['subprocess_pid'] and
        context['config_ref'] == report['evidence_refs']['config'], 'same process/guard/config observation')
    times={key:span(value,key) for key,value in raw['startup_stage_times'].items()}
    full=times['process_full_validation']; public=times['public_calibration_replay']; llm=times['llm_initialization']
    require(full['start_ns'] <= public['start_ns'] <= public['end_ns'] <= full['end_ns'],
        'public replay is contained inside full validation; never add nested durations')
    ct=context['startup_stage_times']
    require(ct['full_entry_start_ns'] == full['start_ns'] and ct['full_entry_end_ns'] == full['end_ns'] and
        ct['public_loader_start_ns'] == public['start_ns'] and ct['public_loader_end_ns'] == public['end_ns'],
        'same actual raw/context startup interval')
    capture=raw['windows'][0]['capture']; warm=raw['warmups'][0]
    require(len(raw['windows']) == 1 and len(raw['warmups']) == len(raw['hot_warmup_observations']) == 1 and
        len(capture['frames']) == len(capture['event_witnesses']) == 128 and len(warm['output_token_ids']) == 128 and
        warm['num_cached_tokens'] == 128 and warm['seed'] == 2829 and capture['selected_offsets'] == [16] and
        capture.get('cross_clock_absolute_mapping') is False and capture.get('no_added_synchronization') is True,
        'original finite warmup/full Event capture and no cross-clock map')
    witness=capture['event_witnesses'][16]
    require(witness['native_step_ordinal'] == capture['frames'][16]['native_step_ordinal'] and
        witness['gpu_elapsed_ns'] == report['selected_gpu_elapsed_ns'] and
        witness.get('cross_clock_absolute_mapping') is False, 'same actual selected Event witness')
    require(raw['engine_config'].get('enforce_eager') is True and raw['engine_config'].get('compilation_config') == 0,
        'actual eager engine configuration')
    return dict(diagnostic_index=index,label=label,public_loader_calls=1,metadata_revalidations=7,
        guard_elapsed_seconds=seconds(guard['elapsed_seconds'],'guard elapsed'),
        startup_spans=times,public_replay_inside_full_validation=True,
        non_public_validation_remainder_ns=full['duration_ns']-public['duration_ns'],
        startup_durations_are_host_wall_not_strategy_CPU_or_GPU_cost=True,
        model_initialization_seconds=llm['duration_seconds'],
        logs=log,eager_disables_torch_compile_and_CUDAGraph=True,
        native_Event_capture_is_not_CUDAGraph_capture=True,
        selected_elapsed_ns=integer(report['selected_gpu_elapsed_ns'],'actual selected Event',1),
        selected_native_step_ordinal=witness['native_step_ordinal'],warmup_requests=1,warmup_outputs=128,
        full_capture_frames=128,cross_clock_absolute_mapping=False,
        JIT_as_cause_of_selected_cost_exceedance_proved=False,GPU_nonoverlap_proved=False,GPU_release_inferred=False,
        source_refs=dict(raw=reader.refs[run+'/details/p4-single-file-runtime-result.json'],
            guard=reader.refs[run+'/result.json'],process_log=reader.refs[run+'/process.log'],
            diagnostic_report=reader.refs[D+'/DIAGNOSTIC_RESULT_'+token+'.json']))


def stats(values):
    return dict(n=len(values),min=min(values),max=max(values),median=statistics.median(values),
        mean=sum(values)/len(values),values=values,population_or_tail_bound_proved=False)


def audit(root,repeat_mapping=None,prior_mapping=None):
    clean_imports()
    repeat=Evidence(root,repeat_mapping); prior=Evidence(root,prior_mapping,prior=True)
    slots=[audit_slot(repeat,index) for index in range(3)]
    source_lock=repeat.read(D+'/COMMON_SOURCE_LOCK.json')
    require(repeat.refs[D+'/COMMON_SOURCE_LOCK.json']['sha256'] ==
        '4aab66888592a8fdf6544513003e9b7382f2194f3da2407b8991073a8e86ffef' and
        len(source_lock['files']) == 4816 and
        [row for row in source_lock['files'] if row['path'] == D+'/process_entry_context.py'] ==
        [repeat.refs[D+'/process_entry_context.py']], 'actual frozen context source row')
    source=repeat.text(D+'/process_entry_context.py'); tree=ast.parse(source)
    creators=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='create']
    require(len(creators)==1, 'actual context create source')
    loader_calls=[node for node in ast.walk(creators[0]) if isinstance(node,ast.Call) and
        isinstance(node.func,ast.Attribute) and node.func.attr=='load_receipt']
    require(len(loader_calls)==1 and '_validation_count += 1' in source,
        'source matches one successful public loader path plus counted metadata validations')
    summary=repeat.read(D+'/REPEATABILITY_SUMMARY.json')
    supervisor=repeat.read(D+'/GPU_SUPERVISOR_RESULT.json')
    repeat.text(D+'/GPU_SUPERVISOR_STDOUT.log'); repeat.text(D+'/GPU_SUPERVISOR_STDERR.log')
    require(summary.get('actual_valid_records') == 3 and summary.get('normal_qualification_passed') is False and
        summary.get('permits_next_mode') is None and supervisor.get('exit') == 0 and
        supervisor.get('active_reservation_after') is None, 'actual completed diagnostic supervisor/summary')
    before=repeat.read(D+'/PREREGISTRATION_LEDGER_SNAPSHOT.json')
    after=repeat.read(repeat.manifest['original_ledger_snapshot_ref']['path'],
        repeat.manifest['original_ledger_snapshot_ref'])
    require(before.get('active_reservation') is None and after.get('active_reservation') is None and
        len(after['events']) == len(before['events'])+3 and after['events'][:-3] == before['events'],
        'actual original append-only three-event ledger prefix')
    for index,label in enumerate(LABELS):
        event=repeat.read('experiments/prefix_io_v1/runs/'+label+'/result.json')
        snapshot=repeat.read(D+'/LEDGER_BEFORE_off'+str(index+1).zfill(2)+'.json')
        require(after['events'][-3+index] == event and snapshot['events'] == after['events'][:len(before['events'])+index] and
            snapshot.get('active_reservation') is None, 'actual each prelaunch prefix and unique completion event')
        require(abs(snapshot['gpu_wall_seconds']-before['gpu_wall_seconds']-
            sum(row['guard_elapsed_seconds'] for row in slots[:index])) < 1e-8,
            'actual each prelaunch cumulative budget fold')
        require(0 < slots[index]['guard_elapsed_seconds'] <= 320, 'original single-attempt allowance')
    consumed=sum(row['guard_elapsed_seconds'] for row in slots)
    start=seconds(before['gpu_wall_seconds'],'ledger start'); end=seconds(after['gpu_wall_seconds'],'ledger end')
    require(abs((end-start)-consumed) < 1e-8 and end <= 28800 and
        supervisor['gpu_seconds_before'] == start and supervisor['gpu_seconds_after'] == end,
        'actual cumulative original eight-hour ledger fold')
    oldrun='experiments/prefix_io_v1/runs/server12-c5-native-normal-off01'
    oldraw=prior.read(oldrun+'/details/p4-single-file-runtime-result.json')
    oldguard=prior.read(oldrun+'/result.json'); oldlog=log_observations(prior.text(oldrun+'/process.log'))
    require(oldraw.get('original_engine_shutdown_returned') is True and oldguard.get('session_drained') is True,
        'actual old completed normal off lifecycle')
    oldelapsed=seconds(oldguard['elapsed_seconds'],'old guard elapsed')
    require(len({row['reservation_id'] for row in after['events'][-3:]}) == 3, 'three unique actual reservations')
    repeat.check_after(); prior.check_after(); actual_imports=clean_imports()
    missing=[
        'temperature/power/clocks/throttle samples bound to each measured frame',
        'SSD/page-cache state or repeatable cache-state witness at the actual read',
        'per-request/per-native-step JIT start/end and kernel signatures mapped to selected Event intervals',
        'explicit host-wall/host-monotonic/CUDA-event correlation; none inferred from absolute CPU timestamps',
        'isolated strategy observation cost and paired off/shadow/on performance comparison',
        'old off public-loader counts and startup spans; no retroactive dedup savings estimate']
    return dict(schema='c5_fixed_startup_evidence_cpu_audit_v1',status='PASS_ACTUAL_STARTUP_EVIDENCE_CPU_AUDIT',
        origin='SHA_verified_actual_GPU_artifact_readonly_CPU_audit',slots=slots,
        loader_observation_source=dict(source_ref=repeat.refs[D+'/process_entry_context.py'],
            exactly_one_public_call_on_successful_create_path=True,metadata_counter_increment_present=True),
        descriptive_startup_statistics={key:stats([row['startup_spans'][key]['duration_seconds'] for row in slots])
            for key in ('process_full_validation','public_calibration_replay','llm_initialization','sdk_private_environment')},
        guard_budget=dict(original_cap_seconds=28800,used_before_seconds=start,used_after_seconds=end,
            actual_three_guard_seconds=consumed,remaining_seconds=28800-end,
            supervisor_elapsed_including_CPU_verification_seconds=supervisor['elapsed_including_CPU_verification_seconds'],
            supervisor_wall_is_not_the_guard_charged_GPU_budget=True,budget_expanded=False),
        old_off=dict(actual_guard_seconds=oldelapsed,startup_stage_times_present='startup_stage_times' in oldraw,
            public_loader_counts_observed=False,logs=oldlog,
            guard_duration_differences_from_new_seconds=[oldelapsed-row['guard_elapsed_seconds'] for row in slots],
            difference_is_descriptive_not_causal_dedup_effect=True,
            source_refs={key:prior.refs[oldrun+'/'+filename] for key,filename in
                (('raw','details/p4-single-file-runtime-result.json'),('guard','result.json'),('log','process.log'))}),
        actual_selected_values_ns=[row['selected_elapsed_ns'] for row in slots],
        original_cost_upper_ns=16238752,original_A_only_budget_ns=13171328,
        audit_conclusion='startup is observed; selected-step cost cause remains unproven; retain failed normal gate',
        missing_evidence_and_limits=missing,minimal_next_observations=missing[:3],
        GPU_rental_decision='No new GPU is needed for this audit. A further bounded diagnostic only becomes informative '
            'after missing step-bound JIT and resource observations can be recorded; do not rerun unchanged samples to seek a pass.',
        policy_candidates_added=0,thresholds_changed=False,calibration_refit=False,
        normal_qualification_passed=False,permits_next_mode=None,P4_strategy_effect_verified=False,
        performance_benefit_proved=False,GPU_operations_this_action=0,model_imports=actual_imports,
        shared_library_loads_this_action=0,forbidden_import_attempts=list(IMPORT_ATTEMPTS),
        repeat_manifest_ref=repeat.manifest_ref,prior_manifest_ref=prior.manifest_ref,
        repeat_local_mapping_ref=repeat.mapping_ref,prior_local_mapping_ref=prior.mapping_ref,
        input_file_refs=list(repeat.checked.values())+list(prior.checked.values()),
        local_archive_maps_used=repeat_mapping is not None or prior_mapping is not None,
        full_model_weight_or_SDK_source_tree_rehashed_this_action=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--repeat-mapping',type=Path)
    parser.add_argument('--prior-mapping',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    clean_imports();sys.meta_path.insert(0,CPUOnlyImports())
    result=audit(args.project_root,args.repeat_mapping,args.prior_mapping)
    root=no_links(args.project_root)
    script=no_links(__file__)
    require(script.is_relative_to(root), 'actual audit source within project root')
    result['audit_source_ref']=bytes_ref(script,script.relative_to(root).as_posix())[1]
    output=args.output.absolute()
    allowed_parent=root/(OUT if args.repeat_mapping is None and args.prior_mapping is None else
        'artifacts/prefix_io_v1_server12_candidates/fixed_evidence_cpu_audit')
    require(output.parent.resolve(strict=True) == allowed_parent.resolve(strict=True) and
        output.name.startswith('STARTUP_') and output.suffix == '.json' and output.parent.is_dir() and
        not output.exists() and not any(path.is_symlink() for path in output.parents),
        'new append-only output within existing non-symlink directory')
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    with output.open('xb') as stream:stream.write(raw)
    print(json.dumps(dict(status=result['status'],output=str(output),bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),actual_three_guard_seconds=result['guard_budget']['actual_three_guard_seconds'],
        public_loader_calls_per_process=[row['public_loader_calls'] for row in result['slots']],
        metadata_revalidations_per_process=[row['metadata_revalidations'] for row in result['slots']],
        causal_cost_exceedance_proved=False,GPU_operations_this_action=0),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
