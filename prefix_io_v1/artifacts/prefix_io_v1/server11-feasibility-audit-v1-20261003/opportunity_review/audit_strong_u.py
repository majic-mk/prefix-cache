"""Recompute P316 U dependency evidence. Stdlib CPU only; no GPU imports."""
import argparse
import collections
import hashlib
import json
import pathlib
import sys


def sha(b):
    return hashlib.sha256(b).hexdigest()


def analyse(d, t):
    probe = d['probe']
    diag = probe['flush_diagnostic']
    assert not diag['faulted'] and diag['errors'] == 0
    assert diag['dropped_records'] == diag['dropped_jobs'] == diag['dropped_causes'] == 0
    waits = [e for e in probe['events'] if e['kind'] == 'native_jobs_to_flush_wait' and e['pending_before'] > 0]
    retired = [r for r in diag['records'] if r['kind'] == 'native_parent_retired']
    blocks = [b for r in retired for b in r['source_blocks']]
    output = {'cohort_seconds': d['cohort_seconds_including_drain'], 'response_seconds': d['response_seconds'],
              'tail_drain_seconds': d['tail_drain_seconds'], 'pending_flush_seconds': probe['pending_flush_wait_seconds'],
              'pending_flush_over_cohort_pct': 100 * probe['pending_flush_wait_seconds'] / d['cohort_seconds_including_drain'],
              'wait_ratio_is_not_a_causal_speedup_or_strict_upper_bound': True,
              'retired_parent_records': len(retired), 'sampled_retired_blocks': len(blocks),
              'sampled_active_ref_histogram': dict(collections.Counter(b['active_refs'] for b in blocks)),
              'sampled_free_queue_linked_count': sum(b['free_queue_linked'] for b in blocks),
              'sampled_reusable_bytes_known_count': sum(b['immediately_reusable_bytes'] is not None for b in blocks),
              'physical_GPU_blocks_freed': None, 'GPU_release_credit': False,
              'foreground_slot_unavailable': probe['foreground_slot_unavailable'],
              'read_bytes': d['cohort_read_bytes'], 'write_bytes': d['cohort_write_bytes'], 'fences': []}
    base = t['start_ts_ns'] / 1e9
    for wait in waits:
        start, end = wait['start'], wait['start'] + wait['seconds']
        batches = [r for r in diag['records'] if r['kind'] == 'flush_batch' and set(r['parent_ids']) == set(wait['ids'])]
        assert len(batches) == 1
        causes = [c for c in batches[0]['causes'] if c['reason'] == 'restore_destination']
        assert len(causes) == 1
        cause = causes[0]
        target = [r for r in d['rows'] if r['internal_request_id'] == cause['trigger_req_id']]
        assert len(target) == 1
        target = target[0]
        ends = sorted(d['rows'], key=lambda r: r['response_end_seconds'], reverse=True)
        fence = {'wait': wait, 'cause': cause, 'target_request_id': target['request_id'],
                 'target_metrics': target['metrics'], 'target_response_end_seconds': target['response_end_seconds'],
                 'last_request_id': ends[0]['request_id'], 'last_request_response_end_seconds': ends[0]['response_end_seconds'],
                 'target_finished_before_last_request_seconds': ends[0]['response_end_seconds'] - target['response_end_seconds'],
                 'first_target_token_after_wait_end_seconds': target['engine_token_timestamps'][0] - end,
                 'frontend_steps_overlapping_wait': [s for s in d['steps'] if s['end'] >= start and s['start'] <= end],
                 'all_requests_token_emissions_during_wait': sum(start <= ts <= end for r in d['rows'] for ts in r['engine_token_timestamps']),
                 'wait_inside_request_path': True,
                 'request_path_explanation': 'restore_destination cause names this pending request; wait returns before its first token.',
                 'cohort_counterfactual_speedup_known': False, 'parents': []}
        for job in sorted(wait['ids']):
            matches = [e for e in t['traceEvents'] if e.get('args', {}).get('job_id') == job]
            ev = {}
            for name in ['py_kvcache.transfer', 'py_kvcache.cuda_staging', 'py_kvcache.file_write']:
                rows = [e for e in matches if e['name'] == name]
                assert len(rows) == 1, (job, name, len(rows))
                ev[name] = rows[0]
            x, copy, write = (ev[n] for n in ['py_kvcache.transfer', 'py_kvcache.cuda_staging', 'py_kvcache.file_write'])
            rr = [r for r in retired if r['job_id'] == job]
            assert len(rr) == 1
            cr = [p for p in cause['parents'] if p['job_id'] == job]
            assert len(cr) == 1
            cr = cr[0]
            assert x['args']['success'] is True
            parent = {'job_id': job, 'req_id': x['args']['req_id'], 'bytes': x['args']['num_bytes'],
                      'parent_age_at_wait_seconds': start - (base + x['ts'] / 1e6),
                      'D2H_host_enqueue_from_wait_seconds': base + copy['ts'] / 1e6 - start,
                      'D2H_event_elapsed_seconds': copy['dur'] / 1e6,
                      'SSD_write_start_from_wait_seconds': base + write['ts'] / 1e6 - start,
                      'SSD_host_interval_seconds': write['dur'] / 1e6,
                      'SSD_write_end_from_wait_seconds': base + (write['ts'] + write['dur']) / 1e6 - start,
                      'parent_transfer_complete_from_wait_seconds': base + (x['ts'] + x['dur']) / 1e6 - start,
                      'scheduler_retirement_record_from_wait_seconds': rr[0]['monotonic'] - start,
                      'cause_owner_state': cr['source_blocks'], 'retired_owner_state': rr[0]['source_blocks'],
                      'no_sampled_free_capacity_witness': all(b['active_refs'] > 0 and not b['free_queue_linked'] for b in rr[0]['source_blocks']),
                      'physical_freed_blocks': None}
            fence['parents'].append(parent)
        fence['protected_parent_bytes'] = sum(p['bytes'] for p in fence['parents'])
        fence['before_first_D2H_host_enqueue_seconds'] = min(p['D2H_host_enqueue_from_wait_seconds'] for p in fence['parents'])
        fence['remaining_after_first_D2H_host_enqueue_seconds'] = wait['seconds'] - fence['before_first_D2H_host_enqueue_seconds']
        fence['after_last_transfer_before_wait_return_seconds'] = wait['seconds'] - max(p['parent_transfer_complete_from_wait_seconds'] for p in fence['parents'])
        first_copy = start + fence['before_first_D2H_host_enqueue_seconds']
        background = []
        for event in t['traceEvents']:
            if event['name'] != 'py_kvcache.transfer' or event.get('args', {}).get('direction') != 'gpu_to_storage':
                continue
            st = base + event['ts'] / 1e6
            en = base + (event['ts'] + event['dur']) / 1e6
            if st < start < en and event['args']['job_id'] not in wait['ids']:
                job = event['args']['job_id']
                competing = [e for e in t['traceEvents'] if e['name'] == 'py_kvcache.cuda_staging'
                             and e.get('args', {}).get('job_id') == job
                             and start <= base + e['ts'] / 1e6 < first_copy]
                background.append({'job_id': job, 'parent_bytes': event['args']['num_bytes'],
                                   'parent_start_before_wait_seconds': start - st, 'parent_complete_from_wait_seconds': en - start,
                                   'D2H_host_enqueue_events_before_first_target': len(competing),
                                   'D2H_bytes_before_first_target': sum(e['args']['num_bytes'] for e in competing)})
        fence['other_store_parents_in_progress_at_wait_start'] = background
        fence['readiness_evidence'] = {'store_readiness_probe_enabled': d['store_readiness_probe'],
                                     'source_GPU_compute_completion_known_at_wait': False,
                                     'source_GPU_completion_to_target_D2H_ready_chain_recorded': False,
                                     'continuous_staging_and_copy_capacity_at_wait_recorded': False,
                                     'scheduler_decision_reasons_before_D2H_recorded': False,
                                     'target_eligible_earlier': None,
                                     'background_progress_is_not_proof_target_was_eligible': True,
                                     'classification_of_pre_D2H_wait': 'UNKNOWN: competing background progress observed, but legal source/dependency wait and schedulable queue wait are not separated.'}
        fence['D2H_timestamp_scope'] = 'Host enqueue with Event elapsed; not device start/end or true DMA overlap'
        fence['not_additive_parent_durations'] = True
        output['fences'].append(fence)
    assert abs(sum(w['seconds'] for w in waits) - probe['pending_flush_wait_seconds']) < 1e-8
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace-root', type=pathlib.Path, default=pathlib.Path.cwd())
    raw = p.add_mutually_exclusive_group(required=True)
    raw.add_argument('--raw-dir', type=pathlib.Path)
    raw.add_argument('--project-raw', action='store_true', help='Read original project-relative paths recorded in selection')
    p.add_argument('--reference-dir', type=pathlib.Path)
    p.add_argument('--checkpoint-manifest', type=pathlib.Path)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    root = args.workspace_root.resolve()
    ref = args.reference_dir or root / 'artifacts/prefix_io_v1_server08_p3_complete_20260930'
    checkpoint = args.checkpoint_manifest or root / 'artifacts/prefix_io_v1_server08_p3_checkpoint_20260930/server08-p3-16-blocked-checkpoint-evidence-v3-manifest.json'
    sources = []

    def read(path, expected=None):
        b = path.read_bytes()
        actual = sha(b)
        if expected is not None:
            assert actual == expected, (str(path), expected, actual)
        sources.append({'path': str(path.resolve()), 'bytes': len(b), 'sha256': actual,
                        'expected_sha256': expected, 'historical_hash_match': None if expected is None else True})
        return json.loads(b.decode('utf-8-sig'))

    selection = read(ref / 'finite-grid-selection-final.json')
    closeout = read(ref / 'p3-closeout-analysis-final.json')
    archive = read(checkpoint)
    index = {e['source_path'].split('/project/', 1)[-1]: e for e in archive['files']}
    runs = []
    for code in ['01', '06']:
        label = 'server08-p3-16-mixed-' + code + '-off'
        s = [e for e in selection['entries'] if e['label'] == label]
        assert len(s) == 1
        s = s[0]
        trace_path = s['result']['path'].rsplit('/', 1)[0] + '/native-cohort.trace.json'
        def raw_path(kind, original):
            if not args.project_raw:
                return args.raw_dir / (code + '-' + kind + '.json')
            path = (root / original).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                raise ValueError('Recorded source path escapes workspace root')
            return path
        d = read(raw_path('result', s['result']['path']), s['result']['sha256'])
        a = read(raw_path('analysis', s['analysis']['path']), s['analysis']['sha256'])
        config = read(raw_path('config', s['config']['path']), s['config']['sha256'])
        t = read(raw_path('trace', trace_path), index[trace_path]['sha256'])
        assert a['source_sha256'] == s['result']['sha256']
        assert d['cohort_seconds_including_drain'] == a['cohort_seconds_including_drain']
        assert d['probe']['pending_flush_wait_seconds'] == a['native_pending_flush_seconds']
        r = analyse(d, t)
        r['label'] = label
        r['historical_identity_crosschecks'] = {'result_analysis_config_match_finite_grid': True,
                                              'trace_matches_original_checkpoint_manifest': True,
                                              'analysis_source_matches_result': True}
        r['source_paths'] = {'result': s['result']['path'], 'analysis': s['analysis']['path'],
                             'config': s['config']['path'], 'trace': trace_path}
        runs.append(r)
    out = {'schema_version': 1, 'scope': 'CPU-only reanalysis of complete real P316 strong U raw; no SSH/GPU/runtime imports',
           'command': sys.argv, 'sources': sources, 'selection': selection['selected_arm'], 'strong_U_runs': runs,
           'unchanged_historical_target_summary': closeout['target_waits'],
           'limitations': ['The two U runs are descriptive development repeats, not a statistical population estimate.',
                           'Pending-flush/cohort ratios are not maximum removable latency or causal speedup bounds.',
                           'Destination fences protect a request path but no counterfactual propagation through subsequent scheduling was measured.',
                           'All parent timing intervals overlap; summing them would double count.',
                           'Host D2H enqueue and CUDA Event elapsed do not establish device start/end or continuous readiness.',
                           'Physical released GPU blocks remain unknown, not zero; positive active refs prevent treating these sampled retirements as free-capacity evidence.',
                           'The current C4 shared idle fix was not present in historical U, so historical waiting durations are not a current C4 forecast.'],
           'paid_GPU_restart_justified_by_this_audit_alone': False,
           'D_J_universally_falsified': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': sha(args.output.read_bytes()), 'historical_raw_files_verified': 8,
                      'runs': [{'label': r['label'], 'wait_s': r['pending_flush_seconds'],
                                'pre_D2H_s': [f['before_first_D2H_host_enqueue_seconds'] for f in r['fences']],
                                'ready_probe': [f['readiness_evidence']['store_readiness_probe_enabled'] for f in r['fences']]} for r in runs]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
