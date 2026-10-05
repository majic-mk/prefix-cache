"""Recalculate finite I-policy evidence using only a bounded archive-member bundle."""
from pathlib import Path
import argparse,ast,hashlib,json,statistics

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence',type=Path,default=HERE/'SELECTED_ARCHIVE_MEMBERS.json')
parser.add_argument('--output',type=Path,default=HERE/'ANALYSIS.json')
args=parser.parse_args()
bundle_bytes=args.evidence.read_bytes()
bundle=json.loads(bundle_bytes);members=bundle['selected_members']
A='artifacts/prefix_io_v1/'
C=A+'server11-p4-single-file-candidate-v4-20261003/'
D=A+'server11-p4-single-file-runtime-v4-20261003/'
N=A+'server11-native-cost-v6-20261003/'
P=A+'server11-p4-retry-cpu-profile-v2-20261003/'
R='experiments/prefix_io_v1/runs/'
checks=[]
def check(value,message):
 if not value:raise ValueError(message)
 checks.append(message)
def get(name):return members[name]['content']
def ref(name):return dict(path=name,bytes=members[name]['bytes'],sha256=members[name]['sha256'])
def match_ref(row):
 check(row==ref(row['path']),'byte/hash reference '+row['path'])
def functions(name):
 text=get(name);check(hashlib.sha256(text.encode()).hexdigest()==members[name]['sha256'],'source text SHA '+name)
 return {node.name:node for node in ast.walk(ast.parse(text)) if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
def function_fact(name,func):
 node=functions(name)[func]
 return dict(path=name,sha256=members[name]['sha256'],function=func,line=node.lineno,end_line=node.end_lineno,
  source=ast.get_source_segment(get(name),node))

q={m:get(D+'QUALIFICATION_'+m+'.json') for m in ('off','shadow','on')}
arms={};raws={}
for mode,qual in q.items():
 name=R+'server11-p4-single-file-'+mode+'-04/details/p4-single-file-runtime-result.json'
 raw=get(name);raws[mode]=raw;match_ref(qual['evidence_refs']['raw_result'])
 window=raw['windows'][0];capture=window['capture'];frames=capture['frames'];witnesses=capture['event_witnesses']
 check(len(frames)==len(witnesses)==128,'128 original frames '+mode)
 check([w['gpu_elapsed_ns'] for w in witnesses]==qual['all_step_gpu_elapsed_ns'],'raw event array equals qualification '+mode)
 check([f['native_step_ordinal'] for f in frames]==[w['native_step_ordinal'] for w in witnesses],'native ordinals align '+mode)
 t=raw['request_and_drain_timing']
 check(t['request_finished_ns']-t['request_started_ns']==qual['whole_request_ns'],'request host interval '+mode)
 check(t['drain_finished_ns']-t['request_started_ns']==qual['total_request_and_drain_ns'],'request-through-drain interval '+mode)
 check(t['drain_finished_ns']-t['drain_started_ns']==qual['drain_ns'],'pure drain interval '+mode)
 io=qual['native_io'];selected=witnesses[16];frame=frames[16];origin=selected['start_record_before_ns']
 events=raw['native_journal']['events']
 for kind,key in (('accepted','accepted_at_ns'),('completed','completed_at_ns')):
  check(len([e for e in events if e['stage']=='ssd_read' and e['kind']==kind and e['at_ns']==io[key] and e['physical_bytes']==917504])==1,'actual native journal '+mode+' '+kind)
 gpu=[w['gpu_elapsed_ns'] for w in witnesses];host=[f['end_ns']-f['start_ns'] for f in frames]
 action=capture['actions'][0]
 check(action['native_step_ordinal']==selected['native_step_ordinal']==frame['native_step_ordinal']==146,'selected original native ordinal '+mode)
 timeline={k:selected[k]-origin for k in ('start_record_before_ns','start_record_after_ns','start_completed_query_ns','end_record_before_ns','end_record_after_ns','end_completed_query_ns')}
 timeline.update(host_start_ns=frame['start_ns']-origin,host_end_ns=frame['end_ns']-origin,
   trigger_before_ns=action['trigger_before_ns']-origin,trigger_after_ns=action['trigger_after_ns']-origin,
   native_read_accepted_ns=io['accepted_at_ns']-origin,native_read_completed_ns=io['completed_at_ns']-origin)
 arms[mode]=dict(raw_ref=ref(name),qualification_ref=ref(D+'QUALIFICATION_'+mode+'.json'),
  full_output_tokens=qual['full_output_tokens'],whole_request_ns=qual['whole_request_ns'],
  total_request_and_drain_ns=qual['total_request_and_drain_ns'],
  pure_drain_ns=qual['drain_ns'],between_request_and_drain_ns=t['drain_started_ns']-t['request_finished_ns'],
  selected_gpu_event_span_ns=gpu[16],selected_host_call_wall_ns=host[16],
  sum_gpu_event_spans_ns=sum(gpu),sum_host_call_wall_ns=sum(host),
  host_interstep_gap_sum_ns=sum(frames[i]['start_ns']-frames[i-1]['end_ns'] for i in range(1,128)),
  first_16_gpu_sum_ns=sum(gpu[:16]),after_111_gpu_sum_ns=sum(gpu[17:]),
  first_16_gpu_mean_ns=statistics.mean(gpu[:16]),after_111_gpu_mean_ns=statistics.mean(gpu[17:]),
  first_16_host_mean_ns=statistics.mean(host[:16]),after_111_host_mean_ns=statistics.mean(host[17:]),
  trigger_callback_wall_ns=action['trigger_after_ns']-action['trigger_before_ns'],
  native_submit_to_complete_ns=io['completed_at_ns']-io['accepted_at_ns'],
  timeline_relative_to_selected_start_record_ns=timeline,
  deferral=qual['deferral'],cost_upper_ns=qual['frozen_cost_upper_ns'],
  cost_covered=qual['frozen_cost_migration_pass'],limited_qualified=qual['qualification_passed'],
  all_steps=[dict(index=i,native_ordinal=f['native_step_ordinal'],gpu_event_span_ns=g,host_call_wall_ns=h)
   for i,(f,g,h) in enumerate(zip(frames,gpu,host))])
 check(sum(gpu)==qual['sum_full_step_gpu_ns'],'complete event span sum '+mode)
check(q['off']['output_token_ids']==q['shadow']['output_token_ids']==q['on']['output_token_ids'],'same complete output')
decision=q['on']['deferral']['actual_decisions'][0]
on_state=raws['on']['p4_after_measurement']['conditional_single_file']
check(on_state['actual_blocked_attempts']==on_state['previews']==decision['count']==181,'actual 181 attempts')
check(on_state['observation_reuses']==180,'actual 180 observation reuses')
on_origin=raws['on']['windows'][0]['capture']['event_witnesses'][16]['start_record_before_ns']
arms['on']['timeline_relative_to_selected_start_record_ns'].update(first_defer_ns=decision['first_defer_ns']-on_origin,last_defer_ns=decision['last_defer_ns']-on_origin)
retry=dict(actual_attempts=181,actual_observation_reuses=180,
 first_to_last_wall_ns=decision['last_defer_ns']-decision['first_defer_ns'],
 first_defer_to_read_accept_wall_ns=q['on']['native_io']['accepted_at_ns']-decision['first_defer_ns'],
 mean_interval_between_first_and_last_ns=(decision['last_defer_ns']-decision['first_defer_ns'])/180,
 per_attempt_actual_cpu_durations_available=False,mean_interval_is_not_per_attempt_CPU_cost=True)
deltas={}
for base in ('off','shadow'):
 d={k:arms['on'][k]-arms[base][k] for k in
  ('whole_request_ns','total_request_and_drain_ns','selected_gpu_event_span_ns','selected_host_call_wall_ns',
   'sum_gpu_event_spans_ns','sum_host_call_wall_ns','first_16_gpu_sum_ns','after_111_gpu_sum_ns',
   'between_request_and_drain_ns','pure_drain_ns','host_interstep_gap_sum_ns')}
 d['total_change_percent']=(arms['on']['total_request_and_drain_ns']/arms[base]['total_request_and_drain_ns']-1)*100
 check(d['sum_gpu_event_spans_ns']==d['first_16_gpu_sum_ns']+d['selected_gpu_event_span_ns']+d['after_111_gpu_sum_ns'],'exact segmented event delta '+base)
 deltas['on_minus_'+base]=d

measurements=get(N+'NATIVE_MEASUREMENTS.json');parent=get(R+'server11-native-cost-six-window-06/details/native-cost-runtime-result.json')
match_ref(measurements['raw_runtime_ref'])
plan=get(N+'NATIVE_COST_PLAN.json');cost=get(N+'NATIVE_CONDITIONAL_COST_RESULT.json')
for value in (cost['evidence_refs']['plan'],cost['evidence_refs']['measurements'],plan['native_source_ref'],plan['collector_source_ref']):match_ref(value)
pairs={};parent_windows={w['capture']['run_id']:w for w in parent['windows']}
for row in measurements['windows']:
 raw_window=parent_windows[row['run_id']]
 check(row['capture']==raw_window['capture'],'measurement matches parent raw capture '+row['run_id'])
 events=row['capture']['event_witnesses'];check(len(events)==128,'full calibration capture '+row['run_id'])
 pairs.setdefault(row['pair_id'],{})[row['arm']]=events[16]['gpu_elapsed_ns']
for key,row in pairs.items():
 row['observed_B_minus_A_ns']=row['action']-row['baseline']
 row['split']=next(e['split'] for e in plan['entries'] if e['pair_id']==key)
calibration_deltas=[r['observed_B_minus_A_ns'] for r in pairs.values() if r['split']=='calibration']
check(statistics.mean(calibration_deltas)==cost['calibration_only_cost']['incremental_or_joint_ns'],'calibration-only incremental mean')
for pair,values in cost['calibration_only_cost']['pair_means'].items():
 check(values==[pairs[pair]['baseline'],pairs[pair]['action']],'calibration estimator pair '+pair)
visible=[x['observed_B_minus_A_ns'] for x in pairs.values()]
profiles={}
for label,name in [('v3',P+'ACTUAL_CPU_full_borrow.json'),('v4',C+'CPU_PROFILE_FULL_BORROW.json')]:
 x=get(name)
 check(x['retries_per_group']==x['semantic_facts']['retries']==155,'profile count 155 '+label)
 check(x['live_capture_is_synthetic'] is True and x['gpu_workloads_run']==0,'synthetic CPU profile scope '+label)
 check(statistics.median(g['thread_cpu_ns'] for g in x['unprofiled_groups'])==x['unprofiled_thread_cpu_ns']['median'],'actual unprofiled median '+label)
 names={r['function'] for r in x['profiled_function_rows']}
 check(not ({'_run','_pump_once','execute_model','sample_tokens'} & names),'profile excludes full reactor/decode loops '+label)
 profiles[label]=dict(profile_ref=ref(name),retries=155,unprofiled_groups=len(x['unprofiled_groups']),
  unprofiled_thread_cpu_ns=x['unprofiled_thread_cpu_ns'],unprofiled_wall_ns=x['unprofiled_wall_ns'],
  profiled_group_timing=x['profiled_group_timing'],synthetic_capture=True,actual_181_covered=False,
  full_reactor_pump_covered=False,concurrent_decode_or_GIL_profile=False,
  profiled_functions_top_12=x['profiled_function_rows'][:12],original_limitations=x['limitations'])
profile_comparison=dict(median_thread_cpu_saved_ns=profiles['v3']['unprofiled_thread_cpu_ns']['median']-profiles['v4']['unprofiled_thread_cpu_ns']['median'],
 percent_saved=(1-profiles['v4']['unprofiled_thread_cpu_ns']['median']/profiles['v3']['unprofiled_thread_cpu_ns']['median'])*100,
 counts_and_state_fixed_at_155_synthetic_only=True,linear_181_extrapolation_used=False)
reactor=C+'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
facts=[function_fact(reactor,n) for n in ('_has_poll_work','_run','_pump_once','_prefix_stage_decide','_prefix_single_file_retry_key','_drain_ready_preload_fds')]
facts += [function_fact(C+'test_single_file_retry_observation.py','repeat'),function_fact(C+'native_full_step_collector.py','current_single_file_step')]
profile_ast=ast.parse(get(P+'profile_retry_hotpath.py'))
check(any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='RETRIES' for t in n.targets) and isinstance(n.value,ast.Constant) and n.value.value==155 for n in profile_ast.body),'profiler source RETRIES=155')
limit=min(visible);upper=max(visible)
result=dict(schema_version=1,scope='CPU_only_independent_runtime_feasibility_audit',
 archive_ref=bundle['archive_ref'],selected_member_count=bundle['selected_member_count'],
 evidence_bundle_sha256=hashlib.sha256(bundle_bytes).hexdigest(),
 source_refs=[dict(path=name,bytes=row['bytes'],sha256=row['sha256']) for name,row in members.items()],
 arms=arms,retry=retry,delta_decomposition=deltas,
 v6_pairs=pairs,v6_observed_increment_min_ns=limit,v6_observed_increment_max_ns=upper,
 scale_comparison=dict(retry_wall_over_max_observed_increment=retry['first_to_last_wall_ns']/upper,
 total_extra_wall_over_max_observed_increment=deltas['on_minus_off']['total_request_and_drain_ns']/upper,
 visible_increment_is_not_guaranteed_recoverable_benefit_or_universal_upper=True),
 profiles=profiles,profile_comparison=profile_comparison,source_path_facts=facts,
 CPU_command=['python','-B','analyze_runtime_evidence.py','--evidence',args.evidence.name,'--output',args.output.name],
 gpu_or_rpc_operations=0,checks=checks,check_count=len(checks),
 verdict=dict(current_active_poll_I_candidate='STOP_UNCHANGED_GPU_REPEATS',
 evidence='Finite lifecycle works, but observed total and selected-step costs are negative; current profile does not explain the live 181-attempt timeline causally.',
 new_performance_candidate_established=False,
 next_CPU_only_diagnostic='A bounded 181-attempt concurrent replay of the original full pump plus an independent scalar-state producer, with thread CPU and wall times, would close a measurement gap; it is not yet a performance-policy candidate.',
 future_change_gate='Do not remove live resource/mandatory/max-wait checks or change sleep/switch interval, thresholds, executor, engine or shared evidence. A proposed value-only optimization must first preserve all checks and demonstrate lower full-path cost with identical producer schedule on CPU; CPU success still cannot establish GPU benefit.',
 causal_limits=['CUDA event duration and host-call wall duration are different clocks/scopes; no absolute cross-clock mapping exists.',
 '181 attempts are aggregated count/first/last, not 181 timestamped CPU profiles.',
 'CPU profile is 155 synthetic ready-drain calls, excluding full pump/OS sleep and concurrent decode producer.',
 'GIL contention, CPU submission gaps, OS scheduling and kernel stalls remain hypotheses, not measured causal fractions.',
 'Paired B-A increments come from distinct finite native runs and are not a guaranteed saved-time ceiling.']))
with args.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status='PASS_RECALCULATED',check_count=len(checks),retry=retry,
 delta_decomposition=deltas,v6_pairs=pairs,profile_comparison=profile_comparison,
 selected_timeline=arms['on']['timeline_relative_to_selected_start_record_ns'],
 selected_host_call_wall_ns={m:a['selected_host_call_wall_ns'] for m,a in arms.items()},
 sum_host_call_wall_ns={m:a['sum_host_call_wall_ns'] for m,a in arms.items()},
 verdict=result['verdict']),ensure_ascii=False,indent=2))

