"""Read-only archived-source audit and explicitly synthetic local CPU replay.

No SSH, CUDA, kernel AIO, model imports, candidate edits or GPU timing occurs.
"""
from __future__ import annotations
import ast
import cProfile
from collections import deque
from dataclasses import dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import threading
import time
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
C = ROOT/'artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4'
RESULTS = ROOT/'artifacts/server11_results/p4_v4'
CP = 'artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003'
DP = 'artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003'
REACTOR = CP+'/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
AIO = CP+'/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/linux_aio.py'
COLLECTOR = CP+'/native_full_step_collector.py'


def require(ok, reason):
    if not ok: raise ValueError(reason)


def digest(raw): return hashlib.sha256(raw).hexdigest()


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value)
    return value


def method(raw, cls, name):
    tree=ast.parse(raw)
    owner=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    return next(x for x in owner.body if isinstance(x,ast.FunctionDef) and x.name==name)


def calls(node):
    return [dict(line=x.lineno,call=ast.unparse(x.func)) for x in ast.walk(node) if isinstance(x,ast.Call)]


def main():
    receipt=json.loads((RESULTS/'ARCHIVE_RECEIPT.json').read_bytes())
    manifest_raw=(RESULTS/'SERVER11_DELTA_MANIFEST.json').read_bytes()
    require(digest(manifest_raw)==receipt['manifest']['sha256'],'archived manifest SHA')
    manifest={r['path']:r for r in json.loads(manifest_raw)['files']}
    archive=RESULTS/'SERVER11_RAW_DELTA.tar.gz'
    with archive.open('rb') as stream:
        require(hashlib.file_digest(stream,'sha256').hexdigest()==receipt['archive']['sha256'],'archive SHA')
    targets=[REACTOR,AIO,COLLECTOR,CP+'/test_retained_idle_wait.py',CP+'/CPU_PROFILE_FULL_BORROW.json']
    targets += [DP+'/QUALIFICATION_'+mode+'.json' for mode in ('off','shadow','on')]
    targets += ['experiments/prefix_io_v1/runs/server11-p4-single-file-on-04/details/p4-single-file-runtime-result.json']
    sources={}
    with tarfile.open(archive,'r:gz') as tar:
        for name in targets:
            raw=tar.extractfile(name).read(); pin=manifest[name]
            require(len(raw)==pin['bytes'] and digest(raw)==pin['sha256'],'archived member: '+name)
            sources[name]=raw
    for relative in ('native_full_step_collector.py','test_retained_idle_wait.py'):
        require((C/relative).read_bytes()==sources[CP+'/'+relative],'local replay source equals archived source')
    require((C/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py').read_bytes()==sources[REACTOR],
            'actual replay reactor equals archived GPU source')
    idle=load('_hotpath_frozen_idle_fixture',C/'test_retained_idle_wait.py')
    collector=load('_hotpath_frozen_collector',C/'native_full_step_collector.py')
    retry=load('_hotpath_frozen_retry_fixture',C/'test_single_file_retry_observation.py')

    # Exercise the real source method with a non-CUDA object. The synthetic
    # origin tag chooses the accessor branch; it is not native GPU evidence.
    class FakeEvent:
        def __init__(self):self.records=0
        def record(self):self.records+=1

    dynamic=[]
    class NotificationTests(unittest.TestCase):
        def test_original_end_record_invalidates_live_state_without_queue_message(self):
            owner=idle.Owner().start()
            try:
                raw_event=FakeEvent();end=collector.EventProxy(raw_event,time.monotonic_ns)
                capture=collector.FullStepCapture(run_id='synthetic-audit',origin='native_gpu_recording',
                    selected_offsets=(16,),action=None)
                start=NS(record_before_ns=1,completed_query_ns=2)
                frame=NS(step_kind='decode',batch=1,active_decode=1,prefill_tokens=0,context_length=144)
                capture.observer=NS(enabled=True,events=NS(active=(5,start,end)),
                    scalar=NS(enabled=True,_adapter=NS(_pending={'phase':'prepared','ordinal':5,'frame':frame})))
                self.assertIsNotNone(capture.current_single_file_step())
                before=owner.pump_count
                end.record()
                self.assertEqual(raw_event.records,1)
                self.assertIsNone(capture.current_single_file_step())
                self.assertEqual(owner._incoming.qsize(),0)
                self.assertEqual(owner.pump_count,before)
                future=owner.request_owner_snapshot()
                self.assertEqual(future.result(timeout=2),{'retained':1})
                dynamic.append(dict(test='synthetic_end_record_does_not_enqueue',record_calls=1,
                    live_value_after=None,incoming_messages_after=0,
                    actual_original_snapshot_producer_then_woke=True))
            finally:
                owner.shutdown(wait=True)

        def test_deferred_ready_keeps_original_poll_predicate_true(self):
            owner=idle.Owner()
            self.assertFalse(owner._has_poll_work())
            owner._ready_fds_preload.append(NS(action='defer'))
            self.assertTrue(owner._has_poll_work())
            dynamic.append(dict(test='deferred_ready_is_poll_work',predicate=True,synthetic_ready=True))

        def test_actual_aio_complete_notifies_its_own_condition_not_incoming(self):
            node=method(sources[AIO],'LinuxAioRing','_complete')
            namespace={}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),AIO,'exec'),namespace)
            incoming=idle.RecordingQueue()
            class Condition:
                def __init__(self):self.notifications=0
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def notify_all(self):self.notifications+=1
            cv=Condition();ring=NS(_cv=cv,_opened={},_done=deque(),_stats={'completed':0})
            namespace['_complete'](ring,NS(kind='read',user_data=7),917504)
            self.assertEqual(cv.notifications,1)
            self.assertEqual(list(ring._done),[(7,917504)])
            self.assertEqual(incoming.qsize(),0)
            dynamic.append(dict(test='synthetic_aio_completion_original_method',ring_notifications=1,
                reactor_incoming_messages=0,physical_io_performed=False))

    suite=unittest.TestSuite()
    for name in ('test_original_submit_job_wakes_existing_queue','test_original_preload_enqueue_wakes_existing_queue',
                 'test_mandatory_unknown_future_wakes_without_creating_job',
                 'test_enqueue_between_predicate_and_queue_get_cannot_lose_wakeup',
                 'test_stop_wakes_and_original_intake_releases_retained_slots_once'):
        suite.addTest(idle.RetainedIdleTests(name))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(NotificationTests))
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    require(result.wasSuccessful(),'CPU source/notification replay failed: '+log.getvalue())

    # Exact 181 decisions using existing scalar CPU fixture, deliberately no
    # timing estimate. cProfile is used only for call counts, never GPU cost.
    owner=retry.Owner(max_wait=10**12);prof=cProfile.Profile();prof.enable()
    for _ in range(181):require(owner.repeat() is False,'original synthetic retry must defer')
    prof.disable();prof.create_stats()
    names={'replace','_prefix_capacity_components','_prefix_single_file_retry_key','preview_issue','issue_preview'}
    counted=[dict(file=file,line=line,function=name,calls=value[1]) for (file,line,name),value in prof.stats.items() if name in names]
    facts=dict(decisions=owner.bridge.single_file_blocked_attempts,previews=owner.bridge.single_file_previews,
        observation_reuses=owner.bridge.single_file_observation_reuses,collects=owner.collects,
        reserves=owner.staging_pool.reserves,releases=owner.staging_pool.releases,
        unchanged_open_ns=owner.ready.open_start_ns,unchanged_sample_ns=owner._prefix_single_file_retry[1].monotonic_ns,
        caveat='181 synthetic CPU retries, live state accessor scalar stub; no measured execution latency')
    require((facts['decisions'],facts['observation_reuses'],facts['collects'],facts['reserves'],facts['releases'])==
        (181,180,1,181,181),'retry semantic counters')
    qualifications={mode:json.loads(sources[DP+'/QUALIFICATION_'+mode+'.json']) for mode in ('off','shadow','on')}
    on=qualifications['on'];decision=on['deferral']['actual_decisions'][0]
    raw_on=json.loads(sources[targets[-1]])
    bridge_after=raw_on['p4_after_measurement']['conditional_single_file']
    count=decision['count'];span=decision['last_defer_ns']-decision['first_defer_ns']
    real=dict(actual_deferrals=count,actual_observation_reuses=bridge_after['observation_reuses'],
        first_defer_ns=decision['first_defer_ns'],last_defer_ns=decision['last_defer_ns'],
        defer_span_ns=span,mean_first_to_last_interval_ns=span/(count-1),
        interval_is_wall_spacing_not_thread_cpu=True,
        selected_gpu_elapsed_ns=on['selected_gpu_elapsed_ns'],
        selected_extra_vs_shadow_ns=on['selected_gpu_elapsed_ns']-qualifications['shadow']['selected_gpu_elapsed_ns'],
        full_request_extra_vs_shadow_ns=on['whole_request_ns']-qualifications['shadow']['whole_request_ns'],
        request_and_drain_extra_vs_shadow_ns=on['total_request_and_drain_ns']-qualifications['shadow']['total_request_and_drain_ns'],
        accepted_at_ns=on['native_io']['accepted_at_ns'],completed_at_ns=on['native_io']['completed_at_ns'])
    historical_profile=json.loads(sources[CP+'/CPU_PROFILE_FULL_BORROW.json'])
    source_nodes={}
    for cls,path,names_list in (
        ('IoReactor',REACTOR,['_has_poll_work','_run','_drain_incoming','_pump_once','_drain_ready_preload_fds',
            '_prefix_stage_decide','_prefix_single_file_retry_key','publish_p4_scheduled_load','_drain_cuda_copies',
            '_poll_ring_completions','submit_job','enqueue_preload','request_mandatory','shutdown']),
        ('LinuxAioRing',AIO,['_complete','poll_all','_run','_wake']),
        ('EventProxy',COLLECTOR,['record','query']),
        ('FullStepCapture',COLLECTOR,['current_single_file_step'])):
        for name in names_list:
            node=method(sources[path],cls,name)
            source_nodes[cls+'.'+name]=dict(source_ref=manifest[path],line=node.lineno,end_line=node.end_lineno,calls=calls(node))
    require(not any('torch'==n or n.startswith('torch.') or 'vllm'==n or n.startswith('vllm.') for n in sys.modules),
            'CPU audit must not import GPU/model packages')
    output=dict(scope='last_cpu_feasibility_hotpath_audit',archive_ref=receipt['archive'],
        independently_verified_members=[manifest[n] for n in targets],real_gpu_evidence_recomputed=real,
        actual_historical_server_cpu_fixture=dict(retries_per_group=historical_profile['retries_per_group'],
            state_accessor=historical_profile['state_accessor'],
            thread_cpu_ns=historical_profile['unprofiled_thread_cpu_ns'],
            not_real_runner_latency=True,profile_times_not_summed=True),
        synthetic_cpu_test_count=result.testsRun,synthetic_cpu_success=True,cpu_test_log=log.getvalue(),
        synthetic_notification_cases=dynamic,synthetic_181_retry_facts=facts,synthetic_profile_call_counts=counted,
        source_method_audit=source_nodes,gpu_operations=0,ssh_connections=0,candidate_edits=0,
        gpu_causal_attribution_proved=False,new_wait_protocol_verified=False)
    with (HERE/'AUDIT_EVIDENCE.json').open('x',encoding='utf8') as stream:
        json.dump(output,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps(dict(real_gpu_evidence_recomputed=real,synthetic_cpu_test_count=result.testsRun,
        synthetic_retry_facts=facts,thread_cpu_fixture=output['actual_historical_server_cpu_fixture']),indent=2))


if __name__=='__main__':main()
