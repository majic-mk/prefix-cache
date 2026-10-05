"""Read-only archived-source audit and explicitly synthetic local CPU replay.

No SSH, CUDA, kernel AIO, model imports, candidate edits or GPU timing occurs.
"""
from __future__ import annotations
import argparse
import ast
import os
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-root',required=True)
    parser.add_argument('--author-root',required=True)
    parser.add_argument('--aio-source')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    global C
    C=Path(args.candidate_root).resolve(strict=True)
    author=Path(args.author_root).resolve(strict=True)
    os.environ['SERVER11_AUTHOR_SOURCE_ROOT']=str(author)
    sources={}
    mappings={REACTOR:C/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py',
              COLLECTOR:C/'native_full_step_collector.py',
              CP+'/test_retained_idle_wait.py':C/'test_retained_idle_wait.py'}
    mappings[AIO]=(Path(args.aio_source).resolve(strict=True) if args.aio_source else
        C/'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/linux_aio.py')
    pins={REACTOR:'96bfd88dcee7f9c7996518ffe762b87be5ad81ef11d598329d41145e2b01fd87',
          AIO:'0a987479722c7d6b520b99c9b283febb55c5145fc17c5623b8b1151c4f45a6d7',
          COLLECTOR:'b8249e9a59a65aa13d452ba24f8fa255ca90ec290f3f1924db883b1fa550a1ce',
          CP+'/test_retained_idle_wait.py':'aa7c372d70cb5d52bb1a259b9f211d17fc213bcbb382826787915f2eb5773462'}
    refs=[]
    for key,path in mappings.items():
        raw=path.read_bytes();require(digest(raw)==pins[key],'actual archived C4 source pin: '+key)
        sources[key]=raw;refs.append(dict(path=str(path),bytes=len(raw),sha256=digest(raw)))
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
    require(not any('torch'==n or n.startswith('torch.') or 'vllm'==n or n.startswith('vllm.') for n in sys.modules),
            'CPU audit must not import GPU/model packages')
    output=dict(scope='frozen_C4_cpu_notification_audit_only',source_refs=refs,
        synthetic_cpu_test_count=result.testsRun,synthetic_cpu_success=True,cpu_test_log=log.getvalue(),
        synthetic_notification_cases=dynamic,synthetic_181_retry_facts=facts,synthetic_profile_call_counts=counted,
        gpu_operations=0,ssh_connections=0,candidate_edits=0,gpu_causal_attribution_proved=False,
        new_wait_protocol_verified=False,timing_extrapolation_to_64898699ns_forbidden=True)
    with Path(args.output).open('x',encoding='utf8') as stream:
        json.dump(output,stream,indent=2,ensure_ascii=False);stream.write('\n')
    print(json.dumps(dict(synthetic_cpu_test_count=result.testsRun,success=True,
        synthetic_181_retry_facts=facts,gpu_operations=0),indent=2))


if __name__=='__main__':main()
