"""CPU source replay: empty native work, retained shared cache, optional bridge.

This counts calls in unchanged extracted functions. The fixture's collector
returns an empty view and exercises the real bridge epoch implementation; it
does not claim a measured GPU profile or simulate actual kernel execution.
"""
from __future__ import annotations
import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS


REACTOR='source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
BRIDGE='source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py'
METHODS=('_has_work','_prefix_p4_read_batch_ids','_prefix_p4_order','_prefix_ordered_stores',
    '_drain_ready_load_fds','_drain_ready_preload_fds','_drain_cancelled_pending','_schedule_work')


def source_methods(path, class_name, names):
    raw=path.read_bytes(); tree=ast.parse(raw.decode('utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
        +selected,type_ignores=[])
    namespace={'time':time}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace,dict(source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),
        source_methods_ast_sha256=hashlib.sha256(ast.dump(module,include_attributes=False).encode()).hexdigest())


def replay(candidate, *, enabled, pumps=10000):
    native,source=source_methods(candidate/REACTOR,'IoReactor',METHODS)
    methods={name:native[name] for name in METHODS}
    Owner=type('OriginalExtractedOwnerPaths',(),methods)
    owner=Owner();owner._active=deque();owner._inflight={};owner._pending_copies=[];owner._copy_ready=[]
    owner._preload_pending=deque();owner._preload_slots={};owner._shared_cached={b'x'*32:object()}
    owner._ready_fds_load=deque();owner._ready_fds_preload=deque();owner._prefix_dispatch_controller=None
    owner._prefix_store_order=None;owner._data_inflight=0;owner.iodepth=4
    owner._can_issue_speculative_preload=lambda:True
    epochs,epoch_source=source_methods(candidate/BRIDGE,'NativeP4Bridge',('epoch',))
    bridge=NS(policy=NS(config=NS(sample_max_age_ns=100000000)),_signature=None,
        _epoch=0,_epoch_started_ns=None,_last_clock_ns=None,_publication=None,_choice_epoch=None,
        _batch_key=None,failures=[],choices=0)
    bridge._owner=lambda:None;bridge.fail=bridge.failures.append
    bridge.epoch=lambda signature,now:epochs['epoch'](bridge,signature,now)
    owner._prefix_p4_bridge=bridge if enabled else None
    count=0
    def collect():
        nonlocal count
        count+=1
        # 30ms of deterministic fixture clock across 30k calls: at most one
        # real signature epoch despite repeated empty-view reconstruction.
        bridge.epoch(((),(),(),917504,False,(),(),(),None),1000000000+count*1000)
        return NS(works=())
    owner._prefix_p4_collect=collect
    before=(tuple(owner._active),tuple(owner._ready_fds_load),tuple(owner._ready_fds_preload),
            tuple(owner._shared_cached))
    assert owner._has_work() is True
    started=time.perf_counter_ns()
    made=sum(owner._schedule_work() is True for _ in range(pumps))
    elapsed=time.perf_counter_ns()-started
    after=(tuple(owner._active),tuple(owner._ready_fds_load),tuple(owner._ready_fds_preload),
           tuple(owner._shared_cached))
    assert before==after and not made and bridge.failures==[] and owner._has_work() is True
    return dict(candidate=str(candidate),bridge_enabled=enabled,pumps=pumps,
        optional_collect_calls=count,actual_signature_epochs=bridge._epoch,choices=bridge.choices,
        native_progress_count=made,retained_cache_keeps_original_has_work=True,
        original_native_queues_unchanged=True,cpu_fixture_elapsed_ns=elapsed,
        actual_gpu_execution=False,does_not_measure_real_collect_cost=True,
        reactor_source=source,bridge_epoch_source=epoch_source)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    rows=[replay(args.candidate,enabled=value) for value in (False,True)]
    result=dict(scope='source_extracted_empty_owner_cpu_replay_v1',runs=rows,
        interpretation='Epoch and choices count state changes and advice, not collect invocations.',
        gpu_performance_causality_proven=False)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
