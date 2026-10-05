"""Read-only frozen-source/CPU progress audit; no receipt or GPU qualification.

Production paths are tried first. Local fallback is the actual verified archive
mapping, with exact map and per-file SHA pins. No reactor/framework is imported.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sys
from types import ModuleType
from dataclasses import asdict, replace

G = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/'
C = G + 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/'
R = G + 'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
MAP = 'artifacts/prefix_io_v1_server12_candidates/native_gpu_delivery/LOCAL_GPU_EVIDENCE_BYTE_VERIFICATION.json'
MAP_SHA = '35d9fac0ab2dbc50694b0a688391939e44aa89aa71e16da4ad2162b93bcb7182'
# Pins are actual rows from the byte-verified native evidence manifest mapping.
PINS = {
 'dependencies.py': (C+'dependencies.py',5772,'41cc50163c51a2f02486230fcfa36c348af95ef687c3f2cb815bdd6797f29481'),
 'dispatch_budget.py': (C+'dispatch_budget.py',11021,'412c43761cb2f59636240e15c884b25d2a1d0ee708684d496259baf2f34d667c'),
 'dispatch_shadow.py': (C+'dispatch_shadow.py',16989,'c3d502423116f8c3b74f976acba092cb574d58d078c7b34f368e9e5b926b8e1b'),
 'simple_stage_policy.py': (C+'simple_stage_policy.py',21557,'8a2dc9586f5e1fe5257f988e44466284d10e0e46a8cee46f876a8febe5caf1a5'),
 'p4_types.py': (C+'p4_types.py',11969,'07bbcb0a449c349f9c63874728e409180c1c9787f3253723403978e17383094e'),
 'p4_cost_table.py': (C+'p4_cost_table.py',5071,'4caae858778182f5575ce2d5807bdc72520ef4c5f32a2c18f02ff2628155d0f8'),
 'p4_policy.py': (C+'p4_policy.py',24976,'bce19c96c7651f69ebb91a515ea3e9e3d6855636b4bd4aa84e60fb2fbc10d673'),
 'p4_eta.py': (C+'p4_eta.py',10679,'befe3d10e3466be8a34d0b4756f993c0100cf909e712b167ec221c37f9064d50'),
 'p4_bridge.py': (C+'p4_bridge.py',25553,'6bec5dd57e25cfc2c0e62540742a781d656ba528cb6c8057cd2402197db93e1a'),
 'reactor.py': (R,185744,'a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47'),
 'native_full_step_collector.py': (G+'native_full_step_collector.py',24553,'bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf'),
 'test_reactor_single_file.py': (G+'test_reactor_single_file.py',12384,'d7387e98bcceb5568eae9827fad3ff2dacc7b2d9bf0a5901322a63eb32cf02a8'),
 'test_single_file_policy.py': (G+'test_single_file_policy.py',10312,'ec61e8c381604e1ee9c7d387a8a8006231af92b9520767276f0d82fd0e633d45'),
 'test_single_file_retry_observation.py': (G+'test_single_file_retry_observation.py',12888,'a24eb1c3d9aadb6e491ad0bfa3f6b123d349ec58a8fb3952fa06f4806e306c71'),
 'test_single_file_wait.py': (G+'test_single_file_wait.py',21605,'7ece9dfb6d7542f467ee08a50897a20b4fe068a0d66d85d3bf020f3433c70a94'),
}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read_sources(root):
    mapping = None
    docs, refs, paths = {}, [], {}
    for name, (relative, size, digest) in PINS.items():
        path = root / relative
        origin = 'actual_project_source'
        if not path.exists():
            if mapping is None:
                mp = root / MAP
                raw = mp.read_bytes()
                if sha(raw) != MAP_SHA:
                    raise ValueError('actual local archive mapping SHA differs')
                mapping = json.loads(raw)
                if mapping['status'] != 'PASS_SERVER_GPU_EVIDENCE_LOCAL_BYTE_VERIFICATION':
                    raise ValueError('unexpected actual archive mapping status: '+mapping['status'])
            rows = [x for x in mapping['files'] if x['path'] == relative]
            if len(rows) != 1 or rows[0]['bytes'] != size or rows[0]['sha256'] != digest:
                raise ValueError('source is absent or differs in actual archive mapping')
            # The original map contains host paths. Only its pinned flat basename
            # is used below the actual mapping's recorded_files directory.
            path = (root / MAP).parent / 'recorded_files' / PureWindowsPath(rows[0]['local_path']).name
            origin = 'actual_verified_archive_flat_mapping'
        if path.is_symlink() or not path.is_file():
            raise ValueError('source must be an actual regular file')
        data = path.read_bytes()
        if len(data) != size or sha(data) != digest:
            raise ValueError('frozen source byte pin differs: '+relative)
        docs[name] = data.decode('utf-8-sig')
        refs.append(dict(path=relative, bytes=size, sha256=digest, read_origin=origin))
        paths[name] = path
    return docs, refs, paths

def node(docs, file, name, owner=None):
    tree = ast.parse(docs[file])
    if owner is not None:
        tree = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == owner)
    found = [x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(found) != 1:
        raise ValueError('ambiguous or missing original function: '+file+'/'+name)
    return found[0]

def section(docs, file, n):
    return '\n'.join(docs[file].splitlines()[n.lineno-1:n.end_lineno])

def static_audit(docs):
    checks, locations = [], []
    def check(label, ok):
        if ok is not True:
            raise ValueError('frozen static contract failed: '+label)
        checks.append(label)
    selected = [
      ('p4_policy.py','issue_preview','P4Policy'),
      ('p4_bridge.py','preview_issue','NativeP4Bridge'),
      ('p4_bridge.py','record_single_file_deferral','NativeP4Bridge'),
      ('reactor.py','__init__','IoReactor'),
      ('reactor.py','_prefix_stage_decide','IoReactor'),
      ('reactor.py','_prefix_ready_read_decision','IoReactor'),
      ('reactor.py','_drain_ready_preload_fds','IoReactor'),
      ('reactor.py','_prefix_single_file_retry_key','IoReactor'),
      ('reactor.py','_prefix_single_file_wait_deadline','IoReactor'),
      ('reactor.py','_prefix_wait_single_file_retry','IoReactor'),
      ('reactor.py','_run','IoReactor'),
      ('reactor.py','_pump_once','IoReactor'),
      ('simple_stage_policy.py','decide','DispatchController'),
      ('dependencies.py','analyze',None),
      ('native_full_step_collector.py','current_single_file_step','FullStepCapture'),
    ]
    text = {}
    for file, name, owner in selected:
        n = node(docs,file,name,owner)
        text[name] = section(docs,file,n)
        locations.append(dict(path=PINS[file][0], function=(owner+'.' if owner else '')+name,
          line=n.lineno, end_line=n.end_lineno,
          function_ast_sha256=sha(ast.dump(n,include_attributes=False).encode())))
    p = text['issue_preview']
    check('policy_progress_precedes_single_file_cost', p.index('native_progress_override') < p.index('if self.single_file is not None'))
    check('policy_age_uses_original_work_created_time', 'now_ns - work.created_ns >= self.config.max_wait_ns' in p)
    check('policy_cost_is_total_upper_vs_a_only_total_budget', 'receipt.cost_upper_ns <= receipt.step_budget_ns' in p)
    check('policy_preconditions_precede_override', p.index('stale_work_generation') < p.index('native_progress_override'))
    check('interference_bridge_has_no_dispatch_controller', 'elif dispatch_controller is not None:' in text['__init__'] and 'dependency-only requires original fixed controller' in text['__init__'])
    check('additional_defer_only_without_native_progress', 'preview.action == "defer" and progress is None' in text['_prefix_stage_decide'])
    check('cost_defer_does_not_create_dispatch_permit', 'return _StageDispatch(_P4Deferred(),work_id)' in text['_prefix_stage_decide'])
    check('unsupported_or_failed_optional_control_uses_original_dispatch', 'return None  # unsupported I/J preserve U' in text['_prefix_stage_decide'])
    check('work_age_not_reset_on_retry', 'single_preload[1] if single_preload is not None' in text['_prefix_stage_decide'] and 'ready.open_start_ns' in text['_prefix_stage_decide'])
    check('defer_retains_original_ready_queue_and_releases_reserved_slot', 'dispatch.action == "defer"' in text['_drain_ready_preload_fds'] and 'self.staging_pool.release(slot.index)' in text['_drain_ready_preload_fds'])
    check('shadow_defer_preserves_native_issue', 'single_file_shadow_preserves_native_issue' in text['preview_issue'])
    check('expired_live_step_does_not_block_issue', 'single_file_step_ended_during_preview' in text['preview_issue'] and 'return None' in text['_prefix_stage_decide'])
    check('mandatory_waiter_becomes_support_progress', 'support = any(self.is_mandatory(job.future)' in text['_prefix_ready_read_decision'])
    check('retry_reuse_excludes_pending_native_work_or_waiters', 'self._active or self._inflight or self._pending_copies or self._copy_ready' in text['_prefix_single_file_retry_key'] and 'self._preload_waiters.get(ready.preload_hash)' in text['_prefix_single_file_retry_key'])
    check('wait_deadline_includes_original_max_wait_and_all_freshness_bounds', all(t in text['_prefix_single_file_wait_deadline'] for t in ['deadline = min(', 'ready.open_start_ns + bridge.policy.config.max_wait_ns', 'snapshot.monotonic_ns + age','snapshot.native_state.captured_ns + age','key[0][2] + age']))
    check('wait_releases_submit_lock_before_original_queue_wait', 'self._incoming.get(timeout=remaining / 1_000_000_000)' in text['_prefix_wait_single_file_retry'] and 'self._intake(item)' in text['_prefix_wait_single_file_retry'])
    check('wake_returns_to_original_pump', text['_run'].index('_prefix_wait_single_file_retry') < text['_run'].index('_pump_once'))
    check('original_pump_keeps_completion_then_schedule_then_submit_then_finish', all(t in text['_pump_once'] for t in ['_drain_cuda_copies','_poll_ring_completions','_schedule_work','_flush_copy_batch','submit_pending','_finish_jobs']))
    check('dispatch_controller_native_safety_precedes_performance_override', text['decide'].index('if active and native:') < text['decide'].index('elif effective_progress is None:'))
    check('release_requires_native_reusable_witness_even_after_parent_completion', 'awaiting_native_release_witness' in text['analyze'] and 'resource.native_reusable is True' in text['analyze'])
    check('live_scalar_read_never_calls_event_query_or_wait', not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('query','wait','synchronize') for n in ast.walk(node(docs,'native_full_step_collector.py','current_single_file_step','FullStepCapture'))))
    return checks, locations

def original_cpu_modules(docs, paths):
    package_name = '_frozen_progress_cpu_audit'
    if any(k.startswith(package_name) for k in sys.modules):
        raise ValueError('fresh original-source namespace required')
    package = ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    result = {}
    order = ['dependencies','dispatch_budget','dispatch_shadow','simple_stage_policy',
             'p4_types','p4_cost_table','p4_policy','p4_eta','p4_bridge']
    for suffix in order:
        module = ModuleType(package_name+'.'+suffix)
        module.__package__ = package_name
        module.__file__ = str(paths[suffix+'.py'])
        sys.modules[module.__name__] = module
        # Compile the full, byte-pinned original module without AST rewrites,
        # global/callable patches, receipt constructors, or framework imports.
        exec(compile(docs[suffix+'.py'],module.__file__,'exec'),module.__dict__)
        result[suffix] = module
    return result

def cpu_replay(mod):
    rows = []
    def check(name, ok, actual):
        if ok is not True:
            raise ValueError('original CPU replay failed: '+name)
        rows.append(dict(check=name, passed=True, actual=actual, origin='explicit_cpu_value_fixture'))
    b, t, s = mod['dispatch_budget'], mod['p4_types'], mod['dispatch_shadow']
    p, bridge, controller = mod['p4_policy'], mod['p4_bridge'], mod['simple_stage_policy']
    run = 'cpu-progress-dependency-audit'
    policy = p.P4Policy(run,t.P4Config('interference',1000,20))
    nb = bridge.NativeP4Bridge(policy); nb.bind()
    state = s.ShadowState(run,30,b.ZERO,917504,0,True)
    snap = t.SystemSnapshot(run,1,30,frozenset(('native_ready_work',)),native_state=state)
    young = t.WorkDescriptor(run,1,0,'preload:cpu-metadata','ssd_read',917504,0,25,False,minimum_unit_bytes=917504)
    # No ExactSingleFileReceipt is instantiated, loaded or forged in this audit.
    check('bridge_without_cost_receipt_keeps_original_native_fallback', (v:=nb.preview_issue(young,snap,now_ns=30)).action == 'native_fallback',asdict(v))
    for progress in ('mandatory','mandatory_support','continuation','shutdown'):
        v = nb.preview_issue(replace(young,progress=progress),snap,now_ns=30)
        check('original_bridge_'+progress, v.action=='issue' and v.progress_override,asdict(v))
    v = nb.preview_issue(replace(young,created_ns=5),snap,now_ns=30)
    check('original_bridge_original_work_age_override',v.action=='issue' and v.reason=='native_progress_override',asdict(v))
    v = nb.preview_issue(replace(young,progress='mandatory'),replace(snap,monotonic_ns=0),now_ns=2000)
    check('stale_snapshot_does_not_become_progress_permission',v.action=='native_fallback',asdict(v))
    config = controller.SimpleStageConfig('fixed',1000000,1,b.ZERO,b.ZERO,b.Amount(),b.Amount(),0,0,0,8,1000000,20)
    dc = controller.DispatchController(run,config); dc.bind()
    value = s.ShadowState(run,10,b.ZERO,917504,0,True)
    v = dc.decide('ssd_read',917504,value,now_ns=10,work_id=('cpu',1))
    check('original_controller_zero_performance_allowance_defers',v.action=='defer' and v.attempt is None,asdict(v))
    v = dc.decide('ssd_read',917504,replace(value,captured_ns=30),now_ns=30,work_id=('cpu',1))
    check('original_controller_age_bypasses_only_performance_allowance',v.action=='issue' and v.attempt.progress=='age',asdict(v))
    dc.accepted(v.attempt)
    check('mock_acceptance_consumes_original_allowance_once',dc.used[0]==b.Amount(1,917504) and dc.performance_override_ops==1,dict(used=asdict(dc.used[0]),overrides=dc.performance_override_ops))
    v = dc.decide('ssd_read',917504,replace(value,captured_ns=31,native_issue_safe=False),now_ns=31,work_id=('cpu',2),progress='mandatory')
    check('mandatory_cannot_override_native_capacity_or_dependency',v.action=='defer' and v.attempt is None,asdict(v))
    d = mod['dependencies']
    rid = d.ResourceId(run,'cpu-fixture',0,0,1)
    resource = d.Resource(rid,917504,0,frozenset((1,2)),'observed_blocking')
    parents = (d.Parent(run,1,1,1,0,True,False,0), d.Parent(run,2,1,1,0,True,False,0))
    v = d.analyze(resource,parents,run_id=run,current_generation=1)
    check('complete_parents_without_native_fence_do_not_release_bytes',v.reusable_now_bytes==0 and v.reason=='awaiting_native_release_witness',asdict(v))
    v = d.analyze(resource,parents[:1],run_id=run,current_generation=1)
    check('partial_parent_closure_does_not_release_bytes',v.reusable_now_bytes==0 and v.reason=='incomplete_parent_closure',asdict(v))
    v = d.analyze(resource,(replace(parents[0],failed=True,done_files=0,inflight_files=1,future_done=True),parents[1]),run_id=run,current_generation=1)
    check('early_failed_future_with_inflight_is_still_draining',v.reusable_now_bytes==0 and v.reason=='failed_draining',asdict(v))
    return rows

def audit(root):
    docs, refs, paths = read_sources(root)
    static, locations = static_audit(docs)
    before_modules = set(sys.modules)
    replay = cpu_replay(original_cpu_modules(docs,paths))
    imported = sorted(set(sys.modules)-before_modules)
    forbidden = [n for n in imported if n.split('.')[0] in ('torch','vllm','py_kvcache','numpy','yaml')]
    if forbidden:
        raise ValueError('framework import is outside this CPU audit')
    after_docs, after_refs, _ = read_sources(root)
    if docs != after_docs or refs != after_refs:
        raise ValueError('frozen source changed during CPU audit')
    existing = []
    for file in PINS:
        if file.startswith('test_'):
            methods = [n for n in ast.walk(ast.parse(docs[file])) if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')]
            relevant = [dict(test=n.name,line=n.lineno) for n in methods if any(x in n.name for x in ('progress','age','deadline','timeout','waiter','queue','end','mandatory','expiry','original_native'))]
            existing.append(dict(path=PINS[file][0],tests_read_only=relevant,executed_by_this_audit=False))
    return dict(schema='c5_frozen_progress_dependency_cpu_audit_v1',status='PASS_READONLY_SOURCE_AND_ORIGINAL_CPU_VALUE_REPLAY',
      origin='actual_frozen_source_sha_and_explicit_cpu_value_fixtures',source_refs=refs,source_sha_unchanged_before_after=True,
      static_checks=static,static_check_count=len(static),source_locations=locations,cpu_replay=replay,cpu_check_count=len(replay),
      existing_test_source_review=existing,forbidden_imports=forbidden,
      original_pure_modules_executed=[x for x in imported if x.startswith('_frozen_progress_cpu_audit')],
      binding_matrix=dict(single_file_interference=dict(p4_bridge='installed_only_when_explicitly_requested',dispatch_controller=None,
        dispatch_ledger_adapter=False,allowance_path='original_native_U',cost_layer='additional_defer_only'),
        dependency_only=dict(p4_bridge='requires_original_fixed_dispatch_controller',allowance_path='original_DispatchController',
        release_credit='requires_owner_published_generation_refs_protectors_and_native_reusable_witness')),
      findings=[
        'ordinary cost defer is conditional, not a permanent prohibition; original progress or immutable arrival age returns to native dispatch',
        'single-file cost gate and generic DispatchLedger are not a linked native permit path; interference requires controller None',
        'original queues own accepted work; a defer releases the temporary staging reserve and retains the original ready FD',
        'narrow optional wait excludes other native pending work and wakes on original Queue messages or finite earliest deadline',
        'native capacity, CUDA dependencies, physical completion and fair owner scheduling remain independent requirements',
        'CPU parent completion cannot establish real GPU allocator release or dependency-order speedup',
      ],
      limitations=[
        'no actual on/shadow GPU execution or live cost-receipt attachment was exercised',
        'the additional single-file defer branch is SHA/AST audited and existing tests reviewed, not replayed with a forged receipt',
        'CPU fixture acceptance is metadata accounting only; no native backend, FD, staging pool, cache engine or CUDA Event was executed',
        'there is no unconditional time-to-completion proof when native safety or backend completion stays unavailable',
        'three actual off jobs with bridge None do not validate this on bridge or an interference permit path',
        'no blocked owner-release witness exists in the current fixed off workload to prove dependency sorting benefit',
      ],
      next_minimal_cpu_validation='review actual existing owner-published dependency interfaces and source-extracted failure/wakeup tests without changing cost bounds, budgets, candidate set or prior results',
      GPU_operations=0,model_execution=False,receipt_constructed=False,authority_issued=False,
      native_execution_qualified=False,on_liveness_gpu_verified=False,P4_strategy_effect_verified=False,performance_benefit_proved=False,
      ordinary_cost_upper_ns=16238752,ordinary_a_only_budget_ns=13171328,ordinary_candidate_admitted=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    report = audit(args.project_root.resolve(strict=True))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as handle:
        handle.write((json.dumps(report,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=report['status'],source_count=len(report['source_refs']),
      static_checks=report['static_check_count'],cpu_checks=report['cpu_check_count'],
      GPU_operations=0,receipt_constructed=False,native_execution_qualified=False,output=str(args.output)),sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
