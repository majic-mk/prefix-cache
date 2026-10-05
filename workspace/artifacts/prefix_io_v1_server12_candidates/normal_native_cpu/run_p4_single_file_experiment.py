"""One original guarded process and one finite off/shadow/on P4 request.

The parent owns permissions, whole-source before/after verification and the
original cumulative GPU guard. This child checks the actual active reservation.
Only original LLM/add_request/step/native preload and shutdown perform work.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types


# This helper stays byte-identical; authority belongs to the new controller.
def notification_adapter():
    name = "_server12_c5_cpu_notification_adapter"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("notification_runtime_adapter.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
DELIVERY = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
CANDIDATE = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate'
OVERLAY = CANDIDATE + '/source'
SCRIPT = DELIVERY + '/run_p4_single_file_experiment.py'
COMMON = 'artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/run_g2_normal_model_lifecycle.py'
SITE_SDK = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004/site_sdk_binding.py'
RUNTIME = 'artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002/g3_calibration_runtime_metrics_v2.py'
ACQUIRE = 'experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py'
REACTOR = OVERLAY + '/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
JOURNAL = OVERLAY + '/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_native_window_journal.py'
LEGACY_COLLECTOR_REACTOR_KEY = 'third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'
GPU_UUID = None
MODES = ('off', 'shadow', 'on')
SAMPLE_MAX_AGE_NS = 100000000
MAX_WAIT_NS = 100000000
MAX_ACCEPTED_PARENTS = 8
PURPOSE = 'FINITE_NORMAL_NOTIFICATION_LIFECYCLE_DIAGNOSTIC_ONLY'
STORAGE = 'experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
ORDER = (('B',),)
PROMPT_TOKENS = 129
OPERATION_COUNT = 1
PROMPT_FIRST = (28100,)
SEEDS = (2829,)
FILE_BYTES = 917504
COST_SOURCE = DELIVERY + '/observation_cpu_cost.py'
CANONICAL = DELIVERY + '/p4_single_file_receipt.py'
CANONICAL_MODULE = 'prefix_io_control.p4_single_file_receipt'
CONTROL_PACKAGE = OVERLAY + '/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/__init__.py'


def require(ok, message):
    if not ok: raise ValueError(message)


def safe(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and '\\' not in relative and all(s not in ('','.','..') for s in relative.split('/')),
            'safe relative artifact path')
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), 'artifact/source symlink refused')
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), 'path outside project')
    return path


def read_json(path):
    require(path.is_file() and path.stat().st_size <= 8 * 1024**2, 'bounded JSON file')
    def unique(items):
        value = {}
        for key, item in items:
            require(key not in value, 'duplicate JSON key')
            value[key] = item
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def write_json(path, document):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, allow_nan=False); stream.write('\n')


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'regular pinned file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): digest.update(chunk)
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(root, refs, relative):
    require(relative in refs and file_ref(root, relative) == refs[relative], 'pinned source drift: '+relative)
    return safe(root, relative)


def load_source(root, refs, relative, name):
    path = checked(root, refs, relative)
    require(path.suffix == '.py' and path.stat().st_size <= 4*1024**2, 'bounded source module')
    require(name not in sys.modules, 'fresh private runtime module')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try: exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
    except BaseException: sys.modules.pop(name, None); raise
    return module


def config_relative(mode):
    require(mode in MODES, 'one explicit finite runtime arm')
    return DELIVERY+'/CONFIG_'+mode+'.json'


def register_canonical_receipt(root, refs):
    """Load one source-verified public class before unchanged policy imports.

    This does not construct a receipt or authorize model execution. The new
    public loader replays the real calibration using explicit history binding.
    """
    require(not any(name == 'prefix_io_control' or name.startswith('prefix_io_control.')
                    for name in sys.modules), 'fresh native control package before canonical registration')
    checked(root, refs, CONTROL_PACKAGE); checked(root, refs, CANONICAL)
    package = load_source(root, refs, CONTROL_PACKAGE, 'prefix_io_control')
    try:
        canonical = load_source(root, refs, CANONICAL, CANONICAL_MODULE)
        require(canonical.__name__ == CANONICAL_MODULE
                and type(canonical.ExactSingleFileReceipt) is type
                and canonical.ExactSingleFileReceipt.__module__ == CANONICAL_MODULE
                and callable(canonical.load_verified_single_file), 'one exact public canonical receipt class')
        require(Path(canonical.__file__).resolve() == safe(root, CANONICAL).resolve(),
                'public canonical class must use the new locked normal source')
        checked(root, refs, CONTROL_PACKAGE); checked(root, refs, CANONICAL)
        package.p4_single_file_receipt = canonical
        return canonical
    except BaseException:
        sys.modules.pop(CANONICAL_MODULE, None); sys.modules.pop('prefix_io_control', None)
        raise


def loaded_native_source(root, refs, name, module):
    """Only the exact public receipt alias may leave the unchanged G overlay."""
    require(type(name) is str and (name in ('py_kvcache', 'prefix_io_control')
            or name.startswith(('py_kvcache.', 'prefix_io_control.'))), 'actual native module name')
    require(type(getattr(module, '__file__', None)) is str, 'actual native module source file')
    require(getattr(module, '__name__', None) == name, 'actual imported native module alias')
    source_path = Path(module.__file__)
    path = source_path.resolve()
    if name == CANONICAL_MODULE:
        require(source_path.absolute() == safe(root, CANONICAL).absolute()
                and path == safe(root, CANONICAL).resolve()
                and type(module.ExactSingleFileReceipt) is type
                and module.ExactSingleFileReceipt.__module__ == CANONICAL_MODULE,
                'only exact source-bound public canonical alias is permitted')
        relative = CANONICAL
    else:
        require(source_path.is_absolute() and source_path.is_relative_to(safe(root, OVERLAY))
                and path.is_relative_to(safe(root, OVERLAY)), 'native package escaped common overlay')
        relative = source_path.relative_to(root).as_posix()
    checked(root, refs, relative)
    return dict(module=name, source_ref=refs[relative])


def native_source_binding(root, refs, reactor, runtime, *, bridge, mode):
    """Tie the actual native owner and loaded modules to the common C4 bytes."""
    require(mode in MODES, 'known actual native arm')
    module = sys.modules[type(reactor).__module__]
    require(type(reactor).__module__ == 'py_kvcache.reactor' and
        Path(module.__file__).resolve() == safe(root, REACTOR).resolve(),
        'actual common reactor source path')
    method = runtime.original_method(reactor, '_run', root, refs)
    require(Path(method.__code__.co_filename).resolve() == safe(root, REACTOR).resolve(),
        'actual original _run must come from C4 source')
    require(type(reactor._max_accepted_parents) is int and
        reactor._max_accepted_parents == MAX_ACCEPTED_PARENTS and
        reactor._prefix_p4_bridge is bridge and ((bridge is None) is (mode == 'off')) and
        (bridge is None or bridge.mode == 'interference'),
        'same bounded common owner and actual requested bridge')
    loaded = []
    for name, value in sorted(sys.modules.items()):
        if name in ('py_kvcache', 'prefix_io_control') or name.startswith(('py_kvcache.', 'prefix_io_control.')):
            loaded.append(loaded_native_source(root, refs, name, value))
    require(1 <= len(loaded) <= 128 and
        any(row['module'] == 'py_kvcache.reactor' and row['source_ref'] == refs[REACTOR] for row in loaded)
        and any(row['module'] == CANONICAL_MODULE and row['source_ref'] == refs[CANONICAL] for row in loaded),
        'bounded actual reactor module list')
    return dict(actual_source_ref=refs[REACTOR], original_run_source_ref=refs[REACTOR],
        original_run_code_verified=True, max_accepted_parents=MAX_ACCEPTED_PARENTS,
        requested_mode=mode, bridge_is_none=(bridge is None), same_requested_bridge=True,
        bridge_policy_mode=None if bridge is None else bridge.mode, loaded_native_modules=loaded)


def collector_source_view(refs):
    """Adapt one old SHA lookup key without rewriting the frozen source map."""
    row=refs[REACTOR]
    require(row['path']==REACTOR,'actual collector native source reference')
    result=dict(refs); result[LEGACY_COLLECTOR_REACTOR_KEY]=row
    return result


def previous_gate(root, config, refs):
    return controller_api().verify_previous_qualification(root,config,refs)


def controller_api():
    name='_server12_normal_mode_controller'
    if name not in sys.modules:
        path=Path(__file__).with_name('control_p4_single_file.py')
        require(path.is_file() and not path.is_symlink(),'new normal authority controller')
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
    return sys.modules[name]


def load_configuration(path):
    config,refs,_authority=controller_api().verify_configuration(ROOT,path)
    require(config['overlay_relative']==OVERLAY and config['purpose']==PURPOSE,
            'actual G common overlay and finite normal purpose')
    for relative in (SCRIPT,COST_SOURCE,DELIVERY+'/notification_runtime_adapter.py',
                     config['collector_relative'],config['runtime_binding_relative'],REACTOR,JOURNAL,
                     CANONICAL,CONTROL_PACKAGE,SITE_SDK):
        checked(ROOT,refs,relative)
    previous_gate(ROOT,config,refs)
    return config,refs


def verify_guard(config):
    actual,_,authority=controller_api().verify_configuration(ROOT,config_relative(config['mode']))
    require(actual==config,'active normal configuration unchanged')
    return controller_api().verify_active_guard(ROOT,config,authority)


def input_groups(root, config):
    document = read_json(safe(root, config['input_manifest']))
    groups = document['groups']; require(type(groups) is list and len(groups)==3, 'three fixed native preload groups')
    hashes = set(); paths = set()
    for index, group in enumerate(groups):
        require(group['pair_index']==index and type(group['pair_index']) is int and len(group['files'])==8,
                'fixed pair/group/file count')
        for file_index,row in enumerate(group['files']):
            require(set(row)=={'path','bytes','sha256','block_hash','producer_file_index'}
                    and type(row['producer_file_index']) is int and row['producer_file_index']==file_index and row['bytes']==FILE_BYTES
                    and type(row['bytes']) is int, 'exact original native payload ref')
            key = row['block_hash']
            require(type(key) is str and len(key)==64 and all(c in '0123456789abcdef' for c in key)
                    and key not in hashes and row['path'] not in paths, '24 distinct original native hashes/files')
            hashes.add(key); paths.add(row['path'])
            rel = config['storage']+'/'+row['path']
            require(file_ref(root, rel) == dict(path=rel,bytes=row['bytes'],sha256=row['sha256']), 'private native input changed')
    return document


def prompt(index):
    values = list(range(1000,1000+PROMPT_TOKENS)); values[0] = PROMPT_FIRST[index]; return values


def config_for_engine(base, storage):
    extra = dict(spec_name='PyKvCacheOffloadingSpec', spec_module_path='py_kvcache.vllm',
        shared_storage_path=str(storage), block_size=16, sync_on_store=False,
        staging_mem=0.125, iodepth=4, enable_preload=True, preload_lookahead_requests=1,
        preload_share_staging=True, staging_cache='lru', load_planner='off', io_backend='linux_aio')
    return dict(base.ENGINE, async_scheduling=False, max_model_len=1040, max_num_batched_tokens=1040,
        max_num_seqs=1, kv_cache_memory_bytes=268435456, disable_log_stats=True,
        prefix_caching_hash_algo='sha256', kv_transfer_config=dict(kv_connector='OffloadingConnector',
        kv_role='kv_both', kv_connector_extra_config=extra))


class SparseOwnerSink:
    """Keep every native stage event; omit redundant per-pump owner frames."""
    def __init__(self, journal): self.journal = journal
    def __call__(self, reactor):
        journal = self.journal
        if not journal.frames or journal.frames[-1].sequence != journal.event_sequence:
            return journal.publish_owner_frame()
        return None


def install_owner_snapshot_observation(reactor, hashes, journal):
    """Read cache membership only inside the existing native owner callback."""
    require('_capture_owner_snapshot' not in vars(reactor),'original owner snapshot not overridden')
    original=reactor._capture_owner_snapshot
    def snapshot(*args,**kwargs):
        result=original(*args,**kwargs)
        require(result.get('owner_capture') is True or result.get('native_shutdown_read') is True,
                'original owner or joined shutdown boundary')
        rows=[]
        for key in hashes:
            raw=bytes.fromhex(key); cache=reactor._staging_cache
            rows.append(dict(block_hash=key,shared_cached=raw in reactor._shared_cached,
                preload_slots=bool(reactor._preload_slots.get(raw)),
                staging_cached=cache is not None and raw in cache,
                pending=reactor._preload_pending_count.get(raw,0),
                inflight=reactor._preload_inflight_hashes.get(raw,0),
                retained_refs=reactor._preload_refcount.get(raw,0)))
        result['diagnostic_input_membership']=rows
        if result.get('owner_capture') is True:
            # Existing inspect_snapshot dispatch already runs in the owner.
            # Exactly one extra immutable boundary frame per explicit outside-
            # window inspection; no main-thread accounting access or new wait.
            frame=journal.publish_owner_frame()
            require(frame is not None,'original owner published diagnostic boundary frame')
            result['diagnostic_owner_boundary']=dataclasses.asdict(frame)
        return result
    reactor._capture_owner_snapshot=snapshot
    return original,snapshot


def require_cold_inputs(snapshot,hashes):
    rows={row['block_hash']:row for row in snapshot['diagnostic_input_membership']}
    for key in hashes:
        row=rows[key]
        require(all(row[field] is False for field in ('shared_cached','preload_slots','staging_cached'))
                and all(type(row[field]) is int and row[field]==0 for field in ('pending','inflight','retained_refs')),
                'selected native preload hash is not cold')
    return [rows[key] for key in hashes]


def native_handler(worker):
    from vllm.distributed.kv_transfer.kv_transfer_state import get_kv_transfer_group
    connector = get_kv_transfer_group(); cw = connector.connector_worker
    registry = cw.worker
    require(type(registry.handlers) is set and len(registry.handlers)==1, 'sole original registered native handler')
    handler = next(iter(registry.handlers))
    mapping = registry.transfer_type_to_handler
    require(set(mapping)=={('GPU','SHARED_STORAGE'),('SHARED_STORAGE','GPU')}
            and all(value is handler for value in mapping.values()), 'original native bidirectional registry')
    return handler


def zero_snapshot(snapshot):
    require(snapshot.get('owner_capture') is True, 'original owner-captured snapshot')
    native = snapshot['native']; aio = snapshot['aio']; stage = snapshot['stage_accounting']
    require(type(native) is dict and all(type(value) is int and value==0 for value in native.values()), 'native work not drained')
    require(all(type(aio.get(key)) is int and aio[key]==0 for key in ('outstanding','pending','ready','unreaped')), 'AIO work not drained')
    require(stage['valid'] is True and all(row['inflight_ops']==0 and row['inflight_bytes']==0
            and row['failed_ops']==0 for row in stage['stages'].values()), 'original stage accounting not drained')
    return snapshot


def original_drain(llm, acquire, worker, journal):
    # Original handler Future/result/wait are confined to between-window drains.
    value = llm.collective_rpc(acquire.worker_control, timeout=45, args=('drain',None))
    require(type(value) is list and len(value)==1, 'one original drain result')
    coordinator = native_handler(worker).coordinator
    deadline = time.monotonic()+10
    while True:
        snapshot = coordinator.inspect_snapshot(timeout=2)
        try: zero_snapshot(snapshot); break
        except ValueError:
            require(time.monotonic()<deadline, 'bounded between-window native drain')
            time.sleep(0.005)
    events, frames, valid = journal.published()
    require(valid, 'native event journal loss or overflow')
    return dict(original_worker_control=value[0], owner_snapshot=snapshot,
                journal_event_count=len(events), journal_owner_frame_count=len(frames))


def frontend_capture(common, engine, sampling, request_id, background_hashes=None):
    """Call the frozen G2 request loop, recording its original step outputs."""
    original = engine.step; require('step' not in vars(engine), 'original engine step not overridden')
    times=[]; steps=[]; last=0; request_hash_rows=[]
    def step(*args, **kwargs):
        nonlocal last
        if background_hashes is not None:
            requests=engine.engine_core.engine_core.scheduler.requests
            selected=[request for request in requests.values()
                      if list(request.prompt_token_ids)==common.PROMPT]
            require(len(selected)==1,'actual sole foreground scheduler request')
            request=selected[0]
            keys=[bytes(key).hex() for key in request.block_hashes]
            require(len(keys)>=8 and all(len(key)==64 for key in keys),'actual foreground block hashes')
            require(not set(keys)&set(background_hashes),'foreground/background original block hashes intersect')
            request_hash_rows.append(dict(request_id=request.request_id,block_hashes=keys,
                prompt_token_ids=list(request.prompt_token_ids)))
        before=time.monotonic_ns(); outputs=original(*args,**kwargs); after=time.monotonic_ns()
        counts=[]
        for output in outputs:
            require(output.request_id==request_id and len(output.outputs)==1, 'actual frontend output cohort')
            count=len(output.outputs[0].token_ids)
            require(last<=count<=128, 'monotonic cumulative original frontend token count')
            for ordinal in range(last,count): times.append(dict(token_ordinal=ordinal,at_ns=after))
            last=count; counts.append(count)
        steps.append(dict(before_ns=before,after_ns=after,cumulative_output_counts=counts))
        require(len(steps)<=4096, 'bounded original frontend steps')
        return outputs
    engine.step=step
    started=time.monotonic_ns()
    try: output=common.run_original_request(engine,sampling,request_id,deadline_seconds=60)
    finally:
        require(vars(engine).get('step') is step, 'frontend step observation identity')
        del engine.step
    require(len(times)==128 and last==128, 'complete frontend token times')
    if background_hashes is not None:
        require(request_hash_rows and all(row['request_id']==output['native_request_id'] for row in request_hash_rows),
                'independent scheduler hash identities match original frontend native request')
    return dict(output=output, request_started_ns=started,request_completed_ns=time.monotonic_ns(),
        token_times=times,steps=steps,original_scheduler_hash_observations=request_hash_rows,
        time_scope='host receipt of original cumulative frontend outputs')


def prepare_window_storage(root, config, inputs, out):
    """Copy the frozen 24-file template; each original process owns new storage."""
    dest=out/'private-storage'; dest.mkdir(exist_ok=False)
    for group in inputs['groups']:
        for row in group['files']:
            source=safe(root,config['storage']+'/'+row['path'])
            target=dest/row['path']; target.parent.mkdir(parents=True,exist_ok=True)
            with source.open('rb') as inp,target.open('xb') as output:
                while True:
                    chunk=inp.read(1024**2)
                    if not chunk: break
                    output.write(chunk)
            relative=target.relative_to(root).as_posix()
            require(file_ref(root,relative)==dict(path=relative,bytes=row['bytes'],sha256=row['sha256']),
                    'fresh window storage exact bytes')
    return dest


def execute_window(root, config, refs, guard):
    require(Path(root).resolve(strict=True)==ROOT,'fixed server normal project')
    actual,actual_refs=load_configuration(safe(ROOT,config_relative(config['mode'])))
    require(actual==config and actual_refs==refs,'direct normal execution configuration/source unchanged')
    actual_guard=verify_guard(config)
    for key in ('id','session_id','process_group','label','gpu_uuid','permissions'):
        require(type(actual_guard.get(key)) is type(guard.get(key)) and actual_guard.get(key)==guard.get(key),
                'direct normal execution original reservation mismatch: '+key)
    LABEL=config['label']; mode=config['mode']; pair_index=0; condition='B'; window_index=0
    actual_uuid=config['gpu_uuid']
    out=safe(root,config['out'])
    require(out.parent.is_dir() and not out.exists(),'original guard created a fresh run root')
    out.mkdir(exist_ok=False)
    config_ref=file_ref(root,config_relative(mode))
    require(not (out/'p4-single-file-runtime-result.json').exists(), 'append-only experiment output')
    inputs=input_groups(root,config)
    storage=prepare_window_storage(root,config,inputs,out)
    common=load_source(root,refs,COMMON,'_server11_cost_common')
    base=common.load_ref(root,refs[common.SMOKE]); base.ROOT=root; base.AUTHOR_ROOT=root/common.AUTHOR
    runtime=load_source(root,refs,RUNTIME,'_server11_cost_lifecycle')
    collector=load_source(root,refs,config['collector_relative'],'_server11_cost_collector')
    cost_module=load_source(root,refs,COST_SOURCE,'_server11_normal_gross_cpu_meter')
    old_env=dict(os.environ); old_path=list(sys.path); old_cwd=Path.cwd()
    old_alias=sys.modules.get('native_gpu_prefix_smoke'); alias_present='native_gpu_prefix_smoke' in sys.modules
    llm=finder=probe=journal=worker=handler=active_capture=None
    notification_audit=None
    native_module=original_factory=factory=None; owner_snapshot_original=owner_snapshot_wrapper=None
    bridge=receipt=runtime_identity=None
    result=dict(status='FAILED_P4_SINGLE_FILE_RUNTIME_QUALIFICATION',purpose=PURPOSE,origin='native_gpu_recording',
        label=LABEL,gpu_uuid=actual_uuid,guard=guard,window_order=[list(pair) for pair in ORDER],
        windows=[],warmups=[],load_planner='off',policy='finite_single_file_original_preload',window_index=window_index,
        mode=mode,config_ref=config_ref,policy_receipt_binding_ref=refs[config['binding_relative']],
        p4_before_measurement=None,p4_after_measurement=None,p4_after_shutdown=None,
        policy_parameters=dict(sample_max_age_ns=SAMPLE_MAX_AGE_NS,max_wait_ns=MAX_WAIT_NS,
            max_accepted_parents=MAX_ACCEPTED_PARENTS),
        baseline_policy_changed=False,production_qualified=False,performance_claim=False,
        curves_exported=False,source_lock=config['source_lock'],actual_guarded_gpu_job_count=1,
        authority_ref=file_ref(root,config['authority_relative']),full_runtime_cost_qualified=False,
        isolated_observation_cost_measured=False,net_observation_cost_qualified=False,
        subprocess_pid=os.getpid(),subprocess_sid=os.getsid(0),fresh_original_process=True,
        private_storage=storage.relative_to(root).as_posix(),
        original_engine_shutdown_returned=False,cache_reset_count=0)
    try:
        sys.path[:0]=[str(root/'experiments/prefix_io_v1/scripts'),
            str(root/OVERLAY/'third_party/work/py-kvcache-p4-02-cpu'),str(root/OVERLAY/'third_party/work/prefix-io-p4-02-cpu/src'),str(root/CANDIDATE)]
        register_canonical_receipt(root, refs)
        sys.modules['native_gpu_prefix_smoke']=base
        acquire=common.load_ref(root,refs[ACQUIRE])
        result['runtime_cache_environment']=common.configure_process_caches(out,base)
        site_sdk=load_source(root,refs,SITE_SDK,'_normal_actual_site_sdk')
        sdk=site_sdk.prepare_site_sdk(root,refs,out); result['sdk_environment']=sdk
        result['ninja_environment']=common.prepare_process_ninja(root,refs,out,sdk)
        optional,absence=common.load_optional_probe(root,refs); result['optional_absence']=absence
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'; os.environ['PYTHONHASHSEED']='0'
        require(os.environ.get('HF_HUB_OFFLINE')=='1' and os.environ.get('TRANSFORMERS_OFFLINE')=='1', 'offline guard environment')
        model_dir,identity=base.validate_local_model(safe(root,common.MODEL),safe(root,common.MODEL_PLAN))
        loader=common.load_ref(root,refs[common.QUALIFY]); finder=loader.BoundAuthorFinder(root,refs)
        require(not any(n=='vllm' or n.startswith('vllm.') for n in sys.modules), 'fresh author import')
        sys.meta_path.insert(0,finder); os.chdir(out)
        import torch,vllm
        from vllm import LLM,SamplingParams
        import vllm.utils.import_utils as utilities
        probe=optional.install_capability_probe(utilities,finder,root,refs)
        require(loader.normalize_uuid(torch.cuda.get_device_properties(0).uuid)==loader.normalize_uuid(actual_uuid), 'actual hardware UUID')
        require(Path(vllm.__file__).resolve().is_relative_to(root/common.AUTHOR), 'actual author vLLM')
        checked(root,refs,JOURNAL)
        from prefix_io_control.p4_native_window_journal import NativeWindowJournal
        journal=NativeWindowJournal(LABEL,refs[REACTOR]['sha256'],enabled=True,max_events=4096)
        from prefix_io_control.p4_single_file_receipt import load_verified_single_file
        from prefix_io_control.p4_types import P4Config
        from prefix_io_control.p4_bridge import make_native_bridge
        receipt=load_verified_single_file(root,config['binding_relative'])
        require(dict(path=receipt.binding_ref.path,bytes=receipt.binding_ref.bytes,
                     sha256=receipt.binding_ref.sha256)==refs[config['binding_relative']], 'exact verified receipt binding')
        if mode!='off':
            bridge=make_native_bridge(LABEL,P4Config('interference',SAMPLE_MAX_AGE_NS,MAX_WAIT_NS,
                receipt.step_budget_ns),single_file=receipt)
        binding_module=load_source(root,refs,config['runtime_binding_relative'],'single_file_runtime_binding')
        sink=SparseOwnerSink(journal)
        native_module=importlib.import_module('py_kvcache.vllm')
        checked(root,refs,Path(native_module.__file__).resolve().relative_to(root).as_posix())
        original_factory=native_module.TransferCoordinator
        coordinators=[]
        def factory(**kwargs):
            require(not coordinators and all(key not in kwargs for key in ('stage_accounting','observation_sink','progress_run_id','max_accepted_parents','p4_bridge')),
                    'sole original coordinator with passive optional observation')
            value=original_factory(**kwargs,stage_accounting=journal,observation_sink=sink,
                progress_run_id=LABEL,max_accepted_parents=MAX_ACCEPTED_PARENTS,p4_bridge=bridge)
            coordinators.append(value); return value
        native_module.TransferCoordinator=factory
        alias=out/base.MODEL_ID; alias.parent.mkdir(parents=True,exist_ok=False); alias.symlink_to(model_dir,target_is_directory=True)
        engine_config=config_for_engine(base,storage)
        result['engine_config']=engine_config; result['model']=identity
        llm=LLM(model=base.MODEL_ID,**engine_config)
        require(llm.llm_engine.log_stats is False and llm.llm_engine.logger_manager is None,
                'optional Prometheus logger disabled equally for all diagnostic arms')
        result['optional_metrics_configuration']=dict(disable_log_stats=True,
            reason='Author in-process OffloadingOperationMetrics dataclass is incompatible with Prometheus dict-only observe; use supported optional logging switch',
            common_to_all_conditions=True,author_files_modified=False,
            native_stage_journal_preserved=True,frontend_tokens_and_full_step_capture_preserved=True,
            performance_benefit_claim=False)
        require(native_module.TransferCoordinator is factory, 'coordinator construction observer identity')
        native_module.TransferCoordinator=original_factory
        require(len(coordinators)==1, 'exactly one original native coordinator')
        workers=[]
        def capture_worker(owner): workers.append(owner); return {'captured':True}
        value=llm.collective_rpc(capture_worker,timeout=10)
        require(type(value) is list and len(value)==1 and len(workers)==1, 'actual in-process Uni worker')
        worker=workers[0]; handler=native_handler(worker)
        require(handler.coordinator is coordinators[0], 'injected original owner is registered handler')
        require((handler.coordinator.reactor._prefix_p4_bridge is bridge) and
            handler.coordinator.progress_bridge_enabled is True,'exact same common native admission and optional P4 binding')
        result['native_source_binding']=native_source_binding(root,refs,handler.coordinator.reactor,runtime,
            bridge=bridge,mode=mode)
        runtime_identity=binding_module.verify_actual_single_file_runtime(worker,handler,receipt,root,refs,
            common=common,base=base)
        result['verified_runtime_identity']=dict(signature_prefix=list(runtime_identity.signature_prefix),
            source_refs=[dict(row) for row in runtime_identity.source_refs],model_path=runtime_identity.model_path,
            geometry=dict(runtime_identity.geometry),same_original_runner=runtime_identity.matches_runner(worker.model_runner))
        for method in ('preload_async','shutdown'): runtime.original_method(handler,method,root,refs)
        runtime.original_method(handler.coordinator,'inspect_snapshot',root,refs)
        require(llm.llm_engine.vllm_config.scheduler_config.async_scheduling is False, 'original synchronous scheduler')
        spec_class=native_module.SharedStorageLoadStoreSpec
        require(handler.coordinator.reactor.file_store.io_size==FILE_BYTES, 'actual physical block quantum')
        mapper=handler.coordinator.reactor.file_mapper
        mapped=[]
        for group in inputs['groups']:
            for row in group['files']:
                actual=Path(mapper.get_file_name(bytes.fromhex(row['block_hash'])))
                expected=storage/row['path']
                require(actual.absolute()==expected,'actual original FileMapper differs from published input')
                mapped.append(dict(block_hash=row['block_hash'],path=str(actual)))
        result['actual_native_file_mapping']=mapped
        owner_snapshot_original,owner_snapshot_wrapper=install_owner_snapshot_observation(handler.coordinator.reactor,
            [row['block_hash'] for group in inputs['groups'] for row in group['files']],journal)
        for index,pair in ((pair_index,(condition,)),):
            common.PROMPT=prompt(index)
            sampling=dict(common.SAMPLING,seed=SEEDS[index])
            primer=llm.generate([{'prompt_token_ids':common.PROMPT.copy()}],
                SamplingParams(**dict(sampling,max_tokens=1,min_tokens=1)),use_tqdm=False)
            result['primer_observations']=[dict(request_id=item.request_id,
                num_cached_tokens=item.num_cached_tokens,prompt_token_ids=list(item.prompt_token_ids),
                output_token_ids=[list(completion.token_ids) for completion in item.outputs]) for item in primer]
            write_json(out/('primer-%d-raw-output.json'%index),result['primer_observations'])
            require(len(primer)==1 and len(primer[0].outputs[0].token_ids)==1,'original prompt primer')
            primer_row=dict(prompt_token_ids=common.PROMPT.copy(),output_token_ids=list(primer[0].outputs[0].token_ids),
                request_id=primer[0].request_id,num_cached_tokens=primer[0].num_cached_tokens,
                drain=original_drain(llm,acquire,worker,journal))
            warm_sampling=dict(sampling)
            warm=llm.generate([{'prompt_token_ids':common.PROMPT.copy()}],SamplingParams(**warm_sampling),use_tqdm=False)
            result['hot_warmup_observations']=[dict(request_id=item.request_id,
                num_cached_tokens=item.num_cached_tokens,prompt_token_ids=list(item.prompt_token_ids),
                output_token_ids=[list(completion.token_ids) for completion in item.outputs]) for item in warm]
            write_json(out/('hot-warmup-%d-raw-output.json'%index),result['hot_warmup_observations'])
            require(len(warm)==1 and len(warm[0].outputs[0].token_ids)==128, 'original full-output warmup publishes all generated KV')
            require(warm[0].num_cached_tokens==128,'original hot warmup cached_tokens expected128 actual='+str(warm[0].num_cached_tokens))
            warmup=dict(pair_index=index,prompt_token_ids=common.PROMPT.copy(),seed=SEEDS[index],
                output_token_ids=list(warm[0].outputs[0].token_ids),request_id=warm[0].request_id,
                num_cached_tokens=warm[0].num_cached_tokens,primer=primer_row,
                drain=original_drain(llm,acquire,worker,journal))
            # As in the pinned G3 acquisition, one original engine request
            # delivers previous store completion metadata before observation.
            flush=llm.generate([{'prompt_token_ids':[32000+index]}],SamplingParams(**dict(sampling,max_tokens=1,min_tokens=1)),use_tqdm=False)
            require(len(flush)==1 and len(flush[0].outputs[0].token_ids)==1,'original one-token completion flush')
            warmup['flush']=dict(prompt_token_ids=[32000+index],output_token_ids=list(flush[0].outputs[0].token_ids),
                                request_id=flush[0].request_id,drain=original_drain(llm,acquire,worker,journal))
            result['warmups'].append(warmup); write_json(out/('warmup-%d.json'%index),warmup)
            group=inputs['groups'][index]; hashes=[row['block_hash'] for row in group['files'][:OPERATION_COUNT]]
            source_spec=spec_class([bytes.fromhex(key) for key in hashes])
            pair_outputs=[]
            for condition in pair:
                rid='%s-p%d-%s'%(LABEL,index,condition)
                before=original_drain(llm,acquire,worker,journal)
                cold_membership=require_cold_inputs(before['owner_snapshot'],hashes)
                result['p4_before_measurement']=before['owner_snapshot']['p4']
                def action(offset,ordinal):
                    accepted=handler.preload_async(rid+'-preload',source_spec,profile_tid='native_cost_preload',req_id=rid)
                    require(type(accepted) is bool and accepted, 'original preload accepted once')
                    return dict(accepted=True,hashes=hashes.copy(),preload_id=rid+'-preload',
                                physical_bytes=OPERATION_COUNT*FILE_BYTES,stage='ssd_read',h2d_requested=False)
                result['collector_native_source_binding']=dict(legacy_lookup_key=LEGACY_COLLECTOR_REACTOR_KEY,
                    actual_source_ref=refs[REACTOR])
                cost_sources=dict(runner=refs[SCRIPT],cost_meter=refs[COST_SOURCE],
                    collector=refs[config['collector_relative']],reactor=refs[REACTOR],
                    notification_adapter=refs[DELIVERY+'/notification_runtime_adapter.py'])
                gross_cpu=cost_module.CompletePathCPU(mode=mode,run_id=LABEL,request_id=rid,source_refs=cost_sources)
                gross_cpu.start(handler.coordinator.reactor)
                active_capture=collector.install(worker,common=common,root=root,refs=collector_source_view(refs),run_id=LABEL,
                    event_class=torch.cuda.Event,selected_offsets=(16,),action=action if condition=='B' else None)
                scalar_adapter=active_capture.observer.scalar._adapter
                require(scalar_adapter.native_source_sha256==refs[REACTOR]['sha256'],
                    'actual scalar adapter bound to overlay reactor before all frames')
                result['collector_native_source_binding']['adapter_native_source_sha256_before']=scalar_adapter.native_source_sha256
                if bridge is not None:
                    bridge.attach_single_file_capture(active_capture,runtime_identity=runtime_identity,
                        runtime_refs=refs,shadow=(mode=='shadow'))
                from py_kvcache.reactor import _SingleFileWake
                notification_audit=notification_adapter().prepare_notification_capture(
                    mode,handler.coordinator.reactor,bridge,active_capture,run_id=LABEL,request_id=rid,
                    reactor_source=root/REACTOR,collector_source=root/config['collector_relative'],
                    wake_type=_SingleFileWake)
                request_started_ns=time.monotonic_ns()
                frontend=frontend_capture(common,llm.llm_engine,SamplingParams(**sampling),rid,hashes)
                request_finished_ns=time.monotonic_ns()
                capture=active_capture.export()
                require(active_capture.observer.scalar._adapter is scalar_adapter and
                    scalar_adapter.native_source_sha256==refs[REACTOR]['sha256'] and
                    len(scalar_adapter.frames)==128 and capture['frames']==[
                        dataclasses.asdict(frame) for frame in scalar_adapter.frames],
                    'all 128 actual frames belong to the same correctly source-bound scalar adapter')
                result['collector_native_source_binding'].update(
                    adapter_native_source_sha256_after=scalar_adapter.native_source_sha256,
                    same_adapter_all_frames=True,full_frame_count=128)
                active_capture.detach()
                write_json(out/(rid+'-frontend.json'),frontend)
                write_json(out/(rid+'-capture.json'),capture)
                drain_started_ns=time.monotonic_ns()
                after=original_drain(llm,acquire,worker,journal)
                drain_finished_ns=time.monotonic_ns()
                notification_audit.close()
                notification_evidence=notification_audit.export()
                notification_adapter().validate_notification_evidence(notification_evidence,
                    mode=mode,run_id=LABEL,request_id=rid,require_wait=False)
                gross_cpu.stop(full_frame_count=len(capture['frames']),action_count=len(capture['actions']),
                    notification_evidence=notification_evidence)
                gross_cpu_evidence=gross_cpu.export()
                active_capture=None; notification_audit=None
                result['p4_after_measurement']=after['owner_snapshot']['p4']
                result['request_and_drain_timing']=dict(request_started_ns=request_started_ns,
                    request_finished_ns=request_finished_ns,drain_started_ns=drain_started_ns,
                    drain_finished_ns=drain_finished_ns)
                row=dict(pair_index=index,condition=condition,split='new_runtime_qualification',mode=mode,
                    request_id=rid,prompt_token_ids=common.PROMPT.copy(),seed=SEEDS[index],
                    prefix_family='first_token_%d'%PROMPT_FIRST[index],
                    workload_sha256=hashlib.sha256(json.dumps(dict(prompt=common.PROMPT,seed=SEEDS[index]),sort_keys=True).encode()).hexdigest(),
                    frontend=frontend,capture=capture,notification_evidence=notification_evidence,
                    observation_cpu_cost=gross_cpu_evidence,
                    native_before=before,native_after=after,
                    cold_input_membership=cold_membership,actual_native_file_mapping=mapped,
                    preload_source_hashes=hashes.copy() if condition=='B' else [])
                row['independent_payload']=(dict(ssd_only=True,cache_miss_before_submit=True,
                    physical_bytes=OPERATION_COUNT*FILE_BYTES,operations=OPERATION_COUNT,preload_key_sha256=hashes.copy(),
                    request_prefix_key_sha256=frontend['original_scheduler_hash_observations'][0]['block_hashes'][:8])
                    if condition=='B' else None)
                result['windows'].append(row); write_json(out/(rid+'-window.json'),row)
                require(frontend['output']['num_cached_tokens']==128, 'original GPU-hot foreground cached_tokens expected128 actual='+str(frontend['output']['num_cached_tokens']))
                require(frontend['output']['output_token_ids']==warmup['output_token_ids'],
                        'measured full output differs from published original warmup cache')
                require(capture['valid'] is True, 'full original execution capture is invalid')
                pair_outputs.append(frontend['output']['output_token_ids'])
            require(len(pair_outputs)==1, 'one complete measured request in this fresh process')
        result['input_template_unchanged']=(input_groups(root,config)==inputs)
        result['status']='PASS_P4_SINGLE_FILE_RAW_REQUIRES_VALIDATION'
    except Exception as exc:
        result.update(error_type=type(exc).__name__,error_message=str(exc),error_traceback=traceback.format_exc(limit=24))
    finally:
        if active_capture is not None:
            try: active_capture.detach()
            except Exception as exc: result['capture_cleanup_error']=str(exc)
        if native_module is not None and factory is not None and native_module.TransferCoordinator is factory:
            native_module.TransferCoordinator=original_factory
        if llm is not None:
            try:
                if worker is not None and journal is not None:
                    result['final_before_shutdown']=original_drain(llm,acquire,worker,journal)
            except Exception as exc: result['final_drain_error']=str(exc); result['status']='FAILED_FINAL_DRAIN'
            finally:
                try:
                    llm.llm_engine.engine_core.shutdown(timeout=15)
                    result['original_engine_shutdown_returned']=True
                    if handler is not None:
                        result['native_post_shutdown']=runtime.post_shutdown_snapshot(handler)
                        result['p4_after_shutdown']=result['native_post_shutdown']['snapshot']['p4']
                        reactor=handler.coordinator.reactor
                        result['native_tail_assertions']=dict(handler_shutdown=handler.is_shutdown,
                            worker_alive=reactor._worker.is_alive(),aio_worker_alive=reactor.ring._worker.is_alive(),
                            reactor_closed=reactor._closed)
                except Exception as exc: result['shutdown_error']=str(exc); result['status']='FAILED_ORIGINAL_SHUTDOWN'
        if notification_audit is not None:
            try:
                notification_audit.close()
                result['notification_failure_evidence']=notification_audit.export()
            except Exception as exc: result['notification_cleanup_error']=str(exc)
        if journal is not None:
            events,frames,valid=journal.published()
            result['native_journal']=dict(run_id=LABEL,source_sha256=refs[REACTOR]['sha256'],valid=valid,
                events=[dataclasses.asdict(event) for event in events],frames=[dataclasses.asdict(frame) for frame in frames],
                lost=journal.lost,last_reason=journal.last_reason,owner_sink='deduplicated unchanged pump frames; every native stage event retained')
            write_json(out/'native-journal.json',result['native_journal'])
        if handler is not None and owner_snapshot_wrapper is not None:
            reactor=handler.coordinator.reactor
            if vars(reactor).get('_capture_owner_snapshot') is owner_snapshot_wrapper:
                del reactor._capture_owner_snapshot
            else: result['status']='FAILED_OWNER_OBSERVATION_RESTORE'
        if probe is not None:
            try:
                result['optional_probe_restored']=probe.detach()
                require(result['optional_probe_restored'] is True,'optional probe restoration')
            except Exception as exc: result['probe_cleanup_error']=str(exc); result['status']='FAILED_OPTIONAL_RESTORE'
        if finder is not None and finder in sys.meta_path: sys.meta_path.remove(finder)
        if alias_present: sys.modules['native_gpu_prefix_smoke']=old_alias
        else: sys.modules.pop('native_gpu_prefix_smoke',None)
        sys.path[:]=old_path; os.chdir(old_cwd); os.environ.clear(); os.environ.update(old_env)
        if file_ref(root,config_relative(mode))!=config_ref:
            result['status']='FAILED_IMMUTABLE_ARM_CONFIG'
        write_json(out/'p4-single-file-runtime-result.json',result)
    return result



def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args(argv)
    require(args.execute,'only an explicit independently guarded single arm is supported')
    config,refs=load_configuration(args.config); guard=verify_guard(config)
    result=execute_window(ROOT,config,refs,guard)
    print(json.dumps(dict(status=result['status'],mode=config['mode'],completed_windows=len(result['windows']),
        original_shutdown_returned=result['original_engine_shutdown_returned'],performance_claim=False)))
    return 0 if result['status']=='PASS_P4_SINGLE_FILE_RAW_REQUIRES_VALIDATION' else 1


if __name__=='__main__': raise SystemExit(main())
