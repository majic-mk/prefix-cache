"""CPU-only original-loop and process-isolation boundary regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_cost_harness',HERE/'run_native_cost_experiment.py')
M=importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def common_module():
    import os
    source=os.environ.get('SERVER11_G2_COMMON_SOURCE')
    if source is None:
        source=HERE.parents[1]/'prefix_io_v1_server09_candidates/g2_normal_worker_site_cache_v4_final/run_g2_normal_model_lifecycle.py'
    spec=importlib.util.spec_from_file_location('original_g2_test',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class Engine:
    def __init__(self,common,fail=False): self.common,self.fail,self.count,self.active=common,fail,0,False
    def has_unfinished_requests(self): return self.active
    def add_request(self,rid,prompt,sampling): self.rid=rid;self.active=True;return rid+'-12345678'
    def step(self):
        if self.fail: raise RuntimeError('real original step failed')
        self.count+=1;self.active=self.count<128
        return [types.SimpleNamespace(request_id=self.rid,finished=not self.active,
            prompt_token_ids=self.common.PROMPT.copy(),num_cached_tokens=112,
            outputs=[types.SimpleNamespace(token_ids=list(range(self.count)),finish_reason='length')])]


class HarnessTests(unittest.TestCase):
    def test_actual_frozen_g2_request_loop_complete_frontend_timing(self):
        common=common_module();engine=Engine(common)
        result=M.frontend_capture(common,engine,None,'request')
        self.assertEqual(len(result['token_times']),128)
        self.assertEqual(result['output']['output_token_ids'],list(range(128)))
        self.assertEqual(result['output']['native_request_id'],'request-12345678')
        self.assertNotIn('step',vars(engine))
        self.assertTrue(all(a['at_ns']<=b['at_ns'] for a,b in zip(result['token_times'],result['token_times'][1:])))

    def test_original_step_failure_restores_observation_without_retry(self):
        common=common_module();engine=Engine(common,True)
        with self.assertRaisesRegex(RuntimeError,'original step failed'): M.frontend_capture(common,engine,None,'request')
        self.assertNotIn('step',vars(engine));self.assertEqual(engine.count,0)

    def test_existing_step_override_refused(self):
        common=common_module();engine=Engine(common);engine.step=lambda:[]
        with self.assertRaisesRegex(ValueError,'overridden'):M.frontend_capture(common,engine,None,'request')

    def test_sparse_sink_retains_every_changed_owner_sequence(self):
        class Journal:
            def __init__(self):self.frames=[];self.event_sequence=0;self.calls=0
            def publish_owner_frame(self):self.calls+=1;self.frames.append(types.SimpleNamespace(sequence=self.event_sequence))
        journal=Journal();sink=M.SparseOwnerSink(journal)
        sink(None)
        for _ in range(100):sink(None)
        self.assertEqual(journal.calls,1)
        journal.event_sequence=2;sink(None);self.assertEqual(journal.calls,2)
        journal.frames.append(types.SimpleNamespace(sequence=4));journal.event_sequence=4;sink(None)
        self.assertEqual(journal.calls,2) # native accepted/completed already published this exact sequence

    def test_fresh_storage_copy_preserves_template_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();source=root/'template/a';source.parent.mkdir();source.write_bytes(b'payload')
            out=root/'out';out.mkdir()
            ref=M.file_ref(root,'template/a')
            inputs={'groups':[{'files':[dict(path='a',bytes=ref['bytes'],sha256=ref['sha256'],block_hash='0'*64)]}]}
            dest=M.prepare_window_storage(root,{'storage':'template'},inputs,out)
            self.assertEqual((dest/'a').read_bytes(),b'payload')
            (dest/'a').write_bytes(b'changed')
            self.assertEqual(source.read_bytes(),b'payload')
            with self.assertRaises(FileExistsError):M.prepare_window_storage(root,{'storage':'template'},inputs,out)

    def test_parent_failure_stops_remaining_processes_and_inherits_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();out=root/'details'
            config={'out':'details'}; calls=[]
            def fail(command,**kwargs):
                calls.append((command,kwargs));return types.SimpleNamespace(returncode=7)
            with patch.object(M,'input_groups',return_value={}),patch.object(M.subprocess,'run',side_effect=fail),\
                 patch.object(M.os,'getsid',return_value=123,create=True),patch.object(M.os,'getpgid',return_value=123,create=True):
                result=M.execute_parent(root,config,{}, {})
            self.assertEqual(len(calls),1);self.assertEqual(result['original_model_subprocesses_started'],1)
            self.assertNotIn('start_new_session',calls[0][1]);self.assertNotIn('preexec_fn',calls[0][1])
            self.assertNotIn('creationflags',calls[0][1]);self.assertFalse((out/'windows/01').exists())
            self.assertTrue((out/'windows/00/details').is_dir())
            self.assertEqual((out/'windows/00/details/runtime-cache').parent.name,'details')
            self.assertEqual(result['status'],'FAILED_NATIVE_SIX_PROCESS_DIAGNOSTIC')

    def test_fixed_workload_split_has_disjoint_prefixes(self):
        prompts=[M.prompt(index) for index in range(3)]
        self.assertTrue(all(len(prompt)==128 for prompt in prompts))
        self.assertEqual([prompt[0] for prompt in prompts],[8100,9100,10100])
        self.assertEqual(len(set(tuple(prompt[:16]) for prompt in prompts)),3)
        self.assertEqual(M.ORDER,(('A','B'),('B','A'),('A','B')))

    def test_zero_drain_rejects_aio_and_stage_pending(self):
        clean={'owner_capture':True,'native':{'active_parents':0},'aio':dict(outstanding=0,pending=0,ready=0,unreaped=0),
            'stage_accounting':{'valid':True,'stages':{'ssd_read':dict(inflight_ops=0,inflight_bytes=0,failed_ops=0)}}}
        M.zero_snapshot(clean)
        clean['aio']['unreaped']=1
        with self.assertRaisesRegex(ValueError,'AIO'):M.zero_snapshot(clean)
        clean['aio']['unreaped']=0;clean['stage_accounting']['stages']['ssd_read']['inflight_bytes']=1
        with self.assertRaisesRegex(ValueError,'stage'):M.zero_snapshot(clean)

    def test_original_ledger_keys_bind_inherited_child_session(self):
        config=dict(label=M.LABEL,gpu_uuid=M.GPU_UUID,seconds_limit=1200,reserved_seconds=1220)
        active=dict(config,session_id=123,process_group=123,runner_pid=99,
            command=['python','-B',M.SCRIPT,'--execute','--config',M.DELIVERY+'/NATIVE_COST_CONFIG.json'],
            permissions=dict(path='permissions',bytes=1,sha256='0'*64))
        with patch.object(M.os,'name','posix'),patch.dict(M.os.environ,CUDA_VISIBLE_DEVICES=M.GPU_UUID),\
             patch.object(M.os,'getsid',return_value=123,create=True),patch.object(M.os,'getpgid',return_value=123,create=True),\
             patch.object(M,'safe',return_value=Path('ledger')),patch.object(M,'read_json',return_value={
                 'active_reservation':active,'gpu_wall_seconds':18829.894}):
            self.assertEqual(M.verify_guard(config)['ledger_gpu_seconds_used'],18829.894)
            active['seconds_limit']=1201
            with self.assertRaisesRegex(ValueError,'seconds_limit'):M.verify_guard(config)

    def test_owner_boundary_published_only_from_original_owner_snapshot(self):
        class Reactor:
            _shared_cached={};_preload_slots={};_staging_cache=None
            _preload_pending_count={};_preload_inflight_hashes={};_preload_refcount={}
            def _capture_owner_snapshot(self,*,native_shutdown=False):
                return dict(owner_capture=not native_shutdown,native_shutdown_read=native_shutdown)
        class Journal:
            calls=0
            def publish_owner_frame(self):
                from dataclasses import make_dataclass
                self.calls+=1;return make_dataclass('Boundary',[('captured_ns',int)])(123)
        reactor=Reactor();journal=Journal()
        M.install_owner_snapshot_observation(reactor,['0'*64],journal)
        first=reactor._capture_owner_snapshot()
        self.assertEqual(first['diagnostic_owner_boundary']['captured_ns'],123)
        self.assertEqual(len(M.require_cold_inputs(first,['0'*64])),1)
        reactor._capture_owner_snapshot(native_shutdown=True)
        self.assertEqual(journal.calls,1)

    def test_original_scheduler_hashes_reconcile_and_reject_background_overlap(self):
        common=common_module();engine=Engine(common)
        request=types.SimpleNamespace(request_id='request-12345678',prompt_token_ids=common.PROMPT.copy(),
            block_hashes=[bytes([index])*32 for index in range(8)])
        engine.engine_core=types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=types.SimpleNamespace(requests={'id':request})))
        result=M.frontend_capture(common,engine,None,'request',['ff'*32])
        self.assertEqual(len(result['original_scheduler_hash_observations']),128)
        engine=Engine(common);engine.engine_core=types.SimpleNamespace(engine_core=types.SimpleNamespace(scheduler=types.SimpleNamespace(requests={'id':request})))
        with self.assertRaisesRegex(ValueError,'hashes intersect'):
            M.frontend_capture(common,engine,None,'request',['00'*32])
        self.assertNotIn('step',vars(engine))

    def test_config_is_fixed_and_in_source_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            config=dict(root=str(root),label=M.LABEL,gpu_uuid=M.GPU_UUID,purpose=M.PURPOSE,
                seconds_limit=1200,reserved_seconds=1220,active_ledger=M.LEDGER,storage=M.STORAGE,
                out='experiments/prefix_io_v1/runs/'+M.LABEL+'/details',source_lock='lock.json',
                input_manifest='inputs.json',collector_relative='collector.py')
            config_rel=M.DELIVERY+'/NATIVE_COST_CONFIG.json';path=root/config_rel;path.parent.mkdir(parents=True)
            path.write_text(json.dumps(config))
            for rel in (M.SCRIPT,'inputs.json','collector.py'):
                target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('{}')
            refs=[M.file_ref(root,rel) for rel in (M.SCRIPT,config_rel,'inputs.json','collector.py')]
            (root/'lock.json').write_text(json.dumps(dict(files=refs)))
            with patch.object(M,'ROOT',root):
                observed,_=M.load_configuration(path)
                self.assertEqual(observed,config)
                config['seconds_limit']=1201;path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError,'fixed config: seconds_limit'):M.load_configuration(path)


if __name__=='__main__':unittest.main(verbosity=2)
