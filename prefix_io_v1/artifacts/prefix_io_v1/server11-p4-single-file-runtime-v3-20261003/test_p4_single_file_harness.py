"""CPU tests for a finite one-arm launch and preservation of original paths."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE=Path(__file__).resolve().parent
SCRIPT=HERE/'run_p4_single_file_experiment.py'
spec=importlib.util.spec_from_file_location('_p4_single_harness_cpu',SCRIPT)
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
WORKSPACE=next((p for p in HERE.parents if (p/'artifacts/prefix_io_v1_server11_candidates/native_cost_v5').is_dir()),None)
BASE=(WORKSPACE/'artifacts/prefix_io_v1_server11_candidates/native_cost_v5/run_native_cost_experiment.py' if WORKSPACE else
      next(p for p in HERE.parents if (p/'artifacts/prefix_io_v1/server11-native-cost-v5-20261003').is_dir())/
      'artifacts/prefix_io_v1/server11-native-cost-v5-20261003/run_native_cost_experiment.py')


def tree(path):
    return ast.parse(path.read_text(encoding='utf-8-sig'))


def function(path,name):
    return next(n for n in tree(path).body if isinstance(n,ast.FunctionDef) and n.name==name)


def write(root,relative,value):
    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')
    return M.file_ref(root,relative)


class HarnessTests(unittest.TestCase):
    def test_same_original_model_cache_frontend_and_drain_functions(self):
        for name in ('config_for_engine','input_groups','prompt','prepare_window_storage',
                     'frontend_capture','original_drain','native_handler','zero_snapshot',
                     'install_owner_snapshot_observation','require_cold_inputs'):
            with self.subTest(name=name):
                self.assertEqual(ast.dump(function(SCRIPT,name),include_attributes=False),
                                 ast.dump(function(BASE,name),include_attributes=False))

    def test_previous_v5_source_remains_frozen(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(),
            'feb1e3dead5da65141b7f0faadbb857bdcf654e0c5b2291c67ca5b1913bc53dc')

    def test_legacy_collector_lookup_records_actual_overlay_without_mutating_lock(self):
        old={'path':M.LEGACY_COLLECTOR_REACTOR_KEY,'bytes':1,'sha256':'a'*64}
        actual={'path':M.REACTOR,'bytes':2,'sha256':'b'*64}
        refs={M.LEGACY_COLLECTOR_REACTOR_KEY:old,M.REACTOR:actual,'other':{'unchanged':True}}
        adapted=M.collector_source_view(refs)
        self.assertIs(refs[M.LEGACY_COLLECTOR_REACTOR_KEY],old)
        self.assertIs(adapted[M.LEGACY_COLLECTOR_REACTOR_KEY],actual)
        self.assertIs(adapted['other'],refs['other'])
        self.assertEqual(adapted[M.LEGACY_COLLECTOR_REACTOR_KEY]['path'],M.REACTOR)

    def test_import_no_gpu_or_executor_and_no_internal_second_job(self):
        self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
        self.assertFalse(hasattr(M,'execute_parent'))
        calls=[n for n in ast.walk(tree(SCRIPT)) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in
            ('Popen','setsid','killpg','create_subprocess_exec') for n in calls))
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
            and n.func.value.id=='subprocess' for n in calls))

    def test_all_arms_same_single_file_foreground_common_config(self):
        self.assertEqual(M.MODES,('off','shadow','on'))
        self.assertEqual(M.ORDER,(('B',),));self.assertEqual(M.OPERATION_COUNT,1)
        self.assertEqual(M.PROMPT_TOKENS,129);self.assertEqual(M.prompt(0)[0],28100)
        self.assertEqual(len(M.prompt(0)),129);self.assertEqual(M.SEEDS,(2829,))
        self.assertEqual((M.MAX_WAIT_NS,M.SAMPLE_MAX_AGE_NS,M.MAX_ACCEPTED_PARENTS),(100000000,100000000,8))
        native_calls=[n for n in ast.walk(function(SCRIPT,'execute_window')) if isinstance(n,ast.Call)
                      and isinstance(n.func,ast.Name) and n.func.id=='original_factory']
        self.assertEqual(len(native_calls),1)
        self.assertEqual({k.arg for k in native_calls[0].keywords if k.arg is not None},
            {'stage_accounting','observation_sink','progress_run_id','max_accepted_parents','p4_bridge'})

    def test_off_never_constructs_bridge_and_attach_is_before_frontend(self):
        execute=function(SCRIPT,'execute_window')
        guarded=[n for n in ast.walk(execute) if isinstance(n,ast.If) and ast.unparse(n.test)=="mode != 'off'"]
        self.assertEqual(len(guarded),1)
        self.assertIn('make_native_bridge',ast.unparse(guarded[0]))
        source=ast.unparse(execute)
        self.assertLess(source.index('bridge.attach_single_file_capture'),source.index('frontend = frontend_capture'))
        self.assertIn("shadow=mode == 'shadow'",source)
        self.assertIn("before['owner_snapshot']['p4']",source)
        self.assertIn("after['owner_snapshot']['p4']",source)
        self.assertIn("result['native_post_shutdown']['snapshot']['p4']",source)

    def test_common_lock_excludes_later_config_but_pins_executed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            config=dict(root=str(root),label='server11-p4-single-file-off-03',gpu_uuid=M.GPU_UUID,
                purpose=M.PURPOSE,seconds_limit=300,reserved_seconds=320,active_ledger=M.LEDGER,storage=M.STORAGE,
                out='experiments/prefix_io_v1/runs/server11-p4-single-file-off-03/details',overlay_relative=M.OVERLAY,
                collector_relative=M.CANDIDATE+'/native_full_step_collector.py',
                runtime_binding_relative=M.CANDIDATE+'/single_file_runtime_binding.py',
                source_lock='lock.json',input_manifest='manifest.json',binding_relative='binding.json',
                mode='off',previous_qualification_ref=None)
            paths=(M.SCRIPT,config['collector_relative'],config['runtime_binding_relative'],
                   config['input_manifest'],config['binding_relative'],M.REACTOR,M.JOURNAL)
            refs=[write(root,p,{'source_fixture':p}) for p in paths]
            write(root,'lock.json',{'files':refs})
            write(root,M.config_relative('off'),config)
            with patch.object(M,'ROOT',root):
                loaded,actual=M.load_configuration(root/M.config_relative('off'))
            self.assertEqual(loaded,config)
            self.assertNotIn(M.config_relative('off'),actual)
            (root/M.SCRIPT).write_text('changed',encoding='utf-8')
            with patch.object(M,'ROOT',root),self.assertRaisesRegex(ValueError,'pinned source drift'):
                M.load_configuration(root/M.config_relative('off'))

    def test_previous_qualification_true_refs_and_same_binding_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            binding=write(root,'binding.json',{'frozen':True})
            refs={'binding.json':binding}
            prior=write(root,'prior-config.json',{'source_lock':'shared-lock.json','binding_relative':'binding.json'})
            report=dict(scope='server11_p4_single_file_runtime_qualification_v1',mode='shadow',
                native_execution_verified=True,runtime_condition_qualified=True,permits_next_mode='on',
                binding_ref=binding,evidence_refs={'config':prior})
            reference=write(root,'qualification.json',report)
            config=dict(mode='on',previous_qualification_ref=reference,binding_relative='binding.json',source_lock='shared-lock.json')
            self.assertEqual(M.previous_gate(root,config,refs),report)
            for key,value in (('runtime_condition_qualified',False),('native_execution_verified',False),
                              ('permits_next_mode',None),('mode','off')):
                with self.subTest(key=key):
                    changed=dict(report,**{key:value});config['previous_qualification_ref']=write(root,'qualification.json',changed)
                    with self.assertRaises(ValueError):M.previous_gate(root,config,refs)
            config['previous_qualification_ref']=write(root,'qualification.json',report)
            (root/'prior-config.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'prior evidence reference changed'):
                M.previous_gate(root,config,refs)

    def test_actual_original_ledger_keys_and_sid_and_per_arm_argv(self):
        config=dict(mode='shadow',label='server11-p4-single-file-shadow-03',gpu_uuid=M.GPU_UUID,
                    seconds_limit=300,reserved_seconds=320)
        active=dict(config,session_id=88,process_group=88,runner_pid=77,
            command=['python','-B',M.SCRIPT,'--execute','--config',M.config_relative('shadow')],
            permissions={'path':'permissions.yaml','bytes':1,'sha256':'a'*64})
        os_fixture=SimpleNamespace(name='posix',environ={'CUDA_VISIBLE_DEVICES':M.GPU_UUID},
                                   getsid=lambda _:88,getpgid=lambda _:88)
        with patch.object(M,'os',os_fixture),patch.object(M,'read_json',return_value={
            'active_reservation':active,'gpu_wall_seconds':100}),patch.object(M,'safe',return_value=Path('ledger')):
            result=M.verify_guard(config)
        self.assertEqual(result['ledger_gpu_seconds_used'],100)
        active['command'][-1]=M.config_relative('on')
        with patch.object(M,'os',os_fixture),patch.object(M,'read_json',return_value={
            'active_reservation':active,'gpu_wall_seconds':100}),patch.object(M,'safe',return_value=Path('ledger')):
            with self.assertRaisesRegex(ValueError,'exact guarded child argv'):M.verify_guard(config)

    def test_original_final_shutdown_and_all_token_timing_remain(self):
        source=ast.unparse(function(SCRIPT,'execute_window'))
        self.assertIn('llm.llm_engine.engine_core.shutdown(timeout=15)',source)
        self.assertIn('runtime.post_shutdown_snapshot(handler)',source)
        self.assertIn('reactor._worker.is_alive()',source)
        self.assertIn('reactor.ring._worker.is_alive()',source)
        for name in ('request_started_ns','request_finished_ns','drain_started_ns','drain_finished_ns'):
            self.assertIn(name+' = time.monotonic_ns()',source)
        self.assertIn("file_ref(root, config_relative(mode)) != config_ref",source)
        self.assertIn("frontend['output']['output_token_ids'] == warmup['output_token_ids']",source)


if __name__=='__main__':unittest.main(verbosity=2)
