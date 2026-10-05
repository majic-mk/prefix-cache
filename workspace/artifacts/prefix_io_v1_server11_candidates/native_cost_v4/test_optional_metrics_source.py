"""Replay unchanged author metrics and engine-step methods on CPU fixtures.

No backend imports; this proves the exact optional-logger failure and bypass,
not GPU feasibility, cache correctness or experimental performance.
"""
import ast
import contextlib
from dataclasses import dataclass,field
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

HERE=Path(__file__).resolve().parent
ARTIFACTS=HERE.parents[1]
AUTHOR='third_party/work/vllm-author-p4-02-cpu/vllm/'
SOURCES={
 'metrics':(AUTHOR+'distributed/kv_transfer/kv_connector/v1/offloading/metrics.py',
     '5ab3c65af3c0f78b0696bcc60c68ecfe51332c6d5f6684bb2cfa0e0ea9677d73',
     ARTIFACTS/'prefix_io_v1_server09_candidates/g3_calibration_launcher_cpu_v1/gpu-actual-20261002/source-audit/vllm_offloading_metrics.py'),
 'engine':(AUTHOR+'v1/engine/llm_engine.py',
     '2900f6732a1e507f6ced6f75f05f1fcaf2cc2929df0f5066574e9077ad3fbde1',
     ARTIFACTS/'prefix_io_v1_server09_candidates/g2_source_readonly'/AUTHOR/'v1/engine/llm_engine.py'),
 'output':(AUTHOR+'v1/engine/output_processor.py',
     '464c3cda347512cf1090f8e6471d0d3f0f65e286eb9a7b97be52efbc1ffd572a',
     ARTIFACTS/'prefix_io_v1_server09_candidates/g3_current_context_calibration_cpu/source_readonly'/AUTHOR/'v1/engine/output_processor.py'),
}


def source(kind):
    relative,sha,local=SOURCES[kind]
    path=Path(os.environ['SERVER11_AUTHOR_SOURCE_ROOT'])/relative if 'SERVER11_AUTHOR_SOURCE_ROOT' in os.environ else local
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('actual author source drift: '+relative)
    return ast.parse(raw),str(path)


def compile_nodes(nodes,filename,namespace):
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*nodes],type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),filename,'exec',dont_inherit=True),namespace)


def metrics_fixture():
    module=types.ModuleType('_server11_actual_author_metrics_cpu')
    sys.modules[module.__name__]=module
    @dataclass
    class Base:
        data:dict=field(default_factory=dict)
    module.__dict__.update(dataclass=dataclass,KVConnectorStats=Base,KVConnectorPromMetrics=object)
    tree,filename=source('metrics')
    selected=[node for node in tree.body if isinstance(node,ast.ClassDef)
              and node.name in ('OffloadingOperationMetrics','OffloadingConnectorStats','OffloadPromMetrics')]
    if len(selected)!=3:raise ValueError('exact original metrics class set')
    compile_nodes(selected,filename,module.__dict__)
    stats=module.OffloadingConnectorStats()
    stats.record_transfer(917504,0.0005,('GPU','SHARED_STORAGE'))
    observer=object.__new__(module.OffloadPromMetrics)
    class Metric:
        def observe(self,value):pass
        def inc(self,value):pass
    key=(0,'GPU_to_SHARED_STORAGE')
    observer.histogram_transfer_size={key:Metric()}
    observer.counter_kv_bytes={key:Metric()};observer.counter_kv_transfer_time={key:Metric()}
    return module,stats,observer


def engine_step():
    tree,filename=source('engine')
    klass=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='LLMEngine')
    node=next(node for node in klass.body if isinstance(node,ast.FunctionDef) and node.name=='step')
    namespace=dict(record_function_or_nullcontext=contextlib.nullcontext,IterationStats=lambda:object())
    compile_nodes([node],filename,namespace)
    return namespace['step']


class OptionalMetricsSourceTests(unittest.TestCase):
    def test_original_inprocess_dataclass_reproduces_prometheus_assertion(self):
        module,stats,observer=metrics_fixture()
        self.assertIsInstance(stats.data['GPU_to_SHARED_STORAGE'][0],module.OffloadingOperationMetrics)
        with self.assertRaises(AssertionError):observer.observe(stats.data)
        with self.assertRaises(AssertionError):stats.reduce()

    def test_original_step_with_supported_logging_off_preserves_outputs(self):
        module,stats,observer=metrics_fixture();step=engine_step();calls=[]
        original_output=object();original_frontend=object()
        batch=types.SimpleNamespace(outputs=[original_output],timestamp=123,scheduler_stats=object())
        class Processor:
            def process_outputs(self,outputs,**kwargs):
                calls.append(('process',outputs,kwargs));return types.SimpleNamespace(reqs_to_abort=[],request_outputs=[original_frontend])
            def update_scheduler_stats(self,value):calls.append(('update',value))
        engine=types.SimpleNamespace(should_execute_dummy_batch=False,log_stats=False,logger_manager=None,
            engine_core=types.SimpleNamespace(get_output=lambda:batch,abort_requests=lambda ids:calls.append(('abort',ids))),
            output_processor=Processor())
        result=step(engine)
        self.assertIs(result[0],original_frontend);self.assertIs(calls[0][1][0],original_output)
        self.assertIsNone(calls[0][2]['iteration_stats'])
        self.assertEqual([row[0] for row in calls],['process','update','abort'])
        engine.log_stats=True
        engine.logger_manager=types.SimpleNamespace(record=lambda **kwargs:observer.observe(stats.data))
        engine.renderer=types.SimpleNamespace(stat_mm_cache=lambda:None)
        with self.assertRaises(AssertionError):step(engine)

    def test_original_prefill_cached_tokens_and_output_fields_not_log_stats_gated(self):
        tree,_=source('output')
        parent={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        assignments=[]
        for node in ast.walk(tree):
            if (isinstance(node,ast.Assign) and any(isinstance(target,ast.Attribute)
                    and target.attr=='num_cached_tokens' for target in node.targets)
                    and 'prefill_stats' in ast.unparse(node.value)):
                assignments.append(node)
        self.assertEqual(len(assignments),1)
        node=assignments[0]
        while node in parent:
            node=parent[node]
            if isinstance(node,ast.If):self.assertNotIn('log_stats',ast.unparse(node.test))
        values=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and
                isinstance(node.func,ast.Name) and node.func.id=='RequestOutput']
        self.assertTrue(any(any(key.arg=='num_cached_tokens' and ast.unparse(key.value)=='self.num_cached_tokens'
                                for key in value.keywords) for value in values))

    def test_all_diagnostic_arms_use_supported_optional_switch(self):
        spec=importlib.util.spec_from_file_location('current_harness',HERE/'run_native_cost_experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=module.config_for_engine(types.SimpleNamespace(ENGINE={}),Path('/private/storage'))
        self.assertIs(config['disable_log_stats'],True)
        self.assertEqual(config['kv_transfer_config']['kv_connector_extra_config']['load_planner'],'off')
        self.assertIs(config['kv_transfer_config']['kv_connector_extra_config']['sync_on_store'],False)


if __name__=='__main__':unittest.main(verbosity=2)
