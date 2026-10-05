"""Read-only source-flow audit; normalization allows explicit metadata only."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SERVER_OLD='artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'


def old_directory():
    local=HERE.parent/'normal_native_cpu'
    if (local/'run_p4_single_file_experiment.py').is_file():return local
    for parent in HERE.parents:
        candidate=parent/SERVER_OLD
        if (candidate/'run_p4_single_file_experiment.py').is_file():return candidate
    raise ValueError('actual old frozen source directory missing')


def funcs(path):
    return {node.name:node for node in ast.parse(path.read_bytes()).body if isinstance(node,ast.FunctionDef)}


def dump(node):return ast.dump(node,include_attributes=False)


def ref(path):
    raw=path.read_bytes();return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


class NormalizeExplicitEntryMetadata(ast.NodeTransformer):
    def visit_Assign(self,node):
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in ('diagnostic_index','sdk_started_ns','llm_started_ns'):
                return None
            if name=='receipt' and isinstance(node.value,ast.IfExp):
                assert isinstance(node.value.body,ast.Call) and isinstance(node.value.body.func,ast.Name)
                assert node.value.body.func.id=='load_verified_single_file'
                node.value=node.value.body
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Subscript):
            target=node.targets[0]
            if (isinstance(target.value,ast.Subscript) and isinstance(target.value.value,ast.Name) and
                target.value.value.id=='result' and isinstance(target.value.slice,ast.Constant) and
                target.value.slice.value=='startup_stage_times'):
                assert isinstance(target.slice,ast.Constant) and target.slice.value in ('sdk_private_environment','llm_initialization')
                return None
        return self.generic_visit(node)

    def visit_If(self,node):
        if isinstance(node.test,ast.Compare):
            test=node.test
            if isinstance(test.left,ast.Name) and test.left.id=='_context':
                targets=[]
                for value in node.body:
                    assert isinstance(value,ast.Assign) and len(value.targets)==1
                    targets.append(dump(value.targets[0]))
                assert len(targets)==3 and 'process_entry_context_metadata' in targets[0]
                assert 'startup_stage_times' in targets[1] and 'startup_stage_times' in targets[2]
                return None
            if isinstance(test.left,ast.Constant) and test.left.value=='request_and_drain_timing':
                assert len(node.body)==1 and isinstance(node.body[0],ast.Assign)
                assert 'measured_request_existing_monotonic_clock' in dump(node.body[0].targets[0])
                return None
        return self.generic_visit(node)

    def visit_Call(self,node):
        if isinstance(node.func,ast.Name):
            if node.func.id in ('load_configuration','verify_guard'):
                node.keywords=[item for item in node.keywords if item.arg!='_context']
            if node.func.id=='config_relative' and len(node.args)==1:
                arg=node.args[0]
                if isinstance(arg,ast.Name) and arg.id=='diagnostic_index':node.args=[ast.Name(id='mode',ctx=ast.Load())]
                elif (isinstance(arg,ast.Subscript) and isinstance(arg.value,ast.Name) and arg.value.id=='config' and
                    isinstance(arg.slice,ast.Constant) and arg.slice.value=='diagnostic_index'):
                    arg.slice=ast.Constant(value='mode')
            if node.func.id=='dict':
                node.keywords=[item for item in node.keywords if item.arg not in
                    ('diagnostic_index','protocol_ref','startup_stage_times','startup_timing_scope')]
        return self.generic_visit(node)


def audit():
    old=old_directory();oldR=funcs(old/'run_p4_single_file_experiment.py');newR=funcs(HERE/'run_p4_single_file_experiment.py')
    assert set(oldR)==set(newR),'runtime function surface changed'
    metadata={'config_relative','register_canonical_receipt','controller_api','load_configuration','verify_guard','execute_window','main'}
    same=[]
    for name in sorted(set(oldR)-metadata):
        assert dump(oldR[name])==dump(newR[name]),'original runtime helper changed: '+name
        same.append(name)
    new=deepcopy(newR['execute_window'])
    assert len(new.args.kwonlyargs)==1 and new.args.kwonlyargs[0].arg=='_context'
    new.args=deepcopy(oldR['execute_window'].args)
    normalized=NormalizeExplicitEntryMetadata().visit(new)
    assert dump(normalized)==dump(oldR['execute_window']),'non-metadata original execution body changed'
    oldC=funcs(old/'control_p4_single_file.py');newC=funcs(HERE/'control_p4_single_file.py')
    inherited=('require','exact','project','safe','ref','reference_shape','verify_reference','pairs','read','put','now',
        'load_module','rows_map','metadata_reference','number','validate_live_interval','permission_document','closure_digest')
    for name in inherited:assert dump(oldC[name])==dump(newC[name]),'original controller pure helper changed: '+name
    fixed=[]
    for name in ('notification_runtime_adapter.py','observation_cpu_cost.py'):
        assert (old/name).read_bytes()==(HERE/name).read_bytes(),'original helper bytes changed'
        fixed.append(ref(HERE/name))
    tree=ast.parse((HERE/'process_entry_context.py').read_bytes())
    public=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='load_receipt']
    assert len(public)==1,'not one public loader call in actual process-context create'
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('setattr','exec','eval') for node in ast.walk(tree))
    return dict(status='PASS_EXPLICIT_PROCESS_ENTRY_SOURCE_FLOW_CPU_AUDIT', inherited_controller_helpers=list(inherited),
        identical_runtime_helpers=same, explicit_runtime_metadata_functions=sorted(metadata),
        execute_window_body_after_exact_metadata_normalization_equal=True,
        normalized_changes='explicit index/config/guard/context receipt reuse; compact startup host-clock metadata; raw provenance fields',
        byte_identical_original_helpers=fixed, source_refs=[ref(HERE/name) for name in
            ('control_p4_single_file.py','run_p4_single_file_experiment.py','process_entry_context.py')],
        native_execution_verified=False, actual_GPU_runs=0, GPU_operations=0,
        startup_speedup_measured=False, cost_gate_changed=False, full_runtime_cost_qualified=False,
        strategy_effect_verified=False, P4_completed=False)


if __name__=='__main__':print(json.dumps(audit(),sort_keys=True,allow_nan=False))
