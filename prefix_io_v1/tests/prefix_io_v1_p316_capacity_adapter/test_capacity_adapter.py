"""CPU-only capacity-adapter preservation and strict registered budget guards."""
import ast, sys, types
from pathlib import Path
import pytest
import simple_stage_worker_probe_capacity_p316 as probe

SCRIPTS=Path(__file__).resolve().parents[2]/"experiments/prefix_io_v1/scripts"

def tree(name):
    return ast.parse((SCRIPTS/name).read_text())

def normalized_imports(module):
    for node in ast.walk(module):
        if isinstance(node,ast.ImportFrom):
            node.module={
                "run_p3_native_pilot_capacity_p316":"run_p3_native_pilot",
                "flush_cause_worker_probe_capacity_p316":"flush_cause_worker_probe",
                "store_readiness_worker_probe_capacity_p316":"store_readiness_worker_probe"
            }.get(node.module,node.module)
    return module

def test_base_probe_preserves_all_native_code_except_registered_tuple():
    old=tree("run_p3_native_pilot.py");new=tree("run_p3_native_pilot_capacity_p316.py")
    old.body=old.body[1:];new.body=new.body[1:]
    target=ast.dump(ast.parse("dict(kv_budget_bytes=2147483648,staging_budget_bytes=960*1024**2)",mode="eval").body)
    removed=0
    for node in ast.walk(new):
        if isinstance(node,ast.Tuple):
            before=len(node.elts)
            node.elts=[x for x in node.elts if ast.dump(x)!=target]
            removed+=before-len(node.elts)
    assert removed==1
    assert ast.dump(old)==ast.dump(new)

@pytest.mark.parametrize("new,old",[
    ("flush_cause_worker_probe_capacity_p316.py","flush_cause_worker_probe.py"),
    ("store_readiness_worker_probe_capacity_p316.py","store_readiness_worker_probe.py")])
def test_flush_and_readiness_are_exact_original_chain_with_new_import(new,old):
    assert ast.dump(normalized_imports(tree(new)))==ast.dump(tree(old))

def test_simple_capacity_adapter_preserves_timing_observation_and_physical_drain():
    old=tree("simple_stage_worker_probe_p316.py");new=normalized_imports(tree("simple_stage_worker_probe_capacity_p316.py"))
    owner=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=="_worker_probe")
    install=next(n for n in owner.body if isinstance(n,ast.If) and
        ast.dump(n.test)==ast.dump(ast.parse('action == "install"',mode="eval").body))
    assert isinstance(install.body[0],ast.If)
    assert isinstance(install.body[1],ast.Assign)
    target=install.body[1].targets[0]
    assert isinstance(target,ast.Attribute) and target.attr=="_p316_capacity_staging_budget_bytes"
    assert isinstance(target.value,ast.Name) and target.value.id=="worker"
    install.body=install.body[2:]
    replaced=0
    for node in ast.walk(new):
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Attribute) and node.left.attr=="actual_staging_bytes" and isinstance(node.comparators[0],ast.Attribute) and node.comparators[0].attr=="_p316_capacity_staging_budget_bytes":
            node.comparators=[ast.parse('(limits or {}).get("staging_budget_bytes",1073741824)',mode="eval").body]
            replaced+=1
    assert replaced==1
    assert ast.dump(old)==ast.dump(new)

def test_capacity_driver_uses_separate_permit_before_model_or_storage():
    module=tree("run_concurrent_capacity_p316.py")
    imports=[n.module for n in module.body if isinstance(n,ast.ImportFrom)]
    assert "qualify_capacity_p316" in imports and "qualify_concurrent_pilot" not in imports
    calls=[n for n in ast.walk(module) if isinstance(n,ast.Call)]
    gate=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="verify_gate"]
    assert len(gate)==1 and {k.arg for k in gate[0].keywords}=={"capacity_domain_id","manifest_path"}
    clone=next(n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="clone_private_storage")
    model=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="validate_local_model")
    assert gate[0].lineno<clone.lineno<model.lineno
    variants=[n for n in calls if isinstance(n.func,ast.Name) and n.func.id=="expected_connector_variant"]
    assert len(variants)==1
    argument=next(n for n in calls if isinstance(n.func,ast.Attribute) and n.func.attr=="add_argument" and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=="--capacity-domain")
    options={k.arg:ast.literal_eval(k.value) for k in argument.keywords}
    assert options==dict(choices=("cap960-l1","cap1024-l2"),required=True)

class Reactor:
    def __init__(self,actual):
        self.actual_staging_bytes=actual
        self.file_store=types.SimpleNamespace(io_size=917504)
        self._prefix_dispatch_controller=None
        self._observation_sink=lambda r:None
    def inspect_snapshot(self,timeout):return {}

@pytest.fixture
def worker_setup(monkeypatch):
    def make(actual=1006632960):
        reactor=Reactor(actual);worker=types.SimpleNamespace()
        cw=types.SimpleNamespace(worker=types.SimpleNamespace(handlers=[
            types.SimpleNamespace(coordinator=types.SimpleNamespace(reactor=reactor))]))
        module=types.ModuleType("vllm.distributed.kv_transfer.kv_transfer_state")
        module.get_kv_transfer_group=lambda:types.SimpleNamespace(connector_worker=cw)
        monkeypatch.setitem(sys.modules,module.__name__,module)
        expected=SCRIPTS.parents[2]/"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py"
        original_file=probe.inspect.getfile
        monkeypatch.setattr(probe.inspect,"getfile",lambda obj:str(expected) if obj is Reactor else original_file(obj))
        calls=[]
        def original(w,action,limits):calls.append((action,limits));return {"native":True}
        monkeypatch.setattr(probe,"original_probe",original)
        return worker,reactor,calls
    return make

@pytest.mark.parametrize("capacity",[1006632960,1073741824])
def test_exact_two_registered_capacity_budgets_pass_to_original_probe(worker_setup,capacity):
    worker,r,calls=worker_setup(capacity)
    limits=dict(kv_budget_bytes=2147483648,staging_budget_bytes=capacity)
    result=probe.worker_probe(worker,"install",limits)
    assert result["native"] is True and calls==[("install",limits)]
    assert worker._p316_capacity_staging_budget_bytes==capacity
    assert "native_metadata_cpu" not in result

@pytest.mark.parametrize("kv,staging",[(2147483648,1006632959),(2147483648,1073741825),
    (268435456,1006632960),(2147483649,1073741824),(2147483648,536870912)])
def test_unregistered_budget_rejected_before_original(worker_setup,kv,staging):
    worker,r,calls=worker_setup()
    with pytest.raises(RuntimeError,match="registered capacity"):
        probe.worker_probe(worker,"install",dict(kv_budget_bytes=kv,staging_budget_bytes=staging))
    assert calls==[]

def test_960_mib_guard_persists_after_install_when_later_rpc_has_no_limits(worker_setup):
    worker,r,calls=worker_setup()
    probe.worker_probe(worker,"install",dict(kv_budget_bytes=2147483648,staging_budget_bytes=1006632960))
    r.actual_staging_bytes=1006632961
    with pytest.raises(RuntimeError,match="staging"):
        probe.worker_probe(worker,"start")
    assert worker._p316_capacity_staging_budget_bytes==1006632960
