"""Independent read-only load/glue review, AST methods and stdlib CPU only."""
from pathlib import Path
import sys, importlib.abc, ast, copy, json, hashlib, subprocess, threading
from types import SimpleNamespace as NS
from dataclasses import replace
root=Path.cwd()
sys.path.insert(0,str(root/"third_party/work/prefix-io-p4-02-cpu/src"))
attempts=[]
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split(".")[0] in ("torch","vllm","py_kvcache","numpy","cupy","cuda"):
            attempts.append(name);raise RuntimeError("CPU review forbids backend import")
sys.meta_path.insert(0,Guard())
from prefix_io_control.p4_load_observation import SchedulerLoadProducer, SchedulerLoadUnavailable
native=root/"third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
old=root/"third_party/work/py-kvcache-p4-01-cpu/py_kvcache/reactor.py"
def methods(p):
    tree=ast.parse(p.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="IoReactor")
    return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
nodes=methods(native);previous=methods(old)
selected=["publish_p4_scheduled_load","_prefix_p4_load_values","_prefix_p4_read_batch_ids"]
module=ast.fix_missing_locations(ast.Module(body=[
    ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),
    ast.ClassDef(name="ActualAstMethods",bases=[],keywords=[],decorator_list=[],
                 body=[copy.deepcopy(nodes[n]) for n in selected])],type_ignores=[]))
ns={};exec(compile(module,str(native),"exec"),ns)
owner=ns["ActualAstMethods"]()
owner._prefix_p4_bridge=NS(run_id="cpu-run",policy=NS(config=NS(sample_max_age_ns=10)))
owner._submit_lock=threading.Lock();owner._closed=False
producer=SchedulerLoadProducer("cpu-run")
scheduled=NS(num_scheduled_tokens={"r":1})
states={"r":NS(req=NS(num_computed_tokens=128,num_prompt_tokens=128))}
first=producer.capture(scheduled,states,now_ns=0)
assert owner.publish_p4_scheduled_load(first)
assert owner._prefix_p4_load_values(5)[0]==first.signature
unknown=producer.capture(scheduled,{},now_ns=5)
assert type(unknown) is SchedulerLoadUnavailable
owner.publish_p4_scheduled_load(unknown)
assert owner._prefix_p4_load_values(5)==((),None)
try:owner.publish_p4_scheduled_load(first)
except ValueError:pass
else:raise AssertionError("old valid frame restored missing newer state")
third=producer.capture(scheduled,states,now_ns=6)
owner.publish_p4_scheduled_load(third)
assert not owner.publish_p4_scheduled_load(third)
try:owner.publish_p4_scheduled_load(replace(third,geometry_sha256="0"*64))
except ValueError:pass
else:raise AssertionError("conflicting same sequence accepted")
assert owner._prefix_p4_load_values(6)==((),None)
try:owner.publish_p4_scheduled_load(third)
except ValueError:pass
else:raise AssertionError("duplicate restored previously conflicted sequence")
fourth=producer.capture(scheduled,states,now_ns=7)
owner.publish_p4_scheduled_load(fourth)
assert owner._prefix_p4_load_values(6)==((),None)
assert owner._prefix_p4_load_values(18)==((),None)
assert owner._prefix_p4_load_values(17)[0]==fourth.signature
assert fourth.production_gpu_state_qualified is False
assert len(fourth.signature)==11 and fourth.signature[0]=="scheduler-work-v1"

# Off reaches neither optional metadata type check nor owner ready scan.
owner._prefix_p4_bridge=None
class Bomb:
    def __getattr__(self,name):raise AssertionError("off touched metadata or source")
owner._ready_fds_load=Bomb()
assert owner.publish_p4_scheduled_load(Bomb()) is False
assert owner._prefix_p4_load_values(Bomb())==((),None)
assert owner._prefix_p4_read_batch_ids() is None

unchanged=[]
for name in ["_flush_copy_batch","_launch_swap_blocks","_prefix_ready_read_decision",
             "_prefix_stage_decide","_pump_once","_schedule_one","_reserve_foreground_slot","_run"]:
    assert ast.dump(nodes[name],include_attributes=False)==ast.dump(previous[name],include_attributes=False),name
    unchanged.append(name)
# Strip only declared additive P4 hooks; the original common body must match.
def strip(node):
    n=copy.deepcopy(node)
    class Hooks(ast.NodeTransformer):
        def visit_Expr(self,obj):
            if (isinstance(obj.value,ast.Call) and isinstance(obj.value.func,ast.Attribute)
                and obj.value.func.attr=="_prefix_p4_parent_terminal"):return None
            return self.generic_visit(obj)
        def visit_Assign(self,obj):
            if any(isinstance(t,ast.Name) and t.id=="p4_batch_ids" for t in obj.targets):return None
            return self.generic_visit(obj)
        def visit_If(self,obj):
            if (isinstance(obj.test,ast.Compare) and isinstance(obj.test.left,ast.Name)
                and obj.test.left.id=="p4_batch_ids"):return None
            return self.generic_visit(obj)
    return Hooks().visit(n)
stripped=[]
for name in ["_drain_ready_load_fds","_file_terminal","_finish_jobs"]:
    assert ast.dump(strip(nodes[name]),include_attributes=False)==ast.dump(previous[name],include_attributes=False),name
    stripped.append(name)

off_code = """
import sys,importlib.abc
sys.path.insert(0,sys.argv[1])
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,n,path=None,target=None):
  if n.split('.')[0] in ('torch','vllm','py_kvcache','numpy','cupy','cuda') or n in (
    'prefix_io_control.p4_bridge','prefix_io_control.p4_policy','prefix_io_control.p4_eta',
    'prefix_io_control.p4_startup_evidence','prefix_io_control.p4_verified_cost_loader'):
   raise AssertionError('off imported optional policy/backend '+n)
sys.meta_path.insert(0,Guard())
from prefix_io_control.p4_options import parse_p4_options,build_p4_kwargs
o=parse_p4_options({'prefix_io_p4_policy':{'mode':'off'}})
class Bomb:
 def __getattr__(self,n):raise AssertionError('off touched evidence root')
assert 'p4_bridge' not in build_p4_kwargs(o,evidence_root=Bomb())
print('PASS off requires no optional policy/evidence/backend')
"""
p=subprocess.run([str(root/".venv/bin/python"),"-I","-S","-c",off_code,
                  str(root/"third_party/work/prefix-io-p4-02-cpu/src")],
                  capture_output=True,text=True,check=True,timeout=20)
out=root/"artifacts/prefix_io_v1/server08-p4-02-cpu/eta/review-integration/scalar-sequence-and-core-01"
out.mkdir(exist_ok=False)
paths=[
"third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/"+n
for n in ("p4_load_observation.py","p4_options.py","p4_startup_evidence.py","p4_bridge.py","p4_policy.py")]
paths+=["third_party/work/py-kvcache-p4-02-cpu/py_kvcache/"+n for n in ("reactor.py","vllm.py")]
paths+=["third_party/work/vllm-author-p4-02-cpu/vllm/distributed/kv_transfer/kv_connector/v1/offloading/"+n+".py"
        for n in ("common","scheduler","worker")]
receipt=dict(status="PASS_INDEPENDENT_CPU_LOAD_SEQUENCE_OFF_AND_ORIGINAL_CORE_AST",
 source_refs=[dict(path=x,sha256=hashlib.sha256((root/x).read_bytes()).hexdigest()) for x in paths],
 cases=["unknown newer step clears valid state","old sequence cannot restore","conflict stays unknown until higher sequence",
        "future and stale frame not used","scheduler signature is not exact9 active GPU signature","off no scan/import/evidence work"],
 untouched_common_methods=unchanged,common_bodies_equal_after_only_declared_p4_hooks_removed=stripped,
 off_subprocess=p.stdout,forbidden_backend_import_attempts=attempts,gpu_workloads_run=0,
 method_replay_scope="Actual source methods AST-compiled into minimal state holder; no native/GPU backend import",
 scheduling_or_effect_qualified=False)
assert attempts==[]
(out/"result.json").write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
