import pathlib,types,sys,importlib.util,json
base=pathlib.Path("artifacts/prefix_io_v1_server07_p3/third_party/work/py-kvcache-p2-aio/py_kvcache")
pkg=types.ModuleType("py_kvcache");pkg.__path__=[str(base.resolve())];sys.modules["py_kvcache"]=pkg
def load(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
be=load("py_kvcache.break_even",base/"break_even.py")
cm=load("py_kvcache.cost_model",base/"cost_model.py")
lp=load("py_kvcache.load_planner",pathlib.Path("artifacts/prefix_io_v1_server11_candidates/notification_v5_native_cost_preparation_cpu/common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/load_planner.py"))
curve=pathlib.Path("artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/source_inputs/ORIGINAL_PLANNER_CURVES_V2.json")
curves=be.load_curves(str(curve),model_name="Qwen/Qwen2.5-7B-Instruct",kv_dtype="auto")
tables=cm.build_cost_tables(curves,block_tokens=16,max_model_len=1024)
k=lp.CandidateCostInput(0,"K",768,0,())
a=lambda pos:lp.CandidateCostInput(pos,"A",768,48,tuple(i.to_bytes(8,"big") for i in range(48)))
print("STATIC_CPU_FIXTURE_ONLY_NO_ACTUAL_IO")
f=tables.recompute_s(48);g=tables.load_ssd_s(48);m=tables.load_mem_s(48);s=tables.service_s_of(48)
print(json.dumps(dict(f768_s=f,gSSD768_s=g,gMem768_s=m,service768_s=s,second_recompute_s=2*f,slack_s=2*f-g),sort_keys=True))
for candidates,old in [([a(0)],[]),([k,a(1)],[]),([k,a(1)],[1]),([k,a(1)],[4])]:
 p=lp.LoadPlanner(tables,max_preload_slots=142,defer_tolerance=1,time_source=lambda:1.)
 res=p.plan(candidates,outstanding_load_blocks=old)
 print(json.dumps(dict(candidate_names=[c.req_id for c in candidates],fixture_outstanding_load_blocks=old,results={key:dict(decision=value.decision.name,preload_blocks=value.preload_blocks)for key,value in res.items()}),sort_keys=True))
