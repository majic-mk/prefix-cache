"""Append the machine-bound plan, source manifest and direct-user scope."""
import copy,datetime,hashlib,importlib.util,json,pathlib,sys
ROOT=pathlib.Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
D="artifacts/prefix_io_v1/server10-reference-migration-v1-20261003"
OLD="artifacts/prefix_io_v1/server09-g3-calibration-launcher-cpu-v1-20261002"
def ref(rel):
 p=ROOT/rel
 assert p.is_file() and not p.is_symlink()
 b=p.read_bytes()
 return dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(name,value):
 p=ROOT/D/name
 with p.open("x",encoding="utf-8") as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
 return ref(D+"/"+name)
contract_ref=ref(D+"/migration_contract.py")
assert contract_ref["bytes"]==20885 and contract_ref["sha256"]=="1ab4f1a694f695f3736c8d9f1e417a6918f548800af4a4e99f3b7bf54d0e491d"
spec=importlib.util.spec_from_file_location("_server10_prepare_contract",ROOT/contract_ref["path"])
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
ancestor=(ROOT/P.MIGRATION_ANCESTOR_REF["path"]).read_bytes();P._ancestor(ancestor)
old_refs={r["path"]:r for r in json.loads(ancestor)["files"]}
sdk_inventory_ref={"path":"artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CUDA13_SOURCE_INVENTORY.json","bytes":692329,"sha256":"0e7cc1df1d8d7aed7271841d6e9f1e4188785f40ad4c74b4c6c78f670b7794b6"}
sdk_proof_ref={"path":"artifacts/prefix_io_v1/server10-cuda13-cpu-20261003/CPU_COMPILE_LINK_RESULT.json","bytes":5588,"sha256":"ef7a7482cc1cae539e2425d6aefc82ea8160e398ab7d8b14926f76537e614a25"}
assert ref(sdk_inventory_ref["path"])==sdk_inventory_ref and ref(sdk_proof_ref["path"])==sdk_proof_ref
assert ref(P.PERMISSIONS)==P.PERMISSIONS_REF
config=dict(schema_version=1,gpu_uuid=P.GPU_UUID,job_names=copy.deepcopy(P.JOB_NAMES),
 storage={m:P.storage_relative(m) for m in P.MODES},permissions_ref=P.PERMISSIONS_REF,
 sdk_inventory_ref=sdk_inventory_ref,sdk_proof_ref=sdk_proof_ref)
config_ref=write("MIGRATION_RUNTIME_CONFIG.json",config)
model_ref=old_refs[OLD+"/G3_REFERENCE_MODEL_CONTEXT.json"]
assert ref(model_ref["path"])==model_ref
model=json.loads((ROOT/model_ref["path"]).read_bytes())
assert model["original_config_gpu_uuid"]==P.SOURCE_GPU_UUID
model.update(source_gpu_uuid=P.SOURCE_GPU_UUID,execution_gpu_uuid=P.GPU_UUID,source_context_ref=model_ref)
model_ref_new=write("G3_REFERENCE_MODEL_CONTEXT.json",model)
analysis_ref=old_refs[OLD+"/G3_REFERENCE_ANALYZER_PLAN.json"];assert ref(analysis_ref["path"])==analysis_ref
analysis=json.loads((ROOT/analysis_ref["path"]).read_bytes())
analysis["native_label"]=P.JOB_NAMES["cold"];analysis["gpu_uuid"]=P.GPU_UUID
analysis["external_groups"][0]["label"]=P.JOB_NAMES["paired"]
analysis["run_details"]={label:str(ROOT/"experiments/prefix_io_v1/runs"/label/"details/acquisition") for label in P.JOB_NAMES.values()}
analysis["native_storage_path"]=str(ROOT/P.STORAGE_RELATIVE)
analyzer_ref=write("G3_REFERENCE_ANALYZER_PLAN.json",analysis)
source_paths=[D+"/"+s for s in ["run_server10_reference.py","migration_contract.py","test_migration_contract.py",
 "server10_reference_runtime.py","test_server10_reference_runtime.py","prepare_server10_reference.py"]]
source_paths += ["artifacts/prefix_io_v1/server10-sdk-rebind-v1-20261003/rebind_cuda13_cpu.py",
 "artifacts/prefix_io_v1/server10-sdk-rebind-v1-20261003/test_rebind_cuda13_cpu.py",
 sdk_inventory_ref["path"],sdk_proof_ref["path"],P.PERMISSIONS,
 config_ref["path"],model_ref_new["path"],analyzer_ref["path"]]
new_sources=[ref(p) for p in source_paths]
plan=P.build_reference_plan(new_sources,P.INPUT_REF,analyzer_plan_ref=analyzer_ref)
plan_ref=write("G3_REFERENCE_CPU_PLAN.json",plan)
manifest=P.assemble_source_lock(ancestor,new_sources+[plan_ref])
lock_ref=write("gpu-source-lock-server10-reference.json",manifest)
scope=P.scope_template(lock_ref,plan_ref,P.INPUT_REF,analyzer_ref)
human={k:copy.deepcopy(v) for k,v in scope.items() if k not in ("status","human_authorization_record")}
human.update(authorization_origin="direct_human_reply",
 question="根据用户明确的新服务器迁移与GPU系统验证指令，先执行原两次/640秒数值参考包；模型、原异步缓存路径、输入、数值比较、原8小时累计预算和停止规则保持，绑定新机器及真实SDK证明。",
 question_is_scope_summary_not_new_permission_request=True,
 verbatim_user_answer="旧服务器已经关机，已经把代码克隆到新的服务器，接下来验证在新服务器进行gpu实验，完成系统验证",
 credentials_omitted=True,ssh_target="root@connect.westd.seetacloud.com:12351",
 reply_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 observation_timestamp_is_not_user_sent_timestamp=True,
 reply_medium="direct_typed_user_message",question_item_id="not_applicable:direct_typed_user_message",
 form_response_id_fabricated=False,allow_gpu_initialization=True,allow_gpu_runs=True,
 old_reference_jobs_executed=False,fail_any_job_stop_no_retry=True)
human_ref=write("HUMAN_AUTHORIZATION_RECORD_SERVER10_REFERENCE.json",human)
scope.update(status="USER_AUTHORIZED_G3_REFERENCE_DIAGNOSTIC",allow_gpu_initialization=True,allow_gpu_runs=True,
 human_authorization_record=human_ref)
scope_ref=write("SERVER10_REFERENCE_AUTHORIZED_SCOPE.json",scope)
refs={r["path"]:r for r in manifest["files"]}
consistency=P.validate_scope(ROOT,scope,refs)
assert consistency["human_record_matches"] is True and consistency["authorizes_gpu"] is False
ledger=json.loads((ROOT/"experiments/prefix_io_v1/gpu-budget-ledger.json").read_bytes())
assert ledger["active_reservation"] is None
out=dict(status="PASS_SERVER10_FROZEN_MIGRATION_AND_HUMAN_SCOPE_CPU_CONSISTENCY",
 source_lock=lock_ref,source_count=len(refs),new_sources=len(new_sources)+1,scope_ref=scope_ref,
 human_ref=human_ref,CPU_consistency=consistency,full_model_source_verified=False,
 actual_GPU_runs=0,remaining_gpu_seconds=28800-ledger["gpu_wall_seconds"])
write("PREPARATION_RESULT.json",out)
print(json.dumps(out,ensure_ascii=False))

