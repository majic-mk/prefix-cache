import pathlib,json,hashlib,math,sys
root=pathlib.Path(sys.argv[1]);sha="0325cb500f74051dedf6aefe6b51fd9d0b817f831ee45e1c3942df13336b"
sha="0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b"
def read(name):return json.loads((root/name).read_bytes())
def file_ref(name):
 b=(root/name).read_bytes();return dict(path=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
source=read("ACTUAL_RAW_RUN_FILE_REFS.json")
for row in source["files"]:
 f=file_ref("runs/"+row["mode"]+"/"+row["path"])
 assert f["bytes"]==row["bytes"] and f["sha256"]==row["sha256"]
scope=read("GPU_STAGE_AUTHORIZATION.json");human=read("HUMAN_AUTHORIZATION_RECORD_G2.json")
assert len(scope)==19 and scope["source_lock_sha256"]==sha and scope["maximum_jobs"]==2 and scope["maximum_total_planned_reserve_seconds"]==640
assert scope["allow_gpu_runs"] is True and scope["allow_gpu_initialization"] is True
assert type(scope["qualification_context"]["sampling"]["temperature"]) is float
hr=scope["human_authorization_record"];f=file_ref("HUMAN_AUTHORIZATION_RECORD_G2.json")
assert f["bytes"]==hr["bytes"] and f["sha256"]==hr["sha256"]
assert human["authorization_origin"]=="direct_human_reply" and human["source_lock_sha256"]==sha
assert human["question_item_id"]=='["request_user_input_async","call_yaI7LpXW6Zun7uDu8gkf1oQt",0]'
scoperef=file_ref("GPU_STAGE_AUTHORIZATION.json");arrays=[];natives=[];mode_records=[];all_ordinals=[]
guards=[]
for mode in ("off","shadow"):
 guard=read("runs/"+mode+"/result.json");r=read("runs/"+mode+"/details/normal-model-lifecycle-result.json")
 guards.append(guard)
 assert read("ACTUAL_"+mode.upper()+"05_GPU_LAUNCH.json")["result"]["exit"]==0
 assert json.loads(read("ACTUAL_"+mode.upper()+"05_GPU_LAUNCH.json")["result"]["stdout"])==guard
 assert guard["exit"]==guard["child_exit"]==0 and guard["timed_out"] is False and guard["error"] is None
 assert guard["session_drained"] is True and guard["session_members_after_cleanup"]==[]
 assert guard["label"]=="server09-g2-normal-"+mode+"-05" and guard["gpu_job_attempted"] is True
 assert guard["gpu_uuid"]==scope["gpu_uuid"] and guard["elapsed_seconds"]<=320
 assert r["status"]=="PASSED_NORMAL_MODEL_FULL_OUTPUT_ONLY"
 assert r["original_engine_shutdown_returned"] is True and r["source_lock_unchanged_after_original_shutdown"] is True
 assert r["optional_capability_probe"]["restored"] is True and r["optional_capability_probe"]["original_finder_modified"] is False
 assert r["source_lock_sha256"]==sha and r["scope_sha256"]==scoperef["sha256"]
 assert r["framework_import_attempted"] is True and r["gpu_initialization_attempted"] is True and r["actual_guarded_job_count"]==1
 assert r["native_io"]=="none" and r["native_drain"]=="not_applicable"
 for k in ("SSD_qualified","effect_verified","production_qualified","GPU_collector_verified","performance_claim","new_executor"):assert r[k] is False
 assert [p["phase"] for p in r["phases"]]==["cold","repeat"]
 phase_records=[]
 for i,p in enumerate(r["phases"]):
  native=read("runs/"+mode+"/details/"+p["phase"]+"-frontend.json")
  assert native==p["frontend"];f=p["frontend"];tokens=f["output_token_ids"]
  assert type(tokens) is list and len(tokens)==128 and all(type(t) is int for t in tokens)
  assert f["prompt_token_ids"]==list(range(1000,1128))
  assert f["num_cached_tokens"]==[0,112][i] and f["finish_reason"]=="length"
  arrays.append(tokens);natives.append(f["native_request_id"]);ob=p["observation"]
  if mode=="off":
   assert p["installation"]["runtime_hook_status"]=="not_installed" and ob["frames"]==[] and ob["diagnostics"]==[]
  else:
   assert p["installation"]["runtime_hook_status"]=="installed_unqualified"
   frames=ob["frames"];ds=ob["diagnostics"]
   assert ob["valid"] is True and len(frames)==len(ds)==128 and ob["pending_event_pairs"]==0 and ob["open_event_pair"] is False
   reconstructed=[]
   for frame in frames:
    all_ordinals.append(frame["native_step_ordinal"])
    assert frame["gpu_elapsed_ns"] is None and frame["existing_io"] is None and frame["new_io"] is None
    assert frame["start_ns"]<=frame["end_ns"]
    for ident,fragment in frame["outputs"]:
     assert ident==f["native_request_id"] and all(type(v) is int for v in fragment);reconstructed+=fragment
   assert reconstructed==tokens
   assert frames[0]["prepared"]["step_kind"]=="prefill" and frames[0]["prepared"]["prefill_tokens"]==[128,16][i]
   assert all(x["prepared"]["step_kind"]=="decode" and x["prepared"]["active_decode"]==1 for x in frames[1:])
   for d in ds:
    assert d["origin"]=="native_candidate" and d["scope"]=="execute_model_through_sample_tokens_current_stream"
    assert d["gpu_elapsed_ns"] is None and d["mapped_start_ns"] is None and d["mapped_end_ns"] is None
    for k in ("cross_clock_mapping_verified","GPU_collector_verified","production_qualified","performance_claim"):assert d[k] is False
  phase_records.append(dict(phase=p["phase"],output_tokens=128,cached_tokens=f["num_cached_tokens"],frontend_ref=file_ref("runs/"+mode+"/details/"+p["phase"]+"-frontend.json"),frames=len(ob["frames"])))
 mode_records.append(dict(mode=mode,guard_elapsed_seconds=guard["elapsed_seconds"],guard_exit=0,session_drained=True,original_engine_shutdown_returned=True,phases=phase_records))
assert all(v==arrays[0] for v in arrays) and len(set(natives))==4 and all_ordinals==list(range(256))
before=read("gpu-budget-ledger-before.json");afteroff=read("gpu-budget-ledger-after-off.json");final=read("gpu-budget-ledger-after-shadow.json")
assert len(before["events"])==227 and len(afteroff["events"])==228 and len(final["events"])==229
assert afteroff["events"]==before["events"]+[guards[0]] and final["events"]==afteroff["events"]+[guards[1]]
for ledger in (before,afteroff,final):assert ledger["active_reservation"] is None and set(ledger)==set(before)
for ledger in (afteroff,final):
 for k in before:
  if k not in ("gpu_wall_seconds","events","active_reservation"):assert ledger[k]==before[k]
assert math.isclose(afteroff["gpu_wall_seconds"],before["gpu_wall_seconds"]+guards[0]["elapsed_seconds"],rel_tol=0,abs_tol=1e-8)
spent=sum(g["elapsed_seconds"] for g in guards)
assert math.isclose(final["gpu_wall_seconds"],before["gpu_wall_seconds"]+spent,rel_tol=0,abs_tol=1e-8)
post=read("ACTUAL_POST_GPU_COMPLETE_SOURCE_PREFLIGHT.json");assert post["result"]["exit"]==0 and json.loads(post["result"]["stdout"])["source_count"]==4050
raw=read("ACTUAL_SHADOW05_POSTRUN_SNAPSHOT.json");actual=json.loads(raw["result"]["stdout"])
assert file_ref("gpu-budget-ledger-after-shadow.json")["sha256"]==actual["ledger_ref"]["sha256"]
obj=dict(schema_version=1,status="PASS_ACTUAL_G2_OFF05_SHADOW05_FULL_OUTPUT_AND_LIFECYCLE_CLOSURE",
actual_GPU_jobs=2,off_before_shadow=True,retries=0,total_frontend_tokens=512,four_complete_arrays_identical=True,
canonical_output_array_sha256=hashlib.sha256(json.dumps(arrays[0],separators=(",",":")).encode()).hexdigest(),
shadow_frames=256,full_frame_outputs_reconstruct_frontends=True,shadow_pending_pairs=0,shadow_open_pair=False,
source_lock_sha256=sha,post_GPU_source_refs=4050,scope_ref=scoperef,human_ref=file_ref("HUMAN_AUTHORIZATION_RECORD_G2.json"),
modes=mode_records,actual_guard_GPU_wall_seconds=spent,remaining_original_GPU_seconds=28800-final["gpu_wall_seconds"],
ledger_ref=file_ref("gpu-budget-ledger-after-shadow.json"),ledger_events=229,active_reservation=None,
CPU_only_verification=True,new_GPU_jobs_in_this_verifier=0,
normal_model_output_qualified=True,production_KV_IO_qualified=False,SSD_qualified=False,
GPU_cost_or_clock_mapping_qualified=False,running_resource_release_qualified=False,new_strategy_effect_verified=False,
P4_real_strategy_validation_complete=False,P5_performance_experiment_complete=False,performance_claim=False)
with (root/"LOCAL_ACTUAL_G2_RAW_CLOSURE.json").open("xb") as f:f.write((json.dumps(obj,indent=2)+"\n").encode())
print(json.dumps(obj))
