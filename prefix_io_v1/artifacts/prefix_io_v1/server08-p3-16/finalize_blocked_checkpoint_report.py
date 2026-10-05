from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET,datetime
out=Path("artifacts/prefix_io_v1/server08-p3-16")
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(n,d):
 p=out/n
 with p.open("x",encoding="utf-8") as f:
  if isinstance(d,str):f.write(d)
  else:json.dump(d,f,indent=2,ensure_ascii=False)
 return ref(p)
rc=out/"private-cache-merge-dryrun-before-grant-command.json";r=json.loads(rc.read_text());assert r["exit"]==0
plan=json.loads(r["stdout"]);assert plan["apply"] is False and plan["target_files"]==10269 and plan["possible_reclaim_bytes"]==9421848576
audit=out/"private-cache-merge-manifest.json";a=json.loads(audit.read_text());allowed=a["completed_cache_roots_allowlist"];assert a["completed_cache_roots"]==len(allowed)==6
assert all(Path(s).is_relative_to(Path(".").resolve()/"experiments/prefix_io_v1/runs") for s in allowed)
for g in a["proposals"]:
 for t in g["targets"]:assert any(Path(t["path"]).is_relative_to(Path(s)) for s in allowed)
target_details=sorted({str(Path(t["path"]).parents[next(i for i,p in enumerate(Path(t["path"]).parents) if p.name=="details")]) for g in a["proposals"] for t in g["targets"]})
auth=Path("experiments/prefix_io_v1/configs/authorizations/server08_p316_archive_dedup.json");journal=out/"private-cache-merge-apply-journal.jsonl";assert not auth.exists() and not journal.exists()
put("private-cache-merge-reviewable-action.json",dict(schema_version=1,status="READONLY_ACTION_READY_FOR_EXACT_HUMAN_GRANT_NOT_APPLIED",action=plan["action"],audit=ref(audit),dryrun_command=ref(rc),original_tool_sha256="e41f5c2300d1b4dd4cf0724521cd95c049dc54dcdc49055442fe4f7383c43c30",target_files=10269,possible_reclaim_bytes=9421848576,possible_reclaim_gib=9421848576/1024**3,selected_completed_cache_roots=allowed,target_completed_details_roots=target_details,target_roots_count=len(target_details),canonical_scope_note="Six audited private roots; actual replacements appear in five, first root supplies canonical copies retained in place.",human_exact_grant_received=False,authorization_exists=False,apply_journal_exists=False,applied=False,previous_grants_cover_this_manifest=False,actual_reclaim_bytes=None,permissions_or_GPU_budget_changed=False,protected="All original paths/content retained. No model, shared source, code or logs replaced.",next_if_granted="GPU/ledger idle, re-audit exact manifest/stat/bytes/sessions, fresh scoped auth, original dryrun/apply with journal, postverify exact10269+canonical SHA/samefile/no debris and realdf, then8 remainingP3."))
xml=out/"closeout-schema-cpu-02.xml";cases=ET.parse(xml).findall(".//testcase");unique={(c.get("classname"),c.get("name")) for c in cases}
assert len(unique)==len(cases)==30 and not any(c.find("failure") is not None or c.find("error") is not None or c.find("skipped") is not None for c in cases)
guard=json.loads((out/"closeout-schema-cpu-02-cuda-guard.json").read_text());assert guard==dict(cuda_initialized=False,gpu_workloads_run=0,pytest_exit=0)
put("p3-closeout-schema-cpu-qualification.json",dict(schema_version=1,status="PASS_CPU_CLOSEOUT_METADATA_SCHEMA_ONLY",latest_unique_passed=30,latest_skipped=0,latest_failed=0,subtests_passed_separately=50,base_unique_passed=1583,total_unique_passed_without_v2_repeat=1613,base_skipped=16,historical_fixture_cases_in_base=12,scope="Only strict report-field/source identity classification; not additional native lifecycle/GPU cases. Latest30 includes28 repeated fromv2 plus2new boundaries, so earlier28not addedagain. JUnit suite tests80 includes50subtests;30 actual testcase nodes/keys.",command=ref(out/"closeout-schema-cpu-02-command.json"),xml=ref(xml),cuda_guard=ref(out/"closeout-schema-cpu-02-cuda-guard.json"),prior_v2_command=ref(out/"closeout-schema-cpu-01-command.json"),analyzer_v3=ref("experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316_v3.py"),tests_v3=ref("tests/prefix_io_v1_p3_closeout_v3/test_closeout_schema.py"),new_GPU_workloads=0))
v3=json.loads((out/"p3-closeout-analysis-at-storage-block-v3.json").read_text());assert v3["status"]=="P3_INCOMPLETE" and v3["p3_phase_closed"] is False and v3["p4_enabled"] is False
required_pass=["core_models","ordinary_wait_and_resource_path","sparse_decode_conditional_calibration","optional_observation_off_on_CPU_and_end_to_end","actual_patch_roundtrip","real_native_transfer_shutdown_qualification","CPU_matrix_and_supplements","source_locks_budget_idle_and_final_delivery"]
g={x["name"]:x for x in v3["gates"]};assert all(g[n]["status"]=="PASS" for n in required_pass)
put("p3-report-schema-correction-receipt.json",dict(schema_version=1,status="REPORT_ADAPTER_SCHEMA_FIXED_GENUINE_STAGE_BLOCK_RETAINED",v1=ref(out/"p3-closeout-analysis-at-storage-block.json"),v2=ref(out/"p3-closeout-analysis-at-storage-block-v2.json"),v3=ref(out/"p3-closeout-analysis-at-storage-block-v3.json"),schema_CPU=ref(out/"p3-closeout-schema-cpu-qualification.json"),original_results_and_frozen_sources_changed=False,prior_errors=["rawCPUextra scope envelope","int1 vsfloat1.0 exactGiB","calibration runtime15 vs global31withtests","one locked prior AUX JSONreference is metadata not runtime source"],current_unmet_gates=v3["unmet_gates"],actual_GPU_sessions_audited_by_v3=28,all_actual_P316_GPU_wrappers_verified_by_ledger=29,count_difference="FirsthotU is held behind incomplete common winner/hot sequence gate; no runtime inferred from report traversal count.",p3_phase_closed=False,p4_enabled=False))
oldreport=(out/"PILOT_PROBLEM_REPORT-at-storage-block.md").read_text()
add="\n补充：最终CPU报告字段修正版v3通过30项守卫测试及50项子检查；先前v2的28项为重复验证，不再次相加。CPU去重总数为1613，仍保留16skip和基数中12个历史fixture的范围限定。v3确认核心模型、真实普通等待、稀疏条件标定、quiet观测、原生transfer/shutdown、CPU及补丁往返门禁通过；未满足项仍来自8次GPU未执行及有限开发未闭合。首次/二次报告适配失败原样保留，不改任何真实GPU结果。\n\n"
add+="本报告表中的P4仅指pressure reserve4Q候选，不代表实施阶段P4获准开启。quiet观察PASS仍受安静域限制，不授予normal-I/O≤2%的工程资格。\n\n"
add+="缓存合并dry-run实际exit0、apply=false、仍10269目标/9421848576B预计回收；六根被审计，目标替换分布在五根，第一根作为canonical保留。没有auth或applyjournal。待精确人类授权。\n\n"
add+="冻结输入快照v2含2599项，ZIP SHA dcefc31e51208f8ff4e91de34389bd032642d20f43946595617506c07c42fa59；其中一份已锁定AUX历史参考用external_aux_inputs映射，仅复制读取，未写AUX。另有16个ELF完整流式SHA身份审计；不打包模型/私有cache/库二进制。\n"
add=add.replace("bd032","bd032")
report=put("PILOT_PROBLEM_REPORT-at-storage-block-final.md",oldreport+add)
put("REPORT_CORRECTIONS-at-storage-block.md",add)
put("p3-blocked-checkpoint-final-status.json",dict(schema_version=1,status="P3_NOT_COMPLETE_STORAGE_AUTHORIZATION_BLOCKED",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),p3_phase_closed=False,CPU_unique_passed=1613,CPU_skip=16,new_closeout_tests_unique=30,new_subtests_separately=50,historical_fixture_cases=12,GPU_P316_completed_runs=29,GPU_cumulative_seconds=15753.810209191404,GPU_remaining_seconds=13046.189790808596,remaining_GPU_runs=8,research_effect="UNPROVEN",next_allowed_phase="P3 remaining finite3 + capacityU2 + GPUhot3 after exact cache grant",p4_to_p7_enabled=False,report=report,analysis_v3=ref(out/"p3-closeout-analysis-at-storage-block-v3.json"),cache_action=ref(out/"private-cache-merge-reviewable-action.json"),source_snapshot=ref(out/"frozen-input-snapshot-at-storage-block-v2.json")))
statep=Path("experiments/prefix_io_v1/execution_state.json");before=statep.read_bytes();backup=out/"execution-state-before-schema-test-update.json"
with backup.open("xb") as f:f.write(before)
state=json.loads(before);state["latest_cpu_passed"]=1613;state["latest_cpu_scope"]="1583base+30latest report-schema tests; earlier28v2 repeated not added;50subtests separate;12historical base fixtures;16skip retained."
state["latest_additional_cpu_tests_p316_schema"]=str(out/"p3-closeout-schema-cpu-qualification.json");state["latest_delivery_report"]=str(out/"PILOT_PROBLEM_REPORT-at-storage-block-final.md")
state["p316"]["latest_report_schema_CPU"]=dict(unique_passed=30,subtests_separate=50,total_unique_passed=1613,CUDA_initialized=False)
statep.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding="utf-8")
put("execution-state-schema-update-receipt.json",dict(status="CURRENT_STATE_P3_INCOMPLETE_LATEST_CPU_COUNTS_UPDATED",before=ref(backup),after=ref(statep),permissions_changed=False,GPU_budget_changed=False,p4_enabled=False))
print(json.dumps(dict(status="P3_NOT_COMPLETE_STORAGE_AUTHORIZATION_BLOCKED",CPU_unique=1613,report=report,cache_action=ref(out/"private-cache-merge-reviewable-action.json"),schema_qualification=ref(out/"p3-closeout-schema-cpu-qualification.json"))))
