from pathlib import Path
import json,hashlib,datetime,subprocess,shutil
root=Path(".").resolve();out=Path("artifacts/prefix_io_v1/server08-p3-16")
def rd(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def put(p,v):
 p=Path(p)
 with p.open("x",encoding="utf8") as f:
  if isinstance(v,str):f.write(v)
  else:json.dump(v,f,indent=2,ensure_ascii=False)
 return ref(p)
a=rd(out/"p3-closeout-analysis-final.json");audit=rd(out/"source-preservation-final-p3.json")
assert a["status"]=="P3_EVIDENCE_COMPLETE_FOR_ROOT_REVIEW" and not a["unmet_gates"] and len(a["gates"])==15 and all(g["status"]=="PASS" for g in a["gates"])
assert a["actual_GPU_sessions_audited"]==37 and not a["engineering_negative_flags"]
assert rd("experiments/prefix_io_v1/gpu-budget-ledger.json")["active_reservation"] is None
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,gpu_uuid,used_memory","--format=csv,noheader"],text=True).strip()
assert a["descriptive_arm_medians"]["selected_arm"]=="U" and a["target_waits"]["repeated_pending_flush_target"] is True
status="P3_LIMITED_PILOT_COMPLETE_EFFECT_UNPROVEN"
review=dict(schema_version=1,status=status,p3_phase_closed=True,closure_scope="Original bounded P3 simple-baseline/sparse-conditional development requirements; not P4, full-domain optimization or positive research gain.",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),strict_closeout=ref(out/"p3-closeout-analysis-final.json"),strict_closeout_manifest=ref(out/"p3-closeout-manifest-final.json"),strict_analyzer_phase_flag_preserved_false=True,root_review_resolves_phase_flag=True,gates_passed=15,unmet_gates=[],p316_successful_GPU_stages=37,p316_failed_GPU_attempts=1,p316_total_GPU_attempts=38,full_model_cohorts=20,model_output_exact_tokens=25600,CPU_unique_passed=1613,CPU_skipped=16,CPU_subtests_separate=50,historical_fixture_cases=12,cumulative_GPU_seconds=audit["cumulative_gpu_seconds"],p316_GPU_seconds=audit["p316_gpu_seconds"],remaining_GPU_seconds=audit["remaining_gpu_seconds"],registered_source_bytes_preserved=True,source_proof=ref(out/"source-preservation-final-p3.json"),selected_B="U",research_gain_established=False,ordinary_remaining_target=True,immediate_GPU_release_credit=False,production_interference_lookup_qualified=False,engineering_negative_flags=[],next_allowed_stage="Archive/review P3; a separately authorized P4 minimal dependency CPU design may be proposed. No P4 GPU or online activation here.",P4_enabled=False,P5_P7_enabled=False,actual_edits_this_continuation=["Exact authorized10269-cache archive consolidation preserving paths/content; no source/model target.","Two new v3 capacity plans correcting only original permit file pointer, new SHA freezes; failed attempt retained.","Three selected hot-control plans with B=U; no engine changes.","New lineage/commands/report/closeout/root review/state/evidence archive."],commands_index=ref(out/"p3-reproduction-final.json"),problem_report=ref(out/"PILOT_PROBLEM_REPORT-final-evidence.md"),version_lock=ref(out/"current-environment-version-lock.json"),modification_map=ref(out/"lifecycle-and-modification-map.json"))
r=put(out/"p3-final-root-review.json",review)
md="# P3 已完成：系统路径可运行，研究效果未证明\n\n**状态："+status+"；P3阶段已关闭，剩余P3 GPU实验为0。** 严格验收15项全部PASS、0项未满足，原始analyzer保持P3_EVIDENCE_COMPLETE_FOR_ROOT_REVIEW/p3_phase_closed=false，其任务只核验证据；本root裁决完成阶段关闭，没有重写analyzer或放宽门槛。\n\n"
md+="37次成功GPU stage、1次模型启动前路径错误失败，共38次尝试；其中20次完整模型对照，每次1280输出token精确一致。失败13.852598秒已计费保留。CPU1613去重passed、16 skipped，50 subtests单列，12个历史fixture适用范围保留。GPU累计16520.562887秒（4.589045小时），8小时上限；最终GPU空闲、38个session均消失，3048份注册源逐字节匹配。\n\n"
md+="|候选|两次cohort中位数/秒|相对原路径U时长|\n|---|---:|---:|\n"
for k,v in a["descriptive_arm_medians"]["arms"].items():md+=f'|{k}|{v["median_seconds"]:.6f}|{100*v["relative_duration_vs_U"]:+.4f}%|\n'
md+="\n**当前原路径U最好，B=U；新策略的有效提升未获证明。** 这不等于证明所有可能策略无效，也不授权扩大范围。每臂两次、已见开发域，只支持描述性结果，不提供置信区间、未见集/SLO或全局最优结论。容量960/l1和1024/l2独立报告；GPU-hot U/B/B/U实际为四次U。quiet观测+1.23%不能推广为正常I/O总开销≤2%。条件干扰表与host事件可见性资格不构成production lookup或真实DMA overlap资格。\n\n"
md+="真实普通pending flush目标在两次U中仍存在（0.408055/0.641355秒）。简单基线没有将此目标与整体速度同时改善到形成研究优势；不能据此开启提前GPU回收，unknown和gpu_release_credit=false保持保守。\n\n"
md+="本轮新增只有限定缓存合并、容量资格路径计划修正、剩余实际GPU对照、冻结锁及验收/交付文档。既有共同reactor/staging安全修复与fixed/pressure研究控制分开；关闭研究控制回到原stage签发，原精确Prefix、成本准入、共享staging、预加载、复制合并、异步流水线均保留。作者代码/配套vLLM路线未变，没有驱动/系统更改、新模型下载或预算扩展。\n\n"
md+="实际命令见p3-reproduction-final.json与*-command.json；严格验收使用analyze_p3_closeout_p316_v3.py，最新schema资格30passed及50subtests未重复加总。版本锁、能力矩阵和生命周期/修改清单已保留。完整细节见PILOT_PROBLEM_REPORT-final-evidence.md，严格证据见p3-closeout-analysis-final.json，阶段裁决见p3-final-root-review.json。\n\n"
md+="下一允许动作是P3证据归档与研究决策。可以依据真实剩余等待目标提出P4最小依赖调度CPU设计，但进入新阶段须单独授权和依计划设门槛；本次P4-P7保持关闭，没有自动GPU/生产策略激活。\n"
doc=put(out/"P3_FINAL_ROOT_REVIEW.md",md)
d=put("docs/prefix_io_v1/SERVER08_P3_FINAL_ROOT_REVIEW.md",md)
statep=Path("experiments/prefix_io_v1/execution_state.json")
backup=out/"execution_state-before-final-p3.json"
with backup.open("xb") as f:f.write(statep.read_bytes())
st=rd(statep);st.update(status=status,current_phase="P3_COMPLETE",current_server_blockers=[],gpu_hours_consumed=audit["cumulative_gpu_seconds"]/3600,latest_cpu_passed=1613,latest_cpu_skipped=16,latest_cpu_scope="1583 qualified base plus30 strict metadata schema cases;50subtests separate,12historicalfixtures bounded; no addedGPU lifecycle cases",latest_cpu_test_evidence=str(out/"p3-closeout-schema-cpu-qualification.json"),latest_delivery_report=d["path"],current_remaining_work=[],remaining_work=[],P4_enabled=False)
st["p316"].update(status=status,p3_phase_closed=True,gpu_completed_runs=37,gpu_failed_attempts=1,gpu_total_attempts=38,full_model_cohorts=20,remaining_gpu_runs=0,cpu_unique_passed=1613,cpu_skipped=16,root_review=r["path"],strict_closeout=str(out/"p3-closeout-analysis-final.json"),selected_B="U",research_gain_established=False,p4_enabled=False)
for p in st["phases"]:
 if p["id"]=="P3":
  p.update(status="complete_bounded_pilot_effect_unproven",full_stage_complete=True,qualification_scope=review["closure_scope"])
  for path in [d["path"],r["path"],str(out/"p3-closeout-analysis-final.json"),str(out/"p3-tuning-lineage-final.json"),str(out/"conditional-interference-table.json")]:
   assert Path(path).exists()
   if path not in p["actual_outputs"]:p["actual_outputs"].append(path)
  for path in [str(out/"cpu-qualification.json"),str(out/"p3-closeout-schema-cpu-qualification.json")]:
   if path not in p["test_evidence"]:p["test_evidence"].append(path)
 elif p["id"] in ["P4","P5","P6","P7"]:p["status"]="not_started_requires_separate_phase_authorization"
statep.write_text(json.dumps(st,indent=2,ensure_ascii=False),encoding="utf8")
snap=out/"execution_state-final-p3-snapshot.json"
with snap.open("xb") as f:f.write(statep.read_bytes())
put(out/"execution-state-final-p3-receipt.json",dict(schema_version=1,status="PASS_P3_CLOSED_AFTER_ROOT_REVIEW_HISTORY_PRESERVED",before=ref(backup),after_snapshot=ref(snap),actual_state_at_write=ref(statep),root_review=r,GPU_ledger_unmodified=True,permissions_unmodified=True,P4_enabled=False))
lock=rd(out/"execution-lock-11-selected-hot-plans.json")
lock.update(a["evidence_inputs"])
for p in [out/"p3-closeout-manifest-final.json",out/"p3-closeout-analysis-final.json",out/"p3-closeout-final-command.json",out/"p3-final-root-review.json",out/"P3_FINAL_ROOT_REVIEW.md",Path(d["path"]),snap,out/"execution-state-final-p3-receipt.json",out/"assemble_final_p3_evidence.py",out/"assemble-final-p3-evidence-command.json",out/"final-source-audit-executed.py",out/"p3-reproduction-final.json",out/"REPRODUCTION-final.md",out/"p3-closeout-schema-cpu-qualification.json",out/"closeout-schema-cpu-02.xml",out/"private-cache-merge-postverify.json",out/"private-cache-merge-space-after-apply.json",out/"private-cache-merge-apply-journal.jsonl",Path("experiments/prefix_io_v1/configs/authorizations/server08_p316_archive_dedup.json")]:lock[str(p)]=ref(p)["sha256"]
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in lock.items())
l=put(out/"execution-lock-12-final-p3.json",lock)
source=put(out/"source-lock-final-p3.json",dict(schema_version=1,status="PASS_FINAL_P3_INPUT_AND_DELIVERY_IDENTITIES",execution_lock=l,locked_input_count=len(lock),all_hashes_match=True,input_files=lock,calibration_anchor_preserved=ref(out/"source-lock-at-storage-block.json"),registered_source_proof=ref(out/"source-preservation-final-p3.json"),root_review=r,model_and_compiled_binary_payloads_not_packaged=True))
print(json.dumps(dict(status=status,p3_phase_closed=True,root_review=r,doc=doc,source_lock=source,locked_inputs=len(lock),P4_enabled=False)))
