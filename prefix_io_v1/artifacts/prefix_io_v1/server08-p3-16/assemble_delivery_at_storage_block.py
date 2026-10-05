from pathlib import Path
import json,hashlib,statistics,datetime,shlex,sys
out=Path("artifacts/prefix_io_v1/server08-p3-16");root=Path(".").resolve()
def rd(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def put(name,data):
 p=out/name
 with p.open("x",encoding="utf-8") as f:
  if isinstance(data,str):f.write(data)
  else:json.dump(data,f,indent=2,ensure_ascii=False)
 return ref(p)
progress=rd(out/"p3-progress-at-storage-block.json");budget=rd("experiments/prefix_io_v1/gpu-budget-ledger.json");assert budget["active_reservation"] is None
lockp=out/"execution-lock-08.json";lock=rd(lockp)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in lock.items())
put("source-lock-at-storage-block.json",dict(schema_version=1,status="PASS_FROZEN_INPUT_BYTE_IDENTITIES_AT_STORAGE_BLOCK",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_lock=ref(lockp),locked_input_count=len(lock),all_hashes_match=True,input_files=lock,registered_source_proof=ref(out/"source-preservation-at-storage-block.json"),version_lock=ref(out/"current-environment-version-lock.json"),compiled_byte_identities=ref(out/"compiled-backend-byte-identities.json"),p3_phase_closed=False,scope="Current evidence/input identity; not fullP3 completion or whole installed-environment snapshot. Binary bytes identified but not included in source snapshot."))
lineage=rd("artifacts/prefix_io_v1/server08-p3-15/tuning-lineage-summary.json")
dims={}
for name in ["batch","iodepth","cache_capacity","preload_capacity","reserve","stage_quota"]:
 complete=name in ["batch","iodepth"]
 refs=[ref("artifacts/prefix_io_v1/server08-p3-15/tuning-lineage-summary.json"),ref(out/"finite-grid-preregistration.json"),ref(out/"finite-baseline-domain-plan.json")]
 if name in ["cache_capacity","preload_capacity"]:refs += [ref(out/d/"capacity-permit.json") for d in ["cap960-l1","cap1024-l2"]]
 if name in ["reserve","stage_quota"]:refs += [ref(out/(n+"-analysis.json")) for n in ["mixed-03-pressure","mixed-04-pressure","mixed-07-fixed8"]]
 dims[name]=dict(evidence_complete=complete,disposition="BOUNDED_HISTORICAL_QUALIFICATION_LINEAGE_AUDITED_NOT_NEW_GPU_EFFECT" if complete else "UNFINISHED_FROZEN_DEVELOPMENT_STORAGE_BLOCKED",source_refs=refs,limitations="Original C1/C2 and d2/d4/d8 lineage is source/hardware bounded, not a full factorial/global-optimum or newly measured GPU comparison." if complete else "Real independent capacity point gates passed but loaded screens not run; reserve/quota candidates lack the required repeat(s). Resource blocking is not scientific pruning.")
put("p3-tuning-lineage-at-storage-block.json",dict(schema_version=1,status="P3_FINITE_TUNING_INCOMPLETE",dimensions=dims,ten_run_winner=None,remaining_gpu_runs=progress["remaining_gpu_runs"],p4_enabled=False))
events=[e for e in budget["events"] if e["label"].startswith("server08-p3-16-")]
commands=[]
for e in events:
 wrapper=Path(e["evidence"])/"result.json";w=rd(wrapper);assert w["session_drained"] and w["exit"]==0 and w["child_exit"]==0
 commands.append(dict(label=e["label"],wrapper=ref(wrapper),exact_gpu_child_argv=e["command"],seconds=e["elapsed_seconds"],gpu_uuid=e["gpu_uuid"],scope="Executed child argv. Repeat only using original guarded run_gpu_stage plan, a new unique run label/output, storage preflight, current source freeze and remaining budget; do not replay this bare child argv on GPU."))
cpu_commands=[]
for n in ["cpu-qualification.json","finite-capacity-cpu-qualification.json","additional-closeout-cpu-qualification.json"]:
 cpu_commands.append(ref(out/n))
repro=dict(schema_version=1,status="CURRENT_EXECUTED_EVIDENCE_REPRODUCTION_INDEX_NOT_PHASE_COMPLETION",workspace=str(root),source_lock=ref(out/"source-lock-at-storage-block.json"),version_lock=ref(out/"current-environment-version-lock.json"),compiled_binary_identities=ref(out/"compiled-backend-byte-identities.json"),gpu_runs=commands,cpu_qualifications=cpu_commands,remaining_gpu_runs_not_executed=progress["remaining_gpu_runs"],source_reference_full128="experiments/prefix_io_v1/runs/server08-p3-13-native-fullref-01/details/result.json",restrictions=["No current GPU execution without original wrapper guards.","Private cache merge is pending an exact new human grant; no authorization file created or apply performed.","Both capacity models use their v2 plan and independent passed permit; single-token cost references are not full model golden.","Prospective B is unselected until all ten common-domain runs pass; remaining hot-B plans are not generated.","P4-P7 remain disabled."])
put("p3-reproduction-at-storage-block.json",repro)
md="# P3 当前实施交付：存储授权阻塞，阶段未完成\n\n"
md+="当前状态是 **P3_NOT_COMPLETE_STORAGE_AUTHORIZATION_BLOCKED**。在新服务器 20739 上，作者原生 CUDA/LinuxAIO 和完整模型已实际运行成功。当前阻塞是磁盘资源预留检查；不是没有 GPU，也不是已证明方案不可行。P4–P7 保持关闭。\n\n"
md+="本轮新增共同修复限于 reactor.py 与 staging_cache.py：紧凑 staging/pin 计数、共享同一 slot 的消费者去重，以及 STOP/异常时等待原 CUDA/AIO 排空证据再回收。fixed/pressure 控制仍使用已有 simple_stage_policy，不新增缓存引擎、模型执行器、joint scheduler 或第二释放队列。原精确 prefix、成本准入、共享 staging、预加载、复制合并和异步流水线保留。dispatch_controller=None 直接回到原 stage 签发；共同安全修复保留。修改函数及行号见 lifecycle-and-modification-map.json。\n\n"
md+="CPU 去重资格：1583 passed、16 skipped，包含 12 个冻结历史协议 fixture；不能把它们都称为当前生命周期测试。guarded 执行矩阵、finite/capacity、补丁往返和容量分析已完成；此前失败和一次无 CUDA guard 的通过收据全部保留。旧失败：首轮 compact fixture 不合法、旧裸构造 fixture 缺 _stop、旧大 registry fixture 缺新增计数导致保守 unknown；修复及新资格没有回写旧结果。补丁往返覆盖共同增量和研究 patch 的正反应用、七阶段 31 文件身份及 CPU common-only/off，不替代 GPU 运行时回退证明。\n\n"
md+="本轮真实 GPU 共 29 次，全部 wrapper/child exit0、无超时、会话已排空。本轮用时 2653.93 秒；全程累计 15753.81 秒（4.376 小时），8 小时上限余 13046.19 秒。GPU UUID 为 GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8，RTX5090，实际驱动595.71.05。\n\n"
md+="已完成 16-case 原生简单阶段资格、80-member registry 的两类真实共享/staging资格、两轮 decode标定及独立验证（各28 windows）、六次 U/F16/P4 重复、F8首轮、两个容量域各六阶段点资格、首轮GPU-hot U以及四次观测开关对照。case/window不是额外GPU run；旧P315的失败all-hit定义及未执行尾部保留。\n\n"
md+="|共同 1GiB 候选|已完成次数|cohort含排空的中位数/秒|解释|\n|---|---:|---:|---|\n"
for a in ["U","F16","P4"]:md+=f'|{a}|2|{progress["core_medians_seconds"][a]:.6f}|冻结的正常开发流；描述性，无置信区间|\n'
md+="|F8|1|32.931849（单次）|缺重复，不能选赢家|\n|P512|0|未运行|存储阻塞|\n\n"
md+="F16、P4 的两次中位数均慢于 U；尚未证明方法提升。F8真实出现8990次 performance deny decision，包含重复检查，不能称8990个独立parent，也不能继续称全部额度均未binding。普通 U 两次真实 pending flush 为0.408055/0.641355秒；目标存在，不能以“没有普通等待目标”作为止损理由。\n\n"
md+="两个容量域 cap960-l1 与 cap1024-l2 已分别通过原数值/介质/成本门槛；各自 loaded-U 敏感性尚未执行，不能汇入共同候选赢家或称容量调优完成。512/768/896MiB 因整段prefix所需1016文件与可用slot上限不相容而结构剪枝；未改变prefix长度或放宽门槛。容量v1计划曾被误改元数据，已恢复原SHA并保留事件；后续只用新增v2/full128参考。\n\n"
md+="稀疏干扰七个条件cell已通过原25%条件（验证loaded误差、重复spread、首尾drift）；该表仅支持owned/conditional校准域，production-state lookup未完成、失配返回None。没有正式收益、真实DMA overlap或完整状态泛化结论。\n\n"
q=progress["quiet_observer_summary"];md+=f'GPU-hot/观测4次对照测量SSD读写均为0，原流水线开启，warmup写入单独计入。optional观测on/off的cohort中位数 {q["on_median_seconds"]:.6f}/{q["off_median_seconds"]:.6f}秒，描述性差值+{q["relative_delta_percent"]:.4f}%。这只隔离安静域可选sink，不能推断normal-I/O总观测成本≤2%；共同parent/capacity计数两臂均开启，thread CPU窗口也不等于model总CPU或墙时窗口。\n\n'
md+="资源释放能力保持保守：CUDA原事件query/同步、AIO完成及完整parent闭包才可回收；原事件没有证明完成时保留owners/Futures并返回unknown。诊断snapshot明确gpu_release_credit=false；旧立即GPU复用UNKNOWN仍为UNKNOWN。host-hold event facade实际调用原query但可延后可见true，只证明软件等待可见证据，不证明真实DMA忙时延或性能改善。\n\n"
md+="当前PRIMARY空闲11784204288B，下一模型固定预留3221225472B，扣除后小于8589934592B底线，原preflight拒绝。仍缺3次finite（P512/P512/F8）、2次capacity loaded-U、3次GPU-hot B/B/U，共8次。10269个旧P315私有重复缓存清单SHA e692ad4495548d9729423d956ceea405bbcac4202d5b9e50bee2db4ac424e062，预计可回收9421848576B，仅是预计值。该清单没有收到新的精确人类grant，未创建授权或合并；旧8226/8231清单授权不能复用。共享源、模型、代码、日志不在目标内。\n\n"
md+="已重新逐字节核对3048份注册源，共2796552192B，全部SHA匹配、无新增bin；冻结08的2599输入全部匹配。实际版本metadata/作者HEAD/16个ELF字节身份已记录，ELF只读hash不构成独立ABI验证或整个安装环境快照。完整原GPU资格依各wrapper/result判断。\n\n"
md+="实际执行命令均可查GPU result.json、out316/*-command.json、CPU XML与 p3-reproduction-at-storage-block.json。GPU必须经原run_gpu_stage.py、唯一label、同UUID空闲/source冻结/预算/磁盘预留校验；不要裸执行child argv。新closeout文件是本截点报告，不能替代未来完整P3验收。\n\n"
md+="下一允许阶段仍是 **P3剩余8次对照**：收到精确缓存grant后，在GPU完全idle时重审此冻结清单、dry-run、应用并保存10269个verified及完整journal，再逐字节核对所有路径/内容并记录真实df回收量；随后继续原冻结候选。未获grant不降低3GiB预留、不删除缓存、不启用P4，也不写P3完成。\n"
report=put("PILOT_PROBLEM_REPORT-at-storage-block.md",md)
doc=Path("docs/prefix_io_v1/SERVER08_P3_P316_STORAGE_BLOCK_REPORT.md")
with doc.open("x",encoding="utf-8") as f:f.write(md)
put("p3-problem-report-at-storage-block-receipt.json",dict(schema_version=1,status="CURRENT_BLOCKED_PROGRESS_REPORT_WRITTEN",p3_phase_closed=False,report=report,doc=ref(doc),progress=ref(out/"p3-progress-at-storage-block.json"),independent_gate_review=ref(out/"p3-gate-review-at-storage-block.json"),historical_reports_preserved=True))
cap="# 当前能力矩阵与版本边界\n\n"
cap+="|能力|实际证据/状态|限制|\n|---|---|---|\n"
for row in [("精确Prefix Cache/完整模型","7次正常模型+5次GPU-hot各1280输出通过full128 golden","P3开发域，不是未见测试/SLO"),("成本准入","原1GiB permit+两个独立容量point门槛通过","新容量loaded-U未跑"),("共享staging/复制合并/异步流水线","原API与实际AIO/CUDA/完整排空保留，80成员共享资格通过","host query gate不能当真实DMA时延"),("preload","lookahead1在实际模型路径保留；lookahead2 point资格通过","lookahead2 loaded效果未验证"),("fixed/pressure控制","核心重复已完成，pressure普通registry已知；F8首轮真实额度deny","F8重复及P512两次未跑，无有限域赢家"),("资源释放依赖","原事件/AIO/parent边界释放；未知保守保留","没有新的GPU source-safe ACK/提前复用协议"),("稀疏干扰额度","七cell conditional table原阈值通过","未获生产状态资格，unsupported返回None"),("观测","真实thread CPU包裹+quiet开关4次","normal-I/O optional开销未隔离；共同计数常开"),("有限候选调度","P3简单策略有限候选冻结、部分已跑","P4 dependency/joint策略未开启"),("版本锁","214metadata+作者HEAD+16ELF SHA；2599source/input身份","不含模型/库二进制本体或完整安装环境")]:cap+="|"+"|".join(row)+"|\n"
put("CAPABILITY_MATRIX-at-storage-block.md",cap)
readme="# 本截点复现说明\n\n当前P3未完成；不要把证据文件的存在当stage验收。\n\n"
readme+="工作区：/root/autodl-tmp/prefix-io-v1-handoff/project；SSH当前20739，凭据不写入交付包。\n\n"
readme+="每个已执行GPU命令由 p3-reproduction-at-storage-block.json 精确索引，29个原wrapper保留出入时刻、elapsed、exit、drain与命令。对应*-plan.json是唯一可供guarded runner复现的计划；新复跑必须新label/output重新冻结与预留。\n\n"
readme+="CPU完整矩阵用run_aio_cpu_tests.py禁CUDA guard，见master qualification所列4processes与XML；finite/capacity03、patch-roundtrip guards、capacity analysis各命令收据完整保留。单次无guard通过不单独作为guarded资格。不要把旧历史fixture作用域扩大到当前生命周期。\n\n"
readme+="本次closeout analyzer只做纯stdlib元数据核对：.venv/bin/python -I -S experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --manifest artifacts/prefix_io_v1/server08-p3-16/p3-closeout-manifest-at-storage-block.json --output artifacts/prefix_io_v1/server08-p3-16/p3-closeout-analysis-at-storage-block.json。预期P3_INCOMPLETE，缺8次GPU且lineage未闭合；分析器status不自动授予P4。\n\n"
readme+="注册source3048份、model权重/私有cache payload及16个库二进制不打包；源与compiled身份通过SHA绑定现服务器。证据包附精确文件manifest，解压后验证每项SHA。source snapshot保留08所有冻结输入实际内容及alias映射；尚未是完整P3最终snapshot。\n"
put("REPRODUCTION-at-storage-block.md",readme)
checks={}
producer={"source_lock":[out/"source-lock-at-storage-block.json"],"version_lock":[out/"current-environment-version-lock.json",out/"compiled-backend-byte-identities.json"],"GPU_budget_idle":[out/"post-mixed07-storage-gate.json",out/"source-preservation-at-storage-block.json"],"storage_and_source_preserved":[out/"source-preservation-at-storage-block.json",out/"post-mixed07-storage-gate.json"],"problem_report":[out/"p3-problem-report-at-storage-block-receipt.json"],"reproduction":[out/"p3-reproduction-at-storage-block.json",out/"REPRODUCTION-at-storage-block.md"]}
for name,paths in producer.items():checks[name]=dict(evidence_complete=True,producer_refs=[ref(p) for p in paths],limitations="Complete handoff evidence for this blocked checkpoint only; does not claim P3 completion, missing runs or whole binary/model packaging.")
put("p3-delivery-evidence-at-storage-block.json",dict(schema_version=1,status="BLOCKED_CHECKPOINT_DELIVERY_EVIDENCE",p3_phase_closed=False,checks=checks))
t=rd(out/"p3-closeout-manifest-template.json");t.pop("_template")
for entry in t["capacity_runs"]:
 domain=entry["capacity_domain_id"];entry["analysis"]["path"]=str(out/domain/"mixed-U-01-analysis.json")
for entry in t["capacity_gates"]:entry["disposition"]="PASS"
for entry in t["calibration_runs"]+t["native_qualifications"]:
 entry["required_status"]=rd(entry["result"]["path"])["status"]
for entry in t["lineage_receipts"]:entry["receipt"]["path"]=str(out/"p3-tuning-lineage-at-storage-block.json")
for entry in t["delivery_receipts"]:entry["receipt"]["path"]=str(out/"p3-delivery-evidence-at-storage-block.json")
for p in t["native_source_hashes"]:t["native_source_hashes"][p]=lock[p]
missing=[]
def fill(obj):
 if isinstance(obj,dict):
  if set(obj)=={"path","sha256"}:
   p=Path(obj["path"])
   if p.exists():obj["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
   else:missing.append(str(p))
  else:
   for v in obj.values():fill(v)
 elif isinstance(obj,list):
  for v in obj:fill(v)
fill(t)
t["checkpoint_scope"]=dict(status=progress["status"],p3_phase_closed=False,missing_files=sorted(set(missing)),selected_B=None,selection_deferred=True,not_run_status="UNEXECUTED_STORAGE_AUTHORIZATION_BLOCKED")
put("p3-closeout-manifest-at-storage-block.json",t)
statep=Path("experiments/prefix_io_v1/execution_state.json");oldbytes=statep.read_bytes();backup=out/"execution_state-before-p316-closeout.json"
with backup.open("xb") as f:f.write(oldbytes)
state=json.loads(oldbytes);state.update(status=progress["status"],current_phase="P3",current_server="connect.westd.seetacloud.com:20739",current_server_blockers=["Primary fixed3GiB GPU reservation would cross8GiB floor; exact10269-cache grant pending"],gpu_hours_consumed=budget["gpu_wall_seconds"]/3600,latest_cpu_passed=1583,latest_cpu_skipped=16,latest_cpu_scope=progress["cpu_scope"],latest_cpu_test_evidence=str(out/"cpu-qualification.json"),latest_delivery_report=str(doc),current_remaining_work=progress["remaining_gpu_runs"],remaining_work=progress["remaining_gpu_runs"],p316=dict(status=progress["status"],evidence=str(out),p3_phase_closed=False,gpu_completed_runs=29,remaining_gpu_runs=8,cpu_unique_passed=1583,cpu_skipped=16,historical_fixture_cases=12,original_execution_state_backup=str(backup),source_lock=str(out/"source-lock-at-storage-block.json")))
for phase in state["phases"]:
 if phase["id"]=="P3":
  phase["status"]="incomplete_storage_authorization_blocked";phase["qualification_scope"]="P316 compact/physical/common controls, conditional sparse calibration, finite development not yet complete"
  if str(doc) not in phase["actual_outputs"]:phase["actual_outputs"].append(str(doc))
 elif phase["id"] in ["P4","P5","P6","P7"]:phase["status"]="not_started_dependency_gate_blocked"
statep.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding="utf-8")
put("execution-state-update-p316-receipt.json",dict(schema_version=1,status="UPDATED_CURRENT_STATE_P3_INCOMPLETE_HISTORY_PRESERVED",before=ref(backup),after=ref(statep),GPU_budget_ledger_unmodified=True,permissions_unmodified=True,p4_enabled=False))
print(json.dumps(dict(status=progress["status"],report=report,manifest=ref(out/"p3-closeout-manifest-at-storage-block.json"),missing_files=len(set(missing)),execution_state=ref(statep),new_gpu_workloads_run=0),ensure_ascii=False))
