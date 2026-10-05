from pathlib import Path
import json,hashlib,datetime,copy,statistics
root=Path(".").resolve();out=Path("artifacts/prefix_io_v1/server08-p3-16")
def rd(p):return json.loads(Path(p).read_text())
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def put(name,v):
 p=out/name
 with p.open("x",encoding="utf8") as f:
  if isinstance(v,str):f.write(v)
  else:json.dump(v,f,indent=2,ensure_ascii=False)
 return ref(p)
t=rd(out/"p3-closeout-manifest-at-storage-block.json")
q=rd(out/"p3-closeout-schema-cpu-qualification.json")
assert q["total_unique_passed_without_v2_repeat"]==1613 and q["latest_unique_passed"]==30
t["cpu_receipts"].append(dict(kind="command",receipt=q["command"],guard=q["cuda_guard"]))
t["latest_cpu_count_qualification"]=ref(out/"p3-closeout-schema-cpu-qualification.json")
hist=rd("artifacts/prefix_io_v1/server08-p3-15/tuning-lineage-summary.json")
assert all(ref(p)["sha256"]==r["sha256"] for p,r in hist["source_files"].items())
assert hist["tuning_dimensions"]["io_depth"]["eligible_screened_depths"]==[4,8]
assert hist["tuning_dimensions"]["io_depth"]["depth2_blocked_by_unchanged_cost_gate"] is True
audit=rd(out/"source-preservation-final-p3.json");sel=rd(out/"finite-grid-selection-final.json")
assert audit["p316_success_count"]==37 and audit["p316_attempt_count"]==38 and sel["selected_arm"]=="U"
assert audit["all_file_bytes_match"] and audit["locked_input_bytes_match"] and audit["gpu_active_reservation"] is None
for e in t["capacity_runs"]:
 if e["capacity_domain_id"]=="cap960-l1":
  e["label"]="server08-p3-16-cap960-l1-mixed-U-02"
  for k in ["wrapper","result","config"]:e[k]["path"]=e[k]["path"].replace("cap960-l1-mixed-U-01","cap960-l1-mixed-U-02");e[k]["sha256"]=None
  e["analysis"]=ref(out/"cap960-l1/mixed-U-02-analysis.json")
for e in t["hot_runs"]:
 e["arm"]="U";e["policy_mode"]="off";e["control_config"]=ref(out/"off-control.json")
 e["selection_role"]="B" if e["label"].endswith("-B") else "U";e["resolved_selected_arm"]="U"
def fill(obj):
 if isinstance(obj,dict):
  if set(obj)=={"path","sha256"}:
   actual=ref(obj["path"]);assert obj["sha256"] in [None,actual["sha256"]],obj["path"];obj["sha256"]=actual["sha256"]
  else:
   for v in obj.values():fill(v)
 elif isinstance(obj,list):
  for v in obj:fill(v)
# Refresh only actual run refs first; no old lineage/delivery reclassification.
for key in ["core_runs","finite_runs","capacity_runs","hot_runs","observation_runs","calibration_runs","native_qualifications"]:fill(t[key])
lineage_old=ref("artifacts/prefix_io_v1/server08-p3-15/tuning-lineage-summary.json")
common=[*t["core_runs"],*t["finite_runs"]]
caprefs=[ref(out/d/"capacity-permit.json") for d in ["cap960-l1","cap1024-l2"]]
capresults=[e["analysis"] for e in t["capacity_runs"]]
domain_plan=ref(out/"finite-baseline-domain-plan.json")
selection=ref(out/"finite-grid-selection-final.json")
dims={}
for name in ["batch","iodepth"]:
 dims[name]=dict(evidence_complete=True,disposition="BOUNDED_HISTORICAL_QUALIFICATION_LINEAGE_RETAINED_CURRENT_FIXED_DOMAIN",values={"historical_C":[1,2],"adopted_C":2} if name=="batch" else {"historical_registered_depths":[2,4,8],"qualified_screened_depths":[4,8],"depth2_blocked_by_original_25percent_cost_gate":True,"no_depth2_performance_imputed":True,"adopted_depth":8},producer_refs=[lineage_old,domain_plan,selection],limitations="Authenticated historical qualification and current frozen C2/d8 only; d2 was cost-gate blocked, d4/d8 actually screened on the prior server; no current GPU effect or fresh factorial/global-optimum claim.")
for name in ["cache_capacity","preload_capacity"]:
 dims[name]=dict(evidence_complete=True,disposition="FROZEN_FINITE_CAPACITY_POINTS_AND_BOTH_LOADED_U_SCREENS_COMPLETE",values=dict(kv_budget_bytes=2147483648,common_staging_bytes=1073741824,independent_domains=[{"id":"cap960-l1","staging_mib":960,"horizon":1},{"id":"cap1024-l2","staging_mib":1024,"horizon":2}],structurally_pruned_staging_mib=[512,768,896],required_whole_prefix_files=1016),producer_refs=[domain_plan,*caprefs,*capresults],actual_loaded_entries=t["capacity_runs"],limitations="Six independently qualified points plus one loaded U per domain; 960/l1 and1024/l2 differ in both capacity and lookahead, so not a causal single-knob comparison and never pooled in common winner. Whole prefix unchanged; geometric prunes are not runtime failures.")
for name,arms in [("reserve",["P4","P512"]),("stage_quota",["F16","F8"])]:
 relevant=[e for e in common if e["arm"] in arms]
 controls={e["arm"]:rd(e["control_config"]["path"]) for e in relevant}
 dims[name]=dict(evidence_complete=True,disposition="TEN_FIXED_COMMON_DOMAIN_CANDIDATE_REPEATS_COMPLETE_DESCRIPTIVE_WINNER_U",values=controls,producer_refs=[selection,*[e["analysis"] for e in relevant]],actual_entries=relevant,limitations="Each arm has two real previously seen development runs; exact winner U retained. Deny counts include repeated decisions, not independent parents. No added contention, delay, thresholds or candidate refit; no CI/global optimum.")
lineage=put("p3-tuning-lineage-final.json",dict(schema_version=1,status="PASS_BOUNDED_SIX_DIMENSION_P3_LINEAGE",dimensions=dims,selected_B="U",finite_selection=selection,capacity_domains_not_pooled=True,p4_enabled=False))
for e in t["lineage_receipts"]:e["receipt"]=lineage
commands=[]
budgetp=out/"gpu-budget-final-p3.json";budget=rd(budgetp)
for e in budget["events"]:
 if not e["label"].startswith("server08-p3-16-"):continue
 wp=Path(e["evidence"])/"result.json";w=rd(wp)
 commands.append(dict(label=e["label"],wrapper=ref(wp),gpu_child_argv=e["command"],gpu_seconds=e["elapsed_seconds"],session_id=e["session_id"],exit=w["exit"],child_exit=w["child_exit"],drained=w["session_drained"],timed_out=w["timed_out"],disposition="PASSED_GPU_STAGE" if w["exit"]==w["child_exit"]==0 else "FAILED_BEFORE_MODEL_QUALIFICATION_PATH_ERROR_CHARGED"))
repro=put("p3-reproduction-final.json",dict(schema_version=1,status="PASS_EXECUTED_P3_REPRODUCTION_INDEX",workspace=str(root),GPU_attempts=commands,success_count=37,failure_count=1,full_model_cohorts=20,output_tokens_per_cohort=1280,source_lock=ref(out/"execution-lock-11-selected-hot-plans.json"),version_lock=ref(out/"current-environment-version-lock.json"),CPU_receipts=t["cpu_receipts"],patch_roundtrip=t["patch_roundtrip"],capacity_corrected_bindings=ref(out/"actual-capacity-qualification-binding-preflight.json"),failed_capacity_first_attempt_preserved=True,restrictions=["Use original run_gpu_stage.py guarded wrapper, a new unique label/output, approved UUID idle, source hashes, budget and original storage reserve; never replay bare child argv.","No downloads, driver/system modifications, broadened cache merge, P4-P7 or new engine/executor authorized here."]))
repro_md="# P3 最终复现索引\n\n服务器工作区："+str(root)+"。凭据不入报告。\n\n实际 38 次 GPU stage 命令、各自 wrapper SHA、返回码、session 和预算用时见 p3-reproduction-final.json，其中37次成功、1次在模型启动前因容量资格文件路径错误失败。每次复跑都必须经原 run_gpu_stage.py、唯一新label、当前source冻结、GPU空闲、8小时累计预算和原存储预留校验；不要裸运行 child argv。\n\n本次 CPU 保留1613个去重passed、16 skipped；50 subtests不另加；12个冻结历史fixture只支持其历史协议。资格、XML、CUDA禁止初始化guard及补丁正反往返均由closeout manifest逐SHA绑定。纯元数据路径修正通过actual-capacity-qualification-binding-preflight.json验证，没有运行GPU或新增测试计数。\n\n最终严格元数据验收命令：\n\n"
repro_md+="\x60\x60\x60bash\n.venv/bin/python -I -S experiments/prefix_io_v1/scripts/analyze_p3_closeout_p316_v3.py --root "+str(root)+" --manifest "+str(out/"p3-closeout-manifest-final.json")+" --output "+str(out/"p3-closeout-analysis-final.json")+"\n\x60\x60\x60\n\n这是只读stdlib证据核对，不能代替实际CUDA/AIO资格。最终phase裁决以P3_FINAL_ROOT_REVIEW.md为准；模型权重、3048份共享源payload、私有cache和已安装二进制不打包，其身份及资格分别由注册源SHA、compiled byte identities及GPU收据绑定。P4-P7关闭。\n"
repro_doc=put("REPRODUCTION-final.md",repro_md)
md="# P3 实验与实施最终证据报告\n\n"
md+="全部预注册 P3 GPU 工作已经实际执行。此报告提供验收输入；阶段关闭须在严格closeout与root审查实际通过后记录，最终裁决见P3_FINAL_ROOT_REVIEW.md。固定路线继续在作者py-kvcache和配套vLLM上增量修改；没有重写缓存引擎或模型执行器。\n\n"
md+="共同修改限于reactor.py/staging_cache.py的紧凑容量及pin计数、共享同slot消费者去重，以及STOP/异常时按原CUDA/AIO完成与parent闭包等待排空再回收。研究fixed/pressure策略与共同修复分离，dispatch_controller=None回到原stage签发；精确prefix、成本准入、共享staging、预加载、复制合并和异步流水线均保留。运行代码15个native文件，测试16个，不能将31个均称运行代码。完整位置、版本、生命周期见lifecycle-and-modification-map.json与current-environment-version-lock.json。\n\n"
md+="CPU资格为1613个去重passed、16 skipped；50 subtests单列，12个冻结历史fixture适用范围保留。无CUDA初始化guard、CPU协议矩阵、有限候选/容量、容量分析、补丁正反往返与strict closeout schema资格都有命令/XML/guard。未重复累加先前28个schema测试。此前失败及无guard通过收据保留，不冒充新的生命周期资格。\n\n"
md+=f'真实GPU共38次stage尝试，其中37次成功、1次失败；不是38次模型成功。成功包括2次原生资格、3次条件标定、12次容量点资格及20次完整模型cohort，每个cohort的10请求/1280输出均与full128 golden一致。全部尝试已自然排空、无timeout。P316累计GPU {audit["p316_gpu_seconds"]:.6f} 秒；全程累计 {audit["cumulative_gpu_seconds"]:.6f}秒（{audit["cumulative_gpu_seconds"]/3600:.6f}小时），8小时额度剩余 {audit["remaining_gpu_seconds"]:.6f}秒。实际RTX5090 UUID GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8，driver595.71.05。\n\n'
md+="|共同1GiB候选|独立运行数|cohort含排空中位数/秒|相对U时长变化|\n|---|---:|---:|---:|\n"
base=sel["arms"]["U"]["median_seconds"]
for arm in ["U","F16","P4","F8","P512"]:
 value=sel["arms"][arm]["median_seconds"];md+=f'|{arm}|2|{value:.6f}|{(value/base-1)*100:+.4f}%|\n'
md+="\n按事前最小中位数规则 **B=U**。本冻结开发工作负载没有证明新策略提升；F8/F16/P4/P512中位数均慢于原路径。每臂两次仅作描述性判断，不提供置信区间、正式SLO、未见测试收益或全局最优结论。F8首轮8990次performance deny是重复决策次数，不能称独立parent。普通U重复pending flush为0.408055/0.641355秒，目标存在，不以无等待目标作为止损理由。\n\n"
md+="容量960MiB/horizon1和1GiB/horizon2各六点资格及一轮loaded-U均通过原门槛。其loaded cohort时长分别为 "
md+="/".join(f'{rd(e["analysis"]["path"])["cohort_seconds_including_drain"]:.6f}' for e in t["capacity_runs"])+" 秒；两域不汇入共同候选赢家。容量与horizon同时不同，不能归因于单一参数。512/768/896MiB保持原整段prefix几何剪枝，没有缩短prompt或放宽门槛。\n\n"
md+="cap960第一次尝试因冻结计划错误引用permit.json而非capacity-permit.json，在模型启动前失败；13.852598秒已计入预算，旧失败wrapper/log/v2计划保留。只新增v3计划修正路径，并通过禁CUDA CPU实际门槛/SHA守卫；参数、门槛、预算和reserve不变。960成功使用mixed-U-02，1024成功使用mixed-U-01；不是把旧失败重写成成功或科学剪枝。\n\n"
md+="GPU-hot按原U/B/B/U位置完成；由于B=U，实际四行均U/off/observation-on。每行完整prefix16256、golden128、SSD读0，generation写入按原上限计费，预热写入单列，外部流水线开启。不是一个新策略热缓存效果对照。可选观测off/on/on/off四次中位数差约+1.23%，只隔离quiet optional sink，不能推广为normal-I/O总观测开销≤2%；共同计数两臂均开启，wrapped-thread CPU不等于全模型总CPU。\n\n"
md+="稀疏干扰七个conditional cell在原25%门槛内通过；production状态lookup尚不具备资格，unsupported返回None。真实资源释放继续依赖原Event/AIO/parent完成，unknown保持unknown、gpu_release_credit=false；host事件可见性延后只证明软件保护，不证明真实DMA忙时延、overlap或提前释放收益。\n\n"
md+="限定合并依据人类对前一条具体10269/e692清单动作的“继续”回复，原冻结清单重新验证后完成10269文件合并，2060 canonical SHA、每个target同inode/size、无临时残留、完整intent/verified journal均通过；全部路径内容保留，共享源、模型、代码、日志不作为合并目标。实际净回收9409294336B（约8.76GiB），不是预计9421848576B。授权JSON记载原始人类回复及具体上下文，未扩大其他清单授权。\n\n"
md+=f'最终重新逐字节核对3048份注册源/2796552192B，全部SHA一致、无新增bin。11号冻结锁2608项全部一致；全部38个P316 session不再存活、ledger无active reservation、GPU空闲。PRIMARY空闲{audit["primary_free_bytes"]}B，8GiB底线保留。版本与16个ELF字节身份记录不构成整个安装环境snapshot或独立ABI验证；真实原生资格依GPU收据判断。\n\n'
md+="下一允许动作是审阅P3结论并封存交付。P4-P7保持关闭；没有自动启动P4策略的授权，也不通过新增延迟、改变候选或混合容量域制造收益。\n"
problem=put("PILOT_PROBLEM_REPORT-final-evidence.md",md)
problem_receipt=put("p3-problem-report-final-receipt.json",dict(schema_version=1,status="PASS_ACTUAL_P3_EVIDENCE_REPORT_WRITTEN_PHASE_REVIEW_PENDING",report=problem,source_preservation=ref(out/"source-preservation-final-p3.json"),selection=selection,all_attempts_including_failure_reported=True,history_preserved=True,root_phase_review_pending=True))
producer={"source_lock":[out/"source-lock-at-storage-block.json",out/"source-preservation-final-p3.json",out/"execution-lock-11-selected-hot-plans.json"],"version_lock":[out/"current-environment-version-lock.json",out/"compiled-backend-byte-identities.json"],"GPU_budget_idle":[out/"source-preservation-final-p3.json",budgetp],"storage_and_source_preserved":[out/"source-preservation-final-p3.json",out/"private-cache-merge-postverify.json",out/"private-cache-merge-space-after-apply.json"],"problem_report":[out/"p3-problem-report-final-receipt.json"],"reproduction":[out/"p3-reproduction-final.json",out/"REPRODUCTION-final.md"]}
checks={name:dict(evidence_complete=True,producer_refs=[ref(p) for p in paths],limitations="Frozen bounded P3 evidence; no full installed-binary/model package or formal research gain.") for name,paths in producer.items()}
delivery=put("p3-delivery-evidence-final.json",dict(schema_version=1,status="PASS_BOUNDED_P3_FINAL_DELIVERY_EVIDENCE",checks=checks,p4_enabled=False))
for e in t["delivery_receipts"]:e["receipt"]=delivery
fill(t)
t["checkpoint_scope"]=dict(status="ALL_P3_FROZEN_GPU_WORK_EXECUTED_FINAL_ROOT_REVIEW_PENDING",remaining_gpu_runs=0,selected_B="U",selection_deferred=False,failed_attempts=[ref("experiments/prefix_io_v1/runs/server08-p3-16-cap960-l1-mixed-U-01/result.json")],source_preservation=ref(out/"source-preservation-final-p3.json"),capacity_plan_binding_fix=ref(out/"actual-capacity-qualification-binding-preflight.json"),finite_selection=selection,p4_enabled=False)
manifest=put("p3-closeout-manifest-final.json",t)
print(json.dumps(dict(status="FINAL_CLOSEOUT_INPUTS_CREATED_WITH_ACTUAL_FULL_EVIDENCE",manifest=manifest,lineage=lineage,delivery=delivery,report=problem,selected_B="U",attempts=38,successful_stages=37,remaining_gpu_runs=0)))
