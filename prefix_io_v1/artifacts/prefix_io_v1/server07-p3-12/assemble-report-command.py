from pathlib import Path
import json,hashlib,subprocess,time,sys
out=Path("artifacts/prefix_io_v1/server07-p3-12")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
locked=json.loads((out/"source-lock-before-model.json").read_text())
changes=[p for p,d in locked.items() if sha(p)!=d]
assert not changes,changes
summary=json.loads((out/"gpu-summary.json").read_text())
updates={
"docs/prefix_io_v1/CAPABILITY_MATRIX.md":"P312：有限 32-parent mandatory-store-first 简单排序在独立工作树实现并通过真实 CUDA/AIO 六阶段及四次 Qwen ABBA（40 输出/5120 token 精确）。关闭策略回到原遍历；异常退回并拒绝实验通过。均值快 2.883% 但配对方向相反、一轮开启未触发，不能作为收益。四阶段 native caps/intake 与 joint 未接入；P3 未完成。下一模型 GPU 因固定 3 GiB 存储预留超限阻塞。",
"docs/prefix_io_v1/LIFECYCLE_REPORT.md":"P312：仅在 native store pass 调整既有未发完 parent 的遍历顺序；已接受 D2H→SSD、load/preload/fusion、真实物理条件与完整 parent completion 保留。实际 fence 保护的 3 个目的块 active_refs=1，解除保护不等于新增加空闲 GPU 容量。模型热身先排空，再一次设置可选排序，线程 shutdown 后读取统计；无新队列或常驻 Future 引用。",
"docs/prefix_io_v1/PATCH_MAP.md":"P312：新增 third_party/work/py-kvcache-p3-order-cpu（codex/prefix-p3-mandatory-order）；原 P3/P2/vLLM 工作树未改。baselines/0004-native-mandatory-order.patch 是相对原 P3 已验证 reactor/vllm 的两文件增量；0005-mandatory-order-experiment.patch 增加 12 个项目模块/脚本/测试文件。14 文件 apply/逐字节 roundtrip 通过，共同修复与研究策略分离。完整工作树 diff 及源码锁在本轮证据包；禁将 native 增量直接套到裸 author HEAD。",
"docs/prefix_io_v1/PILOT_PROBLEM_REPORT.md":"P312：固定 off/pressure/pressure/off 四轮 cohort=32.704260/27.200766/29.057094/25.223704 秒；全部 40 完整输出精确。两个 off 无目标等待；首个 on 未排序，第二个 on 重排两次并等待 32.171 ms。平均差 2.883% 不能作为稳定提速证据。计划不变但实际 SSD read 与时序变化。保持 P3 gate，下一步先定义可复现的目标等待资格和停止条件，不能挑样本。"
}
for name,text in updates.items():
 p=Path(name);backup=out/"before"/name
 backup.parent.mkdir(parents=True,exist_ok=True)
 if not backup.exists():backup.write_bytes(p.read_bytes())
 with p.open("a") as f:f.write("\n\n## P312 更新（2026-09-30）\n\n"+text+"\n\n详见 [本轮报告](SERVER07_P3_MANDATORY_ORDER_REPORT.md)。\n")
p=Path("experiments/prefix_io_v1/execution_state.json");state=json.loads(p.read_text())
state.update(status="P3_mandatory_order_functional_ABBA_pass_stable_effect_unproven",
 gpu_hours_consumed=summary["gpu_hours_total"],
 latest_delivery_report="docs/prefix_io_v1/SERVER07_P3_MANDATORY_ORDER_REPORT.md",
 latest_cpu_test_evidence=str(out/"cpu-summary.json"),latest_cpu_passed=860,latest_cpu_skipped=16,
 latest_cpu_scope="Latest-identity union of full matrix and 65 new order/contract/model/analysis tests; no GPU initialization",
 current_server_blockers=[
  "Original io_uring remains EPERM; explicitly authorized Linux AIO is the working route.",
  "P3 mandatory-order baseline functional; stable benefit unproven. Full stage native caps/intake and strongest simple baseline qualification remain incomplete; P4-P7 gated.",
  "Next fixed 3 GiB model reservation exceeds auxiliary storage cap and minimum free floor; prior per-manifest cache grants do not cover new files.",
  "Primary disk only about 10 MiB above 8 GiB floor; bulk evidence stays in authorized auxiliary root."])
phase=next(x for x in state["phases"] if x["id"]=="P3")
phase["status"]="in_progress_mandatory_order_ABBA_functional_effect_unproven"
phase["actual_outputs"] += [state["latest_delivery_report"],str(out/"model-analysis.json")]
phase["test_evidence"] += [str(out/"cpu-summary.json"),str(out/"primitive-result.json"),str(out/"gpu-summary.json")]
phase["full_stage_complete"]=False
state["environment_feasibility"]=dict(status="P3_STORE_ORDER_REAL_MODEL_FUNCTIONAL_PASS_EFFECT_UNPROVEN",
 evidence=str(out/"model-analysis.json"),gpu_budgeted_jobs_this_delivery=5,gpu_workload_runs=5,
 actual_system_changes=[],model_download_bytes_this_delivery=0,
 scope="Six real primitive phases plus fixed four-arm Qwen mixed ABBA; no stable effect or joint-policy claim")
state["p312"]=dict(cpu_summary=str(out/"cpu-summary.json"),gpu_seconds_added=summary["gpu_seconds_added"],
 worktree="third_party/work/py-kvcache-p3-order-cpu",default_mode="off",
 real_model_outputs_exact=40,token_events=5120,store_order_gpu_qualified=True,
 full_stage_caps_installed=False,full_stage_complete=False,research_efficacy_verified=False,
 cohort_descriptive_reduction_percent=2.8830700194188346,activated_pressure_runs=1,
 next_model=summary["next_model"],cache_merge_this_turn=False)
p.write_text(json.dumps(state,indent=2,ensure_ascii=False))
repositories={}
for key,rel in [("project","."),("py-kvcache-p2","third_party/work/py-kvcache-p2-aio"),
 ("py-kvcache-p3","third_party/work/py-kvcache-p3-quota-cpu"),
 ("py-kvcache-p3-order","third_party/work/py-kvcache-p3-order-cpu"),("vllm-author","third_party/work/vllm-author-build")]:
 def git(*a):return subprocess.check_output(["git","-C",rel,*a],text=True).strip()
 repositories[key]=dict(path=str(Path(rel).resolve()),head=git("rev-parse","HEAD"),branch=git("branch","--show-current"),status=git("status","--short"))
version=dict(unix=time.time(),repositories=repositories,gpu_uuid="GPU-f8744916-1693-fa6a-93b6-7f503c03459c",
 driver="580.105.08",python="3.12.3",torch="2.11.0+cu130",report=state["latest_delivery_report"],
 old_native_worktrees_unchanged_this_turn=True,new_native_worktree="third_party/work/py-kvcache-p3-order-cpu",
 source_lock_files=None,source_lock_sha256=None)
(out/"version-lock.json").write_text(json.dumps(version,indent=2))
newfiles=[Path(state["latest_delivery_report"]),*map(Path,updates),Path("experiments/prefix_io_v1/execution_state.json"),
 Path("experiments/prefix_io_v1/gpu-budget-ledger.json"),out/"fence-analysis-command.py",
 Path("experiments/prefix_io_v1/scripts/package_store_order_delivery.py"),
 Path("experiments/prefix_io_v1/scripts/analyze_store_order_model.py"),
 Path("tests/prefix_io_v1_store_order/test_order_analysis.py"),
 Path("patches/prefix_io_v1/baselines/0004-native-mandatory-order.patch"),
 Path("patches/prefix_io_v1/baselines/0005-mandatory-order-experiment.patch")]
final={p:sha(p) for p in locked}
for p in newfiles:final[str(p)]=sha(p)
(out/"source-lock.json").write_text(json.dumps(final,indent=2))
print(json.dumps(dict(source_lock_files=len(final),runtime_inputs_unchanged=True,CPU=860,GPU_runs=5)))
