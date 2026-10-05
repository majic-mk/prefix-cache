"""Finalize actual P4-02 CPU receipts and status without enabling GPU."""
from pathlib import Path
import json,hashlib,datetime
ROOT=Path.cwd();OUT=ROOT/'artifacts/prefix_io_v1/server08-p4-02-cpu';ART=OUT.relative_to(ROOT).as_posix()
def load(s):return json.loads((OUT/s).read_text())
def ref(p):return dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
integration=load('current-integration-01/result.json');summary=load('CPU_UNIQUE_TEST_SUMMARY.json')
preserved=load('final-input-and-P3-P4-01-preservation.json');version=load('P4_02_VERSION_LOCK.json')
storage=load('storage-current-final.json');plan=load('gpu-next-day/plan-final.json')
cli=load('gpu-next-day/FINAL_CPU_CLI_RESULTS.json');patch=load('eta/review-integration/patch-roundtrip-06/result.json')
assert integration['status']=='PASS_P4_02_CPU_CURRENT_INTEGRATION'
assert integration['guard_verified'] and integration['ledger_unchanged'] and integration['source_lock_unchanged']
assert integration['new_gpu_runs']==0 and integration['guard']['cuda_initialized'] is False
assert preserved['P3_locked_inputs_unchanged'] and preserved['registered_cache_bytes_unchanged']
assert patch['status'].startswith('PASS_') and cli['ledger_unchanged']
counts=integration['unique_counts'];p4=summary['P4_current_directory_status']
missing=[
 'Install and qualify trusted full decode/sampling/output/full accepted-IO drain collector in the existing author runtime, including host/GPU clock and actual-load binding',
 'Qualify legal real single-stage exact9 cost cells and independent validation uncertainty; coupled D2H-to-write is not two isolated cells',
 'Qualify native ETA context/uncertainty and GPU load producer; prepared semantics/history remain diagnostic-only',
 'Integrate and qualify production table/load/quota/batch activation and actual parent/KV/staging progress, stop and drain',
 'Measure real off/shadow observation overhead and bounded same-base four-arm pilot plus independent U before positive-effect claims']
gpu=dict(new_runs=0,initialized=False,availability_probed=False,availability_report='user_reported_no_GPU',
 budget_reserved_seconds=0,budget_added_seconds=0,total_seconds=plan['budget']['used_seconds'],
 remaining_seconds=plan['budget']['remaining_seconds'],ledger_unchanged=True,downloads_added_bytes=0,
 system_changes=False,cache_merge_or_deletion=False,binary_ABI_qualified=False)
report=f"""# P4-02 无卡阶段实际交付（2026-10-01）

本轮在 connect.westd.seetacloud.com:20739 的 {ROOT} 实际执行，保持 CPU_ONLY。当前可独立用 CPU 闭合的接口准备、语义合同、反例验证与交付工作已完成。真实 GPU 运行0次；完整 P4、生产实现和方法提升均未验证。正常运行时的完整 step/output/drain 采集和生产资格仍未完成。

继续增量修改现成 py-kvcache 与作者 vLLM，保留精确 Prefix Cache、原成本准入、共享 staging、预加载、复制融合、原异步流水线及原 owner/执行器。共同修复、观测、研究建议、证据工具分开。新策略关闭后回到原路径；I/J 未资格仍保持原 U。P3 原有限 pilot 闭合结论保留、U 为原阶段最强独立参照；正向研究优势未证明，P5–P7 未启动。

实际改动位置（项目相对路径）：

| 范围 | 修改位置 | 资格边界 |
|---|---|---|
| 有界 ETA/候选咨询 | third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_eta.py、p4_bridge.py；native reactor.py | 原 whole-parent 成功 drain 才形成历史，无提前释放信用，production ETA=None |
| 原负载消息/启动 | p4_load_observation.py、p4_startup_evidence.py、p4_options.py；native vllm.py；作者 offloading/common.py、scheduler.py、worker.py | scheduled-work-v1 不能冒充 actual GPU decode，原 budget/FD/slot/iodepth 决策保留 |
| 配对语义/离线 CLI | p4_paired_measurement_verifier.py、p4_verified_cost_loader.py、p4_raw_pair_recorder.py；prepare_p4_raw_pair.py | V1/V2 字节/源/阶段/量子/ABBA/拆分绑定，完整轨迹与选择窗口分开，prepared only，production lookup=None |
| 可选观测接口 | p4_gpu_step_observation.py、p4_native_window_journal.py | CPU 注入接点，未安装真实 worker；model_forward 不等于完整 decode step |
| 次日入口 | prepare_p4_gpu_next_day.py、qualify_p4_native_gpu.py、prepare_p4_calibration_startup.py | 纯 CPU 外层门禁在旧唯一 GPU guard 前；当前 --launch 实际拒绝，exit78 |
| 原版本补全 | vllm-author-p4-02-cpu 的92份缺失 Python 源、七个旧 .so 精确字节锁 | 从相同 HEAD 现有 build 复制 Python，未安装/编译/下载；binary 只构造 spec，CPU 未执行 .so，ABI 未验证 |
| CPU 回归/交付 | tests/prefix_io_v1_p4_* 与 scripts 内 run/freeze/audit/storage/version/package_p4_02_* | 实际命令、失败尝试和原始证据保留，重复运行不累计 |

独立审查保留并关闭实际反例：边界 I/O 双计；原生 invalidate 后旧帧有效；host 时间倒退仍归属窗口；完整128输出虚假集中单步。未知状态不授成本/资源释放/生产资格。所有明确纯 decode 步（含非选中步）逐请求最多一token、输出等于 active/batch、batch 不超过完整 run 请求数。unknown/mixed/spec/async/dummy 拒绝；明确 prefill 前置记录只保留完整轨迹，不入选成本窗。native/full-decode 标签本身不证明真实 GPU 来源。

最终统一回归：{counts['passed']}通过、{counts['skipped']}跳过、{counts['failed']}失败，{integration['seconds']:.2f}秒。按唯一(classname,name)计数，当前 P4 目录{p4.get('passed',0)}项通过，单独 agent 重跑及历史12个旧 P3 fixture 不累加。跳过为13项 no-vLLM fallback 专用测试、3项 io_uring_setup errno1。真实 Linux AIO 已测试；CUDA 未初始化、GPU=0、无活动 reservation、进程 session 已完整收敛，最终冻结源与 GPU ledger 前后字节相同。

标量 owner CPU fixture 微基准（200次/轮、7轮）：off 的1/32 parent 中位约0.157/0.167微秒；shadow 的1 parent/1 work约57.296微秒、32 parent/64 work约1.874毫秒。较大 fixture 的成本是明天需要测量的风险；未证明2%开销目标、端到端吞吐/延迟/goodput或方法优势。

保护审计：P3 {preserved['P3_locked_inputs']}锁定输入、原21个 control及原 native、P4-01 {preserved['prior_P4_01_inputs']}项输入保持一致。{preserved['registered_cache_files']}份注册缓存共{preserved['registered_cache_bytes']}字节逐项实际 SHA 保持一致，无额外注册bin。当前 CPU冻结{preserved['P4_02_frozen_inputs']}项、未来GPU来源冻结{preserved['GPU_source_inputs']}项，源锁不授GPU资格。补丁将共同 CUDA UUID/旧flush、92份源补全、观测胶水、研究建议、测量准备分开；P4-01+旧author build 与 P3+author HEAD 两条准确base均实际正逆字节通过，CLI 另有准确旧源补丁证明。

实际版本：project HEAD={version['project_HEAD']}；py-kvcache={version['native_author_HEAD']}；author vLLM={version['author_vllm_HEAD']}。Python3.12.3；安装 metadata Torch={version['package_metadata']['torch']}、pytest={version['package_metadata']['pytest']}、numpy={version['package_metadata']['numpy']}。metadata 查询不加载 GPU backend，不代表 CUDA/ABI 资格。完整 SHA 在 P4_02_VERSION_LOCK.json、test-input-lock.json、gpu-source-lock.json。

空间只读快照：PRIMARY free={storage['PRIMARY']['free_bytes']} bytes，128MiB原生资格预留通过；原模型3GiB预留在8GiB floor下缺{storage['PRIMARY']['model_3GiB_deficit']} bytes（约{storage['PRIMARY']['model_3GiB_deficit']/1024**3:.3f}GiB）。批准 AUX 实际唯一inode占用{storage['AUX']['used_bytes']} bytes；20GiB cap下新3GiB预留缺{storage['AUX']['model_3GiB']['deficit_cap']} bytes（约{storage['AUX']['model_3GiB']['deficit_cap']/1024**3:.3f}GiB），floor亦失败。候选审查只找到7MiB潜在收益、PRIMARY收益0，不足解除模型阻塞；未申请或执行合并。磁盘会变化，明天用实时同一门禁，不任意降低原轮次3GiB预留。

新增GPU秒数/预留均0；账本累计{gpu['total_seconds']:.6f}秒，8小时上限内余{gpu['remaining_seconds']:.6f}秒（约{gpu['remaining_seconds']/3600:.3f}小时）。无GPU探测、模型权重读取、下载、安装、构建、驱动/系统修改、缓存合并/删除/移动。

实际执行主命令见 EXECUTED_COMMANDS.json（含精确命令、返回码和agent记录）。在上述项目 cwd、CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1下执行：

    .venv/bin/python -I -S experiments/prefix_io_v1/scripts/freeze_p4_02_sources.py
    PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/run_p4_02_cpu_qualification.py --name current-integration-01
    .venv/bin/python -I -S experiments/prefix_io_v1/scripts/audit_p4_02_cpu_preservation.py
    .venv/bin/python -I experiments/prefix_io_v1/scripts/record_p4_02_version_lock.py
    .venv/bin/python -I experiments/prefix_io_v1/scripts/check_p4_02_storage.py
    .venv/bin/python -I -S {ART}/EXECUTE_FINAL_CPU_PREVIEWS.py

证据根目录：{OUT}。最终统一 XML、guard、result 在 current-integration-01/；版本/保护/存储在本目录对应JSON。成本最终源及原始CLI证据在 cost-verifier/source-lock-v2-final.json 与 fixtures-v2-cli-01/；独审在 eta/review-cost/v2-independent-02/、v2-cli-readonly-01/，eta/review-integration/observer-counterexamples-03/、nested-binary-spec-01/、patch-roundtrip-06/、FINAL_INDEPENDENT_CLOSEOUT_V2.json。真实5条预览/拒绝命令在 gpu-next-day/FINAL_CPU_CLI_RESULTS.json；空间候选在 storage-next-day/。失败尝试和修复前反例均保留，最终测试数量只来自统一 XML。

明天下一允许阶段：有卡并具备匹配 permissions/批准UUID/源锁/context 的 GPU scope 后，先 G1 新原生off/shadow（各180秒+20秒收敛，共计划400秒、每项128MiB），验证真实来源、binary ABI、CUDA/AIO、完整parent/共享预加载/内容/shutdown。随后在原作者 runtime 补接并资格验证完整step/output/drain、时钟、actual load及合法单阶段成本。自然耦合 D2H→write 不能伪造两种单阶段cell；成本/ETA/load/真实释放安全资格完整后，才可激活研究建议及同一共同固定预算四臂加独立U的小pilot。原模型轮次空间仍阻塞。详细入口与门禁在 GPU_NEXT_DAY_RUNBOOK.md；P4完成及方法提升保持开放。
"""
with (OUT/'P4_02_CPU_DELIVERY_REPORT.md').open('x') as f:f.write(report)
with (ROOT/'docs/prefix_io_v1/SERVER08_P4_CPU_EXTENDED_REPORT.md').open('x') as f:f.write(report)
state_path=ROOT/'experiments/prefix_io_v1/execution_state.json';state=json.loads(state_path.read_text())
with (OUT/'execution-state-before-final-update.json').open('x') as f:f.write(state_path.read_text())
close=dict(schema_version=1,status='P4_02_CURRENT_CPU_PREPARATION_AND_QUALIFICATION_COMPLETE_REAL_GPU_BLOCKED',
 created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),current_permitted_CPU_scope_complete=True,
 full_P4_complete=False,production_implementation_complete=False,positive_method_effect_established=False,
 original_executor_and_cache_retained=True,P3_closed_preserved=True,
 CPU=dict(unique_counts=counts,current_P4_directory_unique_passed=p4.get('passed',0),actual_Linux_AIO_tested=True,repeated_runs_added=False,historical_12_fixtures_rerun=False),
 GPU=gpu,production_missing_work=missing,integration_result=ref(OUT/'current-integration-01/result.json'),
 preservation=ref(OUT/'final-input-and-P3-P4-01-preservation.json'),version_lock=ref(OUT/'P4_02_VERSION_LOCK.json'),
 report=ref(OUT/'P4_02_CPU_DELIVERY_REPORT.md'),next_allowed_stage='Source/context/scope-bound G1 real GPU off/shadow after GPU availability; then existing-runtime collector qualification and bounded P4 calibration/pilot; P5-P7 gated')
names=['P4_02_CPU_DELIVERY_REPORT.md','CAPABILITY_MATRIX_P4_02_CPU.md','LIFECYCLE_AND_PATCH_BOUNDARIES.md','P4_02_VERSION_LOCK.json',
 'current-integration-01/result.json','current-integration-01/cpu.xml','current-integration-01/cuda-guard.json','CPU_UNIQUE_TEST_SUMMARY.json',
 'current-integration-01/cpu-observation-overhead.json','test-input-lock.json','gpu-source-lock.json','final-input-and-P3-P4-01-preservation.json',
 'gpu-next-day/FINAL_CPU_CLI_RESULTS.json','storage-current-final.json','EXECUTED_COMMANDS.json','REPRODUCE_P4_02_CPU.md','GPU_NEXT_DAY_RUNBOOK.md',
 'eta/review-integration/FINAL_INDEPENDENT_CLOSEOUT_V2.json','eta/review-integration/patch-roundtrip-06/result.json','cost-verifier/source-lock-v2-final.json']
close['evidence']=[ref(OUT/s) for s in names]
state['status']=close['status'];state['P4_CPU_enabled']=True;state['P4_GPU_enabled']=False
state['p4_cpu_extended']=close;state['remaining_work']=missing
phase=next(p for p in state['phases'] if p['id']=='P4')
phase['status']='current_CPU_preparation_complete_real_production_GPU_blocked';phase['full_stage_complete']=False
phase['actual_outputs']+=['docs/prefix_io_v1/SERVER08_P4_CPU_EXTENDED_REPORT.md',ART+'/P4_02_CPU_ROOT_CLOSEOUT.json']
phase['test_evidence']+=[ART+'/current-integration-01/result.json',ART+'/eta/review-integration/patch-roundtrip-06/result.json']
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
with (OUT/'P4_02_CPU_ROOT_CLOSEOUT.json').open('x') as f:json.dump(close,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=close['status'],CPU=close['CPU'],new_GPU_runs=0,full_P4_complete=False,report=close['report']),ensure_ascii=False))
