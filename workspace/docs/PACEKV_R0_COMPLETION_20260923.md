# PaceKV 首批无卡准备交付

> 历史交接：本文件记录原 A800/Torch2.2.1 快照，不能用于后续 RTX 5090 启动。5090 的新协议、环境和预检要求见 [5090 租机预检](PACEKV_5090_RENTAL_PREFLIGHT_20260923.md)；旧路径与旧摘要仅供追溯。

完整方案：[R0/R1及后续阶段合同](PACEKV_R0_R1_IMPLEMENTATION_20260923.md)。

## 已完成

- 在旧仓库中新增隔离的`src/pacekv/`与`scripts/pacekv/`，未覆盖旧ProbeKV代码、未提交改动、CacheBlend环境或历史结果。
- 完整prefix链身份、在途所有权、已提交取消保留、GPU/pinned不同释放时点、byte-ns与逐token指标。
- CPU pinned/SSD staged实际KV往返与真实Mistral decode干扰诊断入口；显式GPU启动、限时监督、原始tensor/事件摘要及配对分析。
- 全仓库回归1830项：1829通过，1项本地缺Transformers跳过；服务器新增34项全部通过，其中包含小模型CPU缓存/连续生成接口测试。
- compileall、旧合同validator、`git diff --check`通过。旧合同validator不是新系统GPU资格。
- 本地补装仅`ijson==3.3.0`到已有验证venv；服务器未安装/升级依赖。初次缺accelerate的预检保留，改用不依赖accelerate的CPU-first加载器。
- 全部Mistral权重分片、tokenizer、配置及本批代码做SHA256绑定；本地保留代码ZIP与日志。

## 已部署位置

实例：`connect.bjb1.seetacloud.com:31780`。

```text
/root/autodl-tmp/pacekv/20260922T182149Z-da3746ab6875/
  code/
  preflight/cpu_preflight.json
  preflight/plan.json
  cpu-0.log / cpu-1.log / cpu-2.log
  cpu_execution.json
  deployment.json
```

源码快照SHA256：`da3746ab68754f1fef06697e10bf767831ab9520a669b61daa27aa7fa8a6c965`。

这是包含新增未提交文件的独立快照，不是新的Git commit。基线commit仍为`cc7898b1ba59d89ce7fdbb186ded21880f1adf08`；没有把基线SHA冒充本次新代码。

机器可读交付：[R0_FINAL_HANDOFF.json](../artifacts/pacekv-r0-20260923/R0_FINAL_HANDOFF.json)。

## 仍需明天现场确认

- 无卡容器当前只有2GiB内存，无法加载7B模型；挂卡后要求重新观察至少32GiB主存余量与24GiB空闲GPU显存。CPU代码/资产预检通过不等于资源预检通过。
- 必须为本批明确运行时长；脚本只停任务，不关机、续租或付款。
- A800、CUDA/依赖、GPU占用、全部代码/模型摘要重新核验后才加载模型。
- CPU/SSD往返、真实模型token/logits、GPU异步传输与干扰数据均**尚未执行**。

## 明天只启动R1a

```bash
ROOT=/root/autodl-tmp/pacekv/20260922T182149Z-da3746ab6875
PY=/root/autodl-tmp/probekv_stage1/envs/cacheblend-cu121/bin/python
cd "$ROOT/code"
# GPU_BUDGET_SECONDS须由本批授权确定；结果路径必须是新目录。
: "${GPU_BUDGET_SECONDS:?set the authorized task duration first}"
$PY scripts/pacekv/run_pilot.py --preflight "$ROOT/preflight" \
  --output "$ROOT/gpu-r1a-attempt1" --execute-gpu \
  --max-seconds "$GPU_BUDGET_SECONDS"
# 仅在成功且完整后聚合；失败/超时保留，不覆盖目录。
$PY scripts/pacekv/analyze_pilot.py --input "$ROOT/gpu-r1a-attempt1" \
  --output "$ROOT/gpu-r1a-analysis-attempt1.json"
```

R1a是512/2048-token、四种受控I/O条件的真实模型微诊断，不是自然业务负载、现代vLLM服务吞吐或SCI论文收益证据。96份decode记录中16份是显式warm-up。

## 未完成

现代原生offload后端、真实多请求cache恢复/写回、实际allocator写回保留测量、联合控制器、强基线调优、自然trace净收益和双模型验证。上述内容受R1a/R1b阶段条件约束，不能用当前CPU测试替代。

本批GPU动作=0，正式Profile=false，系统收益未验证，paper_evidence=false。先验证瓶颈与简单策略的不足，再投入完整系统；不保证SCI二区录用。
