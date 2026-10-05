# Prefix I/O V1 本次交付

本次完成服务器 P0 审计、P1 可执行的 CPU 修复和测试，并准备了尚未接入真实引擎的有界观测／释放分析组件。P1 的 GPU／完整集成验收仍为 BLOCKED；没有跳过门禁宣称完成 P2–P7。

服务器目录：`/root/autodl-tmp/prefix-io-v1-handoff/project`。
项目基线：`cc7898b1ba59d89ce7fdbb186ded21880f1adf08`；分支 `codex/prefix-io-v1`。
本地旧研究代码和未提交工作保留；没有合入 Source、repair、Prefix shadow 或模型执行器。

## 核心审计结果

- 固定 py-kvcache 为 `3abba7a502d553f6e7e2e58b92086487e3395d7e`。
- 交接候选 vLLM preload@`d6eadf416bb5234047760bf55d532f2f038cf697` 缺少 Plan API；选定作者已有 with_profiling@`817a7e3124f817cd6e549581d3e5483207a753a4`。真实源码及调用点已核验，但没有认证可用 GPU 安装组合。
- store 的成功父任务完成依赖全部文件完成；D2H 结束不能提前 complete_store。已结束请求的块可进入 free list，覆盖前仍可能需要 jobs_to_flush；活跃引用和多父任务保护必须分别考虑。
- O_DIRECT 独立 CPU 探针完成 4096 字节一致读写。真实 io_uring_setup 返回 EPERM。
- 当前容器无 /dev/nvidia*，cgroup 内存上限为 2 GiB。驱动／Torch 版本存在不代表 GPU 可用。
- 服务端 GitHub 网络失败后，通过本地转运固定作者源码。实施和测试在服务器完成。

## 实际改动

1. `patches/prefix_io_v1/common/`：分离 CPU Torch 与 GPU-copy 能力判断；staging 预算计入保留的对齐 backing，超最小槽预算时在分配前拒绝，并记录实际 storage 字节。原 planner 使用同一容量公式；一项容量断言由 112 改为 111。
2. `src/prefix_io_control/config.py / preflight.py`：严格类型、未知字段、重复 YAML key、null、权限和模式门禁检查。不会从 SSH 授权推导 GPU 权限。
3. `observation.py / dependencies.py`：只允许 owner 线程进行有界观察；缺失值保持未知；按 run/generation、活跃引用、全部父任务闭包和失败排空状态分析释放。当前是 CPU 准备组件，尚无真实 GPU 资源适配。
4. `bridge.py`：独立 CPU 边界测试中 off 直接调用原路径，shadow 观察失败不改变原动作。没有接入真实 reactor；其余模式拒绝激活。
5. 审计、版本锁、能力矩阵、生命周期、逐项测试覆盖和复现脚本已落盘。

尚未实现或验收：真实 mandatory bridge、调度器资源 generation 桥接、普通额度、有限候选在线调度、干扰表、joint 策略与 GPU 实验。这些依赖后续阶段，不以 CPU fixture 冒充完成。

## 执行命令与结果

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project

# P0: 固定源码、真实导入能力及受限 CPU 存储探针
.venv-prefix/bin/python experiments/prefix_io_v1/scripts/audit.py

# 原版：258 passed / 2 failed / 9 skipped（原始失败保留）
PYTHONPATH=third_party/upstream/py-kvcache RUN_E2E_TESTS= \
  .venv-prefix/bin/python -m pytest -q -ra third_party/upstream/py-kvcache/tests \
  --junitxml=artifacts/prefix_io_v1/p0/upstream-baseline.xml

# 最终：328 passed / 0 failed / 8 skipped
PYTHONPATH=src:third_party/work/py-kvcache RUN_E2E_TESTS= \
  .venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1 third_party/work/py-kvcache/tests \
  --junitxml=artifacts/prefix_io_v1/p1/cpu-verified.xml

# 可再次运行，使用独立输出目录，不覆盖本轮证据
bash experiments/prefix_io_v1/scripts/reproduce_cpu.sh
```

本轮所有远端命令由执行封装设置 `CUDA_VISIBLE_DEVICES=""`、`PYTHONDONTWRITEBYTECODE=1`。328 项通过由 261 项作者测试和 67 项项目测试组成。8 项跳过是 5 项 GPU E2E、3 项 io_uring 环境受限测试，绝非通过。

原成本规划、身份／文件映射、调度循环、原 D2H→SSD 续接、复制合并及父任务完成函数保留；逐函数证据见 `native-path-preservation.json`。共同补丁适用于所有未来实验臂，研究路径未启用。

## 证据位置

- `docs/prefix_io_v1/WORKSPACE_AUDIT.md`
- `docs/prefix_io_v1/CAPABILITY_MATRIX.md`
- `docs/prefix_io_v1/PATCH_MAP.md`
- `docs/prefix_io_v1/LIFECYCLE_REPORT.md`
- `experiments/prefix_io_v1/locks/dependency-lock.json`
- `experiments/prefix_io_v1/locks/environment.json`
- `experiments/prefix_io_v1/execution_state.json`
- `artifacts/prefix_io_v1/p0/source-symbols.json`
- `artifacts/prefix_io_v1/p0/capability-report.json`
- `artifacts/prefix_io_v1/p1/cpu-verified.txt / .xml`
- `artifacts/prefix_io_v1/p1/preflight.json`
- `artifacts/prefix_io_v1/p1/native-path-preservation.json`
- `artifacts/prefix_io_v1/command-log.jsonl`（精确命令、退出码和原始输出）

## GPU、预算与下一允许阶段

真实 GPU 运行：**无**。GPU 消耗：**0 小时**。模型下载：**0 GiB**。
permissions.yaml 未扩大；没有租机、付款、改驱动／系统策略、清理共享数据或远端推送。

下一阶段仍是 **P1 原版真实环境／GPU smoke 验收**，前提是明确 GPU 授权和预算、可用设备、io_uring 权限、足够内存及真实依赖安装。随后才允许 P2 mandatory/off-shadow 验证、P3 标定和 pilot、P4 研究调度。当前不运行 GPU，不宣称性能收益、新颖性或论文结论。
