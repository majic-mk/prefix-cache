# 继续交付 01：mandatory 进展桥接的 CPU 预集成

本轮所有代码修改和测试均在服务器 /root/autodl-tmp/prefix-io-v1-handoff/project 执行。P0 审计完成；P1 的 GPU/真实集成门禁仍阻塞。P2 仅增加可安全执行的 CPU 预备证据，没有宣称阶段验收通过；P3–P7 未启动。

## 实际修改与边界

新增隔离工作树 third_party/work/py-kvcache-progress，分支 codex/prefix-io-v1-progress-cpu；基于锁定 py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e，先应用上次两项共同修复，再应用独立 observer/0001-mandatory-progress-bridge.patch。原 common 工作树及作者 vLLM 源码未修改。共同修复、桥接增量和研究策略继续分开，policy_patches 仍为空。

- 在现成 IoReactor/TransferCoordinator 增加可选 progress_run_id，默认 None。正常构造调用没有启用参数，生产桥接仍关闭。
- 现成 handler.wait 在阻塞 Future 前发送标记，通过原有 incoming 队列进入 reactor。标记绑定 reactor 实例 token 与实际 parent Future，不以可复用的整数 job_id 充当身份。
- reactor 只为已接受且尚未排空的父任务记录 mandatory 信号；STOP 为全部已接受父任务标记排空，避免依赖后续 scheduler epoch。
- 父 Future 失败不代表其他已接受 I/O 排空。信号只在原生排空判定后清除；致命线程失败仅清除新增信号引用，不增加资源释放或恢复行为。
- 新增 23 项 CPU/mock 测试，运行实际 handler、coordinator 和 reactor 事件循环。CUDA 事件、文件后端、ring 由夹具模拟；零普通额度由测试专用 _schedule_one 钩子提供，生产代码没有引入额度控制。
- 已有紧凑 observer 在真实 reactor 线程、模拟 I/O 上取得不可变快照；GPU 可立即复用字节和未知阶段深度仍为 None。observer 没有自动安装进生产热路径。

没有重写缓存引擎或模型执行器，没有改变 LoadPlanner、Prefix 身份、源块释放、D2H→SSD 续接、复制合并、预加载/共享 staging 或完成协议。此补丁只是后续限流之前的必要桥接预备，不是完整有限候选/干扰策略，也不构成 live mandatory 已验证。

## 测试与可追溯性

最终完整回归：351 passed，8 skipped，0 failures，0 errors（6.18 秒）。其中原作者测试 261 通过，已有项目 CPU 测试 67 通过，本轮桥接 CPU/mock 测试 23 通过。跳过的 5 项要求 GPU/vLLM，3 项实际 io_uring_setup 返回 EPERM。

覆盖等待先发布再阻塞、无新 epoch 且零普通额度下推进、多父任务、物理 slot 容量、compute/copy event 未完成时不得续接、D2H 后原生写盘、Future 早失败后继续排空、重复 CQE 不重复释放、关闭排空、旧 run/Future 标记拒绝、off 路径、原生 Future 取消语义、owner 线程快照和致命失败。详见 TEST_MATRIX_CONTINUATION_01.md。

原始失败记录全部保留：首轮 3 项失败源于测试 ring 缺少 close()，另有线程异常警告；补齐夹具并将线程异常升级为测试错误后通过。首次合并回归出现 tests 包名冲突，15 项收集错误；使用 pytest 官方 importlib 模式解决测试收集问题，未改作者测试或生产代码。

独立补丁正向检查（common 基础）与反向检查（progress 工作树）均通过。AST 检查确认原生普通调度、pump、D2H→写盘、CQE 完成、复制合并、submit_load/store、shutdown 等指定方法不变。源码相同不是 GPU 运行等价证明。

## 实际执行命令

工作目录均为 /root/autodl-tmp/prefix-io-v1-handoff/project。完整记录位于 artifacts/prefix_io_v1/continuation-01/commands.jsonl，以下是最终测试的同等可执行命令：

```bash
CUDA_VISIBLE_DEVICES="" RUN_E2E_TESTS="" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:third_party/work/py-kvcache-progress \
.venv-prefix/bin/python -m pytest -q -ra --import-mode=importlib \
  -W error::pytest.PytestUnhandledThreadExceptionWarning \
  tests/prefix_io_v1 tests/prefix_io_v1_progress \
  third_party/work/py-kvcache-progress/tests \
  --junitxml=artifacts/prefix_io_v1/continuation-01/cpu-verified.xml
```

再次运行请使用下列脚本，它生成新的证据目录而不覆盖已交付日志：

```bash
bash experiments/prefix_io_v1/scripts/reproduce_progress_cpu.sh
```

独立补丁检查：

```bash
git -C third_party/work/py-kvcache apply --check \
  "$PWD/patches/prefix_io_v1/observer/0001-mandatory-progress-bridge.patch"
git -C third_party/work/py-kvcache-progress apply --reverse --check \
  "$PWD/patches/prefix_io_v1/observer/0001-mandatory-progress-bridge.patch"
```

在另一获准环境重建时，先锁定依赖，应用 common/0001、common/0002，再应用 observer/0001。保留独立 common 基线和 progress 工作树；不将 author vLLM 替换为当前主线。虚拟环境重建方法和原始版本见 REPRODUCE_CPU.md 与 dependency-lock.json。

## 证据位置

- artifacts/prefix_io_v1/continuation-01/cpu-verified.txt 与 .xml：最终逐项回归；
- 同目录 test-results.json、patch-checks.json、native-path-preservation.json：结果、补丁与原路径检查；
- 同目录 environment-current.json、preflight.json：环境与权限门禁；
- 同目录 progress-first/second/reviewed 与 cpu-combined：保留的过程及失败证据；
- patches/prefix_io_v1/observer/0001-mandatory-progress-bridge.patch：独立增量；
- experiments/prefix_io_v1/locks/dependency-lock.json 与 execution_state.json：更新锁和阶段状态。

## GPU、预算、限制与下一允许动作

真实 GPU 运行：0 次，0 GPU 小时；模型下载：0 GiB。permissions.yaml 未更改，allow_gpu_runs=false，GPU 预算 null、设备 ID 和获准根目录未设置。服务器无 /dev/nvidia*，cgroup 内存上限 2 GiB，io_uring 返回 EPERM，真实作者 vLLM runtime/内核/handler 未安装验证。本轮 preflight 仍返回 BLOCKED（exit 2）。没有吞吐、ITL、性能收益或 GPU 观测开销结果。

阶段验收的下一步仍是 P1：在已授权且具备真实设备和 io_uring 能力的环境中完成锁定作者组合的基线资格验证。系统/驱动/容器限制调整未获授权，因此未做修改。GPU 执行还必须同时具备明确权限和有限预算。

此后才可做 P2 真实 mandatory/shadow 和资源生命周期验证，包括强制 flush、失败后资源排空、真实 CUDA/SSD 事件、owner 快照开销及源块释放闭包。CPU/mock 不能替代这些验收。有限候选调度和干扰额度仍等待 P1–P3 通过；本轮不提前填入标定或性能常数。
