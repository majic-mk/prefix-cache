# 继续交付 02：有界观测发布与释放证明收紧

本轮在服务器 /root/autodl-tmp/prefix-io-v1-handoff/project 完成 CPU 开发、回归和微基准。P0 已完成；P1 真实集成/GPU 门禁仍阻塞。P2 增加 CPU 预备证据，尚未验收；没有启动 P3–P7。

## 实际改动

独立工作树 third_party/work/py-kvcache-observer，分支 codex/prefix-io-v1-observer-cpu，仍基于锁定 py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e。应用顺序为 common/0001、common/0002、observer/0001、observer/0002。前两个工作树及作者 vLLM 源码保持原样。

1. observer/0002-bounded-snapshot-hook.patch：现成 reactor 在原生 pump 完成后增加可选 observation_sink；coordinator 仅透传该参数。默认 None，正常部署未开启。观测异常只计数一次并停用 sink，不改变父 Future 的成功/失败、不重复日志、不阻断后续原生传输。
2. src/prefix_io_control/publication.py：在 reactor owner 线程按显式间隔采集，跨线程只发布一份最新不可变快照。消费者持锁时，reactor 丢弃本次观测而不等待；不积压历史或建立第二套任务队列。不追赶错过的采样周期；旧 run、过期/未来时间、跨 reactor 复用和采集失败均不能提供可用快照。
3. observation.py：通过 available_fields 显式区分已观测字段与 None。继续限定 32 个 parent、最多各 64 项在途 I/O/copy 摘要，不扫描 GPU pool，不把观察窗口变成并发限制。
4. dependencies.py：增加只读生命周期分类。更重要的修正是：parent 全完成仍不足以计入立即可复用资源；还必须有原生资源所有者在 fence 解除后的 native_reusable=True 确认。没有确认时仅保留 potential_bytes。失败、活跃引用、旧 generation/run 等仍不能计为释放。当前没有 live GPU owner adapter，因此没有据此新增任何实际 GPU 释放量。
5. 新增 34 项 CPU/mock 测试、CPU 观测微基准和复现脚本；更新锁文件与阶段状态。

没有改变原生调度、复制合并、共享 staging、预加载、成本准入、D2H→SSD 续接或源块完成/释放协议。普通额度和研究策略仍未启用。已有 off/shadow 等配置名不等于真实部署已经连通；这个入口是隔离工作树中的 CPU 预集成 API。

## 测试结果

最终完整回归：**385 passed，8 skipped，0 failed，0 errors**，6.59 秒。组成：作者原测试 261，通过的已有项目 CPU 测试 67，上一轮 mandatory 桥接 23，本轮新增 34。

跳过：5 项需要 GPU/vLLM；3 项在实际 io_uring_setup 返回 EPERM 后跳过。没有将跳过计为通过。本轮首轮聚焦测试 56 项通过；完整回归也一次通过。

新增测试覆盖采样节流、单份快照保留、读端竞争不阻塞 owner、freshness/run/owner 校验、采集异常失效、时钟回退、非法配置、真实 reactor 循环中的观测及异常退化、off 无采集、失败排空状态与原生释放确认。模拟 CUDA/file/ring 的测试仍明确标为 CPU/mock，不能替代真实 backend。

补丁正向/反向 git apply --check 与 git diff --check 均通过。AST 验证发现 observer/0002 只改变两个构造函数和 _run；去掉新增可选观察分支后，_run 与上轮相同。所有其余既有方法以及其他 py-kvcache Python 文件与 progress 工作树一致。该源码检查不是 GPU 等价证明。

## CPU 开销实测

对实际 observer hook 和 publisher，使用合成的 32 parent、64 ring op、64 copy 元数据；每种情况每轮 20,000 次，7 轮，按固定种子随机化顺序。表中数值为各轮平均每调用耗时的中位数，不是单次调用 P50/P95。

| 情况 | wall 微秒/调用 | process CPU 微秒/调用 | 计时期间发布快照数 |
|---|---:|---:|---:|
| 观测关闭的分支 | 0.0367 | 0.0367 | 0 |
| 已启用但尚未到采样期 | 0.2865 | 0.2864 | 0 |
| 1 ms 采样间隔诊断 | 0.2985 | 0.2976 | 44 |
| 每次都采集满窗口 | 69.8544 | 33.5711 | 140,000 |

原始重复结果及最大轮次均已保存，不只报告中位数。例如前两种启用且节流的情况，最大轮次平均 wall 耗时约 3 微秒/调用。容器 cgroup CPU 配额为 50000/100000，即 0.5 CPU，内存上限 2 GiB；这些限制和容器调度会影响结果，不能推广为目标 GPU 机器的性能。

实际测量没有运行原生 pump 的完整工作量、模型、设备 I/O 或 GPU。1 ms 仅是本次诊断点，未冻结为生产参数。没有证明低争用端到端退化低于 2%，没有 decode 干扰标定、ITL 或 goodput 结论。不建议按每次 pump 采集满窗口；本实现按显式间隔采集，生产间隔仍需后续实测冻结。

## 实际执行命令

服务器工作目录：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
```

最终回归（调用环境同时设置 CUDA_VISIBLE_DEVICES=""、RUN_E2E_TESTS=""、PYTHONDONTWRITEBYTECODE=1）：

```bash
PYTHONPATH=src:third_party/work/py-kvcache-observer \
.venv-prefix/bin/python -m pytest -q -ra --import-mode=importlib \
  -W error::pytest.PytestUnhandledThreadExceptionWarning \
  tests/prefix_io_v1 tests/prefix_io_v1_progress tests/prefix_io_v1_observer \
  third_party/work/py-kvcache-observer/tests \
  --junitxml=artifacts/prefix_io_v1/continuation-02/cpu-verified.xml
```

CPU 微基准：

```bash
PYTHONPATH=src:third_party/work/py-kvcache-observer \
.venv-prefix/bin/python experiments/prefix_io_v1/scripts/measure_observation_cpu.py \
  --output artifacts/prefix_io_v1/continuation-02/observation-overhead-cpu.json \
  --iterations 20000 --repeats 7
```

上述证据路径已存在，勿覆盖。再次复现使用 bash experiments/prefix_io_v1/scripts/reproduce_observer_cpu.sh，生成新目录。脚本经 bash -n 验证；首次交付实际命令见 commands.jsonl。异地重建必须先检出锁定作者组合，再顺序应用共同补丁与两项 observer 补丁，保留 common 基线独立。

## 证据与修改位置

- artifacts/prefix_io_v1/continuation-02/cpu-verified.txt、.xml、test-results.json；
- 同目录 observer-first.txt、.xml、observation-overhead-cpu.json；
- 同目录 patch-checks.json、native-path-preservation.json、environment-current.json、preflight.json、commands.jsonl；
- 同目录 *-before.py 和 *-before.json 保存本轮起点，项目增量另见 project-changes.patch；
- patches/prefix_io_v1/observer/0002-bounded-snapshot-hook.patch；
- src/prefix_io_control/{publication,observation,dependencies}.py；
- tests/prefix_io_v1_observer/test_publication.py 与已更新的 test_dependencies.py；
- experiments/prefix_io_v1/scripts/{measure_observation_cpu.py,reproduce_observer_cpu.sh}；
- dependency-lock.json、execution_state.json 仍保留未验证/阻塞标记。

交付包中的旧报告与旧 manifest 是历史记录；本轮 continuation-02/delivery-manifest.json 是本包校验依据。

## 真实 GPU、预算与下一允许阶段

真实 GPU 运行 **0 次、0 GPU 小时**，模型下载 **0 GiB**。permissions.yaml 与原模板逐字一致，GPU 未授权、预算 null、设备 ID 和批准根目录未设置。无 /dev/nvidia*；io_uring 仍 EPERM；作者 vLLM runtime、拷贝内核和真实 handler 没有完成资格验证。只读 preflight 仍返回 BLOCKED（exit 2）。

下一验收阶段仍是 P1：在明确获准且具备实际设备、足够资源及 io_uring 能力的环境中恢复锁定作者组合并验证基线。未修改系统、驱动、容器安全设置，也未扩大权限。

P1 通过后才可验收 P2 的真实 mandatory/shadow 调用链、源块 ownership/fence/generation adapter、真实异常排空及端到端观测开销。当前最新快照不保存完成历史，parent 消失不能当作完成或资源可回收证明。有限候选调度和干扰额度继续等待 P1–P3 的真实证据，不用本次 CPU 微基准充当干扰标定。
