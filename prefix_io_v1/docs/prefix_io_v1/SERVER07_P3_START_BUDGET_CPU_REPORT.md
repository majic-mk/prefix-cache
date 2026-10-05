# Server07 P3：固定／压力启动额度的 CPU 集成

**本轮完成了真实作者 reactor 上的 CPU 薄接口集成，尚未完成 P3 全部基线，也未证明性能收益。** 本轮未运行 GPU；649 项测试通过、0 失败、16 跳过，其中 46 项属于新增专项测试。实际 Linux AIO 读写校验已执行，CUDA 仍为 CPU 替身。

## 实际改动与运行边界

在服务器创建独立工作树 `third_party/work/py-kvcache-p3-quota-cpu`，分支 `codex/prefix-p3-quota-cpu`，基于作者 py-kvcache 的 `3abba7a502d553f6e7e2e58b92086487e3395d7e`。先完整继承 P2 的兼容、正确性及 mandatory 桥接补丁，再增加本轮接口。

上轮通过 GPU 验证的 `py-kvcache-p2-aio` 与作者 vLLM 没有修改；此前锁定的 105 个源文件全部保持相同 SHA-256。没有切换 live runtime、重编译模型执行器、修改驱动或扩大权限。

新增策略／测试文件：
- `src/prefix_io_control/start_budget.py`
- `tests/prefix_io_v1_start_budget/` 内 5 个文件（含初始化与 CPU fixture）

作者源码实际修改仅位于新工作树的 `py_kvcache/reactor.py`，通过 `IoReactor` / `TransferCoordinator` 的可选 `start_budget=None` 接入。新增公共入口默认关闭；原 vLLM 配置适配器仍只开放 off/shadow，没有把 CPU 通过自动变成 fixed/pressure 的 GPU 授权或资格。

独立增量补丁：`patches/prefix_io_v1/baselines/0001-native-start-budget-cpu.patch`。
继承补丁与全量差异分别为本轮证据目录内的 `inherited-p2.patch`、`full-inherited-and-start.patch`。

## 额度究竟控制什么

| 能力 | 本轮实现与证据 | 限制 |
|---|---|---|
| fixed | 每个单调时钟 epoch 共用 1/2/4/8 个启动单位；load/store/preload 不各拿一份完整额度 | 这是文件链路启动数量，不是所有阶段的实际字节上限 |
| pressure | 在 fixed 基础上，为前台保留空闲 staging slots；超龄工作允许继续推进 | 原生 slot、iodepth、事件及所有权检查仍有效；超龄是明确的性能额度例外 |
| epoch | 同一窗口反复 pump 不重复发放，空闲窗口不累积无限额度 | CPU 测试中的纳秒值是合成测试时钟，不能照搬为 GPU 参数 |
| mandatory | 沿用原 native Future 信号；普通额度耗尽且没有下一窗口时仍排空 | mandatory 存在时保守允许 native stores 提供支持，是安全上界，不是依赖排序算法 |
| 已接受链路 | opened-fd→read、read→H2D、D2H→write、完成回收继续使用原路径 | 续接可能跨窗口，因此不能声称逐阶段每窗口字节都受本接口约束 |
| 原生复用与融合 | 两个请求共享一次 preload read，并保持一次合并 H2D launch | 这是 CPU fake-CUDA 路径验证，不是 GPU DMA 吞吐测试 |
| 关闭与故障 | 不创建 ticket、不读策略时钟；可选 gate 出错回原路径并显式置 faulted | faulted 后不再宣称额度约束成立，不能把 fallback 结果用于策略收益 |
| 生命周期 | 所有权仍属于原队列、slot、Future；控制器最多保留 32 个等待时间键和一个 ticket | 不维护第二任务队列，不提前 complete_store，不给未知释放字节记账 |

额度在 native 链路入口启动时记一次账；失败的启动尝试不退还额度。探测到没有可用物理 slot 时不扣费。原生 admission、缓存身份、缓存淘汰、预加载、复制合并、下游回调与释放协议未重写。

去掉新增参数、guard 与薄钩子后，**整个 reactor AST 与继承的 P2 原版一致**，对应专项测试已通过。

## 实际命令与测试

所有代码编辑、CPU 测试和真实文件 I/O 都在服务器完成。本地只接收与校验交付证据。

最终测试命令：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$PWD/third_party/work/py-kvcache-p3-quota-cpu:$PWD/src:$PWD/experiments/prefix_io_v1/scripts" \
AIO_CPU_GPU_GUARD_PATH=artifacts/prefix_io_v1/server07-p3-08/final-cpu-guard.json \
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
  -q --import-mode=importlib \
  tests/prefix_io_v1/test_config.py \
  tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs \
  tests/prefix_io_v1_progress tests/prefix_io_v1_observer \
  tests/prefix_io_v1_aio tests/prefix_io_v1_start_budget \
  third_party/work/py-kvcache-p3-quota-cpu/tests \
  --ignore=third_party/work/py-kvcache-p3-quota-cpu/tests/test_e2e_kvcache.py \
  --basetemp artifacts/prefix_io_v1/server07-p3-08/final-tmp \
  --junitxml artifacts/prefix_io_v1/server07-p3-08/final-cpu.xml
```

完整实际环境、命令、stdout/stderr 和耗时保存在 `execution-environment.json` 与 `final-cpu-command.json`。复跑时必须换新的 basetemp、XML 和 guard 输出路径，避免覆盖已有证据。

最终 **665 项收集，649 通过，0 失败／错误，16 跳过**：
- 46 项新增专项测试全部通过。
- 13 项跳过专门针对“没有 vLLM”的 fallback 分支，当前环境已安装作者 vLLM。
- 3 项跳过源于本容器 `io_uring_setup` 返回 EPERM；本轮使用已授权的 Linux AIO 路径，未更改系统以绕开限制。
- 原作者 GPU e2e 文件被明确排除，不计入通过或跳过数量。
- guard 记录 CUDA 未初始化。本轮早期 41/44 项专项测试及 647 项合并通过数不与最终数字相加。

关键通过项：普通额度耗尽且无下一 epoch 的 native mandatory wait、独立 shutdown、压力保留与超龄进展、真实物理 slot 上限、compute/copy 事件未完成时不提前写盘、短写失败保留原生失败与排空、重复 CQE 不重复释放、共享 preload 多请求与融合、gate 异常／错误 owner 不影响 native 完成、单一 budget 不可复用给第二个 reactor。

真实 Linux AIO 专项使用 4096 字节对齐缓冲，验证了读入的实际磁盘字节与随后写出的实际磁盘文件，accepted/completed/reaped 相等且 outstanding 为 0。仅 GPU copy 操作是明确标注的替身，不能称为生产 KV/GPU 正确性。

## 资源、阶段与下一动作

- 本轮真实 GPU 运行 **0 次**，新增 GPU 预算消耗 **0 秒**，新增模型下载 **0 字节**。
- GPU 账本逐字节内容未变：累计 8560.2893 秒，约 2.378/8 小时。
- 权限合同没有更改；但本轮两次完整 CPU 回归保留了约 444 MiB 临时文件，收尾审计发现项目盘空闲曾降至 8,202,485,760 B，低于 8 GiB。已将本轮自建的三个 scratch 目录复制到授权辅助根目录，逐文件 SHA-256／大小和符号链接清单完全一致后退役原位置；没有删除旧实验、共享数据或模型。项目盘空闲恢复至 8,666,599,424 B。详见 scratch-relocation.json。
- 原运行路径仍保持前轮资格；新 CPU 工作树**没有 GPU 资格**。不能把以前的 GPU 通过次数归到本轮 fixed/pressure 上。
- full P3=false；dependency_only/interference/joint 未激活，P4–P7 未进入。
- 下一允许动作：P3 的剩余 CPU 工作——逐阶段实际字节与 in-flight 计量、与启动额度的清晰映射、有限参数的实验入口和关闭/回退合同。完成后再安排容量允许的小规模 GPU 资格验证。
- 相同混合负载额外回放仍受存储预留限制；这不等于所有 GPU 操作都不可用，也不等于系统失败。目前仍没有研究方法稳定提速的实测证据。

主要证据目录：`artifacts/prefix_io_v1/server07-p3-08/`。
包含原始 before、副本与补丁、版本锁、148 项源文件 SHA-256、最终 CPU JUnit/guard、跳过原因、实际命令和 qualification-summary。


## CPU 临时空间修正与后续预检

本次空间预留遗漏属于执行流程问题，不能藏在“测试通过”里。迁移保留的逻辑数据共 464,014,710 B，现位于 /root/prefix-io-v1-validation/cpu-evidence/server07-p3-08/；清单保存于 scratch-relocation.json。原生测试生成的 current 符号链接按原字符串保留作审计，不作为复跑入口。

新增 preserve_p308_cpu_scratch.py 已实际执行；新增 run_start_budget_cpu_qualification.py 给固定 CPU 矩阵预留 640 MiB，并把后续临时数据和输出直接放到授权辅助目录，启动前检查 20 GiB 总额度及 8 GiB 空闲底线。后一个脚本本轮只实际执行了 --dry-run 预检；完整测试启动分支未另跑，不计入 649 项回归。

实际预检命令：
```bash
.venv/bin/python experiments/prefix_io_v1/scripts/run_start_budget_cpu_qualification.py --label server07-p3-next-cpu-qualification --dry-run
```

预检通过只表示当时容量允许；真正执行前会再次检查并拒绝已有标签。不要直接再次执行上文已用过的 primary basetemp 命令。当前辅助目录约 19.15/20 GiB，目录额度仍不足以预留下一轮混合 GPU 回放所需的 3 GiB。
