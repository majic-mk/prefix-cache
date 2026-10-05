# Server11 修复版 v4：真实 GPU 对照交付（2026-10-03）

本轮已经完成真实 GPU 运行，不再停留在 CPU 准备阶段。结论是：共同底座修复可用，单文件策略能实际延期并正常排空，但新策略没有显示净收益，成本覆盖也失败。不能宣布完整 P4 通过，更不能进入 P5 性能结论。

服务器：connect.westc.seetacloud.com:26909；工作区 /root/autodl-tmp/prefix-io-v1-handoff/project。凭据不写入报告。设备为 RTX 5090，UUID GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9。原作者 py-kvcache / vLLM 路线、模型执行器与缓存所有权不变。

## 实际改动

1. 共同修复：在私有 C4 reactor 中新增 _has_poll_work，仅改变原 _run 是否使用已有 incoming Queue 等待的判断。只剩 retained cache 时不再持续空循环；原 _has_work、pump、STOP、清理及 0.5 秒队列超时保持不变。off / shadow / on 均使用此修复。
2. 紧凑观测：bridge 复用相同不可变原快照、完整 live state 和 runtime prefix 所派生的快照。原策略、每次实时资源检查、时效性检查、原延期记录、原分配/释放仍每次执行。没有缓存动作或发放资源额度。
3. 重新绑定成本：新 native v6 六窗口实际加载 C4，核验 _run 的编译来源、原生包路径、前后 adapter SHA 与 128 帧。新 receipt 重放完整原始数据并运行原成本公式，禁止把旧 v5 结果冒充新底座资格。
4. 实际加载核验：P4 runtime 也证明正在运行的 reactor 确实来自 C4，而非只证明 C4 文件在锁中。新增反例能拒绝“旧、新源码同时在继承锁里但实际导入旧版本”的情况。

改动位于新增私有目录：
- artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003
- artifacts/prefix_io_v1/server11-native-cost-v6-20261003
- artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003

原作者工作树与此前 v1/v2/v3/v5 证据未改写。p4_policy.py 字节完全不变，SHA-256 bce19c96c7651f69ebb91a515ea3e9e3d6855636b4bd4aa84e60fb2fbc10d673。共同修复、观测简化与 receipt 接线分别提供补丁；27 个 hunk 严格回放、独立 Git apply 后 21/21 文件与 C4 相同。关闭新策略时 bridge 为 None，仍走原执行路径及统一共同修复。

## CPU 实际测试

| 服务器测试组 | 结果 |
|---|---|
| C4 原策略/bridge、单文件、生命周期与新增反例 | 197/197 |
| 原生 CPU 集成回归 | 547/547，3 条既有 JUnit record_property 格式警告 |
| native v6 成本及来源验证工具 | 57/57 |
| P4 runtime 与实际来源验证工具 | 44/44 |

这些组有范围重叠，不把总数宣称为互不重复的用例数。v6 第一次服务器测试因未设置原公式文件路径而有 4 个错误；补上 NATIVE_ORIGINAL_ESTIMATOR 环境变量后 57/57 通过，没有为此消耗 GPU 作业或改动公式。新增独立本地复核记录另存，不能替代服务器结果。

相同 155 次 CPU 重试回放中，replace 从 156 次降至 2 次。完整 live-borrow 路径的 41 组未加 profiler 的线程 CPU 中位数从 8.950252 ms 降为 7.259697 ms（约 18.89%）。原 collect=1、preview=155、延期=155、reserve=release=155 均保留。该测量使用明确的 CPU 资源/事件夹具，不能当作真实 GPU 加速结果；cProfile 的重叠累计时间没有相加或混入此中位数。

## GPU 校准和真实对照

本轮一共 4 个原 guard 作业、9 个独立模型进程；每个进程 primer 1 + warmup 128 + flush 1 + measured 128，共 258 个输出 token。所有作业 exit=0、无超时、原 shutdown 返回、OS 会话排空。没有下载、系统/驱动/已安装包变更、旧数据删除或付费平台操作。

native-cost-six-window-06 按 AB / BA / AB 完成：前两对拟合，最后一对仅留出验证。每个 B 只执行原生 SSD 读取 917504 字节，完整观测 128 帧。所用参数为 Qwen2.5-7B、BF16、eager/Triton、batch=1、KV=256 MiB、staging=128 MiB、parent cap=8、iodepth=4。保留原单文件条件、种子与原公式，不因新结果修改门槛。

| 窗口 | 选定步骤 CUDA Event 时间 ms |
|---|---:|
| A0 | 13.025376 |
| B0 | 18.865088 |
| B1 | 15.533440 |
| A1 | 13.035968 |
| A2（留出） | 12.802240 |
| B2（留出） | 16.355232 |

仅校准数据计算得成本上界 18.865088 ms；留出未低估，差额为 -2.509856 ms。内部 A-only 步预算 13.035968 ms，来自两份校准 A，不是用户服务 SLO，也未用 B/留出调宽。新条件资格通过，生产资格与策略效果资格仍为 false。

随后严格执行 off04 → shadow04 → on04，三组同代码、同输入、同 seed、同单文件工作，128 个输出 token ID 完全相同：

| 模式 | 选定步骤 ms | 完整请求+排空 s | 对 off 变化 | 成本覆盖 | 有限功能/生命周期 |
|---|---:|---:|---:|---|---|
| off | 15.546464 | 1.678502543 | 基准 | 通过 | 通过 |
| shadow | 15.636128 | 1.677100006 | -0.084% | 通过 | 通过 |
| on | 68.464035 | 1.800676040 | +7.279% | **失败** | 通过 |

on 相对 shadow 也慢 7.368%。即使只看完整请求、不计随后排空，off / shadow / on 分别为 1.664287303 / 1.662794556 / 1.775296238 秒，结论方向不变。

shadow 提出 1 次延期建议但不拦截。on 真实拦截 181 次，首次至末次延期约 64.899 ms；该步骤最后仍完成原 SSD 读取，保留 original shutdown 与会话排空证据。有限 qualification_passed=true 只表示这个狭窄动作与生命周期通过；frozen_cost_migration_pass=false 和 strategy_effect_verified=false 同时保留，不能把前者偷换成性能成功。

选定步骤之后的 111 步平均 Event 时间分别为 off 12.7854 ms、shadow 12.7713 ms、on 13.2595 ms。共同底座已没有上一版约 18 ms/步的持续现象，但这是不同版本、单次顺序运行的描述性观察，不能当作随机化因果估计。尤其不能把共同修复后相对旧 off 的加速，算成 on 策略收益。

CUDA Event 时长包含主机提交间隙，不能解释为纯 kernel 计算时间。host end / event record 时间也不能单独证明 GPU 已经完成。本轮没有 D/J 资源释放 credit，没有新增虚构的资源释放或 GPU 无重叠结论。

## 失败原因与阶段判断

已排除本轮的模型启动、CUDA/Ninja 接线、无卡环境、输出不完整和 shutdown 不排空问题；真实模型与原生 I/O 均能运行。共同 retained-cache 空转及重复快照构造已修复，shadow 的净开销在这一次对照中接近零。

当前剩余问题出现在 active 延期阶段：ready 工作存在时，原 reactor 继续轮询，on 多次重试与主线程提交并行；受影响步骤及后续完整请求实际变慢。实测只能定位到该执行阶段，尚不能把全部差额精确归因给 GIL、某个函数或某个 GPU kernel。CPU 简化没有转化为此条件下的策略收益，且原生无策略校准上界不能覆盖开启延期后的 68.464 ms。

结论边界：
- 系统运行可行、共同修复能保持生命周期和输出正确性：本轮有真实证据。
- 此单文件 on 策略有效提升：本轮否定；目前不能据此进入性能主张。
- 整个研究思想在所有工作负载上不可行：本轮无法证明。
- 完整 P4、D/J 原生释放依赖资格、P5 SLO-goodput：仍未完成/未验证；正式 SLO 仍为空。诊断中的 load_planner='off' 不等于验证了生产正常准入流程。

遵守事先写下的 V4_PILOT_DECISION_RULE.md：on 再次成本不覆盖且完整请求慢于两组对照，停止本轮 GPU 迭代，不改预算逼出改善，不自动追加重跑。下一允许工作是无卡分析 active 重试/主线程提交关系及这类条件是否有足够收益空间；只有形成新的、符合原范围且可比较的候选，才应制定下一份冻结 GPU 对照。没有依据直接进入 P5 或承诺性能提升。

## 实际执行命令与证据

以下为已经执行的关键命令形式；变量均指上述同一服务器工作区。不要用相同 label 再次运行，控制器与 guard 会拒绝重复作业。

```sh
ROOT=/root/autodl-tmp/prefix-io-v1-handoff/project
C=artifacts/prefix_io_v1/server11-p4-single-file-candidate-v4-20261003
V=artifacts/prefix_io_v1/server11-native-cost-v6-20261003
R=artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003
.venv/bin/python -B -I -S "$C/run_original_cpu_regression.py" --project-source "$ROOT"
.venv/bin/python -B -I -S -m unittest discover -s "$V" -p 'test_*.py'
.venv/bin/python -B -I -S -m unittest discover -s "$R" -p 'test_*.py'
.venv/bin/python -B -I -S "$V/control_native_cost_job.py" prepare
.venv/bin/python -B -I -S "$V/control_native_cost_job.py" launch
.venv/bin/python -B -I -S "$V/control_native_cost_job.py" after
.venv/bin/python -B -I -S "$R/control_p4_single_file.py" freeze --calibration-lock-sha256 53fe06db2572b6d07b5dfdaac31ca46aed62fe0710bbb63e006a4d7d089d8e22
# mode 依次 off / shadow / on，逐臂 prepare、launch、after、独立 verify。
```

CPU 命令的实际环境变量、全部参数、stdout/stderr/exit 在各目录 *_COMMAND.json、*_RESULT.json 与日志中；native prepare 的完整 plan/scope、独立证据重放命令也已保存。本段不是省略授权/门禁的可直接批量重跑脚本。

主要证据：
- 本地 artifacts/server11_results/p4_v4/V4_FINAL_EVIDENCE.json（聚合及原资格结果；SHA-256 92f67dccc221e2ed85c71a29ae8e72f14da2c9d3b2bd97bea9b35eb849308c7e）。
- 服务器 artifacts/prefix_io_v1/server11-p4-single-file-review-v4-20261003。
- 四个 guard 原始记录：experiments/prefix_io_v1/runs/server11-native-cost-six-window-06，以及 server11-p4-single-file-{off,shadow,on}-04。
- native v6 锁：53fe06db2572b6d07b5dfdaac31ca46aed62fe0710bbb63e006a4d7d089d8e22，4,655 项前后通过。
- P4 v4 共同锁：76474f66c3b632850eb1a8ece3650fb44321d95614dd4299bd352a4a1d5b25e5，4,670 项逐臂前后通过。
- receipt binding：013a14f48d9b90f98667b9f6ea6b94898ea42d2844af8987c77bc1e5f0f3f3d9。
- 补丁、两份来源/交互复核与结果复核均单独保留；旧主报告及 v1–v3 原始备份不覆盖。

本轮实际 GPU guard 累计 1021.929462 秒（约 17 分 02 秒）；原 8 小时累计现用 22380.561258 秒，剩余 6419.438742 秒（约 1.78 小时），active reservation=null，nvidia-smi 无 compute 进程。此预算是 guard 计时，不等于云平台账单。汇总时数据盘可用 57.153 GiB，无需为当前剩余文件扩容；没有删除实验代码或数据。

完整原始数据的本地备份路径为 p4_v4/SERVER11_RAW_DELTA.tar.gz；归档及本地逐成员核验结果以 ARCHIVE_RECEIPT.json / LOCAL_ARCHIVE_VERIFICATION.json 为准。完成核验后可将实例切回无卡模式。
