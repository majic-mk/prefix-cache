# V4 单文件 I 策略：无卡可行性审计

裁决：**停止现有 active-poll 延期实现的无变化 GPU 重复试跑。** 这轮已证明有限条件下的原生输出、调度和排空能运行，但没有净收益。现有 CPU profile 没有覆盖真实 181 次重试与 decode 线程并发；因此也不能断言“GIL 已被证明是根因”。目前没有新的性能候选达到再次 GPU 验证的条件。

## 来源与复算范围

以 `SERVER11_RAW_DELTA.tar.gz` 为唯一原始归档，SHA-256：
`a52b9de286c46fd342592565bb9dd0550d1c82791ca0a8e94e2c9e7cd8f1e8b6`，276,693,033 bytes。

使用 `tarfile.open(..., 'r|gz')` 流式扫描 1,573 个文件，仅读取 22 个白名单成员；没有全量解压。被读取成员共 11,648,973 bytes，转存的审查 JSON 为 5,825,776 bytes。逐项成员原始字节 SHA 保留在 `SELECTED_ARCHIVE_MEMBERS.json` 和 `ANALYSIS.json` 中。

实际本地分析完成 **75 项一致性检查**：原始 event arrays 与 qualification 一致、完整 host 请求时间重算、实际 I/O journal 接收/完成项吻合、原始六窗口 capture 与测量表吻合、profile 中位数复算、固定 155 次 profile 范围及源函数路径检查。此处没有重新运行模型或原始 GPU qualification。

## 三个时间量不能相加

| 指标 | 独立复算 | 含义 |
|---|---:|---|
| 首次至末次延期 | 64.898699 ms | 181 次实际尝试的首尾墙钟跨度；不是累计 CPU 占用 |
| on 受控第 17 步 CUDA 事件区间 | 68.464035 ms | 原 execute-through-sample 事件区间；不是纯 kernel 时间 |
| on 受控步骤 host-call wall span | 68.393572 ms | 主执行路径 start/end 间墙钟；不是主线程 CPU 时间 |
| on-off 完整请求差 | +111.008935 ms | 完整生成请求时间差 |
| on-off 请求至排空结束差 | +122.173497 ms | 本轮总差，+7.278720% |

off/shadow 的受控事件区间分别为 15.546464 / 15.636128 ms；host-call wall span 为 15.491117 / 15.579073 ms。on-shadow 的请求至排空结束差为 +123.576034 ms，+7.368436%。

on-off 事件区间总和增加 109.233283 ms，其中前 16 步累计 +3.694592 ms、选中步 +52.917571 ms、后 111 步累计 +52.621120 ms。选中步并不能解释全部差值。host-call wall span 总和也增加 109.032545 ms。请求完成至 drain 开始的中间记录区间增加 9.555166 ms，真正 drain 函数增加 1.609396 ms；不能把两者都叫作 I/O drain。

以 on 选中步的 start-record-before 为相对零点（下列时间均属 host monotonic 时间）：

| 时间点 | 相对毫秒 |
|---|---:|
| start query 确认完成 | 0.493017 |
| preload 回调开始 / 返回 | 0.497849 / 0.547446 |
| 首次延期 | 3.427513 |
| 末次延期 | 68.326212 |
| 主路径结束 | 68.416129 |
| end record 调用前 / 后 | 68.450217 / 68.485144 |
| 原生 SSD read 接收 | 69.747227 |
| 原生 SSD read 完成 | 70.802010 |
| end event 后来被 query 确认 | 1562.985667 |

首次延期至原生接收为 66.319714 ms；接收至完成为 1.054783 ms。read 在 host end 后 1.331098 ms 被接收，但 end event 的确认发生得更晚，**不能据此证明 GPU 完全不重叠**。跨 CPU/GPU 时钟的绝对映射不存在。

181 次尝试对应 180 次不可变观测复用。首尾间平均间隔约 0.360548 ms 也只是节奏统计，没有逐次 CPU 时间采样，不能作为每次策略开销。

## 可见 I/O 增量的量级

从 v6 六份原始 capture 重算相同配对的 B-A：

| 配对 | A (ms) | B (ms) | B-A (ms) | 用途 |
|---|---:|---:|---:|---|
| pair-0 | 13.025376 | 18.865088 | 5.839712 | calibration |
| pair-1 | 13.035968 | 15.533440 | 2.497472 | calibration |
| pair-2 | 12.802240 | 16.355232 | 3.552992 | heldout |

校准增量均值仍为 4.168592 ms。上述 2.497–5.840 ms 是有限配对中观察到的增量，不是可保证回收的收益，也不是所有工作负载的理论上限。64.899 ms 重试跨度约为最大观察增量的 11.1 倍，122.173 ms 总退化约为其 20.9 倍，仅用于工程量级判断。

on 的 68.464035 ms 仍超过冻结成本上界 18.865088 ms，故成本覆盖为 false；limited qualification 为 true 只说明条件调度、完整输出和排空通过。不能用有限资格掩盖性能负结果。

## CPU profile 覆盖了什么

归档中 C4 的 `CPU_PROFILE_FULL_BORROW.json` 固定 **155 次**，41 组非 profile 计时：

- V3 thread CPU 中位数：8.950252 ms。
- C4 thread CPU 中位数：7.259697 ms。
- 减少 1.690555 ms，即 18.888351%，仅适用于这份同条件 CPU fixture。
- C4 开启 cProfile 后该组 thread CPU 为 23.906951 ms；profile 改变开销，不能与非 profile 或 GPU 窗口混算。

实际 profile 执行 `Owner.repeat → _drain_ready_preload_fds → reserve / _prefix_stage_decide / preview / release`，使用真实源 AST 和值对象，但 slot、file、clock 和 live capture state 是合成的。full_borrow 使用真实只读 accessor 读取合成 weakref/scalar，并没有运行真实模型线程。

源与 profile 均证明它**没有包含**完整 `_run / _pump_once / sleep(0)`、真实 decode producer、CUDA 执行、线程竞争或 GIL/OS 调度；也没有覆盖真实 181 次。计时期间 fixture 的逻辑钟每次仅增加 100 ns，不能重放真实约 65 ms 的状态演变。cProfile 的 cumulative 时间相互嵌套，禁止相加。

真实源路径则显示：ready preload 仍在原队列时 `_has_poll_work` 为 true；原 reactor 每轮继续 drain incoming、poll/schedule/flush/finish，没有进展时保留原 `sleep(0)`。本轮共同 idle 修复只排除“仅保留缓存”的空闲工作，不能阻止 active 延期重试。每次重试仍做真实资源、安全、mandatory、最大等待及状态复核；这些不能为提速直接删掉。

## 哪些因果仍未证明，是否有下一候选

时间线上，长 host-call wall span、长 CUDA span 和反复延期同时出现；缺少实际双线程 CPU 占用、GIL 等待、OS 调度及 kernel/submission 时间分解。不能把全部 +122.173497 ms 归因于 GIL、观测，或 SSD；也不能把 7.259697 ms 乘以 181/155 当作 GPU 真机 CPU 测量。本报告未使用这种外推。

**现有证据足以停止当前实现的相同 GPU 重跑，尚不足以认定所有 I 策略都不可行。** 已实现的快照复用有 CPU fixture 收益，但没有建立 GPU 净收益；继续降低少量派生值构造开销不是现成可行的新性能候选。

唯一具体的下一 CPU 工作是“测量缺口候选”：以独立 producer 发布标量边界，在保持原完整 pump、原 Queue、原 live/mandatory/max-wait 校验的条件下，对 181 次并发重放分别记录 reactor 与 producer 的 thread CPU/wall time，按相同 producer 时间表比较 off/shadow/on。它用于区分重复决策工作与协作/调度延迟，**不是新调度算法或 GPU 效果证明**。若要提出新值缓存优化，应先证明语义不变和完整路径成本下降；不得靠更改睡眠、switch interval、执行器、资源拥有者或抬高阈值制造通过。本审计不批准新的 GPU 实验。

## 实际命令与服务器重放

本地执行过以下命令（Python 3.12 路径为 `C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe`）：

```text
python -B -I -S .../runtime_review/collect_archive_evidence.py
python -B -I -S .../runtime_review/analyze_runtime_evidence.py --evidence .../runtime_review/SELECTED_ARCHIVE_MEMBERS.json --output .../runtime_review/ANALYSIS.json
python -B -I -S .../runtime_review/collect_archive_evidence.py --help
```

初次流式采集约 57.13 s；随后为服务器重放补充显式 CLI 路径参数并检查帮助。分析命令 exit 0，75 项一致性检查通过。两个脚本不依赖本机绝对路径；服务器可从同一归档重放，输出应选尚不存在的新文件：

```bash
python -B collect_archive_evidence.py --archive /path/to/SERVER11_RAW_DELTA.tar.gz --output SELECTED_ARCHIVE_MEMBERS.server.json
python -B analyze_runtime_evidence.py --evidence SELECTED_ARCHIVE_MEMBERS.server.json --output ANALYSIS.server.json
```

新增写入仅限本审查目录；没有改动旧冻结代码、原始结果，没有 SSH/GPU，没有全量解压归档。
