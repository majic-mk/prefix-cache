# P3 强 U 基线剩余机会：原始证据复核

2026-10-03。本审计只读取已有文件并运行 Python 标准库分析；没有 SSH、CUDA 初始化、GPU 作业、模型调用或原冻结源码修改。root 另行从现有无卡服务器取回八份原始文件，本审计逐项核验其历史 SHA，未把新运行数据混入旧实验。

**裁决：当前证据不支持立即恢复 GPU 调参或继续开发性能策略。** 保留一个具体但未证实的问题——目的地保护所依赖的小写回，是否在已合法 ready 时受到后台大 parent 的可避免排队。当前缺少区分“合法依赖等待”和“可调度排队”的记录，所以这还不是一个已成立、预期有净收益的候选。D/J 未被普遍否定；也不能用它们尚未验证来承诺未来收益。

## 实际新增证据

此次读取最终强 U 的两次完整结果、原生 trace、冻结配置和独立分析，共八份。result / analysis / config 对照 `finite-grid-selection-final.json`，trace 对照原 checkpoint manifest；八份 SHA 全部匹配。分析还核对 analysis 指向的 result SHA、cohort 和 pending flush 数值。机器可读结果在 `STRONG_U_REANALYSIS_FINAL.json`。

| 真实原始运行 | mixed-01-off | mixed-06-off |
|---|---:|---:|
| cohort 含排空 | 32.923586 s | 32.209402 s |
| pending flush | 408.055 ms | 641.355 ms |
| 首个目标 D2H host enqueue 距等待开始 | 370.058 ms | 574.030 ms |
| 此前间隔占该次等待 | 90.688% | 89.503% |
| 最后目标 transfer 完成后至等待返回 | 0.308 ms | 0.620 ms |
| 等待期间全部请求的 token 输出事件数 | 0 | 0 |
| 目标首 token 距等待返回 | 1.755 s | 2.072 s |

两轮等待均保护 parent 23、25、27，每个 917,504 字节，总计 2,752,512 字节。原因均为 `restore_destination`，指向请求 c2-5。等待落在一个前端 step 内，这两个 step 分别长 690.325 和 992.163 ms，返回事件均包含 c2-4；它们不能全归因于 flush。

这是请求推进确有保护等待的证据。c2-5 最终却分别比最后一个请求 c2-9 早 7.404 和 8.198 秒完成；没有反事实运行能证明提前完成该保护会等量缩短最终 cohort。pending flush / cohort 的 1.239% 和 1.991% **仅为描述性比例，既不是可获得加速，也不是严格的可省时上界**。

## 为什么仍不能把 90% 等待当作可消除机会

两个强 U 原始配置均为 `store_readiness_probe=false`。trace 只有真实的 host enqueue、Event 时长和 transfer/write 区间，没有以下完整链条：

- 源 GPU 计算完成事件在每个候选决策点是否已经满足；
- 目标 D2H 在等待开始前后何时实际成为合法 ready；
- 当时 staging 和 copy/AIO 容量的连续、因果对应状态；
- 每个尚未签发候选的具体拒绝或等待原因。

确有后台活动：parent 22 总大小 932,184,064 字节，在等待开始时未完成。首个目标 D2H 之前，这个 parent 分别新增 117 / 221 次 D2H host enqueue，合计 107,347,968 / 202,768,384 字节。它证明背景大 parent 在推进；**不证明那三个目标 parent 当时也已合法 ready，也不证明大 parent 顺序是唯一原因**。不能将这段时间直接改名为调度饥饿、GIL 延迟或纯 SSD 等待。

目标三个 parent 的 Event 时长约 0.088–0.229 ms，SSD host 区间约 22.057–51.488 ms。时间段互相重叠，不能相加成总耗时；host enqueue 加 Event elapsed 也不是校准的设备开始/结束时间轴。等待到最后一个 transfer 真正结束才返回，符合原完成依赖，不支持提前释放。

## 真实资源生命周期

两次 flush 的三个目的块完全同型：

| block | 源 generation | 当前 destination generation | 保护撤销后的 active_refs | free_queue_linked |
|---|---:|---:|---:|---|
| 1312 | 3086 | 4408 | 1 | false |
| 310 | 4104 | 4407 | 1 | false |
| 308 | 4106 | 4406 | 1 | false |

在等待前有各自 protecting parent，retirement 记录中 protecting_jobs 清空，但 generation 已属于新目的地，active_refs 仍为 1。**这是解除目的地覆盖保护，不能记为新增可分配 GPU 块。** 两次各 38 个退休 parent 的 59 个采样块全部 active_refs=1、没有 free queue 链接、`immediately_reusable_bytes=null`。它们是有界采样，不能推广为系统从未释放块；本审计的 `physical_GPU_blocks_freed` 保持 null，而不是伪造为 0。

retirement 采样出现在等待开始约 0.962 / 1.203 秒后，比等待返回晚约半秒；这是调度侧观察时刻，不是设备资源实际释放时刻。原生 AIO/copy 完成、parent 完成、调度 ACK、解除保护、GPU 空闲容量是不同状态，不能互相替代。

## 继续与止损边界

已有强简单基线 U 胜过 F16、P4、F8、P512，不能把更长 prefix、提高压力或再次简单限流当作尚未尝试的新贡献。两轮强 U 的目标确实出现，故不能以“没有目标等待”为理由结束；但目标可控性、实际资源收益和净收益仍不成立。

建议停止当前性能策略投入、保留 off、封存全部正负证据。只有在独立的后续决策中确认了上述具体 ready/owner/资源链，并证明一个有限调度动作能提前推进且费用足够小，才有理由再制定冻结 GPU 对照。不能为填这个证据缺口而默认无限追加 GPU 诊断。本审计没有构成重启 GPU 的通过门槛，也没有否定所有 D/J 场景。

另一个限制是这些强 U 运行早于 C4 的共同空转修复。历史等待量不是当前 C4 的等待预测；共同修复的效果不能计入研究策略优势。

## 命令、身份与结果

本地执行退出码 0，标准库运行约 0.54 秒，八份历史源身份验证和两组完整重算均通过：

```text
D:/dev/python3.8.7/python.exe -B -I -S artifacts/prefix_io_v1_server11_candidates/feasibility_audit_v1/opportunity_review/audit_strong_u.py --workspace-root . --raw-dir artifacts/prefix_io_v1_server11_candidates/feasibility_audit_v1/raw_p3 --output artifacts/prefix_io_v1_server11_candidates/feasibility_audit_v1/opportunity_review/STRONG_U_REANALYSIS_FINAL.json
```

脚本 SHA-256：`b08e8a536087aebdc6099b8aff2576968d8564382850f551e3e18edbbf280b3e`。
最终本地 JSON SHA-256：`b55e8889111cc54d710fab237f9c8747019ad4838cefbf352767a97dceb11e4f`。

脚本也支持 `--project-raw`，直接读取冻结清单记录的服务器项目路径；无需把 33 MB 原始数据传回服务器。使用 `--reference-dir` 与 `--checkpoint-manifest` 指向三份小参考 JSON 即可。该模式由 root 另行执行，不能把本地通过写成已经完成服务器重放。

参考清单：

- `artifacts/prefix_io_v1_server08_p3_complete_20260930/finite-grid-selection-final.json`
- `artifacts/prefix_io_v1_server08_p3_complete_20260930/p3-closeout-analysis-final.json`
- `artifacts/prefix_io_v1_server08_p3_checkpoint_20260930/server08-p3-16-blocked-checkpoint-evidence-v3-manifest.json`

八份 raw 的完整绝对路径、字节数、实际/预期 SHA 和匹配结果均保存在最终 JSON 的 sources 中。旧本地 checkpoint 的四份失败传输 ZIP 无法打开，未用于本次结论；实际使用的是重新取回且对照历史身份通过的八份原始文件。中间 JSON 另存，未覆盖历史记录。
