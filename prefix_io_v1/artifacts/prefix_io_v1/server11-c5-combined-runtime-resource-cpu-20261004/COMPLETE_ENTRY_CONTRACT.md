# C5 完整入口 CPU 资源与有限验证约束

本目录是新的 `server11-c5-combined-runtime-resource-cpu-20261004` 的资源先决条件与只读协议。只读取实际 cgroup、affinity、源锁和 GPU ledger；不配置作业、不启动测量、不调用模型或原生 GPU，不重放或重新解释旧性能分数。解析器 tests 的资源文本属于 synthetic；resource_preflight 的服务器 resource_facts 才是当次实际只读采样。

旧 `CPU_COMPARISON_PROTOCOL.json` 的 SHA-256 固定为 `6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218`。服务器精确路径是 `artifacts/prefix_io_v1/server11-p4-notification-benchmark-v5-cpu-20261003/CPU_COMPARISON_PROTOCOL.json`。两入口要求该文件在本轮外层闭包中并验证原字节；本机可用 `--old-protocol` 指定同哈希归档路径。旧 132 次中 48 次 warmup、84 次 scored、42 对的结果不重新跑、不筛选或放宽；0.5 核条件下此前完整资格未通过的事实不变。

`cpu.max` 必须是可解析、正整数 period，quota 为正整数或明确的 `max`。用于进入后续测量的最低资源条件是有限 quota >= period，即至少 1 核，或者实际明确的无限额 `max`，且 affinity 和所需 cpu.stat 计数器均可读。合法 `max 100000` 是已知的无限额配置；缺失、坏文本和缺计数器仍返回 RESOURCE_UNKNOWN；当前 `50000 100000` 返回 RESOURCE_LIMITED。此最低条件是新 preflight 的资源就绪判断，绝不是新的性能资格阈值；满足 1 核或无限额也不能免除原配对零节流检查。历史累计节流计数不作为某次新配对的观测或分数。GPU ledger 存在 active reservation 时记录 RESOURCE_BUSY_LEDGER_ACTIVE，不进入测量；原预算字节前后必须不变。设备节点仅列出，不打开，不查询 CUDA。

| 完整入口所需组件 | 实际边界与保持条件 | CPU 能证明的范围 |
|---|---|---|
| 正常 launcher helper | 原 Stage A `notification_runtime_adapter.prepare_notification_capture` 在正常 measured callsite；capture.run_id 使用 owner LABEL，外部请求 RID 与实际 native_request_id 保留 | 实际 helper/source 调用与拒绝边界；不能用 common 六窗 capture=外部 RID 的契约替代正常 launcher |
| canonical typed policy/bridge | 新 Stage C 的同一 `prefix_io_control.p4_single_file_receipt.ExactSingleFileReceipt` 被原 policy/bridge 模块引用；保持精确类型检查和原算法 | 源闭包、同模块类型、拒绝旧来源；所有 CPU synthetic 值不能颁发有效 native receipt/budget |
| 实际 collector.install API | 原 C5 collector 以 origin=cpu_fixture 调用，worker connector/Event 对象明确 synthetic；保留 prepare 的一次原调用、128 帧边界和正常 detach | collector 的实际 Python API；不能证明真实 worker、torch.cuda.Event、模型、AIO 或 128-token 真机输出 |
| 四个绑定 hook | observer.invalidate、observer.detach、observer.scalar._fail、observer.scalar.detach 均为各自实例的 bound method；安装后保留四个 weak bindings | 实际原 attachment 的绑定身份和异常清理；不能伪造四个 hook 占位来声称完整接线 |
| 原 Queue 的观测 wrapper | 原 queue.Queue 实例，原 Queue.get unbound 方法一次调用、原 block/timeout 参数不变；16 条 value witnesses、8 条 failures 上限保持 | 实际安装/原调用/恢复和有限 value evidence；没有 GPU 完成证明或观测开销测量 |
| 正常退出 | 原队列排空后 registration/armed token 清除，capture detach 后 audit.close 恢复自己的 Queue.get，不覆盖后装 wrapper | Python 生命周期条件；不能提前直接清空状态来伪造正常 drain |
| 原生共同成本与 on 成本 | common 校准 bridge=None；正常 on helper/Queue wrapper 成本需要单独真实覆盖 | CPU 只验证接线。source 闭包包含 helper 不代表其成本已测；共同成本不能覆盖 active notification 或完整 on 生命周期 |

下一轮完整入口需要的有限语义案例计划如下。它们是额外的完整入口检查内容，尚未执行的新案例不计作本轮分数；不追加新性能阈值、等待时间或成本数值。

| 案例 | 需要覆盖的完整入口行为 |
|---|---|
| off 与 shadow | 实际 collector.install → 正常 helper；off 无 bridge，shadow 用原 shadow bridge；均不装新等待或 Queue observer；原返回与 detach 保持 |
| on 匹配唤醒 | 实际 collector.install → 原 bridge attachment → 四 bound hooks → helper/Queue observer；原 incoming queue 消费相同 value token，正常排空并恢复 |
| on 原截止到达 | 原 freshness/arrival/max_wait 的最早截止保持；不得 CPU fixture 制造成本凭据来授权真实等待 |
| mandatory / parent pending / physical pending | 原有强制与资源依赖优先，不能把未释放资源当完成或把可选等待延长 |
| stale wake 与 invalidation | 旧 token 不当新完成；任一原 hook/invalidation 正常取消当前等待，且保持原函数一次调用 |
| STOP / finish / late wrapper | 原 drain/finish/STOP 返回；weak registration 排空；detach/audit.close 不覆盖后装 wrapper |

本目录不会执行上述性能比较或发放 GPU 授权。新的 CPU combined 结果、旧 132 次比较、后续原生六窗与 off→shadow→on 的证据应各自保留身份。全部 GPU/native/full-runtime/on-observation 资格标志保持 false，UUID、有效 receipt/cost/budget 保持 null。

推进顺序必须避免循环依赖：首先通过 CPU resource preflight（仅资源先决条件）；然后在实际同 UUID 授权与新冻结闭包下运行真实 GPU common 六窗校准，按原公式/holdout签发真实原生成本 receipt；之后才运行 off→shadow→on 的真实生命周期并核验完整 CPU/Queue observer 成本及通知覆盖；这些条件满足后才能开始 P4 正式对照。common 校准 bridge=None，因此不依赖 on 安装成功；完整 on 成本资格不能冒称已在 common 校准之前通过。严格 canonical receipt 拒绝 CPU synthetic 值时，报告该边界，不私造 typed receipt、不绕过 issuer，不把 on 未安装的 CPU timing 当完整 on 资格。

```sh
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$RESOURCE/resource_preflight.py" \
  --project-root "$ROOT" --source-lock "$LOCK" --output-dir "$RESOURCE/SERVER_PREFLIGHT_01"
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$RESOURCE/run_cpu_resource.py" \
  --project-root "$ROOT" --source-lock "$LOCK" --output-dir "$RESOURCE/SERVER_CPU_RESOURCE_01" --location server_cpu
```

各 output-dir 必须尚不存在；所有外层闭包行在操作前后核验。第一条命令只采样一次，不循环等待资源变化；RESOURCE_LIMITED/RESOURCE_UNKNOWN 的退出码 0 表示报告成功生成，resource_ready 仍为 false，不能当作资格通过或 GPU 操作许可。
