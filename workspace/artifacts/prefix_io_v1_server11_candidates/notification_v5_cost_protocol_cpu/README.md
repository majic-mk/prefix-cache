# C5 共同成本与通知资格：有限后续协议，CPU-only

结论：原六窗口是 **共同 I/O 条件成本** 的校准结构，不能证明 on 通知及 Queue 观测开销已经测量。共同校准、严格 receipt、off→shadow→on 的实际通知生命周期资格必须依次进行。当前没有新的 C5 原生数据、可消费成本 receipt 或 GPU 授权；本目录只定义/校验协议，没有实现新 GPU runner，没有启动作业、扩大预算、重跑性能 score 或修改任何冻结源。

## 由实际源码确定的边界

| 冻结源码位置 | 已确定的事实 | 不能推出的结论 |
|---|---|---|
| native_cost_v6/run_native_cost_experiment.py:424，native_source_binding:122 | 原 coordinator 明确 `p4_bridge=None`，最大 parent=8；六窗代码没有 attach_single_file_wait 调用 | 源码锁里列了 adapter/collector，不代表执行了 active policy、通知注册或 armed Queue.get |
| 同文件:40、503–523 | AB/BA/AB；每窗新进程、原 preload，选 offset16，128 个原始事件帧 | 旧 C4 六窗结果不能作为新 C5 数据；现 runner 仍有 C4 的硬路径绑定，需未来独立新薄接线/冻结 |
| native_conditional_cost.py:490–516 | 只用前两对 calibration 拟合原公式；最后一对独立 holdout 检查，不重拟合 | holdout 通过不是概率/SLO 保证，也不是通知策略性能提升 |
| C5 p4_single_file_receipt.py:120–138、236–275 | A-only budget=两 calibration A 的向上取整均值+最大正残差；当前工厂硬绑 C4/v6 | 不能私构 issuer、换 job 名或塞入 CPU 数值形成新 C5 receipt |
| C5 p4_policy.py:298–319 | 同条件下 upper≤A-only budget 时 issue，否则仍需全部 live checks 才 defer | 不允许调阈值、放宽条件、改工作量以人为触发等待 |
| runtime_v4/verify_p4_single_file.py:38–44 | off/shadow 的 selected GPU duration 必须被 common upper 覆盖；on 返回语义资格不要求该覆盖 | on 语义通过不能解释为 common upper 已包含 on 成本 |
| 新 preparation launcher:603；adapter prepare_notification_capture | 仅 on 实际 attach 通知及原 Queue.get 观测；共同六窗没有这个调用 | 22 项 CPU 接线通过或原完整 pump 比较不等于该最终 runtime 额外观测开销已经合格 |

这些 12 个实际文件的字节/哈希在 `SOURCE_PINS.json`；CPU runner 前后核对，并用 AST 确认原六窗只传一次 `p4_bridge=None` 且无通知安装调用。

## 保持原范围的依赖顺序

1. **先完成 CPU 环境资格与最终 source/factory 准备。** 0.5 核/2 GiB 环境不能把已有不合格 timing score 变成合格。后续充分资源下的 CPU 资格必须包含最终 adapter/launcher 的实际路径；本轮不重跑。冻结最终同一 reactor/collector、原 model/Event/runner、原引擎选项以及实际 runtime 观测接线。最终同源并不代表所有列入文件均已在共同六窗执行。
2. **另获限定授权后，重新执行 C5 共同六窗。** 可以继承原 AB/BA/AB、两对拟合+一对 holdout、原 128 帧/实际原生 I/O 验证和原数学。仍 bridge=None、不装通知、不装 Queue observer；因而会测到 C5 未安装等待的共同分支，但不会测到 active 额外路径。新 raw parent、六个 fresh child、实际模块路径/`_run` code filename、实际 collector 每帧归属、前后 source refs、原预算 guard、输出相等与原 shutdown 全部必须重做。只改常量或用旧 summary 不能通过。
3. **严格工厂消费实际新 raw，再发行有限 common receipt。** common upper=`baseline + incremental_or_joint + uncertainty`，仍用冻结 `ceil_calibration_means_plus_max_positive_residual_v1` 数学；A-only budget 不含 B 或 holdout。collector 继续在 overlay refs，额外严格要求其 ref 等于新 calibration plan 的实际 collector ref，不能移出 overlay 破坏原 bridge 检查。receipt 只支持该 common I/O 条件，没有通知成本、生产资格或启动权限。
4. **另获限定 runtime 授权后，按 off→shadow→on 验证。** off/shadow 保持原 migration gate；前一臂失败停止后续。每 fresh reactor 一个 128-token 测量 capture。owner/capture run_id 使用 LABEL，外部 request 仍 RID，逐帧 native_request_id 保留原值。原校准 serializer 的 capture=外部 RID 契约不变：不能直接拿新 runtime 输出冒充六窗 raw，不能删掉身份检查来混合两种契约。
5. **on 只在原策略有资格且真实触发时取得通知生命周期证据。** 证明同 capture/reactor/bridge 安装、matching original Queue wake 或原最早 deadline、原 reserve/preview/live checks 重走、无注册泄漏/故障/溢出、原 drain/shutdown。记录 selected 和全 128 GPU 时长、整请求与 drain 的 host 时长、Queue 值证据；这些是该次组合 runtime 的观测，不能从一组 off/on 差值分离“纯通知/纯观测成本”，也不能直接称为性能改善。旧 on gate 不提供 cost coverage。

因此没有数学上的循环：共同六窗不消费 policy receipt，先得到 common receipt，再允许有限策略生命周期验证。通知额外开销仍未知时，只能在新的明确授权下做有界资格诊断，不能把它提前称为已获成本资格。不得把 on 时长或 synthetic CPU 开销加回 upper、重拟合 receipt 或改变原 A-only budget。若研究要求完整策略的有效提升，单次有限资格之后仍缺独立、预先确定的性能对照协议与授权；本轮不扩展该实验。

若新 upper≤A-only budget，原 policy 正常 issue。此条件下没有目标延期可供通知优化，输出 `NOT_EXERCISED_NO_FORCED_DEFER`，停止目标通知晋级，不强造一个 on 等待。若 upper>budget 但实际运行因 live state 失效从未进入合法 wait，输出 `ON_NOTIFICATION_NOT_DEMONSTRATED`；这是未触发/条件不足，不能算通知已验证。

## 固定条件与预算边界

`PROTOCOL.json` 固定 129 prompt、128 output、offset16、单次 917504-byte SSD read、parent8、batch1/active_decode1/prefill0/context144、无原在途 I/O、100ms sample freshness 与 max_wait，保留原共同三组 prompt-first/seed 及 runtime prompt-first/seed。共享 staging、精确 Prefix Cache、原 preload/复制合并/异步流水线均沿用源码，不增加 D/J 或其他负载。

预算字段只是旧代码的上限参考：原累计上限 28800 秒、共同六窗 1200+20 秒、runtime 每臂300+20秒。当前剩余、分配与新授权均为 null。它们不继承旧作业授权、不创建 scope，也不承诺未来一次用足这些额度；必须由父流程根据真实 ledger 和用户限定授权再次决定。本协议没有 GPU 作业名称或启动命令。

## 可执行的 CPU 协议检查

`check_cost_protocol.py` 严格核对 schema、固定条件、原公式、共同/active 路径区分、无 cost 数字、无权限提升及无 receipt 依赖环。`simulated_next_obligation` 只接受显式 `synthetic_protocol_test` 的枚举关系，不接收数值成本；即便所有 symbolic stage 都通过，也永远返回 gpu_launch_allowed/native_execution_verified/receipt_issued=false。

服务器由 root 执行（先上传本目录六个源/协议/文档文件）：

```sh
ROOT=/root/autodl-tmp/prefix-io-v1-handoff/project
D="$ROOT/artifacts/prefix_io_v1/server11-c5-cost-protocol-cpu-20261004"
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$D/run_cpu_protocol.py" \
  --source-lock "$LOCK" --project-root "$ROOT" \
  --candidate-root "$ROOT/artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003" \
  --native-root "$ROOT/artifacts/prefix_io_v1/server11-native-cost-v6-20261003" \
  --preparation-root "$ROOT/artifacts/prefix_io_v1/server11-c5-runtime-preparation-cpu-20261004" \
  --runtime-v4-root "$ROOT/artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003" \
  --output-dir "$D/SERVER_CPU_PROTOCOL_01"
```

服务器 `LOCK` 必须指向 root 冻结的 `c5_cost_binding_cpu_source_lock_v1` 外层 CPU 锁。runner 在前后检查其中全部 files（包含本目录六文件和12个实际来源），拒绝遗漏、重复路径、改字节、符号链接或 native/GPU 资格声明；保存 source_lock_sha256，必须 source_before==source_after。本机先前无外层锁结果会明确记 null，不冒充服务器闭包。

输出目录必须新建；保留完整 18 项反例日志、前后 source hashes、命令与实际统计。测试只验证协议，未调用六窗 runner、真实 collector.install、torch/vLLM 或任何 GPU API，没有生成性能 score。当前尚待实现和真实授权的内容：最终 C5 严格工厂/新 raw 接线、充分 CPU 资源下最终路径资格、新共同原生校准、三个 runtime 臂及其独立结果审查。旧 preparation 的 GPU 硬阻断仍有效。
