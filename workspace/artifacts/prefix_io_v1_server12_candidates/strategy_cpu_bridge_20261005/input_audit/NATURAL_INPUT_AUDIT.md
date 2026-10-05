# 自然输入与开发阶段 SLO 的只读 CPU 审计

2026-10-05。本审计读取现有小型源文件、真实服务器只读清单及原交接 ZIP 成员，没有连接服务器、执行 tokenizer、加载模型或共享库、运行 GPU、修改旧冻结源。13 份本地输入的字节 SHA 在审计前后相同；完整来源与范围在 `NATURAL_INPUT_AUDIT.json`。

## 已有材料与检查范围

真实 Qwen tokenizer 资产存在，可以作为后续实际 CPU 分词的输入：

- `models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444/tokenizer.json`：7,031,645 B，SHA-256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`。
- 同目录 `tokenizer_config.json`：7,305 B，SHA-256 `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583`。
- 作者 `third_party/upstream/kvcache-experiments/scripts/shared_storage_trace_replay.py`：11,845 B，SHA-256 `83713db1b011a954616896ec0ccfa05dc766805e80ed8defdef0da65100ff99e`；`common/prefix_cache_common.py`：9,566 B，SHA-256 `a331d1595b815c3c8fc63d3e2f236264c2f181ece7a9de51f50beb4703338df5`。这两份实际源码可作为下一版本追加的源叶，保留 V10 原有全部行。

这些是父代理实际服务器检查返回的来源记录；本代理没有再次远程哈希。tokenizer 文件存在不等于已经执行实际分词或发行家族收据。

V10 中相关 43 个源引用没有实际自然请求数据或其分词收据。服务器有界文件名检查在 `experiments/prefix_io_v1` 查看 53,094 个文件，144 个候选全部属于 `build-cache` 工具、测试或算法示例；剔除后 0 个。另一范围为 `third_party/upstream`，排除 vendor 元数据、私有模型和构建目录，查看 9,647 个文件，4 个候选是上述作者源码与两份 573 B 的 OpenAI batch 示例。本审计不能证明范围外或非匹配文件名中的数据不存在。

旧 `low_contention.json` 与冻结的 12 个 P3 请求是受控资格输入；原协议明确 `natural_trace_bound=False`、`fit_or_evaluation_input_allowed=False`。batch 示例、单元测试里的 alpha/beta 输入和手写 receipt/result 也不能变为自然评测证据。

## 开发阶段目前多了一道正式 SLO 门槛

`formal_trace_binding.py:326` 对所有 partition 调用 `require_independent_service_SLO`，因此真实 development 输入即使有独立 deadline，也会因 `service_SLO=None` 被拒绝。该文件的 370–373 行同时声明支持 development/off/U 与 development/shadow/I。

原 `prerental_protocol.py:72–79` 明确允许 `service_SLO=None`；328–340 行仍要求正的独立 full-control-window deadline、实际 authority 字节和同 host/boot 的 monotonic clock。原交接包第 03 文件第 121 行允许在 development 选择 SLO，第 203 行要求正式 evaluation 前冻结 TTFT 与请求内 ITL 阈值。

建议只在新 CPU 修订中允许开发阶段不提供正式 service SLO：已有 SLO 必须按原严格合同校验，独立 deadline、authority 与时钟证明继续强制。正式 effect/evaluation 的 U/off 和 I/on 继续要求完整且独立冻结的 SLO；缺 SLO 不得报告 formal goodput。不能从 A-max、B/on 观测或首次效果结果反推 deadline，也不能填一个方便通过的数值。旧冻结文件保持原样。

## 最少仍缺的真实输入

1. 完整自然请求数据及来源；真实会话、文档、公共前缀家族元数据。保留作者接受的所有请求与原顺序。
2. 在现有 tokenizer 资产上实际执行的 CPU 分词结果，闭合实现、模型与资产 SHA，记录每个原请求的 token IDs 与家族证据。当前源码只有 receipt 消费合同，没有发现生产收据的实际执行记录。`family_proof` 字符串和跨分区家族标签检查不能自行证明标签来自真实会话；文本或 token 哈希也不能代替此来源。
3. 在新 GPU 结果之前冻结的 calibration/development/evaluation 分区和调度声明，拒绝家族泄漏，不注入前缀、不逐请求重置缓存、不删除困难请求。
4. 独立的 full-control-window deadline 及真实 authority、同 host/boot 时钟声明。开发可以暂缺正式 TTFT/ITL SLO，但此独立期限仍必需。
5. 两臂独立的全分区命名空间、实际强基线与源绑定。CPU 可以准备；实际 I/shadow development reserve 需要后续 GPU 原始输出、完整控制区间、CUDA/I/O、排空、guard 与源回放，不能在 CPU 伪造。

已完成的 cal05 只发行一个精确成本 cell。它没有提供自然工作负载、I-development reserve 或正式 SLO，策略 GPU 次数仍为 0，不能据此声称方法性能提升。下一允许步骤是补齐真实 CPU 输入与有限合同修订；仅开 GPU 无法补足缺失的数据或独立期限。
