有界公开请求数据的 CPU 取得、检查和处理已有授权，无须再按数据文件逐项请求许可。此次审查没有取得数据，也没有运行 GPU。

实际 `permissions.yaml` 与迁移保留的原文件字节相同，SHA-256 为 `795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50`。第 6–8 行明确允许项目内修改、CPU 测试和公开来源读取。用户采用实施交接包并持续要求完成 CPU 验证、所有无卡工作和租 GPU 的前置准备，因此公开请求及其来源元数据的有界取得、项目内留存和 CPU 处理属于必要准备工作。

交接文件的指令与用户授权须区分：用户明确采用 `04_CODEX_EXECUTION.md` 的项目实施范围；文档不能自行产生新的支付、租用、删除或系统修改授权。后来的直接用户指令“授权gpu，只要有gpu就用，以后无需授权”保留在持续授权 amendment 中，已替代旧逐轮 GPU 许可提问；它不增加原八小时预算，也不改变实验有效性门禁。

单次 GPU 作业的限制不能扩大成全项目的 CPU 禁令。原 `run_gpu_stage.py` 第 44–55 行针对非原始 permission 路径要求 `allow_model_downloads=false` 等较窄能力；这里的字段明确是模型下载。旧有 GPU no-download 约束仍完整保留，但不撤销一般项目的 `allow_public_source_read=true`。持续 GPU amendment 的“不扩展 downloads”表示该 GPU 指令没有增加下载能力，不表示撤销早已存在的 CPU 公开来源读取。公开数据须在 GPU child 之外取得并在启动前冻结，不能借 CPU 准备让 GPU child 改为在线下载。

| 输入缺口 | 门禁是否合理 | 能否继续 CPU 工作 |
|---|---|---|
| 实际 dataset 为空 | 合理；当前 raw producer 的 `UNBOUND_NO_ACTUAL_DATASET` 说明没有真实输入，并非权限不足 | 可以自主检查、取得合适的公开请求源，再按原字节绑定和分词 |
| 会话／文档／公共前缀家族闭包 | 对正式分区与收益声明合理，避免开发和评估泄漏；token IDs 或任意 labels 不能替代来源 | 可以从公开数据真实元数据核验，不要求只能由用户手填；缺证据仍可生成 raw IDs，但保持家族证明未绑定 |
| 独立 deadline／SLO | 对策略预算和正式评估合理；不能由 A/B 成本、Amax 或已观察收益倒推 | 可以整理已有服务或独立实验要求，事前冻结有依据的目标；它是实验输入，不是另一份 GPU 人工授权 |

真实 `freeze_trace` 检查全部请求完整保留、连续 calibration/development/evaluation 分区，以及家族／相同 prompt 不跨分区。来源可从真实公开 metadata 取得；若现有工具只输出 raw IDs，真实家族闭包 producer 仍是代理可以完成的 CPU 准备工作，不能永久把“提供完整 receipt”推回给用户。对没有支持元数据、无法连续切分的来源，应如实拒绝，不能造独立家族。

deadline 的现有 authority 契约核验实际字节、schema、事前独立 origin、相同正数 deadline 和 SLO，没有要求人类签名或私有外部 issuer。因此可以依据可核验的服务目标或独立前置实验要求准备声明；不能编造“现实服务的既定目标”，也不能把大于实测成本的任意数字包装为独立来源。预算 reserve 仍必须来自真实 development 的四类控制区间，CPU 不能代替。

发现一处范围说明差异：`P4_NEXT_INPUT_REQUIREMENTS.json` 表示 development SLO 可能未绑定，但现有 `formal_trace_binding.py:321–334` 对每个正式分区都强制有效 service SLO。当前真正入口因此比说明更窄。应按实际 validator 报告阶段缺口，不能绕过检查；这不阻止公开 CPU 数据读取、作者 parser 检查或 raw 分词，也不是缺少用户许可。

下一步允许自主选择符合现有边界的完整公开输入文件，记录 URL／版本／实际 SHA 与完整原记录映射，冻结后运行作者 parser 和现有 raw CPU tokenizer。保留 32 MiB、完整接受请求最多 96、每区最多 32、每 prompt 最多 4096 tokens 等既有界限；不能截尾、重排、删慢请求、人工追加前缀或改家族 label 来过关。新数据应使用项目内新路径，保持既有存储限制与余量。原模型执行器、缓存底座和旧冻结代码不修改。

本审查只读权限、交接指令、授权 amendment、旧 effective GPU permission、原 guard、protocol、formal consumer 和 raw producer。没有 SSH/RPC、下载、分词、模型导入或 GPU 操作；只新增本目录审查文件。未创建自然数据、家族 receipt、SLO 或策略收益证据。
