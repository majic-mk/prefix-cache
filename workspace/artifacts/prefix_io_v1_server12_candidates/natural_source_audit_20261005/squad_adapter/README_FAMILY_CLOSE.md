# 实际 SQuAD CPU 分词与家族回执闭合

`close_squad_tokenizer_family_cpu.py` 是一个有限 CPU 闭合工具，复用已冻结 raw producer 的真实本地 backend，不实现分词器、缓存引擎、模型或调度器，不下载、不调用 SSH/GPU，不声明 deadline、SLO 或性能提升。

输入 request 固定为 `CPU_squad_tokenizer_family_close_request_v1`，精确字段如下：`source_ref`、`selection_contract_ref`、`trace_ref`、`mapping_ref`、`adapter_report_ref`、`adapter_source_ref`、`raw_producer_source_ref`、`protocol_source_ref`、`author_trace_ref`、`author_common_ref`、`raw_tokenizer_request_ref`、`raw_result_relative_path`、`block_size=16` 和 `schema`。各 ref 必须精确 `{path, bytes, sha256}`，全部进入 V13；固定 raw 输出路径只在 actual raw CPU 作业完成后按真实字节取得 ref，避免 raw 结果引用 V13、V13 又引用结果的循环。closer 自身也必须进入 V13。

V13 必须为原 append freezer 产生的严格 CPU schema；核验实际 V12 metadata 原 SHA `23ab34887c7f4340a37f2814936c639c46890ca31e4b3ba1b07359d41fc13f77`、全部 4,953 个原引用完整逐行相等继承、addition manifest 与原 `63db...` append freezer 的真实引用，且只有声明叶，没有同名伪锁、未知额外叶或 self-cycle。该步复验 metadata 继承，不重新读 16 GB 模型/SDK；实际使用的有限来源叶均前后真实 hash。

官方源额外固定为 payload SHA `95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9`，重算 Git blob SHA `e9a3f913ad1468ebe105b891334ca7b0bc0e2510`，而非仅信来源自报。取得过程与选择合同先序时刻仍由根执行者独立保存的真实证据说明。

adapter v2、raw producer、原 protocol 都从已经核验的精确 SHA 源字节直接 compile/exec，不读取 pycache。首先重新执行完整 `replay_adapter`；原 source、问题映射、分区、来源家族、raw/accepted prompt SHA 都要一致。随后用原 `inspect_trace` 与原作者 AST 确认全部 accepted prompt。actual raw result 必须真实成功、synthetic false、未导入模型/GPU、backend 非 mock、源锁/请求/模型/完整来源集合和 inspection 精确相同；raw artifact 的 `family_proof_available=False` 等原字段只读保留，不重写原结果。

再次通过原 `actual_local_backend` 编码全部 accepted prompt，对比每份原 token IDs，并单独编码三份真实 context 的原起始边界。来源 family 由重放的 source/article/context 身份推导；同 article ancestry 不能跨 split，重复 context 不能跨 split。每条 prompt 的完整首 16 个 token 必须被该真实 context 覆盖，家族内首块一致；任何跨分区相同首 16-token 完整缓存块拒绝。该检查只支持块大小 16，不偷偷变更分区或选其他文章。

通过后写两个原协议可消费的 CPU 文件：`actual_cpu_tokenizer_result_v1` 与 `cpu_actual_tokenizer_family_receipt_v1`。记录只有原 raw 行字段加 `prefix_family`，结果中的 `tokenized_records_sha256` 是完整含 family 行的摘要。receipt 的源码与结果引用使用实际绝对路径；额外 evidence 保留独立 source/mapping/adapter report/raw-result/原请求与 V13 ref，以及真实再编码和首块检查。证明范围为整个派生的已选 trace，不能扩展为完整公开语料或生产会话。

结果与 receipt 各以 `xb` 追加一次，父目录必须已存在且输出互异。所有实际输入校验通过前不写任何输出，任何失败不写 receipt；首次结果写出后再次核验全部实际输入，若外部并发漂移，则保留已写文件供审计而不发布 receipt。已完成两份输出须进入**后继源锁**才可供正式消费者使用，不能反写进原 V13。

```bash
python -B -I -S <已冻结closer路径> \
  --project-root <实际项目根> \
  --source-lock <实际PRERENT_SOURCE_LOCK_V13.json相对路径> \
  --request <已冻结family请求JSON相对路径> \
  --result-output-relative <既有artifacts目录中新结果JSON> \
  --receipt-output-relative <既有artifacts目录中新receiptJSON>
```

成功 exit 0；缺真实输入或任一门失败 exit 78。CPU 文件不授予 GPU/effect 资格，完整 trace 声明、独立 deadline/SLO、实际模型输出/资源生命周期和策略结果仍必须后续独立验证。

`test_close_squad_tokenizer_family_cpu.py` 的 16 个测试只用显式 synthetic source、纯函数记录、mock header predicate 和实际源字节 loader fixture；不执行真实 tokenizer 库，不调用完整 receipt writer。测试覆盖来源身份、article/context 泄漏、跨分区同首块、实际 context 覆盖、缺/改/错序记录、布尔/非法/短 token、raw fixture/mock/继承资格拒绝、合法 timestamp pyc 拒用，以及实际 V12 4,953 行 metadata 继承/伪造 schema/缺祖/缺叶/改叶/重复/bool/未知叶拒绝。metadata fixture 不复制、读取或加载模型字节，也不是实际 V13 冻结证明。CLI UNBOUND 子进程不写文件。

本 agent 实际 tokenizer 库执行、网络、SSH/RPC、GPU 和 actual family receipt 交付均为 0；实际 server raw+family CPU 作业由根执行者在 V13 冻结后执行并单独记录。
