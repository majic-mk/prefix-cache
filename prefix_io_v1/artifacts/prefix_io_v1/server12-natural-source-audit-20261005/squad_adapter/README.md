# 固定官方 SQuAD 来源的 CPU 适配与回放

本模块只做来源适配，不下载、不使用 SSH、分词器、模型或 GPU，不产生正式 tokenizer/family receipt、自然工作负载 manifest、deadline/SLO 或性能资格。

根执行者必须先在下载之前冻结 `prospective_official_squad_fixed_articles_v1` 合同，之后取得原始官方文件、保存实际获取证据并把源和合同加入实际源锁。固定 commit 是 `eee5fdbf62f8613a7812b03419e6b29617b74fd1`，仓库 `rajpurkar/SQuAD-explorer`，文件 `dataset/dev-v1.1.json`。模块通过源字节 SHA 绑定输入，但不能独立证明网络获取真实性或前瞻合同时间；报告对此明确返回 false。

选择规则全部固定于 `FIXED_CONTRACT`：原 `data` 的索引 0、1、2 三篇文章，各自只取 `paragraphs[0]`，依原文章顺序映射 calibration、development、evaluation。每段全部 qas 原序保留，1..32 条，合计不超过 96。无效或空字段、重复 qid、缺少首段、错误标注 span、超限均整体失败，不能替换文章、改选其他段落、选前 N 条或按长度/表现筛选。

每条 prompt 严格为原 `context + '\n\n' + question`，不附加回答，不执行 trim/改写。若原作者 `.strip()` 会改变构造后的 prompt 字节，就整体拒绝。源最多 16 MiB，必须是完整 UTF-8 SQuAD v1.1 JSON，重复 JSON key 也拒绝。

`validate_adapter(source_bytes, selection_contract)` 返回 `{trace, mapping, report}`：

- `trace` 是原作者接受的 JSON list，每条只有 `prompt`。
- `mapping` 保留整个源的字节 SHA/大小、全部文章的原序 title 元数据、所选三段的全文 context、逐条 raw position/article/paragraph/qa 索引、原 qid/question 与 SHA、prompt SHA、分区。原始源全文由根原样保留；本模块不再复制包含答案的完整源到输出。
- 所有来源分组都由 raw-source SHA、固定 commit、article index/title SHA、paragraph index、context SHA 与 partition 重建。`source_prefix_family` 是该身份的确定性摘要，绑定真实文章和上下文；它仅为来源分组，**不是已验证的 token 前缀覆盖**。
- `report` 明确 `formal_receipt_ready=False`、`tokenizer_prefix_coverage_proven=False`、`natural_production_traffic_claim=False`、`GPU_launch_allowed=False`，实际 GPU/网络操作均为 0。公共 QA 文档不能被称为真实生产请求流量。

`replay_adapter(raw, contract, trace, mapping, report=None)` 从原始源重新构造完整所选结果，逐字节等价比较 canonical JSON。它不会把提供的分区/家族标签当作证明。漏问、改问、重排、上下文注入或来源分组漂移均拒绝。

CLI 只读现存文件，输出通过 `xb` 各追加一次。写入前检查三个输出均新、路径互异、父目录已存在；不创建目录，不覆盖旧结果。验证失败时不写结果。

```bash
python -B -I -S squad_source_adapter_cpu.py \
  --source <实际原始官方JSON> \
  --selection-contract <下载之前冻结的实际合同JSON> \
  --trace-output <既有目录中新的traceJSON> \
  --mapping-output <既有目录中新的mappingJSON> \
  --report-output <既有目录中新的CPU报告JSON>

python -B -I -S squad_source_adapter_cpu.py \
  --source <同一原始官方JSON> \
  --selection-contract <同一前瞻合同JSON> \
  --replay-trace <实际traceJSON> \
  --replay-mapping <实际mappingJSON> \
  --replay-report <实际CPU报告JSON>
```

成功 exit 0；缺真实源/合同或校验失败 exit 78。占位参数不是实际获取/实验命令。

`test_squad_source_adapter_cpu.py` 中全部 source 和 QA fixture 明确为 synthetic；没有取得官方源。20 项本地标准库测试覆盖固定完整顺序、全部标题/上下文、来源分组复验、漏问/改问/注入/分区/原位漂移、重复 qid、空字段、33 条拒绝、32×3=96 条完整保留、禁止换篇改段改上限、正确 span、原作者 `.strip()`、16 MiB 和版本/duplicate-key 拒绝，以及 CLI 实际子进程的适配→回放、失败不写和追加不覆盖。另用已冻结原协议及原作者 CPU AST 验证每条生成 prompt 原样被接受，三份既有源 SHA 保持不变。

本地测试日志和 manifest 是实际 CPU 接线证据；不属于正式数据回执或 GPU 结论。后续允许阶段是根执行者的真实官方获取记录核验、服务器 CPU suite、实际适配/回放与源追加冻结；分词与 token 前缀覆盖必须随后独立验证。
