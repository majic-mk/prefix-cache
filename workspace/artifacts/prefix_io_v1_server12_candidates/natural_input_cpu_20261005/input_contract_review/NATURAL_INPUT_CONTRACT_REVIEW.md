# 真实自然请求输入：文件格式与家族来源要求

2026-10-05。本轮只读原作者 parser 缓存、协议、formal validator、上一轮输入审计和原交接 ZIP。没有造数据或家族标签、下载、RPC、分词、模型导入或 GPU 操作，旧数据和源锁未改。6 份来源前后 SHA 一致；细节见 `NATURAL_INPUT_CONTRACT_REVIEW.json`。本机 `docs/prefix_io_v1/` 不存在，因此原文依据实际读取的原用户 ZIP 成员，未声称已读取不存在的本机目录。

## 需要提供什么文件

请提供**已经存在的真实请求文件的实际路径**，以及来源说明和真实会话／文档／公共前缀关联。优先保留原 `.jsonl` 或 `.json` 文件，不为了通过测试重写 prompt 或生成一批请求。

作者 `shared_storage_trace_replay.py:33–97` 实际接受：

| 文件 | 接受格式 | 注意 |
|---|---|---|
| `.jsonl` | 每个非空行是 JSON 字符串或对象 | 对象取首个非空字符串字段：`prompt`、`text`、`input`、`content`，按此顺序；之后才尝试 `conversations` |
| `.json` | 顶层列表，或顶层对象中的 `data` 列表 | 单个顶层 `prompt` 对象不作为一个请求读取 |
| 普通文本 | 每个非空行是一个 prompt | CSV/TSV 没有专门解析；标题和逗号会变成 prompt 内容；压缩文件也没有专门支持 |

建议真实文件使用 UTF-8、无 BOM，并确认服务器的实际默认编码。原 parser 的 `open()` 未显式设置编码。JSONL 对象只能靠上述实际字段解析；OpenAI batch 的 `body/messages` 或普通 `messages` 数组不会自动转成请求。

直接字符串会删除首尾空白。`conversations` 只保留 `from.lower()` 为 `human` 或 `user` 的 `value`，删除首尾空白后用两个换行连接；assistant、system 轮次被忽略，没有 chat template。它将多个用户轮次合成一个 prompt，不是完整多轮会话回放。角色值带首尾空白不会被自动纠正，非字符串 `value` 会经 `str()` 转换。因此实际 CPU 检查必须记录这种解析，不得把结果声称为原对话的忠实重放。

原文件中空或不支持的项不产生请求。解析错误会中止；重复 JSON 键也没有原生拒绝机制。正式工具应报告原记录到接受记录的映射，拒绝歧义，不能悄悄抹掉不合格式的记录后宣称原始数据全部保留。

## 哪些来源足以支持分区

parser 输出的是接受顺序中的 `request_id=0,1,...` 和处理后的 prompt；它不保留原 ID、行号、会话、文档或到达时间。真实元数据必须能够闭合“原文件 SHA／原行或数组位置／原 ID → 作者接受序号 → 实际处理后 prompt SHA”。用户已有的字段名称可以保留；下面是证据要求，**不是已实现的侧车 schema，也不是要手填 family 标签**。

- **会话**：真实日志或导出中的稳定 session/conversation ID、原记录对应关系，以及覆盖整个输入的导出来源。同一会话的多个请求或轮次不能被任意拆成独立家族。
- **文档**：实际 document ID、文档字节或版本／谱系清单、请求到所用文档的映射。同一文档家族不能跨分区；仅换一个记录编号不代表新文档。
- **公共前缀**：实际使用的服务模板、system/public prefix 来源字节或冻结清单，以及哪些请求使用它的对应关系。保留真实渲染 prompt 的证据，不能向请求追加一个新前缀来制造复用。
- **确实没有某类关系**：需要真实源／导出格式或独立语料说明支持这种缺失。某个 JSON 字段不存在，不能自动证明请求独立。

分区必须同时遵守这些真实关系和相同处理后 prompt 的关系；多种关系连起来形成的完整组不能拆开。只 tokenize 能得到 token IDs 和精确文本关系，不能证明所有会话、文档和公共前缀关系。`family_proof` 字符串、每行一个任意 label、`synthetic_fixture=False` 都不产生真实来源证明。

当前旧 `freeze_trace` 要求并封存 `prefix_family`，检查同一家族／相同 prompt 不跨分区，但**不直接读取元数据侧车，也不从侧车重建家族闭包**。给 receipt 多加一个 `family_source_refs` 字段不会自动获得校验；实际 CPU 生产工具必须先核验来源字节和完整映射，才能生成来源支持的结果。

如果只有 prompt 文本而没有可核验的家族来源，可以继续 CPU 检查与真实分词，但家族证据保持 `UNBOUND`。如果真实组交错导致原顺序不存在合法连续切分，应报告当前数据不满足当前协议；不得重排、删请求或改 label 解决。

## 当前输入规模与到达计划

协议使用 `max_prompts=None` 读取作者接受的完整输入，再按原顺序作 calibration、development、evaluation 三个正数连续分区；切分数之和必须等于全部接受数。每区最多 32 个接受请求，整份源最多 96 个；文件不超过 32 MiB，每个 prompt 最多 4096 个 token，且必须满足冻结模型上下文容量。

当前 V3 固定完整生成 128 tokens，还要求分区估算完整采集步数不超过 4096。32 是行数上限，并不保证 32 条一定满足步骤上限。不能截尾、减少输出或删除长请求来通过检查。

请求到达计划来自作者 `build_global_specs`：正 arrival rate 使用冻结 seed 的累计指数间隔，否则全部到达时间为零。它不重放原文件时间戳。因此可称“真实 prompt 的作者计划 trace replay”，不能宣称已保留真实生产到达时序。

## 可操作的路径与下一步

服务器项目根是 `/root/autodl-tmp/prefix-io-v1-handoff/project`。可把**已有真实文件按原字节复制**到新的 `experiments/prefix_io_v1/datasets/natural/<真实输入标识>/raw/`，保持原文件名；实际来源和关联文件放在同批次 `provenance/`。这是存放位置建议，本审计未创建这些数据目录或文件，也未声明已有自然数据。

随后应记录实际路径／字节数／SHA，运行原协议的 CPU `inspect`，核对完整原记录映射，再进行真实 CPU 分词和家族来源校验。CPU inspect 命令模板及两个实际作者源路径在 JSON 报告中；模板没有执行，也没有填造数据路径或 SHA。

用户目前最需要提供的是：现有真实输入文件路径、来源，以及真实 session/document/public-prefix 关系文件或可核验的不存在说明。独立 deadline／authority 和前瞻切分也须在真正 development 前完成；开发阶段可以缺正式 service SLO，正式效果评测仍需要冻结的 TTFT 和请求内 ITL P95 目标。仅开 GPU 无法补足这些缺失输入。
