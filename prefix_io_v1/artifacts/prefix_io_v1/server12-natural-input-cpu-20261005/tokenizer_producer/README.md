# 原作者解析规则下的独立 CPU 原始分词入口

`raw_tokenizer_cpu.py` 只读取已经在本地、已经进入当前 CPU 源锁的真实文件，按原作者接受的全部 prompt 保序保存 token IDs。它不产生原 `actual_cpu_tokenizer_result_v1`、`cpu_actual_tokenizer_family_receipt_v1`、自然工作负载 manifest、家族证明、划分声明、到达时间、deadline、成本表或 GPU 资格。当前未提供真实数据集，入口处于 **UNBOUND**。

成功文件使用独立 `actual_cpu_raw_tokenization_v1`。每行只有 `request_id`、`prompt_sha256`、`prompt_token_ids`，使用独立 `raw_records_sha256`；`family_proof_available`、`formal_receipt_ready`、`gpu_eligible`、`formal_effect_qualified` 始终为 false，实际 GPU 操作为 0。这些 raw IDs 不能直接替代已有正式家族 receipt。

## 有界来源与执行

请求 schema 为 `bounded_author_cpu_raw_tokenization_request_v1`。绑定请求严格包含以下字段：

| 字段 | 输入要求 |
| --- | --- |
| `dataset_ref` | 明确指定的实际完整数据集；缺失或 null 时，任何分词器导入之前返回 UNBOUND |
| `protocol_ref` | 已冻结原 `prerental_protocol.py`，保留既有 SHA |
| `author_trace_ref` / `author_common_ref` | 已冻结原作者源码；仅由原 `author_cpu_namespace` 提取 CPU AST |
| `model_manifest_ref` | 既有 `modelscope-download-plan.json` 的实际冻结引用 |
| `tokenizer_json_ref` / `tokenizer_config_ref` | 实际 Qwen2.5-7B 本地资产，保留已核验 SHA |
| `tokenizers_package_root` | 项目相对 POSIX 路径，目录名固定为 `tokenizers` |
| `tokenizers_dist_info_root` | 同一 site-packages 下的 `tokenizers-0.22.2.dist-info` |
| `synthetic_fixture` | 必填布尔值；诊断文本、单元 fixture 必须为 true |

所有 `*_ref` 都必须精确为 `{path, bytes, sha256}`，与当前锁的叶完全相等，并在本机实际验证字节。请求 JSON、producer 自身也必须进入该锁。源锁本身通过调用者传入的实际 file ref 闭合。锁继承与完整全项目审计由根执行者完成；本入口只复验本次实际使用的有限叶，不重新 hash 模型权重。

入口枚举 package 下所有 `.py`/`.so`/`.pyd` 与 dist-info 下所有非 pyc/pyo 常规文件，逐项要求当前锁已有引用。四项已实际审计的 backend 核心叶有固定 SHA：`__init__.py`、`tokenizers.abi3.so`、`METADATA`、`RECORD`；还要求 `WHEEL` 闭合。既有 model-plan、tokenizer 资产、原协议和两份作者源也有固定 SHA。解析与分词完成后，所有使用的叶和完整 package 树再次只读复验。

原 `inspect_trace` 与 `load_trace_prompts(str(dataset), None)` 保留作者的 JSON/JSONL/文本和 ShareGPT 规则。作者自然拒绝空 prompt 的行为属于原解析规则；入口不额外挑选、排序或修改文本。不超过 32 MiB 和 96 个作者接受的请求；超限或空集合时整体拒绝，不能选前 96 条。每份 prompt 必须生成 1..4096 个原始整数 IDs，范围 `[0, 152064)`；越界、布尔值、空编码均整体拒绝。

真实 backend 仅调用本地 `tokenizers.Tokenizer.from_file(tokenizer.json)` 和 `encode(prompt, add_special_tokens=False)`。不调用 `from_pretrained`、chat template、transformers、模型或网络客户端。不添加 BOS/EOS，不截断、不补齐，原 added-token 行为由精确冻结 JSON 保留。这是作者接受的纯文本到 IDs；不能声称等同于原 HTTP chat 接口可能添加的模板。

实际导入需要干净的 `tokenizers` namespace，验证 version、实际加载的 package/二进制来源，并临时禁止 framework/model/network-client 导入。临时 source-only finder 将 `tokenizers.*` 的每份 Python 模块直接从已验证的当前 `.py` 字节 compile/exec，扩展仍通过 CPython 原 extension loader 加载已验证 `.so`；原协议也直接 compile 已 pin 字节。已有合法时间戳的 `__pycache__/*.pyc` 不会作为执行来源，也不删除或修改这些文件。退出恢复 `sys.path`、导入 gate、finder 和 bytecode 设置，清除本次 tokenizers 模块。源导入禁止写 pyc。输出采用 `xb` 追加一次，只允许已经存在的 `artifacts/` 目录，不创建缓存/模型目录，不覆盖旧文件。没有真实输入或任何前置检查失败时不写输出。

## 命令

拟部署到 `artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/natural_input_cpu/raw_tokenizer_cpu.py`；实际部署相对路径由根加入新的源锁。

```bash
python -B -I -S artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/natural_input_cpu/raw_tokenizer_cpu.py
```

未指定请求时 stdout 返回 `UNBOUND_NO_ACTUAL_DATASET`，exit 78，分词器不导入且不写文件。`--help` exit 0。

只有真实完整数据集和绑定请求进入新锁之后，才使用：

```bash
python -B -I -S <已冻结的producer路径> \
  --project-root <实际项目根> \
  --source-lock <当前CPU锁的项目相对路径> \
  --request <实际绑定请求JSON的项目相对路径> \
  --output-relative <既有artifacts目录中的新raw结果JSON>
```

占位参数不是实际实验命令。本阶段不创建假数据集引用，也不替用户生成家族关系。

## 本地测试与证据范围

`test_raw_tokenizer_cpu.py` 使用标准库，复制原协议和作者源码的真实字节，fixture 数据和 fake backend 都显式标记。mock `.so` 只是永远不加载的文本占位；模型资产和 backend pins 只在测试对象里替换。positive producer 输出必须 `synthetic_fixture=True`；注入 backend 不能生成 actual 实际输入结果。backend 分派测试只用 import spy，绝未加载真实分词器库。

本地执行 `python -B -I -S test_raw_tokenizer_cpu.py -v`；24 个测试覆盖 UNBOUND 在所有源/模型导入前短路、原作者保序与重复、source 自身/字节/新增包叶闭合、97 条和超大数据拒绝、禁用截断、非法 token IDs、编码期间数据或包叶变动、append-only 输出，以及实际导入分支的来源/恢复 spy。新增可执行的缓存 fixture 重现默认 loader 会读取合法 timestamp 的未冻结 pyc，证明 source-only finder 与原协议路径拒用该缓存、cached-only 子模块拒绝且保留缓存文件。CLI 未绑定分支还通过独立子进程验证。

`LOCAL_CPU_TEST_RESULT.json`、测试 stdout/stderr 和 CLI stdout 是本机实际 CPU 接线检查，**不是实际自然 trace、真实分词回执或 GPU 实验证据**。真实 tokenizers CPU smoke 和服务器部署/新冻结由根执行者单独记录。本 agent 实际 GPU 操作、RPC 操作和真实 tokenizer 库执行均为 0。
