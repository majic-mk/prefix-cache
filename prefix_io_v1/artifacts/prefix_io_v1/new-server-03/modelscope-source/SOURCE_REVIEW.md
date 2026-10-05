# 固定 Qwen 模型：官方 ModelScope 渠道与版本审查

本次已核实可用的官方 ModelScope 发布渠道，固定模型仍是 Qwen/Qwen2.5-7B-Instruct，BF16、无量化，不改变作者 py-kvcache / vLLM 路线。只读取小量来源和模型 metadata；未下载权重、未运行 GPU。所有网络查询已停止，权重下载交由主线程预算执行器处理。

## 官方身份与原始需求

Qwen 官方 2024-09-19 发布博客 https://qwenlm.github.io/blog/qwen2.5/ 的 MODELSCOPE 链接直接指向 https://modelscope.cn/organization/qwen。服务器实际取得原始 HTML，SHA256 与 URL 已保存。ModelScope 官方模型 API 返回 Path=Qwen、Name=Qwen2.5-7B-Instruct，Organization.Name=Qwen，GithubAddress=https://github.com/QwenLM；这形成作者页面到官方组织再到指定模型的来源链。

原始交接包 03_IMPLEMENTATION_AND_EXPERIMENTS.md:102 固定首模型，04_CODEX_EXECUTION.md:25 约束权限/预算，05_SOURCES_AND_AUDIT.md:86 引用 HF config，:90 明确运行时还须冻结 revision。审查所列原始文档未发现“只能从 Hugging Face 下载”的硬约束。来源引用不自动等于唯一分发渠道；本次 ModelScope 身份独立核实，不称第三方镜像，也不冒充 HF 的 commit。完整检查范围/命中行见 document-source-constraints.json。

## 可冻结身份

- provider: modelscope
- source: https://modelscope.cn
- model_id: Qwen/Qwen2.5-7B-Instruct
- revision_namespace: modelscope_git_commit
- revision: 16c174980d8a1492910551634b4969e69cdc2444

官方 repo/files 的 master 结果显示最新 README 对应上述完整 revision，LatestCommitter.ShortId=16c17498。随后显式用该 40-hex revision 读取 repo/files，返回 200，完整文件路径/大小/SHA256/LFS 标记与观测到的 master tree 一致。负对照全零 revision 返回 404，而非悄悄回退 master。config、index、tokenizer_config 均通过同一固定 revision URL 实读，大小和内容 SHA256 与固定 tree 完全匹配。

Git smart refs 尝试返回 421，未据此宣称 Git 协议可用；固定 revision 的 REST API 已独立验证。没有核实这一 ModelScope commit 与任何 HF commit 的哈希或树等价，因此 hf_revision_equivalence_verified=false。

## 精确文件与字节

modelscope-download-plan.json 含 11 个必要文件、全部精确大小和内容 SHA256、固定 revision URL：

- 四个原始 safetensors 分片；
- config.json、generation_config.json；
- tokenizer.json、tokenizer_config.json、vocab.json、merges.txt；
- model.safetensors.index.json。

总计 15,242,788,168 字节，即约 14.19595 GiB，低于 20 GiB 总额度。四分片均由官方 metadata 标记 IsLFS=true；所有必要文件（包括非 LFS 小文件）都有内容 SHA256，无需混用 Git blob SHA1。

已实读配置为 Qwen2ForCausalLM、qwen2、bfloat16、28 层、4 KV 头；无 quantization_config 或 auto_map。大 tokenizer 文件和权重本身尚未下载；不能把 metadata hash 当作本地文件已经核验。源文件配置未经改写，仍须下载后逐文件校验。

## 预算与请求结果

metadata-accounting.json 的最终汇总：

- 全部响应 payload：115,928 B；
- 其中官方来源 HTML：51,224 B；
- ModelScope metadata 与已完成错误响应：64,704 B；
- 请求总数：11；
- 不确定请求：0，未结算不确定上限：0；
- 权重 body：0 B；
- GPU 操作：0。

主线程预留的 metadata 上限为 64 MiB；应按上述实际 payload 结算，保守可将官方博客 HTML 也全部计入模型下载额度。本记录不把 HTTP/TLS 协议开销混称模型 payload。

最后一次针对首分片的请求仅 HEAD，未读取 body。返回 200，但没有 Content-Length、ETag 或重定向信息，因此不能据此证明最终 CDN host 或 Range 支持。主线程下载器应从固定官方 API URL 开始，在收到实际 GET 响应头时验证状态/长度/重定向和批准 host；遇到未知来源或不确定预算时停止。没有预先许可未知 CDN，没有改系统 DNS、代理、路由或关闭 TLS 验证。

## 下游 schema

下载 plan 保持 schema_version=1、model_id/revision/files，其中 files 为 path/bytes/hash_algorithm/hash/url，hash_algorithm 均为 sha256；source/provider 区分平台。provenance.evidence 的角色为 official_release/provider_model/pinned_files/config/safetensors_index/tokenizer_config，每条含 url、项目相对 path、sha256、bytes。

输出仍是 download_execution=UNEXECUTED、download_ready=false、budget_reservation=NOT_RESERVED；它不是本代理取得权重下载额度的声明。root 负责共享账本与实际下载，build_review 已收到 schema 和证据路径用于 smoke 的严格 provider 校验。

主要证据：modelscope-download-plan.json、metadata-requests.jsonl、requests-started.jsonl、metadata-accounting.json、原始 HTML/JSON/config/index、invalid-revision-rejection.json、weight-head-only.json。所有原始响应保留。
