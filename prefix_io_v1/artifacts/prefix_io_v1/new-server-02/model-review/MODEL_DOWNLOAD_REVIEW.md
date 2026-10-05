# 固定 Qwen 模型下载预备审查

状态：**BLOCKED / UNEXECUTED**。本轮没有取得可验证的官方 revision、精确文件列表、总字节或 LFS SHA，因此没有冻结模型版本，也没有生成可执行的真实下载清单。GPU 操作 0 次，模型权重下载 0 字节。

## 服务器位置检查

有限检查了 /root/autodl-tmp/project、/root/autodl-tmp/models、/root/autodl-tmp/huggingface、/root/.cache/huggingface 和当前项目/models，均不存在。HF_HOME、HF_HUB_CACHE、TRANSFORMERS_CACHE 均未设置；/root/autodl-tmp 顶层仅 .autodl 和 prefix-io-v1-handoff。此结论只覆盖所列位置，不声称已扫描整个磁盘。

## 官方源与诊断

- 首次官方 metadata URL：https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct?blobs=true；返回 URLError / Network is unreachable。
- 系统 resolver 返回 IPv4 31.13.87.9、IPv6 2a03:2880:f102:183:face:b00c:0:25de。IPv4 TCP443 在 2 秒期限内未连接；IPv6 立即 Errno101。最终错误来自 IPv6，但 IPv4 也未连通。
- 经主线程允许追加一次 Cloudflare HTTPS DNS 查询，连接被 reset，未获得可信候选 IP。未运行 --resolve 请求。
- 没有修改 DNS、代理、路由或模型源，没有替换镜像，已停止重试。原始失败均保留。

## 最小计划脚本

新增 experiments/prefix_io_v1/scripts/prepare_qwen_download.py。它只读取已经获得的官方 metadata、明确 40-hex revision、同 revision 的 config/index 和项目现有严格权限配置，不含网络或下载代码。

它校验模型 ID 固定为 Qwen/Qwen2.5-7B-Instruct；revision 不接受 main，须匹配 metadata.sha；config/index 的实际大小和 Git blob SHA1 / LFS SHA256 须匹配；架构为 Qwen2ForCausalLM、model_type 为 qwen2，不允许 auto_map/remote code；从经验证的 safetensors index 得到精确 shard 集合。每个必需文件都须有精确正整数字节数与校验哈希，权重须有 LFS SHA256。所有文件仅指向官方固定 revision URL。

总字节不能超过当前权限预算与 20 GiB 的较小值。即使元数据完整，输出仍为 download_execution=UNEXECUTED、download_ready=false、budget_reservation=NOT_RESERVED、ledger_status=NOT_VERIFIED。此离线清单绝不代表取得下载额度；外层执行器仍须在共享账本锁内重新验证权限、已有内容和剩余字节，先预留，再签发下载，并持久记录实际写入及未结算预留。并发或失败账目不确定时必须停止。遵照本轮缩小后的范围，没有另造复杂下载器、网络后端或修改账本。

schema：
- schema_version: 1
- model_id、revision、trust_remote_code: false
- files: [{path, bytes, hash_algorithm, hash, url}]
- hash_algorithm 为 sha256 或 git_blob_sha1；后者的 SHA1 输入依次是 ASCII 头部（blob、空格、十进制字节数）、一个 NUL 字节（0x00）、原文件内容。不能用裸内容 SHA1 代替。
- total_bytes、total_gib、authorized_ceiling_bytes
- input_sha256 记录本次元数据/config/index 文件的内容 SHA256。

当前 pending-model-plan.json 只是阻塞记录，revision、files、total_bytes 均为 null，明确不是可执行下载 manifest。CPU 测试的 synthetic revision/hash 不得作为真实模型版本。

## CPU 验证

26 项 CPU 测试通过；覆盖不完整 revision/大小/LFS hash、错误模型、hash 不一致、缺 shard、重复 metadata、预算无效或超限、禁用权限、remote-code 配置以及永不启用下载的输出。首轮存在测试夹具根目录计算错误，已修正，原 collection-error 证据保留。

命令和结果：commands.jsonl、plan-cpu-tests.txt/xml（首轮错误）、plan-cpu-verified.txt/xml（26 passed）、plan-cli-help.txt。无需权重、网络、GPU 或模型初始化。

下一步骤：恢复可验证的官方 metadata 访问后，取得真实完整 revision 及同版本 config/index，再生成清单并交外层授权预算执行器。当前不能运行模型 smoke 或宣称模型阶段通过。
