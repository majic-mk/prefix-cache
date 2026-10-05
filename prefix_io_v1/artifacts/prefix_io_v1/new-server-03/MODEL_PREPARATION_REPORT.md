# 模型准备完成报告：固定官方 ModelScope 来源与完整本地权重

本报告只汇总模型来源、下载、CPU 文件校验和模型下载预算，不作 GPU 推理或 Prefix Cache 验收结论。实际操作在服务器 connect.westc.seetacloud.com:24801 的 /root/autodl-tmp/prefix-io-v1-handoff/project 完成。下文相对证据路径均以 artifacts/prefix_io_v1/new-server-03/ 为基准。

**结果：qwen-ms-final-04 实际 exit 0，状态 VERIFIED_LOCAL_FILES；11 个固定文件全部按精确大小和 SHA256 复验通过。** 独立 header/index 审计通过：339 个 tensor 全部 BF16，共 7,615,616,512 个参数元素。完整模型文件共 15,242,788,168 B；这与应用层下载 payload、保守预算收费和保留失败文件的磁盘占用是不同数值。

## 固定身份与来源

| 字段 | 实际固定值 |
|---|---|
| model_id | Qwen/Qwen2.5-7B-Instruct |
| provider / source | modelscope / https://modelscope.cn |
| revision_namespace | modelscope_git_commit |
| revision | 16c174980d8a1492910551634b4969e69cdc2444 |
| manifest | modelscope-source/modelscope-download-plan.json |
| manifest SHA256 | 9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017 |
| 本地模型目录 | models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 |

Qwen 官方发布博客链接到其 ModelScope 组织，官方模型 API 的 Qwen 身份与 GithubAddress 形成可复核来源链。固定 40-hex revision 的 REST 文件列表、config/index/tokenizer_config 已实际取得并核验，全零 revision 负对照返回 404。Git smart refs 的 421 失败保留；该失败不被写成 Git 协议成功。

此 revision 属于 **ModelScope Git commit 命名空间**。没有证明它与任何 Hugging Face commit 等价，不能把它标记为 HF revision 或跨平台等价 SHA。模型仍是原定 Qwen 模型、BF16、无量化；来源适配只为同一作者原生 vLLM 提供已验证本地模型目录，没有引入 ModelScope SDK 或另一模型执行器。来源详情与原始证据见 modelscope-source/SOURCE_REVIEW.md。

## 实际执行过程与失败保留

| 实际作业 | 退出码与结果 | 对预算有影响的事实 | 主要证据 |
|---|---|---|---|
| 官方来源和 metadata 审查 | 请求结束并结算 | 应用层 payload 115,928 B；无权重 body | modelscope-source/metadata-accounting.json、budget-settlement.json |
| qwen-ms-01 | exit 1，FAILED | 两个小文件成功；首权重未知重定向被拒，权重 body=0，却按完整预留保守扣 3,945,441,441 B | download-launch.json、downloads/qwen-ms-01/result.json、qwen-download-01.log/.exit |
| 精确 CDN 头审查 | 8 个实际请求，body=0 | 四个固定官方 GET 首跳与四个 CDN HEAD；payload/charge=0，单独记录响应头 9,680 B | redirect-review/redirect-review-public.json、budget-settlement.json |
| qwen-ms-02 | exit 124，FAILED，原 1800 秒时限未延长 | 前三分片完成；第四分片 SIGTERM 时应用层已返回 3,321,888,768 B，仍按完整第四分片+1 扣 3,556,377,673 B | download-retry-launch.json、downloads/qwen-ms-02/result.json、download-timeout-settlement.json、qwen-download-02.log/.exit |
| qwen-ms-resume-03 | exit 0，VERIFIED_RESUMED_FILE_ONLY | 从固定 offset 补回 234,488,904 B，成功按实际数扣费；仅证明第四分片完整 | resume-launch.json、downloads/qwen-ms-resume-03/result.json、qwen-resume-03.log/.exit |
| qwen-ms-final-04 | exit 0，VERIFIED_LOCAL_FILES | 复用并核验 9 个文件，补 tokenizer.json 7,031,645 B、vocab.json 2,776,833 B，再复验全部 11 文件 | final-download-launch.json、downloads/qwen-ms-final-04/result.json、qwen-final-04.log/.exit |

首个失败收费没有因修复重定向而返还；第二个超时收费没有因成功续传而返还。两次 FAILED 结果和对应旧 label 证据保持原样，不覆盖为成功。

CDN 审查只批准精确 hostname cdn-lfs-cn-1.modelscope.cn。下载器仅从固定官方 URL 开始，采用默认 TLS 验证，限制官方主站→该 CDN、该 CDN→自身，最多 3 跳；每跳验证 manifest 内容地址与对象 query，关闭旧响应且不读跳转 body。新的请求重新取得临时签名，不复用历史 auth_key；未扩大通配主机或关闭证书验证。

实际续传前，主线程确认旧进程退出、原账本已自动清除该预留，并冻结失败 partial 的大小和 SHA256：
533b7c828ecc2fa7641c61c5063b4efb3037e0153d81d3a90204ad52d3976214。
续传入口在共享锁内绑定原失败 result、已结算 ledger event、固定 manifest、精确 partial 和 offset；保留原件，复制到新 label 的 partial。请求前 fsync 预留 234,488,905 B，单次请求 Range: bytes=3321888768-。本次实际成功执行通过了固定脚本的 HTTP 206、Content-Range bytes 3321888768-3556377671/3556377672、Content-Length 234488904 检查；完整重组文件的 SHA256 通过后才发布。200 或不一致头会在读取 body 前失败关闭，未放宽检查或自动重试。

## 完整本地文件与 header 审计

完整模型包含四个 safetensors 分片、模型/生成配置、权重索引及 tokenizer 所需文件，共 11 个。全量验证结果是 downloads/qwen-ms-final-04/result.json，不能用此前只验证一个分片的 resume 结果代替。

model-header-audit.json 的 PASSED_HEADER_AUDIT 核验索引与 tensor 所属分片一致、无重复或遗漏、shape/offset/文件跨度一致，并确认 339 个 tensor 全部为 BF16。四分片 tensor 数依次为 82、97、97、63；逻辑 tensor payload 总计 15,231,233,024 B。该 header 审计仅读文件头，没有加载 tensor payload；内容 SHA256 由完整下载器另行校验。它不构成模型执行证明。

为保留失败证据，模型目录仍有原第四分片 partial：

| 本地内容 | 文件逻辑字节 |
|---|---:|
| 已验证完整模型的 11 个文件 | 15,242,788,168 |
| 保留的 qwen-ms-02 第四分片 partial | 3,321,888,768 |
| 目录文件逻辑字节合计 | 18,564,676,936 |

保留 partial 是磁盘上的重复数据，不是额外下载 payload。上述为文件大小之和，不是文件系统块分配或 du 实测值；没有为节省磁盘而删除原失败文件。

## 最终模型下载预算口径

授权上限 20 GiB，即 21,474,836,480 B。以 model-completion-accounting.json 与最终下载 result 为结算依据：

| 会计量 | 字节 |
|---|---:|
| S：完整模型文件逻辑大小 | 15,242,788,168 |
| P：累计应用层返回的 response payload | 15,242,868,376 |
| C：累计保守预算收费 | 19,422,798,722 |
| 本模型准备结算后的剩余下载额度 | 2,052,037,758 |

本次模型下载作业均已结算；该剩余值为上限−C，不把后续独立 GPU reservation 混入模型下载字节。它是本次模型准备完成快照，不表示未来可以不再读取共享账本。

P 比 S 多 80,208 B：metadata/来源/错误响应合计 115,928 B，其中 35,720 B 的已验证 config、index、tokenizer_config 被本地复用进模型目录，因此不会再次下载。C 比 P 多 4,179,930,346 B，来自：

- qwen-ms-01 首权重拒绝重定向：0 B body，保守收费 3,945,441,441 B；
- qwen-ms-02 第四分片超时：收费 3,556,377,673 B，应用层已返回 3,321,888,768 B，两者差额 234,488,905 B。

成功续传的 234,488,904 B 和最后两个小文件 9,808,478 B 都另外实际读取并收费，没有抵销旧失败收费。metadata 的 64 MiB 上限按 115,928 B 结算并释放未用部分；header-only 审查的 64 KiB 预留按 0 B 结算并释放。初始 metadata 已完成请求是在有界审计过程中归集结算，不能追溯声称每个初始请求前都已有预留。

P 的准确含义是程序计数的应用层已返回 response 字节，不是 HTTP/TLS/TCP wire 流量。SIGTERM 可能涉及内部 read 缓冲，不能把应用层计数宣称为网络物理流量精确值。header-only 的 9,680 B 响应头仅作独立诊断，也未冒充 payload。

## 命令与可追溯证据

四次下载/续传的**实际完整 argv**分别保存为 download-launch.json、download-retry-launch.json、resume-launch.json、final-download-launch.json；每个表中 label 的日志、exit 文件和 result 互相对应。它们是已经执行的参数记录，不是未来复现示例。不要复用这些旧 label 重跑。

header 审计实际 argv 为：

~~~bash
.venv/bin/python experiments/prefix_io_v1/scripts/audit_local_model_headers.py --model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444 --output artifacts/prefix_io_v1/new-server-03/model-header-audit.json
~~~

它由服务器根执行日志 /root/autodl-tmp/prefix-io-v1-handoff/commands.jsonl 第 327 行的 subprocess argv 调用，父命令 exit 0，结果同时保存 model-header-audit.log/json。对外交付应使用脱敏副本；原始证据不改写。

关键结果 SHA256：

| 文件 | SHA256 |
|---|---|
| model-completion-accounting.json | e8f1407d5631663f0404678a8e959c4532206dac24cb66c8c7f7723585f58d3a |
| model-header-audit.json | 87fc23a94917333db7ec90f9c95a15c8abef3b1dcf8e9c270ec8596aebf65efe |
| downloads/qwen-ms-resume-03/result.json | a1b8357d58bfcfb56d3b48d46aa2bd4ce2abade1d1c2e07a3cbf8e2a1e75f873 |
| downloads/qwen-ms-final-04/result.json | e0900bba5237baab647f5be1ab1153c2698b5bd149e059ecb75e1059b7f42783 |

下载/续传安全边界的 CPU 证据见 resume-review/first.xml：109 项通过，包含早期 65 项；来源/KV metadata suite 当时为 49 项、交付脱敏 15 项，合计快照 173 unique。后续 TMP 或其他改动的 CPU 累计由主线程单独维护，本报告不将该历史快照冒充最终全任务测试总数。

本模型准备消除了固定模型文件缺失这一前提障碍。GPU 模型行为、原生 Prefix 命中以及 io_uring/SSD 端到端路径的状态和结果由各自实际阶段报告给出，不从文件验证推断。
