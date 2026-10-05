# P0 无卡交接与全新目录部署

这一步只部署已验证源码、校验模型文件和补丁、运行 CPU 测试；不初始化模型，不申请 GPU，
不付款，不用历史 GPU UUID 或历史价格填补本轮授权。

## 源码身份

未提交工作树不能用单个 HEAD SHA 代表。交接同时绑定：

- 实际 base_commit；
- 验证报告的 worktree_digest 和逐文件 SHA256；
- CPU 报告、逐测试记录、原始日志和命令摘要；
- 用户交接任务书的固定文件校验；
- ZIP 自身的 SHA256。

`scripts/prepare_decoupled_v2_cpu_handoff.py` 的 `build` 子命令需要显式 workspace、
evidence、archive，以及 max-archive-bytes、max-uncompressed-bytes、max-members。
evidence 必须恰含 cpu_report、worktree_files、test_results、commands、package_checksums
五个实际 `{path, sha256}` 引用。打包前检查当前文件与报告一致；代码修改后必须重跑验证。

输出布局：`handoff.json`、`workspace/<仓库相对路径>`、`evidence/<原样CPU证据和日志>`。
不带 `.git`、凭证、模型、数据行、旧实验输出或用户未纳入验证的 bundle。
允许移植的文件按目录/扩展名白名单控制；路径穿越、符号链接、重复或大小写冲突成员拒绝。
所有限制在解压前核验，verify 子命令不解压、不执行包内代码。

被排除的已追踪历史文件仍列在 excluded_worktree_members；它必须可从相同 HEAD 原样重建，
或仅采用声明的 LF→CRLF checkout 变换后恢复原始摘要。不能遗漏后声称整个工作树相同。

## 接收顺序

1. 当前已授权的无卡服务器上确认磁盘、路径与旧仓库身份，只选全新目录。
2. 原仓库只作为 `git clone --no-hardlinks` 的来源，不修改其分支、index 或工作树。
3. 如缺基线对象，将本地生成的增量 Git bundle 上传到新目录，核验文件摘要和 prerequisites；
   只在新 clone 中 fetch，再 detached checkout 真实 base_commit。
4. 先核验 ZIP 的外部 SHA 与全部 member；仅向新 clone 写入 workspace 成员。
   还原明确列出的 excluded tracked 文件，逐项核对原始 worktree_files。
5. 本地验证证据原样留档；其中 Windows 绝对日志路径不能改写后沿用原 SHA。
   服务器以 `CUDA_VISIBLE_DEVICES=''` 重新运行 CPU 回归，生成服务器自身的新路径证据。

本地 `git diff --check` 不覆盖未追踪文件；compileall 和完整 unittest 会实际读取新增实现。
交接 zip 的逐文件摘要使新增文件也被绑定，不能只部署 Git HEAD 丢掉它们。

## 补丁与扩展

固定补丁基础为任务书现有 CacheBlend base；从旧补丁仓库 clone 到另一个全新目录，
只在该新目录应用当前 manifest 中有序基础补丁和明确 extras。独立核验调用
`scripts/server/verify_cacheblend_patch.py`，输出全新 patch_audit.json。
必须是 `independent_clean_clone_ordered_patchset`，且 actual/expected tree 一致。

verifier 使用临时 index 但会写 Git object，故不能直接以历史补丁目录为核验目标。
其输出函数不禁止覆盖；调用者必须先拒绝已有 output。

Git patch tree 不涵盖未追踪 `.so`。扩展另记实际来源、SHA、文件字节数及
Python/Torch/CUDA ABI 元数据；若复制到新目录，要校验复制前后字节一致。
这只是二进制移交证据，不证明新进程 CUDA import 或真实执行通过。
只设置本次进程的 PYTHONPATH，禁止修改全局 editable 安装来掩盖路径差异。

## 模型与隔离池

模型只从用户已经提供的 snapshot 读取，不下载/替换。可调用已有 audit_snapshot 对
config、tokenizer、权重分片做文件校验；读盘 SHA256 不等于加载模型，更不等于 GPU 正确性。
不得读取锁定测试行来构造 P0 样本。

隔离池/registry 用显式预算和权限域初始化，当前 native 绑定不全时不生成假的可执行
manifest。受控输入和数值政策须在结果前确定；Source birth 后才能绑定真实 ID/digest。

## 停止条件

CPU 部署、patch rebuild、模型文件核验成功，仍不授予 GPU 或 P1 资格。
实际 GPU、当前单价、最大时长/费用、执行窗口、numeric policy 和执行动作文件缺一项时，
保持 BLOCKED，保留具体文件/条件清单。连接失败不自动换服务器，网络恢复不自动租卡。
