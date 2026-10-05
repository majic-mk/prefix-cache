V13 工具已完成 CPU 开发与只读自查。本目录没有执行真实 V13 全源冻结，没有连接服务器、下载数据或使用 GPU。

`append_public_source_lock_v13.py` 固定继承实际 V12 锁 SHA `23ab34887c7f4340a37f2814936c639c46890ca31e4b3ba1b07359d41fc13f77`，要求 4,953 个原叶及 idle 账本 SHA `774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b`。每个原叶必须流式重算 SHA、核对完整 ref，并维持文件身份／大小／mtime／ctime 稳定；最终逐条要求原叶完全保留，末尾再检查全部叶的 stat 与祖锁、manifest、账本字节。Windows 路径 stat 与 fd stat 的 ctime 含义差异分别按各自接口前后检查；跨接口仍核对 mode/dev/ino/size/mtime。

新增来源仅由明确 manifest 选择，不扫描目录，不纳入旧残留 `.partial`、`.part` 或 `.tmp`。manifest 示例：

```json
{
  "schema": "bounded_public_source_append_manifest_v1",
  "files": [
    "artifacts/prefix_io_v1/server12-natural-source-audit-20261005/<actual-source.py>",
    "experiments/prefix_io_v1/datasets/<actual-complete-input.json>"
  ]
}
```

示例中的尖括号不是可执行路径。根代理须用实际已稳定的 source/data/request/adapter 文件替换。`files` 必须是唯一、非空的项目相对字符串列表；任何重叠叶必须与已记录 ref 完全相同。工具自动绑定实际 V12 锁自身、manifest 和自身源码。manifest 可以带说明 metadata，但它不会获得 GPU 能力。

manifest 与两个输出均须位于 `artifacts/prefix_io_v1/server12-natural-source-audit-20261005/`，输出父目录须已经存在。输出必须是不同、全新 `.json`，以 `xb` 独占写入。输出不得作为输入叶；源路径的 symlink、目录和遍历路径均拒绝。执行的工具必须就是该项目内的 `source_freeze/append_public_source_lock_v13.py`。工具只导入标准库；模型与 SDK 只进行字节读取，不导入或加载。

服务器 CPU 命令（根代理自行设置真实变量）：

```sh
A=artifacts/prefix_io_v1/server12-natural-source-audit-20261005
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$A/source_freeze/test_append_public_source_lock_v13_cpu.py"
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$A/source_freeze/append_public_source_lock_v13.py" \
  --project-root . \
  --addition-manifest "$A/<actual-additions>.json" \
  --source-lock-output "$A/PRERENT_SOURCE_LOCK_V13.json" \
  --proof-output "$A/PRERENT_SOURCE_PROOF_V13.json"
```

Windows 小型 fixture 实际运行 7 个测试，exit 0：6 项通过，1 项实际 symlink 测试因 WinError 1314 无创建权限而跳过；Linux 服务器应执行全部 7 项。已覆盖真正 CLI 参数处理、完整继承、重叠、独占输出、文件内容变化、重复祖叶、重复追加、路径越界、残留 partial、目录、末尾来源变化与账本变化、active ledger 拒绝。fixture 会在内存替换 ancestor SHA/count 等常量以使用极小测试资产，明确不是真实 V12 证明或 GPU receipt；生产 CLI 不提供这些 override 参数。`--help` 实际通过。

只读自查结论：无旧源／账本写入、无目录 glob、无第三方／模型／GPU 导入、无网络或子进程调用；输出声明 CPU-only、非 GPU authority、非 family receipt。真实 16 GiB 源资产未在本机重复检查。下一步须由根代理在全部选定源稳定后，执行一次真实服务器 CPU 全源冻结；只有该实际 PASS 才能供后续 raw tokenizer 使用。
