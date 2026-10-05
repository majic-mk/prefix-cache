# C3 → C4 源码补丁与精确重建证明

只读取本地 `p4_single_file_candidate_v3` 与 `p4_single_file_candidate_v4`。
这份交付未修改候选、v6 标定或 P4 运行器，未连接服务器，未执行 GPU 操作。

## 修改分类

| 分类 | 生产代码变化 | 配套变化 |
|---|---|---|
| `01-common-idle` | reactor 新增 `_has_poll_work`，仅令 `_run` 使用它决定原 incoming Queue 是否阻塞。原 `_has_work` 继续决定停止和清理。 | retained-idle 测试、已有源码边界测试、说明 |
| `02-compact-bridge` | 相同不可变 snapshot、真实 step state 和 runtime prefix 时复用派生 snapshot；每次仍执行策略判断，失效/结束时清空。 | 派生 snapshot 测试与说明 |
| `03-v6-receipt-binding` | receipt 改为重放 v6 原始六窗口证据，检查实际 C4 reactor 及共同源码；不将 v5 成本迁移到新底座。 | receipt 反例测试 |
| `04-candidate-metadata` | 无生产代码；更新候选 manifest。 | manifest |

`p4_policy.py` 字节完全不变，SHA-256：
`bce19c96c7651f69ebb91a515ea3e9e3d6855636b4bd4aa84e60fb2fbc10d673`。
局部 collector 和 runtime-binding 文件也未变化。

## 实际验证

- C3 为 17 个文件，C4 为 21 个文件；10 个文件发生修改或新增，11 个字节完全相同。
- `build_patch_map.py` 使用原始 bytes 生成 unified diff，并严格按每个 hunk 的旧/新行号、上下文及增删行数逐项回放。共 27 个 hunk 全部通过；不使用偏移、模糊匹配或换行规范化。
- `PATCH_MAP.json` 记录每个原文件、目标文件、patch 的 bytes/SHA，以及每个 hunk 的精确恢复证明。重建覆盖全部 C4 文件，包括原样保留文件。
- `verify_with_git.py` 另复制 C3 到新目录，真实执行 Git `apply --check` 和 `apply --no-index`，然后逐字节比较完整结果与 C4。21/21 文件一致，见 `GIT_APPLY_VERIFICATION.json`；重建文件保留在 `git_reconstructed_candidate_v4/`。

实际执行命令（工作目录为共享 workspace）：

```powershell
& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B artifacts/prefix_io_v1_server11_candidates/v4_source_patches/build_patch_map.py --base artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v3 --target artifacts/prefix_io_v1_server11_candidates/p4_single_file_candidate_v4 --output artifacts/prefix_io_v1_server11_candidates/v4_source_patches
& 'C:/Users/mamengkui/AppData/Roaming/uv/python/cpython-3.12.14-windows-x86_64-none/python.exe' -B artifacts/prefix_io_v1_server11_candidates/v4_source_patches/verify_with_git.py
```

复现时使用一个新的输出目录，并把两个验证脚本复制过去，调整上述路径。
脚本使用追加式输出，拒绝覆盖已有证明或重建目录。
Git 的完整实际 argv、cwd、stdout/stderr 和返回码已写入验证 JSON。

## 使用限制

在干净的 C3 副本上选择 **完整 `C3_TO_C4.patch`、分类 patch、逐文件 patch 其中一种**；不能重复应用。
本证明只覆盖本地候选文件。服务器完整包中的原始复制文件、GPU 源锁、真实运行和预算由主流程独立校验。
精确源码重建不能证明功能正确、成本资格通过、GPU 加速或完整 P4 完成。
