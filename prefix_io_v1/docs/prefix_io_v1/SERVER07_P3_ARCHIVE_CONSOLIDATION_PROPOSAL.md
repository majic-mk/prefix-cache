# 旧实验缓存副本合并提案（未授权、未执行）

## 可复核范围

只读 SHA-256 审计识别了 10,457 个已结束实验的私有 .bin 缓存副本，预计释放 9,594,339,328 B（约 8.94 GiB）。
完整逐文件清单位于 artifacts/prefix_io_v1/server07-p3-09/private-cache-duplicate-audit.json。
清单 SHA-256：63908528c8ef3ec31be98a2e4ba5a84aa4097c543fdd97b390b097977ebb6d55。

目标仅在 /root/prefix-io-v1-validation/runs/ 已成功结束的实验 storage 目录内，且每个目标当前只有一个硬链接。
不会替换模型、已有共享 inode、原始结果／trace、CPU 测试临时数据或已发布缓存源。
去重后每个原路径仍存在、字节不变；物理空间由内容相同的文件共同使用。

## 拟执行动作

1. GPU 必须空闲且账本无活动预留；重查权限、路径、inode、大小、mtime 和已结束任务回执。
2. 每一项重新校验 canonical 和 target 的 SHA-256；不接受内容变化或目标已共享。
3. 为目标原 inode 创建临时备份硬链接；为相同内容的 canonical 创建临时链接。
4. 原子替换目标路径，复核 samefile 与内容 SHA-256，通过后才退役旧私有 inode；失败恢复原 inode。
5. 保留逐项 intent／verified 日志和最终容量记录。不扩大 20 GiB 目录额度、8 GiB 空闲底线或 GPU 预算。

变化在于归档文件将共享 inode。如果今后有人原地编辑其中一份，会影响相连文件。因此应将这些已结束实验缓存按只读归档使用；需要编辑时先制作独立副本。工具不 chmod，不改变系统或驱动权限。
异常终止可能留下带 dedup-backup/dedup-link 后缀的安全副本，须依据 journal 核对，不自动进行广泛清理。

## 已完成验证

- 只读内容审计完成，数据修改 0。
- 全部 10,457 个目标的路径／元数据／已结束任务条件预演通过。
- 7 项临时目录测试通过，包含私有副本合并、拒绝已有共享文件、内容不一致拒绝、替换前异常与替换后复核失败恢复。
- 本轮未运行 --apply，也未创建批准记录。

## 为何需要单独确认

04_CODEX_EXECUTION.md 的工作区要求为：“保留所有用户工作，不执行清理、强推或破坏性重置。”
permissions.yaml 同时设置 allow_shared_data_deletion: false。

本提案保留内容和路径，但会退役旧私有 inode、形成共享关系，因此应得到仅针对这份冻结清单的例外授权；不能把常规“继续实验”解释成对旧归档数据组织方式的修改许可。一般禁止删除共享数据的设置无需改成 true。

拟限定授权 action：
deduplicate_completed_private_cache_copies_preserving_paths_and_bytes
并绑定上述 audit SHA-256。只有用户明确批准后才写入独立授权记录并执行。

## 待授权命令

~~~bash
.venv/bin/python experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py \
  --audit artifacts/prefix_io_v1/server07-p3-09/private-cache-duplicate-audit.json \
  --authorization experiments/prefix_io_v1/configs/authorizations/server07_p309_archive_dedup.json \
  --journal artifacts/prefix_io_v1/server07-p3-09/dedup-approved-journal.jsonl \
  --apply
~~~

这条命令目前不能运行：授权文件尚不存在。确认只适用于清单内内容完全相同的已结束实验缓存副本，不适用于其他删除、驱动或系统修改。
