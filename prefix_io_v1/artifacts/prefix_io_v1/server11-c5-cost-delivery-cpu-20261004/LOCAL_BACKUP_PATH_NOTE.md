本轮验证后的完整本地提取目录为 `C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/c5_cost_verified_20261004`。

第一次尝试先通过全部归档/成员SHA校验，但写入较深的 `server_replay/.../SOURCE_DEPENDENCY_SNAPSHOTS/<sha>.bin` 时触及本机 Windows 路径长度限制，未产生成功回执。未修改系统设置，未覆盖或删除该部分写入目录。随后以同一归档和receipt提取到新的较短工作区目录，通过全量写后字节核验，成功结果保存于 `LOCAL_BACKUP_VERIFICATION.json`。这两次仅为本地备份处理，未重跑CPU测试、正式基准或GPU。
