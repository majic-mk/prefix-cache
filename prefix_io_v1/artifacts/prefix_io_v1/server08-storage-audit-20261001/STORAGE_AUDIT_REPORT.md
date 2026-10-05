# server08 数据盘只读清理审查（2026-10-01）

服务器 connect.westd.seetacloud.com:20739，项目 /root/autodl-tmp/prefix-io-v1-handoff/project。
本次仅做审查，删除0、合并0、GPU操作0。模型权重与真实KV载荷未读取；仅对已确认CPU自造临时数据读取SHA。

数据盘总量100GiB，已用约90.5GiB，剩余约9.5GiB。主要占用：experiments/prefix_io_v1/runs约59.3GiB、模型约17.3GiB、.venv约8.0GiB。真实实验cache是大头，不能按目录大就删除。

建议先清理严格清单内的78个CPU临时子树，预计回收3.000271GiB（3221516288字节regular allocated；目录块另有约0.7MiB，不计入保守预计）。当前快照free=10173935616字节；预计清理后约12.475GiB，实际以删除后df为准。

| 限定候选 | regular allocated |
|---|---:|
| P3-15/16四轮cpu-evidence/shadow-pytest-tmp | 950,218,752 B |
| P4-01与P4-02最终回归cpu-evidence/pytest-tmp | 572,395,520 B |
| 旧mixed-before/after的36个r*-s*-d*人工输入子目录 | 839,729,152 B |
| 旧mixed CPU两份pytest-mixed/pytest-release临时树 | 423,636,992 B |
| 旧AIO三轮microbench的27个block-*-depth-*子目录 | 70,447,104 B |
| 旧AIO六份pytest-*临时树 | 141,475,840 B |
| new-server-07/pytest-cpu | 223,612,928 B |
| 合计 | 3,221,516,288 B |

逐对象清单在 CPU_TEMP_CLEANUP_PROPOSAL.json：
SHA-256 b0c347fc4fc50384c088b8539b63ac09d77446a9563f33fba83f7844514c410e，文件9832220字节。
包含8423个普通文件路径、8405个唯一普通inode、822个内部pytest符号链接、6个人工FIFO反例；普通文件逐SHA冻结。所有候选与P3/P4四份保护锁的规范路径及受保护inode无交集，无外部hardlink，/proc cwd/fd/maps没有引用且无权限漏查。这些都是已结束CPU测试/微基准的自造输入，删除后重测可生成新fixture，随机fixture不承诺恢复相同字节。

严格保留：
- 模型、.venv、作者旧7个.so与原Git/worktrees。
- P3原始GPU结果/trace/config/source与2724锁定项，P4-01的70项输入、P4-02的2198测试/2048未来GPU来源。
- 3048份注册KV共2,796,552,192字节及来源manifest/registration，全部真实GPU私有缓存本轮也不列删除。
- benchmark父目录内的5份results.json、外层命令/exit/报告/XML/guard/log/patch/证据包。
- P4-02 patch-roundtrip复制树与reviewed-baseline原始快照本轮保留（另约0.343GiB可再议，不混入建议批次）。
- P3同级cpu-evidence的plan/log/XML/guard/result；不能删除整个cpu-evidence或runs。

此批只释放PRIMARY数据盘，不释放AUX或更改20GiB cap。预计PRIMARY的3GiB新轮次预留可过8GiB floor，但原模型驱动若仍写批准AUX，其cap/floor阻塞不会因此自动解决。仍需实时门禁，不能据此宣称GPU阶段全解锁或方法结果成立。

原交接04第13行要求“保留所有用户工作，不执行清理、强推或破坏性重置”；此前CPU_ONLY授权的allow_storage_deletion_or_merge为false。当前用户请求是查看，因此清单尚未执行。执行需要用户明确同意本批清单，且应在删除前重验路径、逐SHA、inode/链接闭合、活动FD、保护锁；删除符号链接不能跟随目标。旧缓存去重授权仅保留全部路径/内容，不能外推为删除授权。

实际命令：df -B1 -T /root /root/autodl-tmp /root/prefix-io-v1-validation /tmp；du -x -B1 --max-depth=1/2 指定项目与runs目录；以及 AUDIT_CPU_CLEANUP_CANDIDATES_EXECUTED.py 内纯stdlib审查，CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1、.venv/bin/python -I -S。没有执行rm/unlink或GPU命令。审查通过不作为新性能测试计数。

证据：AUDIT_SUMMARY.json、CPU_TEMP_CLEANUP_PROPOSAL.json、AUDIT_CPU_CLEANUP_CANDIDATES_EXECUTED.py、04_CODEX_EXECUTION_REFERENCE.md、CPU_ONLY_AUTHORIZATION_REFERENCE.json、DISK_DU_SNAPSHOT.json。
