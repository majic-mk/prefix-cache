# Server08 限定 CPU 临时清理交付
本轮按用户明确授权清理已冻结清单的严格子集，状态：PASS_APPROVED_CPU_TEMP_CLEANUP。数据盘实测可用空间从 9.442 GiB 增至 12.382 GiB，净增加 2.940 GiB；清单内实际删除普通文件占用 3.000248 GiB，空目录额外占用 614400 B。净增加量扣除了新保存的审计清单、日志及校验凭证，并可能包含文件系统即时统计差异，不能把稀疏文件逻辑大小当作释放量。

用户授权原文：“确定，但是要保证后续后续需要的实验数据和代码不被删除”。授权对应冻结清单 SHA-256：b0c347fc4fc50384c088b8539b63ac09d77446a9563f33fba83f7844514c410e，执行子清单 SHA-256：41611b8f173c78d66579086734abaf1d4166606e4d62ef291d1ab6d6a22e4e12。原 permissions.yaml、原 CPU 授权和所有源码锁保持不变；本轮授权单独记录在 APPROVED_CPU_CLEANUP_AUTHORIZATION.json。

实际改动：
- 删除 8417 个普通 CPU 临时文件、822 个仅指向候选内的符号链接和 6 个测试 FIFO；只对 11300 个已列入清单且实际为空的目录使用 rmdir。
- 原 78 个候选根中 72 个已移除，6 个保留。6 个零字节 aligned 测试文件出现 allocated_bytes 4096→0 漂移，按约定保留文件及 12 个祖先目录，保留文件 SHA 与身份及目录身份事后再次一致。
- 临时 .py/.so 是 pytest 复制件、人工构造的二进制反例或合成输入；实际实验源码、7 个真实运行 .so、模型、环境、原始及私有真实 KV、GPU 结果/trace、回归日志/XML、patch reviewed baseline 与归档证据均保留。
- 生产源码与研究策略没有修改；只新增清理执行脚本、授权、检查和交付证据。CPU 临时随机输入被丢弃，后续测试可由保留的源码重新生成测试用输入；未声称能复原已删临时随机字节。

校验：
- P3 锁中全部 2724 个相对及绝对路径都实际重验 SHA（原只读审查脚本只提取绝对键，执行门禁已补强并保留旧证据）。
- P4-01 70 项、P4-02 test-input-lock 2198 项、GPU-source-lock 2048 项均实际重验。去重后 7813 个受保护文件在执行前后 SHA、身份及大小等记录一致。
- 3048 份注册 KV 合计 2796552192 B，在执行前后全部实际读取 SHA，一致、无缺失/多余 .bin。
- 项目与 AUX 共 1020725 个非候选对象执行前后元数据一致；删除/增加/变化均为 0。当前审计文件夹的新增证据、候选与审计祖先目录因自有子项变化而改变的目录尺寸/nlink/mtime 不在该对照中，目录 device/inode/mode 仍核对。
- 13 份 benchmark 源码、results 和执行记录，P3/P4 结果及同级日志证据保留。清单外 1111 个符号链接另经独立审查，无目标落入候选。
- 冻结清单 SHA、inode/mtime/大小/模式/nlink、硬链接完整闭包和 /proc cwd/fd/maps 活动引用门禁通过。实际 unlink 使用锚定目录描述符与 O_NOFOLLOW、即时 SHA/身份检查；rmdir 只移除空目录。
- 独立审查完成 AST/差异校验，V2 隔离模拟覆盖硬链接、元数据/SHA 不符禁止删除、符号链接/FIFO 不打开、非空目录拒绝。V3/V4 仅做真实 SHA/AST/差异及保护函数 AST 等价复核。V5 另有 9 项已保存的隔离流式测试，exit=0，覆盖清单范围/元数据等价、12027行跨块排序、路径标点、新增/缺失/变化、超过1000差异的完整计数、空清单与乱序/重复/错误数量拒绝。Windows 实际符号链接夹具未能创建（false），如实保留；生产1111个符号链接的只读检查另行实际完成。未声称运行完整模型/回归套件。
- 本轮未重新运行此前 1956 passed / 16 skipped 的完整 CPU 回归；它的结果和代码证据保持，清理的校验不充当模型性能实验。

实际命令（工作目录 /root/autodl-tmp/prefix-io-v1-handoff/project，CUDA_VISIBLE_DEVICES=''，PYTHONDONTWRITEBYTECODE=1）：

- /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-storage-audit-20261001/EXECUTE_APPROVED_CPU_TEMP_CLEANUP.py preflight
  exit=1; 51.03s；凭证 PREFLIGHT_COMMAND_01.json。

- /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-storage-audit-20261001/EXECUTE_APPROVED_CPU_TEMP_CLEANUP_V2.py preflight
  exit=1; 78.28s；凭证 PREFLIGHT_COMMAND_02_V2.json。

- /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-storage-audit-20261001/EXECUTE_APPROVED_CPU_TEMP_CLEANUP_V3.py preflight
  exit=0; 609.33s；凭证 PREFLIGHT_COMMAND_03_V3.json。

- /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-storage-audit-20261001/EXECUTE_APPROVED_CPU_TEMP_CLEANUP_V4.py execute
  exit=-9; 253.65s；凭证 EXECUTE_COMMAND_04_V4.json。

- /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-storage-audit-20261001/EXECUTE_APPROVED_CPU_TEMP_CLEANUP_V5.py execute
  exit=0; 635.52s；凭证 EXECUTE_COMMAND_05_V5.json。

第一轮预检因 6 个零字节文件块分配漂移而安全中止（没有删除）；第二轮因 1 个 AUX 历史结果引用不在初设项目路径范围而安全中止（没有删除），随后仅扩展该单文件的只读保护，不扩展删除。V3 预检最终 exit=0，608.71s；SSH 360s 等待超时已单独记录，服务器检查持续完成并保存实际最终凭证，不能把通信超时当作通过。V4 在删除前被 SIGKILL（exit=-9）终止，253.65s，journal 未创建、没有删除；只读诊断发现 cgroup 内存上限 2 GiB、内存限制事件较多，但 oom_kill=0，不能声称已确认 OOM 原因。V5 改用最大12000行的分块流式对照，复用 SHA 冻结的一百万项预检基线，比较窗口始于更早的预检；执行时仍重新验证所有 SHA、候选和活动引用，事后重新扫描全部对象。新分块仅位于本轮审计目录并作为证据保留，无系统配置改动。

真实 GPU 情况：本轮 GPU 初始化 0、运行 0、预算新增 0 秒；没有模型执行、GPU 探测、构建安装、驱动/系统修改。gpu-budget-ledger SHA：31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1，permissions.yaml SHA：795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50，均保持。AUX 回收量 0，20 GiB AUX 限额没有改变。

下一允许阶段：CPU 临时清理已完成。有卡后仍须按现有授权/预算完成 GPU 资格与空间门禁、限定 P4 运行；AUX 额度及真实模型执行/端到端收集等原阻塞不会由本轮清理消除。P4_GPU 仍未完成，本轮没有新增方法有效性结果。

操作边界：Linux unlink 本身没有原子的 inode 比较删除接口，候选生成器在整个操作期间须停止。目录锚定、即时复核和活动引用门禁减小竞争窗口；结果还经过完整保留对象事后对照，所有保护检查均通过。本轮没有扩大到其他数据清理。

证据位于本目录：APPROVED_CPU_CLEANUP_RESULT.json、ACTUAL_CLEANUP_COMMANDS.json、EXACT_DELETION_JOURNAL.jsonl、NONCANDIDATE_METADATA_COMPARISON.json、PROTECTED_BYTES_AFTER_CLEANUP.json、RETAINED_SIX_ZERO_BYTE_FIXTURES_AFTER.json、RETAINED_TWELVE_DIRECTORY_IDENTITIES_AFTER.json、各版独立审查记录，以及两个百万条对象 gzip 清单、V5流式分块与执行脚本 V1/V2/V3/V4/V5 历史。UTC 完成时刻：2026-10-01T05:20:16.010774+00:00。
