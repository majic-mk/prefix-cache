正常入口 CPU 准备和服务器整合验证已完成。当前服务器 connect.westc.seetacloud.com:26909 处于无卡模式，无 NVIDIA 设备节点，配额 0.5 核 CPU、2 GiB 内存。本轮真实 GPU 作业、GPU 秒数和模型进程均为 0。

实际新增和修改位于新副本 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004。原 G、GPU03 源码和原始证据、作者模型/缓存执行器保持原字节；保留精确 Prefix Cache、共享 staging、预加载、复制合并、异步流水线和原 drain/shutdown。共同修复为历史标定的真实账本追加校验、合法空源码引用检查、正常入口来源冻结及授权接线。通知适配器、研究策略、成本上界和预算均未改动；off 保持无策略桥路径。

各模式独立核验源锁、现场、直接人类授权、有效 permissions、原预算守卫及前驱结果。原 GPU03 单次授权已经执行完，本轮没有生成新的 GPU 授权。三模式共用同一共同锁，可变累计账本、每模式授权/配置/证明不进入共同锁。

观测计量使用三模式同一完整请求与原 drain 区间，记录进程 CPU、主线程 CPU 和固定原 owner/AIO 线程的 /proc 读数。线程 CPU 与进程 CPU 重叠、不相加，缺失保持未知。gross CPU 包含模型和原生 I/O；未来单次 on-off 差值只能作描述，不能称净观测上界、完整成本、SLO 或性能提升。

实际服务器结果：100/100 新测试通过，失败/错误/跳过均为 0。4779 个完整真实来源文件测试前后逐字节核验一致；禁止的 Torch/vLLM/native backend 导入及导入尝试均为 0。G 的 64 个 Python 文件保持冻结字节，47 处原模型/原 drain/数值/序列化/类型逻辑 AST 一致。serializer 与通知适配器字节相同。execute_window 的授权门、公共类型注册及 gross 计量明确改动，没有声称整个函数 AST 不变。

CPU 完整公开重放原 GPU03 原始数据返回真实 typed receipt，不使用 fixture 或字典提升。共同锁与 off 配置成功生成；保留原上界 20.456415 ms、A-only 预算 12.957824 ms，未重标定或调阈值。上界超过预算，不能据此宣称策略已经准入、提速或具备论文级收益。

共同锁 SHA-256：cbdaf335d5c260cfd4936ee981448fc1c1fd430121ef642c4447272221adc433。NORMAL_RUNTIME_BINDING SHA-256：c8db1e47e2da477b1abcd265c2a9886422c03332d779116444b5f939e0fee995。CONFIG_off SHA-256：dd51ce53c9ccc6fade2596aeff5b742f0dab77f7dcd8630de06793e2a53c56ec。恢复保留既存 7,767 字节 binding、原始数据及真实公开类型；未覆盖 binding。原校准的全部 4,741 个源码资产逐字节校验并冻结为叶，其内旧文档历史路径不提升为当前依赖；实际新增校准/raw/child/guard/proof 全部递归核验。恢复 helper 自身进入共同锁。同函数仅复用刚返回的真实 typed receipt 准备 off 元数据，不跨进程序列化资格、不授予 GPU 权限。

实际命令、stdout/stderr、退出码在本交付目录各 *_COMMAND.json、*_STDOUT.log、*_STDERR.log、*_RESULT.json。所有命令设置 CUDA_VISIBLE_DEVICES 空字符串和 PYTHONDONTWRITEBYTECODE=1；03 超时另有 03_FREEZE_TIMEOUT_RESULT.json。关键成功命令：

    /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-delivery-20261004/resume_cpu_freeze_v2.py
    /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004/run_cpu_normal_native.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --source-lock /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004/COMMON_SOURCE_LOCK.json --output-dir /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-delivery-20261004/SERVER_CPU_01
    /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-cpu-20261004/audit_normal_sources.py --root /root/autodl-tmp/prefix-io-v1-handoff/project --output /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-delivery-20261004/SOURCE_ANCESTRY_AUDIT.json
    /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server11-c5-normal-native-delivery-20261004/verify_unauthorized_cpu_entry.py

05 恢复/完整回放/来源冻结/配置退出 0，366.269687 秒；06 新服务器测试和完整前后来源校验退出 0，86.426551 秒；07 原逻辑审计退出 0，0.650688 秒；08 真实 frozen 正常模型入口带 --execute 和 CONFIG_off，准确在缺少 AUTHORITY_off 时拒绝，导入/导入尝试、模型/GPU/新运行目录均为 0，6.374342 秒。最后一项只是拒绝边界验证，不能当正常模型运行通过。

失败 CPU 记录保留：01 合法空 __init__.py 引用检查错误，修复仅涉及新 controller 和对应新测试；02 共同锁未生成时提前执行审计的前提失败；03 无卡配额下 120 秒 CPU 超时并保留部分生成的 binding；04 完整标定回放通过后，元数据递归误升格历史路径而拒绝，未写共同锁；05 已完成修复后的整合。未运行复制保留的旧测试或旧 CPU benchmark，未把这些失败记作 GPU 实验。

原累计 GPU 上限 28,800 秒，实际已用 23129.437302809 秒，剩余 5670.562697191 秒（约 1 小时 34 分 31 秒）。账本 SHA-256 4fd592aaecf9371ae050c65035e7cf037b56d876653aac2e49fa65f5fc5c3da8 本轮前后完全相同，active_reservation 为 null。服务器数据盘实际可用 60585283584 字节，原存储上限继续生效。未删除旧代码、实验数据和模型；未修改系统、驱动或已安装包。服务器 tracked Git 改动为空，用户本机此前 14 个修改文件的实际 SHA 均保持不变。

下一允许阶段见 NEXT_GPU_STAGE_PROPOSAL.json：需切回 GPU 模式并取得一次限定批次授权，同一真实 GPU UUID 上 off→shadow→on，最多三个独立原模型作业，各 300 秒执行加 20 秒收尾，总预留 960 秒。off/shadow 真实资格及原 shutdown/session drain 未通过，就停止后继；每模式一次、不重试。现有 Qwen2.5-7B，固定 warmup/measured 各完整 128 tokens，保留原 1-token primer/flush、一个 917,504 字节 SSD 操作和相同前景输出。每次 PRIMARY 预留 128 MiB、保留 8 GiB，不下载、不修改系统/驱动/包/作者文件、不删除数据。现场 UUID 不匹配、资源/预算不足则在预约前停止。

未来这一批 GPU 仅验证正常路径和通知的实际可运行性并采集 gross CPU 诊断。等待或通知没有真实触发，就记 NOT_EXERCISED、on 资格 false。P4 正式性能、完整观测成本和论文效果尚未验证，不能把 CPU 结果或有限标定覆盖当策略收益。

证据位置：SERVER_CPU_01/CPU_RESULT.json、TEST_STDOUT.log；SOURCE_ANCESTRY_AUDIT.json；UNAUTHORIZED_ENTRY_CPU_RESULT.json；SERVER_PRESEAL_STATE.json；独立源码复核；LOCAL_USER_WORK_BEFORE/AFTER.json；NEXT_GPU_STAGE_PROPOSAL.json。封存仅含本次小源码和 CPU 证据，逐成员复验 SHA，不打包模型权重或私有 KV；保留失败日志。GPU03 原始证据另见既有 GPU03_NATIVE_VALIDATION_EVIDENCE.tar.gz 及原交付报告。
