# AutoDL AIO CPU 验证交付（2026-09-29）
结论：CPU_FUNCTIONAL_PASS；PERFORMANCE_SCREEN_ONLY；GPU_UNVERIFIED。可以继续在现有 AutoDL 上推进兼容路线，但不能把 CPU 通过当作 P1 全部通过，也不能宣布无性能损失或新策略有效。

## 实际环境与范围
- connect.westd.seetacloud.com:40168，autodl-container-mz3u56hqpn-f855e672。
- Linux 5.15.0-78-generic，实验目录在 /dev/md0 的 XFS 文件系统。
- 固定作者 py-kvcache HEAD：3abba7a502d553f6e7e2e58b92086487e3395d7e。
- 用户“可以，先进行cpu验证”授权了此前 AIO 提案的 CPU 实施与验证。本轮不延伸至 GPU、模型下载、系统修改或其他研究策略。
- 工作目录：third_party/work/py-kvcache-aio-cpu，独立 detached worktree；先应用现有共同修复，再增加可选 AIO 补丁。原 third_party/work/py-kvcache 未改。
- 当前导入日志提示 libcuda.so.1 文件过短、vLLM CUDA 扩展未载入；本轮是 CPU 模式，未修复或绕过驱动，GPU 能力没有重新验证。

## 实际修改
1. 新增 py_kvcache/linux_aio.py：Linux x86_64 原生 AIO 的适配层，CPU 实测通过。专用线程提交内核 I/O，最多 2×depth 个未回收请求描述符、最多 depth 个内核在途数据请求；元数据工作线程默认 2、上限 8。
2. 使用 eventfd/selector 等待真实完成和新提交通知；保留有界重试，仅在临时提交失败或模拟内核时使用短时重试。不是将读写改为同步串行执行。
3. fs_config.py 新增显式 io_backend=io_uring|linux_aio 和 aio_metadata_workers；默认 io_uring，错误不会自动回退；原位置参数顺序保留。
4. reactor.py 仅增加后端构造分支。仍由原 reactor 执行读写完成、文件发布、staging 引用、GPU 复制衔接和任务完成。
5. 新增 tests/prefix_io_v1_aio、CPU 测试保护脚本和有限 I/O 微基准。
6. 兼容补丁为 patches/prefix_io_v1/compatibility/0001-optional-linux-aio-cpu.patch；共同修复单独保存在 common-before.patch。未增加研究策略。

## 最终验证
最终命令（完整环境与命令在 REPRODUCE.md 和 release-qualification-importlib.json）：
CUDA_VISIBLE_DEVICES='' PYTHONPATH=third_party/work/py-kvcache-aio-cpu:src .venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_aio third_party/work/py-kvcache-aio-cpu/tests --ignore=third_party/work/py-kvcache-aio-cpu/tests/test_e2e_kvcache.py

最终一次联合运行：310 passed，16 skipped，0 failed，0 errors。
- 新增真实 CPU 文件 I/O：17 项。覆盖 4KiB/64KiB/1MiB、深度 1/4/16、直接读写、对齐错误、短读、缺失文件、重复发布、FD 与线程重复生命周期。
- 实际作者完成函数 + 真实文件 I/O + 模拟 CUDA 接口：5 项。成功读取只移交复制、不提前释放 slot；短读不移交复制；成功写入后才发布；失败写入清理并拒绝发布。不是生产 GPU KV 验证。
- CPU/mock 故障注入：26 项。覆盖部分/零提交、EAGAIN/EINTR、永久错误、乱序/异常完成、预算满、阻塞提交隔离、关闭排空、构造/元数据/销毁失败。
- 配置和实际构造选择：14 项，包含默认原生路径失败时不得静默回退。
- 作者原有回归：248 项通过。13 项仅用于无 vLLM fallback 的测试按原条件跳过；3 项 io_uring 测试因 EPERM 跳过。模型端到端测试明确排除，没有启动。
- 合并测试首次遇到两个仓库同名 tests 包引起的 14 个收集错误；修正测试导入模式为 importlib 后最终全组通过。失败原始记录保留，不计为最终代码通过结果。
- release-inputs.json 锁定源码/脚本/测试 SHA256；最终检查确认未变化。release-qualification-importlib-gpu-guard.json 确认 cuda_initialized=false。

## 资源生命周期
入队或内核接受不等于完成。缓冲指针按原合同由调用者保活至完成或安全排空，适配层不分配第二套 KV/staging。
短 I/O、负 errno 原样进入作者完成处理；不把异常包装为成功。
关闭时先拒绝新请求、提交已接受的工作并排空，再销毁内核 context 和通知 FD。
尚未交给调用者的成功 open FD 在关闭时回收；已交付 FD 仍属调用者。
无法确认内核排空时保留请求引用并报致命错误，要求停止拥有者进程，不能继续复用缓冲。
实际 CUDA event、pinned staging、D2H/H2D 和生产 KV 生命周期仍未通过 GPU 验证。

## 性能初筛
执行 aio_cpu_microbench.py：块大小 4KiB/64KiB/1MiB，深度 1/4/16，两次重复，两种方式，最终共 72 个样本。
对照是直接调用同一 Linux AIO 数据接口，只有提交/回收，没有异步元数据或完整 reactor；它用于检查适配开销，不是研究基线或 io_uring 性能。
最终轮在计时前对文件统一预写并 fsync，模式顺序交替。使用固定偏移、预打开小文件、短时间测试，结果包含 Python 轮询，不能代表冷 SSD、持续压力、带宽上限或生产请求流。
- 4KiB、深度1读取：适配约 13.79–13.88 MiB/s；直接参考约 46.43–59.63 MiB/s。小块适配开销明显，不能称低开销。
- 1MiB、深度1写入：适配约 328.04–329.37 MiB/s；参考约 323.43–328.55 MiB/s。仅限这组短样本。
- 1MiB、深度16写入：适配约 363.26–391.05 MiB/s；参考约 363.33–395.01 MiB/s。
- 大块读取波动较大，不能据个别更快数字宣称收益。
- 调用者 submit_pending 返回较快，是阻塞提交被移到后台线程的表现；不等于总 I/O 延迟降低，也不代表 GPU 干扰已经减少。
本轮三个微基准轮次共计 1,920 MiB 逻辑读写，最终轮额外预写 23,482,368 字节；正确性测试另有少量 I/O。所有轮次保留，早期轮次为诊断过程，不与最终轮混合汇总收益。
最终测量进程峰值 RSS 约 844 MiB，包含 Torch/vLLM 的 CPU 导入开销，不是适配器自身占用。

## 预算、原路径与阶段门槛
GPU 工作负载 0、模型加载 0、模型下载 0、系统/驱动更改 0。
权限文件与账本 SHA256 前后不变。历史累计 GPU 285.247483 秒（约 0.079235 小时）/8 小时；模型下载累计 19,422,798,722 字节/20GiB，不重置。
后端开关和策略开关独立：linux_aio+off 是兼容基线；io_uring+off 才是原路径，而原路径当前仍被环境阻塞。
共同兼容后端必须一致用于所有研究实验臂，并重新标定成本，不能把较慢后端制造的差距作为算法贡献。

## 下一允许阶段
CPU 功能门槛通过；本轮停在 CPU 授权边界。可继续 CPU 的生产块尺寸/元数据混合负载验证和准备 GPU 驱动命令。
GPU 阶段需先核对实际 GPU 身份、明确授权与剩余预算，再进行短时真实 pinned staging、CUDA copy、生产 KV 往返与 mandatory drain 验证。当前 GPU 是否可用也需重新检查。
P1 未完成，P2–P7 不开启；成本标定、TTFT/ITL、goodput、策略收益与原 io_uring 性能差距均未测得。
