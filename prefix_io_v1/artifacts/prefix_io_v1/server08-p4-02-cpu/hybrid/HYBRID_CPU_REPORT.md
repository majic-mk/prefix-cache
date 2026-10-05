真实 Linux AIO 的原 reactor CPU 验证已通过，最终定向运行 3 passed、0 failed、0 skipped。GPU 工作负载为 0，CUDA 未初始化，预算 ledger 未改变。这不是 GPU DMA、模型推理或方法收益实验。

本轮新增仅为 tests/prefix_io_v1_p4_hybrid/__init__.py 和 test_real_aio.py。fixture 复用新 02 NativeRig 和作者 IoReactor，保留原 intake、pump、复制映射/合并、父 admission、SSD 完成、终止与 shutdown；窄替换旧 fake 文件/ring 为现有 LinuxAioRing、DirectIoFileStore、FileMapper，并分配实际对齐 CPU staging。复制函数是对拥有范围内 CPU 指针执行 memmove，CUDA stream/event 是明确 fake。没有重写执行器或新增工作队列。

off 与 shadow 两种路径各执行两个不同 payload 的真实 O_DIRECT store/read roundtrip。每阶段 d2h、ssd_write、ssd_read、h2d 记录 8,192 accepted bytes；两个 payload 的磁盘/回读字节 SHA 相同。实际 Linux AIO accepted=completed=reaped=8，最大 kernel inflight=2、最大 outstanding=2（capacity=4），最终 closed/drained=true、outstanding=unreaped=0。假复制事件未 ready 时原 Future 与 staging slot 保持保护。此处父计数由原 fixture pump 在 reactor owner 线程捕获，停止后通过 native_shutdown snapshot 读取。

真实 240 字节文件的 4,096 字节读取触发原短读错误，accepted=completed=reaped=3。原父失败 drain 完成后 admission 归零，未提交 H2D，CPU destination 保持原字节。原错误字符串仍写 io_uring read transferred，实际本用例 backend 是 LinuxAioRing；不以错误文案推断后端。

实际命令在 hybrid/run-04/command.json，入口脚本 hybrid/run-04-executed.py。入口先执行 experiment_storage.preflight(scratch,128 MiB)，再使用冻结项目 venv 和既有 run_aio_cpu_tests.py、CUDA_VISIBLE_DEVICES=''、禁 bytecode/offline 设置，只运行 tests/prefix_io_v1_p4_hybrid。guard/result/junit/process.log 与 9 个 before/after 源 SHA 均存于 hybrid/run-04。原/native/control 源未被本轮修改。

失败证据完整保留：run-01 三个 pytest 失败后因真实 ring 未关闭导致 90 秒进程 timeout；原因是测试从非 owner 调用 owner-only admission API、旧 wait=False fixture cleanup 未关闭真实非 daemon AIO worker，以及短读 regex 文案不符。run-02 捕获 owner snapshot 的测试 callback 未接受原 pump keyword 参数；run-03 测试将 rig.h2d_launches 误写为 reactor.h2d_launches。修复均只在新增测试 fixture，改用原 shutdown(wait=True) 完整关闭。最终 run-04 全通过。所有失败 guard 也记录 CUDA initialized=false、GPU=0，未隐去失败。

这补足真实文件/AIO/原生命周期的跨组件验证；fake CUDA 不授权生产 ETA/干扰额度/即时 GPU 回收，也不能用于声称性能提升。下阶段交由 root 完成冻结 CPU 统一回归；明天真实 GPU 操作继续遵守 permissions.yaml/ledger 和冻结入口。成本独审最新结果见 eta/review-cost/final-counterexample-replay-03/result.json，旧漏洞均已拒绝，生产门禁保持关闭。
