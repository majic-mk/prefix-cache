已修复准备阶段的无效等待。新 CPU 入口 g3_fast_reference_preflight.py 已部署至服务器 /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g3-reference-latency-cpu-v1-20261002，默认先进行小范围环境核验，满足条件才调用原完整 CPU 预检。它不接受 GPU 启动/执行/scope 参数，也不创建授权或修改原 GPU 包。

根因有实数支持：原源锁 4,088 项共 15,807,096,855 字节（14.72 GiB），其中 11 个模型文件共 15,242,788,168 字节，占 96.43%。原预检先读完整模型，最后才检查驱动，真实耗时 96.574216 秒后发现驱动为 0 字节；随后重复源审计又花 85.356682 秒。合计约 182 秒用于这种可提前发现的环境失败。当前容器实际 cgroup CPU quota 为 50000/100000，即 0.5 核，内存上限为 2,147,483,648 字节（2 GiB）。这些是只读快照，未修改资源配额。

修复先以固定字节/SHA 绑定原 8443f45d… GPU 源锁、SDK helper、inventory 和原 CPU proof，然后检查原驱动的规范路径、文件类型和字节数。当前 0 字节挂载立即返回 BLOCKED_DRIVER_ASSET_GATE / exit 78，并明确完整源与 runtime 资产仍未验证。读取量从模型全量扫描变为 1,624,919 字节的小文件，模型读取 0、完整源核验调用 0。大小匹配只给 CONTINUE，不给 ready；默认继续原完整预检，同大小但 SHA 错误仍会被原全 SHA 校验拒绝。

三个真实服务器进程的壁钟耗时为 0.185476 / 0.294537 / 0.212278 秒，中位数 0.212278 秒。相较此前 96.574216 秒，当前失败路径的判定耗时缩短约 455 倍。这是环境失败检测的耗时改进，不能用于推理吞吐、TTFT、缓存策略或论文性能结论。原成功 GPU 路径的父启动、guard 子执行和父结束完整检查全部保留；本次没有测量该成功路径的加速。

实际服务器命令如下，以 /root/autodl-tmp/prefix-io-v1-handoff/project 为工作目录：

```sh
D=artifacts/prefix_io_v1/server09-g3-reference-latency-cpu-v1-20261002
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/g3_fast_reference_preflight.py" --project /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S "$D/test_g3_fast_reference_preflight.py" -v
```

第一条实际执行三次，均以正确的环境阻塞状态退出 78；第二条在 Linux 服务器执行一次，16/16 PASS、0 failure、0 skip，0.522 秒（含进程壁钟 0.756056 秒）。本机曾有 1 项符号链接权限 skip，服务器实际补全覆盖。原 85 项测试未因无变化重复运行。独立只读审查通过，完整日志及测试命令见 ACTUAL_SERVER_FAST_PREFLIGHT_BENCHMARK.json、ACTUAL_SERVER_FAST_PREFLIGHT_CPU_TESTS.json 和 INDEPENDENT_FAST_PREFLIGHT_BOUNDARY_REVIEW.json。

服务器原系统 driver 路径及 libcuda.so/libcuda.so.1 别名仍指向 0 字节文件。现有 compat 570 库不是冻结要求的 595.71.05 资产，未使用它替换原驱动，也未改系统或 SDK 引用。原 GPU 源锁 SHA 8443f45dd4dd6c1c64595199051f5e88449268af347bcbd836c927b754eabe30、原入口 SHA 1bd502be997fd1a784ab49aa0262d45dabeca2374a53312d4c96dbdc504ea7a6、原 guard SHA 3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a 均保持不变。已有两次/640 秒的授权提议不变；这次 CPU 修复不需要更换其 GPU 源锁。

本轮实际 GPU 运行、GPU 元数据查询、GPU 初始化与研究策略启用均为 0；没有新 GPU 授权、没有下载、删除数据、修改系统/驱动/安装包/作者源码。GPU 账本前后均为 be14eb7e34ccde501481777340ec3baf62e1f8511436614324ff976585ebf0f3，剩余约 3.05 小时。

后续工作采用 NEXT_ALLOWED_FAST_CPU_COMMAND.json 的入口：等待环境变化时只运行 --quick-only，不为相同的 0 字节驱动错误重复全模型扫描。该探测若返回 CONTINUE，也不证明 GPU 可运行；在原待确认授权和环境门禁均满足后，直接使用原 --launch，让原完整检查在必要关口执行，避免之前多做一遍独立全量预检。当前 GPU 阶段仍阻塞于原驱动挂载，没有编造 GPU 效果结果。
