已在新建 SSH 连接中确认：当前连接的容器没有任何 /dev/nvidia* 设备节点，mountinfo 中无 libcuda/nvidia 挂载，CPU quota 为 0.5 核、内存上限 2 GiB，冻结595.71.05驱动为0字节。用户随后直接确认“仍是无卡模式，我会切到 GPU 模式开机”。因此本次GPU不可用与已确认的无卡模式一致，不把0字节现象解释为驱动损坏。

实际执行快速CPU检查：CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S artifacts/prefix_io_v1/server09-g3-reference-latency-cpu-v1-20261002/g3_fast_reference_preflight.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --quick-only。返回78 / BLOCKED_DRIVER_ASSET_GATE，fresh连接内部耗时0.026653秒，模型读取0、完整源扫描0。还只读核验原源锁/入口/guard及预算账本，原8443f45d…源锁与账本均保持不变，活动预算预约为null，剩余10990.859632秒，PRIMARY可用12599889920字节。

没有重复此前已经通过的85项reference及16项fast gate CPU测试。CPU实施、接口审查、输入与交付验证已完成；下一项必要事实是GPU下两个有限数值参考作业的完整输出及原top5比较，CPU夹具不能替代。待用户以GPU模式启动同一实例后先快速检查挂载，再按原源码、原GPU UUID、人类scope、8小时累计预算和存储门禁执行；这里没有创建新授权或启动作业。

本轮GPU运行0、CUDA/NVML/nvidia-smi调用0、模型后端导入0、系统/驱动/安装包更改0、数据删除0；只有文件系统只读环境审计及新CPU证据写入。详见CONFIRMED_CPU_MODE_AND_NEXT_STAGE.json与ACTUAL_CONTINUATION_CPU_COMMANDS.json。

