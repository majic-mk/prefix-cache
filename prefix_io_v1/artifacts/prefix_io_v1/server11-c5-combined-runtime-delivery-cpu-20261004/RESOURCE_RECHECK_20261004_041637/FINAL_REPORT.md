# 2026-10-04 服务器资源复查

已实际连接指定服务器 connect.westc.seetacloud.com:26909，在 /root/autodl-tmp/prefix-io-v1-handoff/project 执行项目 Python 标准库只读检查。现场资源快照时间为 2026-10-03T20:16:37.749505+00:00。

GPU 设备节点仍为空，CPU cgroup 配额 50000 100000（0.5 核），内存限制 2147483648 字节（2 GiB），模型进程为空。数据盘可用 56.87 GiB。CPU affinity 可见多核并不改变半核实际配额；累计 nr_throttled 不解释成本轮测量。

本轮没有修改源码、权限或预算，没有重跑已通过的 65 项 CPU 测试，也没有执行正式性能对照。测试新增 0，GPU 运行 0 次、消耗 0 秒。现存源锁 SHA-256 bb7522102bddee35e5878551e98d23ae475897e24663d4c3aa9ae235329b176d 与已封存交付一致；这里只核对锁文件本身，没有宣称重新核验锁内全部来源。预算账本原始 SHA-256 bef6d78e43058eaad811ab5ecb08937356c911784f9000e76eea9de0276c0b72 未变，剩余 6419.438742311206 秒（约 1.78 小时），活动预留为 null。

实际执行方式为 .venv/bin/python -B -I -S -c；完整命令及退出码保存在 COMMANDS.json，现场结果保存于 RESOURCE_SNAPSHOT.json，权限及预算派生字段保存在 AUTHORITY_SNAPSHOT.json。两份有效结果对应命令退出码均为 0；这表示只读检查成功，GPU/性能资格仍为 BLOCKED。另一次只读检查打印完整历史账本导致本地输出截断及 JSON 解析失败，随后改为有界摘要输出；该命令及真实失败原因也保存于 COMMANDS.json。这不是 CPU 测试或 GPU 实验失败。

04_CODEX_EXECUTION.md 的要求是 GPU 实验只在权限已有明确授权、目标环境可用且预算不为空时运行。当前目标设备不可见；CPU Stage C/D 版本也刻意保持 GPU 入口阻断，不能使用旧 UUID 或简单移除阻断启动实验。没有生成 GPU 配置、scope、作业、凭据或预算预留。

独立只读审查确认下一步顺序：足够 CPU/内存且有可见 GPU 的现场资源核验；在新目录准备真实 common 六窗口 GPU 修订并绑定现场 UUID、资产、源码、作业及授权；执行真实标定和严格 canonical receipt；再进行 off/shadow/on 生命周期及完整入口成本；验收通过后 P4 公平对照。新 GPU 修订需恢复真实 native serializer/receipt 路径，保留已冻结 CPU 版本，固定六窗、128 完整输出、917504 字节读取、8 个 accepted parents 和原成本公式。无实际 defer 记 NOT_EXERCISED。

现在需要用户把当前实例切换为 GPU 模式并告知完成；若换实例，则需提供新 SSH 地址。切换后先核验现场资源，不能承诺立即进入 P4。没有 C5 GPU 收益或系统最终可行性的新结论；现存负结果保留。本阶段无需扩容或删除数据。
