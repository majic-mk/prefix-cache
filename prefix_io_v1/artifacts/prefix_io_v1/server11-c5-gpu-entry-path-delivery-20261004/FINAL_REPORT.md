# GPU 现场验证与路径修复交付（2026-10-04）
当前服务器 connect.westc.seetacloud.com:26909，项目 /root/autodl-tmp/prefix-io-v1-handoff/project。当前 RTX 5090 UUID GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9 已处于 GPU 模式，实际可见字符设备 /dev/nvidia5；空闲显存 32110 MiB，CPU 25 核配额，内存配额 90 GiB、可用约 89.6 GiB，PRIMARY 可用约 56.8 GiB，无其他 compute 进程。
本次仅做公共六窗成本标定入口与现场资格，未运行 P4 性能对照，没有 C5 GPU 收益证据。
## 实际修复与保留
F（server11-c5-gpu-entry-device-revision-20261004）兼容数字 NVIDIA 字符设备，不再固定 nvidia0；nvidiactl 也须真实字符设备，拒绝普通文件、目录、符号链接、UVM/caps。所有 GPU/CPU/内存/磁盘/空闲进程门槛保持。
F 的唯一真实 guard 启动作业 gpu02 在模型导入前失败：argparse 将相对配置参数转换为 pathlib.Path，而 validator 把所有 Path 当成绝对路径，错误返回 configuration outside project。这是我方入口接线错误，非 GPU、驱动或 AutoDL 平台失败。
G（server11-c5-gpu-entry-path-revision-20261004）将 relative Path 转为项目相对 POSIX 路径，再进入唯一 CONFIG+safe 校验；absolute Path 仍须在指定 root，旧路径、越界、遍历及符号链接仍拒绝。新增实际 argparse 相对/绝对/越界形态测试。
旧 E/F 源码、冻结包、权限与失败证据保留，均未覆盖。G 为新目录和新标签 gpu03，canonical pins 随路径刷新；63 个公共文件字节、32 个函数 AST 和原数学/holdout/六窗设计保持。未改模型执行器、缓存引擎、原 GPU guard、权限预算、负载或验收门槛。
## 命令与测试
F 实际执行 freeze_device.py、run_cpu_native_entry.py、run_cpu_entry_binding.py、control_native_cost_job.py context/bind、prepare_and_verify_native_cost.py --prepare、control ... scope/launch/after。23+44=67 项服务器 CPU 测试通过，无失败/错误/跳过。
唯一 guard 作业为 server11-c5-native-common-cost-gpu02：exit=2，elapsed=0.1312496941536665 秒，模型子进程 0，完成窗口 0，session_drained=true，无残留 reservation。failed after 的完整源码核验成功写入 SOURCE_AFTER_VERIFICATION.json，随后严格成功资格校验按预期拒绝失败 guard（exit2），没有有效成本凭据；并非另一次模型尝试。
G 实际执行 freeze_path02.py（未执行的初稿 freeze_path.py 保留）、run_cpu_native_entry.py、run_cpu_entry_binding.py、control_native_cost_job.py context。23+51=74 项服务器 CPU 测试通过，无失败/错误/跳过；CPU 锁 479 引用前后核验一致。此次新测试有真实 CLI argparse 产生的 Path 形态。
全部 full argv/cwd/stdout/stderr/exit 在对应 SERVER_* 命令文件与日志；F GPU 启动命令在 device delivery/GPU_LAUNCH_CONTROLLER_COMMAND.json。
## 真实资产与预算
G context 先真实资源探测，再完整重核验旧 4655 行资产及增量，4735 个源/资产引用全通过；没有下载、复制权重、删除数据或系统/驱动修改。
G revision source lock：4372cb6bb43399aededd3509305691dabb88cfbb16962ddecbb0887697c5a305。
G live context：0bb9efcdb5d340a81b9dfd1079c3e2fc91d9e6dfeee23e1b53cfcdcd0077ef02。
G CPU source lock：4f8f58d713a488ff9b526fbc946428276e3953bc8e63225e8d61f64faa205e4b。
账本由原 guard 从 22380.561257688794 变为 22380.692507382948 秒，差额 0.1312496941536665 秒。剩余 6419.307492617052 秒（约 1.78 小时），无活动预留。新账本 SHA 45af5eacc6b137f3aa9c2b20ee8d2755d9a42892b578ee1ef4f83e26d76145c2。
## 下一允许动作
F 的 max_attempts=1 已消耗。G 当前只有现场 context/revision source lock，未创建 HUMAN_GPU_GRANT、EFFECTIVE_GPU_PERMISSION、GPU_ENTRY_BINDING、NATIVE_COST_CONFIG、SITE_SOURCE_LOCK、scope、launch intent，gpu03 run 不存在。
已向用户提出具体单次修正版授权：同现场 UUID、当前 G source/context、一次公共六窗、最多 6 子进程、1200+20=1220 秒、原累计预算、失败停止且不自动重试。未获新答复前禁止执行。
获得后严格依次记录真实人类授权 → bind → plan → scope → launch原guard → session排空 → after完整source复核 → native --verify。真实结果合格后才准备正常 off/shadow/on 生命周期与后续 P4，本轮授权不覆盖这些扩展。
现场 GPU 可以运行不等于模型或成本已通过。当前系统最终可行性、C5 GPU 改善和论文收益均未验证。

