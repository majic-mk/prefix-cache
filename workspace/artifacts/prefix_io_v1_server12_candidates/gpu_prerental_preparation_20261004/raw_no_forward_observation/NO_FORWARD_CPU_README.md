# 初始无模型执行调用的观测修复

新增 `runner/bounded_native_full_step_collector_v2.py`，固定委托原 collector SHA `9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d`。仅在作者原 SchedulerOutput/CachedRequestData 类与冻结源一致、严格模型工作字段为空、尚无模型帧且无标量或 CUDA 事件 pending 时，直接调用原 execute_model 一次。非空 KV metadata 和 finished 通知继续由作者原方法处理。正工作、不确定 metadata、已有模型步骤或 pending 均走原观测路线。新增首调用诊断只复制类型、长度、整数与布尔值，不保存请求 ID 或资源所有者。

实际本机 Python 3.12.14 的最终 16 项 CPU 测试全部通过，结果为 `LOCAL_CPU_RESULT_FINAL.json`。执行命令为 `python -B -I -S test_initial_no_forward_cpu.py --output LOCAL_CPU_RESULT_FINAL.json`，环境 `CUDA_VISIBLE_DEVICES=''`；该命令在脚本目录内使用实际现代 Python 路径执行。测试脚本支持明确的本地和服务器布局，并对使用的真实作者与原观测源码核验 SHA。

测试回放完整作者 execute_model AST：初始零工作保留原计数增长、KV 回调、原返回对象和原异常对象；正工作实际走到作者原 prepare，随后以明确 CPU 探针停止。原 scalar/worker execute 的真实 AST 证实无 prepare/sample 的旧观测会关闭，新观测只绕过初始零工作。纯 CPU Event 是显式探针，没有创建 CUDA Event。原 FullStepCapture.export 实际拒绝 0 模型帧，仍要求精确 128 真实帧；未伪造、补齐或重编号任何模型步骤。

首轮测试因 AST 查找没有搜索 try 内的嵌套函数而失败，日志 `LOCAL_CPU_RESULT_01.json` 保留；修正仅限新测试脚本，13 项复验 `LOCAL_CPU_RESULT_02.json` 和 16 项最终验收均保留。未修改原 collector、标量适配器、worker、issuer、validator 或作者 executor。

本轮 GPU 操作、RPC、真实 CUDA Event 均为 0。实际 CAL03 失败帧此前未持久化，无法据此确认现场执行了零工作分支；本 CPU 回放只证明该源码因果路径成立。新 shim 的真实安装、完整 128 CUDA witness 和成本单元资格仍须后续 GPU 验收，不构成策略效果或性能收益证据。
