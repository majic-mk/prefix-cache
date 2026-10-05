# CPU 协议交付 2026-10-04

实际新增仅在本 protocol 目录：前瞻家庭/运行清单、标准库严格验证器、保留驱动 AST 能力记录、逐 token 时刻校验、独立开发预算冻结校验、实际 source/CPU/guard byte join 和拒绝测试。没有修改既有实验记录、源代码、旧 U、旧 A预算、GPU ledger、系统/驱动或模型。

实际本地 CPU 测试 22/22 PASS，失败 0、跳过 0。测试包括 7 类 entry join 拒绝子场景；子场景不另增 unittest 测试计数。stdout/stderr、命令、输入 bytes/SHA 和 exit=0 已记录在 `CPU_PROTOCOL_FINAL_TEST_RESULT.json`。fixture 仅用于算术与拒绝路径，不是 GPU 数据。

实际命令（Windows Python 3.12.14，全部 -B -I -S）：

```text
python -B -I -S prepare_protocol_cpu.py --project-root <实际本地工作区> --output-dir <本 protocol 目录>
python -B -I -S -m unittest discover -s <本 protocol 目录> -p test_i_pilot_protocol.py -v
python -B -I -S i_pilot_protocol.py --protocol <本目录>/I_PILOT_PROTOCOL.json --output <本目录>/FINAL_CPU_PROTOCOL_DECISION.json
```

首次执行 20 项初版测试通过；加入真实 source/guard join、布尔声明不能替代证据及七类 join 拒绝场景后，终版 22 项通过。`prepare_protocol_cpu.py` 实际读取保留规范 02/03/04 全文并核验 UTF-8 bytes/SHA。实际 AST 读取六份已保留 runner/controller，不导入 Torch/vLLM/模型/本地 native库。

实际发现保留 calibration/single-file 的 `load_planner='off'`；本轮校准入口只是直接 preload 机制诊断。完整强 U/I 原 LoadPlanner 请求流未绑定。保留 frontend 以累计输出记录 token 时刻，未来必须逐步核验每次只新增一个内容 token；不能让一次 chunk 返回多个 token 冒充多个独立 ITL。

当前本目录 standalone 判定：`CPU_PROTOCOL_COMPLETE_RUNTIME_BINDING_REQUIRED`，效果 `BLOCKED_FOR_EFFECT`，正式 goodput 不允许。只有 actual server 新 source lock、完整 source proof、真实 CPU 入口测试和原 guard preflight 的 refs 完整连接，才会输出 `CPU_READY_FOR_GPU_CALIBRATION`。这只准备 cal01，不准备 on/评估自动执行；当前九槽 dev/on/eval 部分是前瞻预算规划。

真实 GPU 运行：0。新 GPU 消耗：0。未预留新的 GPU budget。原历史剩余 4121.104761 秒，九槽计划最高 3780 秒；原 guard 在每次实际运行前读取实时 ledger 并独占预留。持续 GPU 授权保持，不要求按新源锁再次审批。

下一允许阶段：root 将本协议源锁加入新 calibration_v2 的真实 CPU 交付，并在实际服务器执行 byte join。其通过后，在GPU可用且原 guard 的现有现场条件通过时，仅能执行新 cal01 校准诊断。旧失败不被改写；独立 development deadline/余量、普通候选、shadow/on 生命周期、完整 U/I 入口与配对评估继续作为效果门槛。
