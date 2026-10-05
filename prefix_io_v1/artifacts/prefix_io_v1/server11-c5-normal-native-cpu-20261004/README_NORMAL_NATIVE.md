# 正常模型路径的 CPU 接线准备

此目录是 StageD 的新增副本。此前 StageD、G 和 GPU03 源码、授权、原始记录保持不变。
本目录仅推进正常 off/shadow/on 接线、真实标定来源绑定和观测成本计量，未取得新的 GPU 执行授权。

固定复用 G common 与原模型执行器；新版公开 canonical 接口只补历史账本回放校验。
旧 G 完整源码及证据保持不变，新校验复用原始序列化和成本公式。真实 GPU03 结束时的
只读账本快照用于核对后续账本是否完整追加，绝不改写当前账本或提供新执行授权。
生成的正常 binding 可以在 CPU
重放既有 GPU03 数据以核实类型、来源和成本；这不构成本目录 normal/on 的原生运行资格。

各模式使用同一 COMMON_SOURCE_LOCK；每个模式的配置、权限、授权、前驱资格和 scope
在 common lock 外单独固定，避免改变锁或循环引用。旧 GPU03 授权已消耗，不能复用。

CPU 检查通过 run_cpu_normal_native.py 只运行新增的控制器/运行时/历史回放测试，核验完整真实来源
前后不变、GPU 预算账本不变、没有导入 Torch/vLLM/native backend。本目录复制保留的
test_combined_runtime.py 和 run_cpu_combined_runtime.py 是历史 StageD 合同，未作为本轮验证运行。

观测计量只可陈述实际覆盖的入口；CPU fixture 和 synthetic 时钟不得设置 native/on/full-cost
资格。正常 GPU 仍需 off 生命周期通过后才 shadow，shadow 通过后才 on，实际等待未触发时
标记 NOT_EXERCISED。不得修改成本上界、工程预算、SLO 或负载来制造收益。

实际 GPU03 上界 20.456415 ms 高于 A-only 工程预算 12.957824 ms，当前标定条件的严格
成本准入应拒绝；这不是策略性能结论。完整正常观测成本及 P4 提速尚待原生运行验证。

