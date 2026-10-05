# P4 CPU 能力矩阵

| 能力 | CPU 当前证据 | 生产/GPU 状态 |
|---|---|---|
| 原作者引擎、LoadPlanner、Prefix、模型执行器 | 原字节保持，当前相关回归通过 | P3 历史资格保留，新 P4 未获 GPU 资格 |
| 32 parent / 64 work / 3 depth / 8 closure | 严格界限、超限原路径通过 | 不截断原生任务 |
| run/epoch/generation/capability 与完整父 AND | policy/ABI/native fixtures通过 | 无真实 GPU owner 复用接口，credit未知 |
| CPU staging/cache/shared consumer 生命周期 | fake 事件、STOP、异常排空通过 | 潜在回收不是立即释放；需真实资格 |
| off/shadow、策略与公共基础分离 | 原路径/额度、common-only off、patch字节往返通过 | 真实开销未测 |
| fixed/pressure | 原 P3 路线保留与回归 | P3 有界历史资格，不外推新 P4 |
| dependency_only | 纯值算法、固定普通额度、真实字段绑定通过 | native ETA未知，不模拟真实重排 |
| interference/joint | 算术、有限批量、精确状态/单位与回退、preview seam通过 | 实际回U；producer/verifier/loader/生产应用未完成 |
| 生产候选表入口 | 严格 schema/上下文/哈希/证据链、伪资格拒绝通过 | 结果恒blocked；不执行引用verifier |
| 强制进展与接受计费 | mandatory/continuation/age/STOP、fused bytes、accepted失败通过 | 没有绕过设备/事件/原生安全约束 |
| Linux-AIO | CPU真实字节路径通过 | io_uring 3项被环境拒绝；不修改系统 |
| 新P4真实CUDA/模型/输出/IO干扰/live shadow | 未执行 | BLOCKED |
| 方法提升/正式goodput/P5-P7 | 没有新增结果 | 未验证/未开始 |

唯一统一测试为1588 pass、16 skip、0 fail，179项P4新增。CPU/mock不等于GPU资格；此前独立重复运行不叠加计数。
