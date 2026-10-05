# P4-02 当前能力与资格

本轮仅允许服务器 CPU 开发和测试；真实 GPU 运行为 0。实现存在、CPU 合同通过和生产资格成立是三个不同条件。

| 接点 | 当前实现 | CPU 可验证范围 | 当前生产资格 |
|---|---|---|---|
| 精确 Prefix Cache、原成本准入、共享 staging、预加载、复制融合、原流水线 | 继承现成 py-kvcache 与作者 vLLM | 原回归与真实 Linux AIO；CUDA 使用明确 fake | P4 新源码真实 GPU 资格待验证 |
| 有界候选与 I/D/J 合同 | 原 P4-01 增量及本轮原 owner 接线 | 新旧合同、异常、关闭回退、真实 owner 时序 | 新策略 GPU 效果未验证 |
| Scheduler 负载元数据 | 原 scheduler→worker 消息通道；一个替换式标量单元 | 源绑定、过期、冲突、unknown、原预算及融合保持 | scheduled-work-v1 **不是** actual active GPU decode；不授予 production_gpu_load_state |
| 完成时间历史 | 原完整父任务 first-ready→实际 drain 成功钩子；有界标量历史 | 因果、失败排除、未完成排除、窗口上限、经验裕量 | 仅诊断；production ETA=None；经验裕量不保证覆盖率 |
| 有限 batch 建议 | 原 SSD read batch 前缀咨询；仍走原 slot/iodepth/accounting | 原排列、有限前缀、不可分的 continuation、伪资格拒绝 | 未资格时返回 None；当前 I/J 保持原 U 路径 |
| 真实原始记录 ABI | 四类 raw role 的追加式字节绑定工具 | 标量格式、实际文件 SHA、AB/BA、trace/workload/family split | 未实现/验证可信 GPU 单步窗口采集；native 标签不等于实际运行证据 |
| 配对成本语义校验与 builder | 一份统计计算实现；验证/构造/loader 共用 | 原始字节、隔离、顺序、阶段/量子、上下文、源漂移反例 | prepared only，gpu_qualified=false，production lookup=None |
| 原作者执行器启动准备 | 读锁定 ENGINE/SAMPLING/模型计划构造单阶段配置 | 配置及来源、合法候选；无新模型执行器 | 预览不代表已启动或 ABI 通过 |
| 新原生 G1 启动准备 | off/shadow 原 FileMapper/TransferCoordinator + 旧唯一 GPU guard | 先纯 CPU scope/source/storage/budget gate；明确拒绝测试 | GPU identity、CUDA、AIO/owner、新 Python/旧 binary ABI 尚未通过 |

D 的显式共同固定预算与原基础控制器绑定；I/J 未资格时不会因为声明 capabilities 或装入 PreparedCostTable 而改变预算。只有真实 GPU 来源、上下文、阶段量子、load state、成本、误差及生命周期资格完整时，才可讨论生产激活。CPU fixtures、成本 native-origin 标签、scheduler 元数据和原 P3 aggregate 均不能授予资格。

P3 保持有限 pilot 已闭合、方法收益未证明、U 为当前最强独立参照。P4 未完成；P5–P7 未启动。
