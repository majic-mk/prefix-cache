# 新协议初版拒绝探针（终版修复前记录）

实际在本机 Python 3.12 `-B -I -S` 导入纯标准库协议模块并运行边界调用，未执行其 CLI、GPU guard、RPC、引擎、模型、数据写入或真实开发预算冻结。所有探针均为注明的 CPU fixture，不能作为实验或资格收据。目标源码运行前后 SHA 相同：`1a448386120e51b9ceb681fb7d8481434bc3dfed03b3817033109d7d863707bd`。完整结果保存在 `PROTOCOL_INITIAL_REJECTION_PROBES.json`。

- `rental_decision` 将 `remaining_seconds=NaN` 和 `Infinity` 视为可进入有界强 U off 诊断。应验证剩余值类型、有限性及 0..28,800 范围，不得让比较运算的 NaN 语义打开门。
- `freeze_development_budget` 接受非十六进制的 64 字符 SHA `z*64`，且接受任意未校验 SLO 对象并输出 `formal_goodput_allowed=true`。应拒绝无效 SHA/SLO，且纯预算算术不能单独赋予正式 goodput 资格；最终还需实际来源字节、开发/评估隔离、原准入及生命周期联接。
- trace/free-qualification 目前接受输出长度 2，与上一轮实际强 U/I 构造合同至少 128 tokens 不一致；应在当前路线统一合同或明确拒绝交叉消费。
- 客户端或迁移服务器后的 monotonic 声明值不能直接与服务器开发起始值比较；应要求同服务器、boot/clock scope 的前瞻证据，或使用可核验的冻结记录顺序。

以上是初版问题，不预断终版仍存在。修复后应实际执行相应拒绝用例，并保留新源码 SHA 与结果；不得删除本初版记录。当前 22 个测试通过不覆盖这些缺口，不能替代终版独立验收。
