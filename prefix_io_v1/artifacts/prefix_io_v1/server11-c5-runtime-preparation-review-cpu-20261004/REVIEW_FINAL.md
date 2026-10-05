# 新 C5 运行入口的独立 CPU 审查

结论限定为 **CPU 接线正确性准备通过，GPU 资格仍阻塞**。本目录只增加反例、运行器与审查证据；不改冻结 C5、C4、原 runtime_v4/native_cost_v6 或历史实验结果。最终同源执行记录见 `LOCAL_REVIEW_01/TEST_RESULT.json`、`TEST_STDERR.log`，源锁及每份实际读取文件的 SHA/字节数在该 JSON 的 `source_before` / `source_after`。服务器重放结果必须另存，不能把本地结果标成服务器结果。

审查的实际入口是 `notification_v5_runtime_preparation/`，路径下行号如下：

- `run_p4_single_file_experiment.py:593` 使用 owner `LABEL` 安装 capture；原 frontend request `rid` 不变。`:603` 先经过原 bridge attachment 后调用统一通知 helper。独立测试执行这些真实 AST 调用片段，分别验证 off/shadow/on，未执行模型或 GPU 初始化。
- `verify_p4_single_file.py:285` 校验 owner run、外部 request 与 native request 三个身份；`:290` 继续调用原 `native_cost_v6.validate_capture` 完整检查 128 帧。合成 128 帧正例与错误 run、请求身份、缺帧反例均执行原函数，未删除旧约束。
- `notification_runtime_adapter.py:205` 只在 on 安装原 C5 等待注册；off/shadow 保留未安装的原队列。实际源 SHA、原方法 `co_filename`、wake 类 identity、已有 bridge/capture identity 都先校验。
- 同文件 `:138` 只包装原 Queue 实例的 `get`，保留原 unbound `Queue.get` 并原参数调用一次。队列 FIFO、返回值、原异常保持；观测异常记录后不阻止原 get。保存弱引用及最多 16 条值观测、8 条失败；恢复时不覆盖后来安装的其他 wrapper。
- 同文件 `:258` 不以 deferral 代替等待证明；旧 wake 或一般 native message 不能单独晋级为通知已运行。原 deadline timeout 只说明原截止时间兜底，不说明 end 通知交付、GPU 完成或资源释放。`performance_claim`、GPU 完成与原生成本资格均必须为 false。

两项实际竞态已独立强制复现并验证：

1. current-check 返回 True 后、get 前，end.record 已清注册并将匹配 wake 入队。允许该行 registration=None，但只能用同 token 的真实匹配 wake 证明；把该行改成 native message、旧 wake 或 timeout 都会拒绝。
2. capture.detach 只发布 wake，不能直接清 reactor arm。真实双线程测试暂停在原 get 前，此时 detach 后仍有 arm 和 pending 行，提前 export 无资格；get 实际消费后才清 arm。启动器成功路径 `:619`→`:623`→`:625` 为 detach→原 drain→关闭/导出观测；异常路径也在原 drain/shutdown 之后关闭。未向 owner state 写入伪清理值。

本轮独立测试为两个 suite：冻结入口 8 项、实际准备入口 14 项，共 22 项。GPU 模样的 event、native metadata、128 帧数据均是明确标注的 CPU 合成输入；真正执行的是原 Python Queue、冻结源 AST 控制分支和原纯 CPU 验证器。没有执行实际 `collector.install`、vLLM 模型或 CUDA。实际测试数、失败数、平台、命令及前后 SHA 一致性以最终运行 JSON 为准。

尚未形成真机资格的部分：

- C5 receipt 文件与 C4 字节相同，仍绑定 C4/native-v6/job06；当前准备目录的 launch/control/verify 入口无条件阻塞。不能用旧 v6 成本、旧 GPU 输出或 CPU fixture 解除阻塞。
- 未来新 receipt/ref 工厂需要显式绑定同一最终 reactor、collector 与完整启动器/观测源锁。collector 仍属 overlay refs，再要求它等于新计划的 `collector_source_ref`；直接改成 common 会违反冻结 bridge 的来源查找约束。
- 原先约 73.6% CPU 计分只覆盖当时 C5 reactor/collector 基准，不覆盖本轮 Queue.get 观测包装和新 runtime。新增观测不是零成本；本轮没有完整运行入口的成本计分，没有性能提升结论。
- 后续若继续，先完成新严格凭据工厂与完整源锁的 CPU 准备；新的原生校准和 off→shadow→on 真机资格必须使用新明确授权及原累计预算。继续保持原模型、128 tokens、单文件工作量、原成本估计和等待/新鲜度上限，不提高阈值、不伪造释放信用。

早期本地调试中先修正了测试 fixture 的动态模块注册和跨线程 owner 使用；这些不是候选修复，不计入最终成功运行。第一次最终归档尝试遇到旧准备锁正在迁移至 `LOCAL_FREEZE_V1_PROVENANCE`，在运行测试前因缺锁退出，未生成成功记录。最终归档只绑定新冻结锁。
