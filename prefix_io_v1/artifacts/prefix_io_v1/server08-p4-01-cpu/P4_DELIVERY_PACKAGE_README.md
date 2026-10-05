# P4 CPU 交付包

这是当前 server08 的增量 CPU 合同、fake/native 事件与证据交付，不是完整 P4/GPU/提速验收。

优先读取 `artifacts/prefix_io_v1/server08-p4-01-cpu/P4_CPU_DELIVERY_REPORT.md`、`P4_CPU_ROOT_CLOSEOUT.json` 和 `audit-review/P4_CPU_STATIC_REVIEW_V2.md`。实际统一结果1588通过、16跳过、0失败；新P4测试179项。GPU新增0，累计账本未变。

包内保留70份已测试源码、当前源码锁、CPU XML/guard/实际命令、独立审查及历史失败、两份分离patch与byte roundtrip、版本/生命周期/能力矩阵、P3保全和GPU阻塞计划。前述交接文档与模板也附入。临时测试数据、模型权重、私有KV缓存与编译ELF未打包。

统一回归要求现有已克隆项目的冻结P3测试/依赖和当前.venv；这是增量包，不是整个运行环境镜像。旧P3完整包及其SHA已在P3报告记录。生产ETA producer、真实paired语义verifier、qualified loader与I/J生产状态应用尚未完成；GPU资格和效果仍BLOCKED。

`server08-p4-cpu-evidence-v1-manifest.json` 列出每份打包文件大小/SHA；服务器还实际重新打开zip验证每份内容。新策略关闭可回原路径，P3原始代码/数据保持。没有扩大权限、GPU预算或研究范围。
