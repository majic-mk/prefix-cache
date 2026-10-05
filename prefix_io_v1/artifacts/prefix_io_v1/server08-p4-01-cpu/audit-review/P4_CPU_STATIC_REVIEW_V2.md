# P4 CPU 独立静态复核 V2

结论：冻结的隔离源码通过 CPU 合同与安全边界复核；根代理完整当前 CPU 回归实际为 1588 个唯一通过、16 跳过、0 失败，XML 总数 1604，其中 P4 新测试 179=88 policy/candidate+79 bridge/native/options+12 stage。所有回执 GPU 初始化为 false、GPU 运行 0；独立重证与历史重复测试没有叠加计数。本结论不是完整 P4 生产验收，也不证明方法已经提速。

完整 00–05 和五份模板的逐项审计见 P4_CPU_CONTRACT_AUDIT.md、p4-cpu-contract-audit.json。本次基于实际 p4_types/policy/cost_table/production_table_contract/bridge/options、隔离作者 py-kvcache reactor/vLLM 修改和已完成测试复核。输入路径、字节数、SHA-256、缺口和 AST 修改清单在 p4-cpu-static-review-v2.json 冻结。

- 严格 run/epoch/source/capability/generation 合同、父作业全部保护者 AND、未知保持未知、原 clean reclaim 优先、32 父/64 工作/8 闭包/3 阶段边界已经复核。观察边界没有截断或丢弃原生已接收工作。
- 跨 run 身份、无 joint 标定却相加干扰、无关父作业冒充 restore 完成、直接 issue_preview 绕过前置校验等真实反例已关闭。失败证据保留，独立修复重证为 counterexamples-03/04。D 不使用 interference 信号，单位/误差区间/原生优先序保持分离。
- 生产桥首稿可将 native None 预测伪造为 150，并改动 D 排序；负证据在 p4-review-forecast-counterexample-01.json。最终 bridge e5b26a3c… 对相同 ownership+虚构预测严格拒绝，最终重证 p4-review-forecast-counterexample-03.json 断言退出 0。当前没有合格预测 producer 时，CPU 模拟预测不得进入生产路径。
- 64 个 H2D generation 满窗口时省略额外可选 witness，保留全部 native ready 工作，记录标量 omission。共享/cache 槽证明绑定原 registry 对象，使用 _slots.get 不触碰 LRU。实际 reactor fixture 证明采样后校验时间、原 job.profile.start_ns 年龄逃逸；无到达时刻证据的普通 preload 保持原路径。
- off 在时钟、快照、候选扫描前短路。shadow 与未知 I/J 保持原 U 额度，只有 D 安装原 fixed controller。桥没有新工作队列、缓存/作业所有者、资源释放或额度账本。同 epoch 一次完整 advice，子集只投影；缓存 advice 每次仍复核当前快照新鲜度。
- mandatory、continuation、shutdown/STOP、age 维护原生进展和设备/槽/事件约束。H2D fusion、D2H→SSD write 续接、父完成协议不拆。CPU fixture 验证零额度等待、shared/cache consumer 保护、部分复制失败和 accepted 后 end-event 失败：完成未知时保留所有者，完成证明前不发布父完成或复用复制源。
- production candidate parser 严格绑定模型/GPU/layout/kernel/source/geometry/load/cost basis/uncertainty/measurement roles/独立 verifier 来源。parse 与 SHA 来自相同 bounded raw 字节并结束重验。它不执行引用 verifier；CPU/conditional/P3、自报 PASS、布尔声明或哈希都不能授予 GPU 资格；结果恒为 BlockedProductionCandidate。
- C01 仅 interference 开，C10 仅 dependency 开，标签已恢复原 03 合同；四组固定额度保持匹配，独立 U 参考另列。

实际独立服务器命令均通过 -I -S 与空 CUDA_VISIBLE_DEVICES 执行：

    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S artifacts/prefix_io_v1/server08-p4-01-cpu/audit-review/p4-review-counterexamples-03.py
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S artifacts/prefix_io_v1/server08-p4-01-cpu/audit-review/p4-review-counterexamples-04.py
    CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S artifacts/prefix_io_v1/server08-p4-01-cpu/audit-review/p4-review-forecast-counterexample-03.py

对应 JSON 实际保存。独立重复例子不叠加为更多唯一单元测试。root 当前回归已审读 current-integration-01/result.json：进程退出 0、CPU 子进程清空、70 输入前后 SHA 一致、ledger 未变；12 个冻结历史 fixture 被排除且没有计入当前通过。policy/candidate 回执为 policy-production-contract-cpu-tests-02.json；桥回执为 native-bridge/guard-05/{results.xml,cuda-guard.json,pytest.log}。历史 import/fixture 失败保留，不能计为通过。

patch-roundtrip-01/result.json 实际 PASS：52 原文件→58 最终文件，8 文件变化，observer/native-glue 与 bounded research 两份 patch 独立正/反应用均字节精确；common-only off 没有 strategy/backend 导入。原 P3 bytes 保留；这证明隔离和可退回，不授予生产策略或 GPU 资格。

剩余工作同时包含实现缺口与 GPU 验证，不能说只差 GPU 数据：

1. D 的实际 native 父完成/解除阻塞估时仍为 None，合格 ETA/误差 producer 尚未实现且未获 GPU 资格。CPU 排序测试不证明真实预测排序已经接通。
2. I/J 独立真实 paired 测量的语义 verifier、可授予资格的 CostTable loader 尚未实现。schema/字节绑定成功不证明测量来源或其统计含义。
3. native load_signature 目前为空，真正生产状态匹配的 I/J 降额/额度应用、表接入和原 source/geometry 运行时资格链未完成。safe preview 接点与人工 CPU defer fixture不能替代；当前 I/J 走 U，没有新增性能额度。
4. 真实 GPU allocator generation、active refs、protectors、立即复用容量、CUDA/模型/完整输出、真实重叠干扰、live shadow、真实开销尚未验证，GPU release/reuse credit 保持未知。
5. 对独立 U 的方法提升和 P5 因果验证仍未成立。P3 根结论仍为 limited pilot complete/effect unproven；旧 conditional 表没有 P4 production 资格。

本 reviewer 没有运行源码修改、GPU 操作或状态探测、下载、驱动/系统修改、缓存合并/删除。source-preservation-during-review-02.json 实际复核 31 份原 P3 native/test 与 21 份原 control，共 52 份字节保持，GPU ledger SHA 保持 31199998…efc1；这个选择复核不是全库锁审计。本阶段新增 GPU 运行 0。后续允许的仍是已经授权的 CPU 工作；GPU 阶段按明确授权、设备可用性、存储和预算门禁继续阻塞，本审核不授权 GPU 或 P5–P7。
