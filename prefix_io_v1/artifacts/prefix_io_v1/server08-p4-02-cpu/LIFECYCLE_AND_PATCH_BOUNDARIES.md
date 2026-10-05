# P4-02 生命周期与修改边界

继续采用原 py-kvcache 引擎和作者 vLLM 执行器。真实工作仍在原 accepted-parent、stage accounting、reactor ready/IO/copy 队列内；本轮没有第二个工作队列、资源 owner 或执行器。

1. 原接纳、FD/slot/iodepth 和四阶段 accepted accounting 决定工作是否能开始。预算咨询不能提前消耗、退款或替代原提交语义。
2. scheduler 仅收集当前 scheduled 工作的标量；通过已有 connector metadata/worker message 发布。reactor 在原 submit lock 下替换一个不可变值，过期/冲突/不完整值退回 unknown；不把 unknown 当零负载。
3. epoch 绑定原父任务/物理操作几何、四阶段真实 inflight Amount.nbytes 及 scheduler sequence。窗口变化使旧建议失效。源 SHA、几何和精确成本表九维 cell 不被 scheduler 十一项 signature 偷换。
4. 可选 batch 咨询只作用于已有 SSD read 原排列的有限前缀。原 progress/已接纳 continuation、共享 staging 和 H2D 融合不被拆分；原预算/slot/iodepth 仍在后面独立执行。当前没有生产资格，咨询返回 None。
5. ETA 历史从父闭合工作全部 ready 的首次观测到成功 whole-parent drain；失败、短读、drain unknown 不作为成功样本。parent/Future 未完成不产生资源释放信用；历史仅存标量，窗口省略不丢真实工作。
6. 真实 Linux AIO + CPU fake CUDA 的 hybrid 测试验证复制事件完成前不 Future/不 staging slot release，短读失败在原父 drain 后收敛。它不能证明真实 CUDA DMA 或 GPU KV block 已释放。
7. 额外观测接口只绑定原 owner 标量，边界同一笔 I/O 不双计，原生 invalidate 保持 sticky unknown 且不释放未完成 counter；host 时钟倒退不能归属窗口。forward event seam 保留原返回值，不视为完整 decode step，也未安装真实 vLLM runtime。
8. shutdown(wait=True) 仍由原 owner 完整收敛并关闭真实 ring。新诊断错误和缺资格不能取代原错误/取消/停止语义。

共同基础、观测胶水和研究策略分开保存。新工作区继承原 P3/P4-01 共同修复；作者 HEAD 内已有 canonical layout 不重复算新增。作者 CUDA UUID 映射与原 flush probe 基础有独立补丁；新增三处 metadata 接线与研究控制分开。正逆字节补丁证据位于 patch-roundtrip 目录。

控制/负载的 off 在时钟、导入、候选扫描和元数据采集之前返回。注入式 forward seam 的关闭状态直接调用原函数，不采事件。shadow 可以声明与 D 一致的原固定预算；I/J 未资格保持原 U。关掉新策略后仍保留已完成的共同修复和原缓存/模型执行路径。

原 P3 和 P4-01 源、权限、预算及注册缓存字节在本轮最终 preservation audit 逐项核对。没有缓存合并/删除/移动、安装、构建、下载或系统/驱动修改。
