# C5 实际入口 CPU 准备：独立审查合同

本目录仅新增 CPU 反例与审查证据。不修改冻结 C5、runtime_v4、native_cost_v6、历史 GPU 结果或 score，不连接服务器、不调用 GPU。

已确认的实际断点：旧 runtime_v4 在 run_p4_single_file_experiment.py:471 以 LABEL 构造 bridge，:566/:578 以 LABEL-p0-B 构造 capture；C5 reactor.py:2329 要求相同 owner/capture run_id。直接追加 attach 不能工作。

审查目标：

1. 新实际入口显式使用 owner LABEL 作为 capture.run_id，原请求 rid 及原 128 帧中的 native request_id 保留；独立 validator 必须区分三种身份而不删除任何校验。
2. on 路径实际调用原 C5 attach，off/shadow 不注册等待；仅声明 strategy='on' 或 defer>0 不能证明通知运行。
3. 用同一个原 Queue 的真实 get/返回、原截止时间、真实通知/入队事实证明停车和唤醒；timeout、旧消息、defer 和调用次数不能伪装为 end wake。
4. 所有原方法只调用一次；返回值及原异常保持。观测自身失败不得吞掉或替换原异常。原队列、资源 owner、pump 顺序不改变。
5. 观测只保留有界标量、弱引用与原 unbound function；detach 只恢复自己安装的 wrapper。原 C5 不允许同一 reactor 再次 attach，入口必须限定 fresh reactor 的单次 capture。
6. 合成事件/模型 fixture 不生成 GPU/native execution 资格。冻结 C5 receipt 与 C4 同 SHA，仍硬绑定 C4/native-v6/job06；新入口必须明确缺少 C5 原生校准。
7. 未来真实 GPU 接线需要另建冻结 overlay 中的严格 receipt 工厂与新共同源锁，使所有校准/运行臂使用同最终 reactor、collector；不能改常量、私构 receipt 或移植旧 v6 数值绕过来源约束。

所有通过结论限定为源码与 CPU 调用链。服务器及 GPU 状态必须单独说明，不复用上一轮计分或历史成功计数。
