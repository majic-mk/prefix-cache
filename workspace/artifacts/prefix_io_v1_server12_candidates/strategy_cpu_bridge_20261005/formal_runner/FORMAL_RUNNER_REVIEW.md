正式输入候选已接入原 token-ID drive，并完成 32 项本地 CPU 分派/拒绝测试。原资格校验、原 drive 函数源码、guard、engine、采样/模型执行/缓存生命周期不改。正式开发 U/off 是同真实输入的最小先决基线；I/shadow 使用真正完成并关闭的正式开发 U。效果 U/off 和 I/on 共同受真实 private issuer、独立 SLO 与原 native reserve 只读重验约束。

namespace 的新薄 adapter 只改变新独立 F 模块实例的 metadata 回调：当前 arm 永远新，首 arm 的两臂都新，已完成 peer 必须具备真实 source/config/child/guard/output/CUDA/native-tail 闭合后才可保留，支持 AB/BA，不删路径、不重置。开发诊断可采用原协议允许的 SLO=None；所有独立 deadline/authority/host clock 条件仍保留，evaluation 的 SLO=None 仍拒绝。

测试包括实际 native_runtime_v3 源码 header/status 表达式、peer消费者实际 runtime export、token-ID及全局到达计划传递、原 drive源码一致、外域/错误deadline/缺真实reserve/字段篡改/部分输出/顺序改变/当前arm旧namespace/无闭合peer拒绝。测试使用标识明确的 CPU spy，仅证明分派合同，不能当作 GPU /正式输入/提升证据。

未执行任何 GPU/RPC 或模型/backend导入，没有生成 dataset、deadline、SLO、reserve、可执行正式GPU配置。真实输入仍 UNBOUND。已暴露 verify_closed_peer_document 纯校验 API，为根代理的纯CPU单次append闭合薄器服务：parent本身不需预写/先引用，所有真实child+guard+config+原始观测字节仍必须实际存在且闭合。消费者从真实parent引用读取后委托同一纯API。producer CLI由根代理其它候选补充；此目录不声称该组合已在服务器通过，缺真实数据不能以标签补齐。

推荐runner部署到 preparation/runner/strong_trace_runner_v3.py，namespace_bridge与test部署 preparation/formal_runtime_bridge/。根代理负责服务器CPU测试、完整继承V10的追加源锁与交付核验。
