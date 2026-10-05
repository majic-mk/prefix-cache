`close_formal_peer_cpu.py` 补齐正式流的实际 child → 原 guard 闭合 parent 生产接口。建议部署到 preparation/formal_runtime_bridge/close_formal_peer_cpu.py，单独作为 CPU 命令执行；不覆盖旧 runner、runtime、作者代码或预算。

CLI 接受实际 project root、当前完整 source-lock 文件、实际 child/guard/config 的 project-relative 文件路径和一次性 output-relative。路径会转成实际文件 `{path, bytes, sha256}`，所有已有输入必须在当前 source closure 内且全字节核验。producer 自身、原 runner 和 namespace pure verifier 也必须冻结。缺真实输入即拒绝，不写 draft parent。

执行时只读持有原 guard 已存在的 `gpu-budget-ledger.lock` 共享、非阻塞锁。账本须 idle，实际 guard reservation 事件须唯一且完整 JSON 与账本一致。原 guard 必须自然结束、无 timeout/error/signal，实际 `/proc` session 在纯校验前后均为空；账本原 bytes 前后相同，不增加预算事件，不改变 GPU 预算。

parent 仅深拷贝实际 child，把 `os_session_drained=False` 改为 True，新增 `completed_guard_ref`、`child_result_ref`、`formal_parent_closure_schema` 三项。构成后先在内存调用原 namespace `verify_closed_peer_document`，完整重验原 child/config/guard/source、全部 token-ID outputs/CUDA alignment、native tail、保留的真实命名空间。effect I 的真实私有表与 development reserve 仍由原 helper 纯读重验。pure checker 无权重写 parent；通过后仅一次 `open('x')` 写入，不覆盖或创建任何缓存目录。

输出 parent 不在自己的输入 source lock 中；成功后必须由根再冻结其实际 ref，未来另一臂才可使用。producer 不授予新的 GPU、策略效果或 goodput 资格。正式 development 只允许 U/off parent；effect 支持 U/off 和 I/on，以保留预先固定的 AB/BA 顺序。

16 项 unit tests 使用明确 `CPU_TEST_MOCK_ONLY`、`NOT_ACTUAL_GPU_EVIDENCE` 标记的内存 checker、guard 和 source-call fixtures，测试单次 append、child 字段未变、缺 raw、guard 未结束、OS session 不空、账本变化、输出重写/循环引用拒绝等边界。mock checker 不验证真实 GPU，不产生 CostTable、CUDA frames 或真实 receipt。CLI `--help` 与缺真实输入的退出码 78 也单独记录；这不是实际 formal parent 成功。

使用顺序：真实 formal run 经原 guard 结束 → 把实际 child/guard/config/raw outputs/capture 全部加入新 source lock 并 CPU 全校验 → 调 producer（全部真实证据满足才成功）→ 把实际输出 parent ref 加入后续 source lock → 执行下一臂原入口。当前没有真实 formal child/guard/development reserve，因此本轮实际 formal parent 数为 0、GPU 操作数为 0。
