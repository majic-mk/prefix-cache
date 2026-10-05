# Server09 生命周期与修改位置

本轮保留原 reactor、原 LoadPlanner、共享 staging/preload、复制融合、异步流水线及作者模型执行器。

原 store：scheduler _build_store_jobs → worker prepare_store_kv 延后原提交 → handler.transfer_async → 原 compute_event 保护 → D2H → _on_store_copy_done 立即续接 SSD write → _on_write_complete 检查完整写入/发布/slot 结算 → 全部 child 终结后父任务成功 → worker completed_jobs → scheduler complete_store/fence 解除。不能用 D2H 结束或 Future.done 换取提前回收。

原 load：全部被选文件和 H2D 完成后父任务才能 ready。共享 slot 的 retained cache 与 copies_inflight 由原 owner 管理。GPU 活跃引用、全部 parent fences、allocation generation、原生 reusable 确认必须同时满足；free queue 或消失于有界观察窗口不是释放证明。

已接受任务的 mandatory wait、原下游续接、completion、fusion 与错误后排空保持原语义。失败关闭不能冒充实际完成；短读/短写的 completed_requested_bytes 也不能证明 transferred_bytes 成功完整。

| 本轮位置 | 改动类别 | 激活状态 |
|---|---|---|
| artifacts/prefix_io_v1/server09-migration-20261001/ | 迁移审计、实际命令/CPU证据、源锁与受限 GPU 提案 | 追加新证据，旧结果保留 |
| gpu-scope-candidate/candidate/qualify_p4_native_gpu.py | 共同部署适配：可选 --permissions-path，scope/源锁/guard/child 参数一致，PRIMARY-only | 候选已 CPU 验证，尚未替换当前脚本 |
| gpu-scope-candidate/candidate/run_gpu_stage.py | 共同部署适配：可选权限文件、单账本、同一 reservation 的真实权限字节绑定 | 候选已 CPU 验证，尚未替换当前脚本 |
| collector-candidate/ | 紧凑正常 execute→sample 标量观测生命周期准备 | 不安装 worker hook、不执行模型、不计 GPU 时间、不提交 I/O |

G1 候选 native qualification、source loader、原记账与 session cleanup 函数 AST 保持不变。默认权限路径和原 guarded argv 保持。候选不借用旧 GPU 的 AUX 许可。

新 GPU 授权后部署前，保存两份原脚本字节/原路径→历史副本映射；生成真实 server09 当前新锁（显式包含两个脚本和有效权限文件）。历史锁证明历史版本，不能在版本演进后声称同一旧路径字节仍匹配。当前本轮尚未改变这两个脚本，旧锁实际仍一致。

原 permissions.yaml、CPU_ONLY_AUTHORIZATION.json、累计 ledger、作者 worktree、Prefix identity、准入、attention/sampling/executor 与生产 I/D/J 没有改动。新授权文件必须 allow_remote_push=false，且不改变原总预算。

当前完整 P4 与方法效果保持未验证；G1 off/shadow 只资格验证原生 store/restore/shared/drain，不覆盖零普通额度研究策略或生产模型完整成本。
