# 最终协议的独立科学边界复核

已只读核验最终 I_PILOT_PROTOCOL.json 和 i_pilot_protocol.py 字节，SHA 分别为 f4ef557d8612af959ed89496c7a97eeb1b8647e15c6500657b81dfbe8172ea6e 与 aa4c3041a8c2b7c50cf098468ae4d98beff3d4256dc4be3ee5ca7ec10a2caadc。没有追加测试循环、改源码或启动 GPU。

当前可以交付“CPU 准备已具备有界机制校准入口”。不能交付“正式 U/I 效果入口已全部完成”。协议的 readiness join 只准入 cal01，明确 paired_effect_runner_bound=false、GPU_launch_allowed_by_CPU_join=false 和 effect_status=BLOCKED_FOR_EFFECT；没有伪造新 GPU 授权或效果。

calibration_v2 保留原 native runner 的全部函数 AST，同时保留 LoadPlanner=off；serializer 也要求真实子报告 load_planner=off。这只能称为固定原预加载单文件机制/成本校准。正式强 U/I 对照必须在同一作者 LoadPlanner=on、精确 prefix、成本准入、共享 staging、预加载、复制合并及异步执行器上，仅切换新干扰策略。旧 planner-off 窄 receipt 不自动覆盖 planner-on；目前这种正式入口及其配置对应的真实成本/source 验证仍未绑定。

协议保留旧 U=16,238,752ns、原 A-only=13,171,328ns、全部覆盖失败、no-refit/no-widening；新 families 不被用于替换旧反例。服务 SLO、独立开发 deadline、reserve 和内部 budget 仍 null，缺少前瞻独立 deadline 时必须停止普通 on。不能用任意 20ms 或重新构造开发 A-max 来通过门槛，也不能把 progress fallback 当成策略收益。

九个槽位的最多 3,780 秒是条件预算规划，不是执行 authority，更不是完成系统验证的保证。它们仍需原 guard、实时设备/预算/源校验和阶段门。未来 planner-on 配置的额外资格证据必须重新核算预算，不能假定本次 planner-off 机制校准已经承担这部分工作。两对诊断评估也不支持自然请求流、总体或论文级收益声明。

机器拒绝清单见 SCIENTIFIC_BOUNDARY_REJECTIONS.json。以上限制阻止效果推进或错误结论，不阻止本次 CPU 交付或合法的 cal01 机制校准；它们也不证明方法永久不可行。
