本次独立审查截至 patch-roundtrip-05 所记录的 P4-02 源。它是后续 observer/measurement V2 增量之前的可复现基准，不是本轮最终全量冻结或 GPU 资格。

load/options/startup/batch/reactor/author 元数据接线的定向只读审查通过。实际 bounded batch 接口在 p4_bridge.py::batch_prefix（不存在 p4_native_bridge.py）。off 的 options 不构造 P4 bridge，manager/handler/reactor 在时钟、可选 import、evidence root 或请求扫描前返回；author scheduler/worker 增加的是可选值 relay。scheduled-work-v1 记录真实作者 scheduler 当前字段和分布几何，能力仅 scheduled_work_observation，不是 production_gpu_load_state、实际 active GPU decode 或 exact9 表签名。空工作是真实空调度，也不证明 GPU 空闲。

prepared cost loader 在启动校验严格 raw/plan/candidate refs，摘要留作准备状态，table 不安装到 native P4Policy。I/J 无生产资格时保持原 U 路径，声明固定 caps 不代表生效许可；D/shadow 仍使用已声明的原 common fixed budget。batch_prefix 对 continuation/mandatory/shutdown 保守不切割；生产选择还要求政策资格、batch 资格与 GPU-load capability，当前均关闭。ready read 的原 iodepth、slot reserve、原 stage 决策/结算和 H2D 融合继续权威。

scalar-sequence-and-core-01 使用实际源 AST 编译的窄方法，仅 stdlib+control：组合重放 newer unavailable、旧序列不能恢复、同序列冲突保持 unknown 直到更高序列、未来/过期过滤；fresh signature 长度11且 false GPU state。独立 off 子进程禁止 policy/ETA/startup-evidence/backend import，实际通过。8个原 common 方法 AST 与 P4-01 完全一致：_flush_copy_batch/_launch_swap_blocks/_prefix_ready_read_decision/_prefix_stage_decide/_pump_once/_schedule_one/_reserve_foreground_slot/_run；read drain 和两个 terminal 方法只去除声明的 P4 hook 后与01一致。生产权能/效果未取得；此 AST fixture 不冒充完整 GPU/native 执行。

两套 patch 字节正逆 roundtrip 在 patch-roundtrip-05/result.json、commands.json（真实check/apply/check-reverse/apply-reverse命令）：
- P4-01增量基准：control_01/native_01 + 已有 author-build 的 vllm 包1844Python源。共1902文件，目标1908，13变更，3patch：观测/胶水、有限研究值、测量准备。已继承的旧 CUDA/flush/runtime bytes 在基准里保留。
- P3+作者锁定 HEAD 全量包源码基准：control P3根src、native P3-16、author HEAD 817a7e3124f817cd6e549581d3e5483207a753a4 的1752个 tracked vllm 包Python源。共1804→1908，110变更，6patch，依次为旧 common CUDA UUID修复、旧 flush观测、新P4观测/胶水、有限研究值、测量准备、92份已锁定runtime Python补源。
每个正向步骤检查路径/字节精确一致，全部逆向后恢复所有基准路径/字节；03空__init__路径也验证存在。没有将补丁应用到生产源或删除旧证据。

author 未提交 tracked 修改全列（相对817HEAD）：platforms/cuda.py；offloading/common.py/scheduler.py/worker.py。其中旧 author-build 实际只修改 cuda.py 的 NVML UUID映射和 scheduler.py 的 opt-in flush probe。common.py 在旧 author-build 与817HEAD字节一致，其 canonical layout基础已在该HEAD中；fork新增的 common.py变化仅P4 metadata字段，不能伪标为旧未提交共同修复。新author overlay的92纯Python补源与旧build逐bytes/SHA一致；来源清单 author-overlay-inventory.json、复制收据 gpu-next-day/AUTHOR_PYTHON_OVERLAY_COPY_92.json；它们独列runtime supplement，不混作策略贡献。

此 source patch 证明范围为 control/native 全Python与author vllm包Python；未改变的仓库资源、编译二进制和系统依赖不在源码patch中。运行仍需要锁定的既有二进制 ABI fallback，未构建/安装或声称仅这些patch即可独立编译镜像。

补丁失败审查 run-01至04全部保留：空→空 difflib 漏新增空文件；header换行转义；混用unified/Git边界；父repo前缀过滤使git exit0但scratch未变。严格树相等断言拒绝全部假成功，最终05统一Git边界并用新scratch Git仓库实际收敛。没有GPU操作，预算ledger未改变。后续另两名agent新增V2/窄诊断observer后须以本基准重新锁和review，不将本证明冒充未来源的资格。
