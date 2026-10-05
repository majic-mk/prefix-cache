# Server09 迁移与 P4 CPU 续行实际交付

实际服务器 connect.westc.seetacloud.com:29815，项目 /root/autodl-tmp/prefix-io-v1-handoff/project。本轮迁移检查、CPU 回归与受限 G1 启动准备完成；真实 GPU 初始化/运行均为 0，完整 P4、生产成本/ETA资格和方法提升尚未验证。已发出新卡 G1 限定授权请求，答复前不激活新 GPU 权限。

克隆后的四份冻结锁实际 SHA/字节全部一致，2,724 / 70 / 2,198 / 2,048 引用完整。7,813 个受保护文件实际读取 3,283,686,731 字节并校验 SHA；3,048 份注册 KV 的 2,796,552,192 字节和成员集合完整。上轮清理的 72 根目录实际不存在，6 个空夹具/12 个目录仍保留。历史 AUX result 与七个旧 .so 字节一致；二进制没有执行，不构成新 ABI 资格。Qwen config/index 与四分片元数据存在，本轮没有全量哈希模型权重。

当前版本：project cc7898b1ba59d89ce7fdbb186ded21880f1adf08；py-kvcache 3abba7a502d553f6e7e2e58b92086487e3395d7e；作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4。Python 3.12.3；安装 metadata Torch 2.11.0、vLLM 0.1.dev1+g817a7e312、pytest 9.1.1、Transformers 4.57.6。GPU 模块/实际路径和 ABI 的生产资格必须真机验证，不能由 metadata 代替。完整版本、SHA、Git dirty/untracked摘要见 SERVER09_VERSION_LOCK.json 和 G0_RESOURCE_AND_VERSION_AUDIT.json；现有未提交内容保留。

实际测试结果：

| 套件 | 通过 | 跳过 | 失败 | 边界 |
|---|---:|---:|---:|---|
| 新服务器原当前 CPU 集成 | 1956 | 16 | 0 | 29.528 s；P4 的547项包含在1956内；原12项历史P3排除不累加 |
| 可选 GPU 权限路径候选 | 21 | 0 | 0 | Linux实际符号链接拒绝已覆盖；授权文件为临时合成fixture，不能授权GPU |
| 完整 execute→sample 标量帧候选 | 26 | 0 | 0 | 0.012 s；真实worker hook未安装，GPU时间/I/O=None |

16 个跳过：13 个 no-vLLM fallback 专用测试，3 个 io_uring_setup errno1。真实 Linux AIO CPU 路径已测试；没有修改 io_uring 系统设置或替换缓存/模型执行器。CPU CUDA 守卫 cuda_initialized=false、gpu_workloads_run=0，进程 session 完整收敛，原测试输入源锁与 GPU ledger 前后字节相同。

新增内容全部追加于 artifacts/prefix_io_v1/server09-migration-20261001，原生产源码没有替换：

- gpu-scope-candidate/：仅两个脚本的共同部署适配候选，为 qualifier 与现有预算 guard 增加可选 --permissions-path。默认路径/原 guarded argv、原 native qualification、唯一账本与 session cleanup 保持；显式权限的路径/真实字节、source-lock、reservation、child argv绑定一致。新分支仅 PRIMARY，不借旧卡 AUX许可。实际21项服务器CPU测试及2057条真实当前/候选源引用验证通过。后续新锁必须显式包含两个修改脚本和有效权限文件，allow_remote_push=false。
- collector-candidate/：紧凑标量观测生命周期，跨原 execute_model 返回 None 与随后 sample_tokens，验证实际 CPU token IDs、连续ordinal/上下文、128完整输出与真实排空证据形状。无模型执行、GPU计时、I/O提交或资源所有权。off 在读取对象前返回。短读/短写 completed_requested_bytes 不能冒充 transferred_bytes 成功完成。
- 原权限、账本、CPU_ONLY授权、P3/P4历史材料、作者worktrees/LoadPlanner/Prefix identity/shared staging/preload/copy fusion/异步流水线均保持。

查明两个后续采集要求：原 ENGINE 未显式设置 async_scheduling=False，而作者配置可能将 None 自动转 True；新 G2 计划必须冻结并审计该值。真实冷 prefill 的 pre_context=0 必须保留，另记 computed+scheduled 输入序列长度；旧 v2 验证器要求每帧 context>=1，故不能资格验证真实冷首帧。需要独立版本化完整trace契约，不能删首帧、伪填1或改成本上下文维度。本轮未实现该新版 verifier，也未安装真实worker collector；这些明确列在 plan，不能称 G2 完成。

当前新卡只读取 /proc 与 /dev CPU元数据：nvidia6 对应 RTX5090 / GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2，驱动 proc 595.71.05，cgroup内存90GiB。未运行 nvidia-smi/NVML/CUDA，尚未验证空闲显存、占用或GPU ABI。

空间：CPU回归后 PRIMARY free=13,008,269,312 B（约12.115GiB），原3GiB模型reserve＋8GiB floor通过。AUX唯一inode实际占用20,095,844,352 B，20GiB cap/8GiB floor下3GiB仍不通过，不扩大其许可。CPU测试新增scratch保留，未删既有代码、模型、实验来源或结果。

累计GPU账本保持原 SHA 31199998369e35fcd40daddd7af340353b0feaf134bbe37d2dd8f1f35391efc1。已用16,520.562887秒，8h总上限内余12,279.437113秒（约3.411h），active_reservation=null；本轮新增GPU秒/预留均0，模型下载字节新增0。没有新租用、付款、远端推送、驱动/系统修改或既有数据删除/合并。

实际主命令（cwd均为上述服务器项目；完整参数与返回码在 EXECUTED_REMOTE_COMMANDS_FINAL.json）：

    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S -c '<VERIFY_SERVER09_CLONE_READONLY.py 的原始脚本全文>'
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S -c '<AUDIT_SERVER09_G0.py 的原始脚本全文>'
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/run_p4_02_cpu_qualification.py --name server09-cpu-regression-01
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TMPDIR='artifacts/prefix_io_v1/server09-migration-20261001' .venv/bin/python -I artifacts/prefix_io_v1/server09-migration-20261001/gpu-scope-candidate/test_permissions_adapter.py
    CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -S artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/test_p4_full_step_frame_adapter.py

CPU回归原XML/log/result/guard在 artifacts/prefix_io_v1/server08-p4-02-cpu/server09-cpu-regression-01/（沿用原runner输出父目录，name标明新服务器），本轮本地副本在 cpu-regression-01/，6份文件SHA已逐项校验。命令包装说明/第一次缺PYTHONPATH、一次大型stdout截断、一次本地上传size预检失败均保留，不混入测试结果。G1提案的实际 --launch 已在纯CPU门禁拒绝（exit78），未调用guard或GPU，见 G1_UNAUTHORIZED_LAUNCH_REJECTION.json。

下一允许GPU阶段是 G1：新卡off/shadow各一次、每次180s运行＋20s收敛，共计划400s，16MiB staging/iodepth1、每次PRIMARY128MiB reserve。授权依据缺口是旧 permissions.yaml 只批准 GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8，最新CPU_ONLY_AUTHORIZATION又禁止GPU初始化；新服务器登录指令没有明确解除该限制。具体可审查提案 G1_GPU_SCOPE_PROPOSAL.json；尚为 PROPOSED_NOT_AUTHORIZED。部署须保留两个旧脚本历史副本、冻结新源/权限/context绑定，只走外层 --launch，原总账本不重置。详细步骤见 NEXT_ALLOWED_STEPS.md。

G1通过也只证明原生store/restore/shared/parent/drain和新源码/ABI的有限资格。随后才进入G2完整正常模型采集与成本/ETA/load/时钟资格，仍不能伪造自然耦合D2H→SSD单阶段cell；生产D/I/J不具备资格时回退原U。P3有界pilot已闭合、方法收益未证明；P4完整未完成，P5–P7未启动，正式SLO仍null。

