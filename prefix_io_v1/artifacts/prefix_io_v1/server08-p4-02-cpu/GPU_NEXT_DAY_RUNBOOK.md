# 2026-10-02 有卡阶段处理顺序

当前服务器 connect.westd.seetacloud.com:20739，项目 /root/autodl-tmp/prefix-io-v1-handoff/project。本轮真实 GPU 运行为 0；“明天有卡”没有触发定时任务或今天的 GPU 操作。

## G0：先用纯 CPU 检查门禁

读取本轮最终源锁、permissions.yaml、GPU budget ledger 及存储收据。CPU_ONLY 收据保持现状；实际 GPU 阶段须有与用户授权、源锁 SHA、现有批准 UUID 和精确原生 primitive context 绑定的新 scope 收据。设备变化不能自动替换批准 UUID。

外层 --launch 在任何 GPU probe、旧 guard 或预算 reservation 之前验证 scope/source/storage/budget；子 --execute 在旧 guard 的真实 session/reservation 内再验证。禁止跳过外层复制 internal guard_argv。preview 和旧 P3 CLI 只是参考，不能授予新 P4 资格。

ledger 原累计 16520.562886646483 秒（4.589045 小时），8 小时上限下余 12279.437113353517 秒（3.410955 小时）。本轮新增/预留均为 0。明天使用 ledger 的实时值，不把此快照当永久预算。

## G1：新源码最小原生 off / shadow 资格

新原生 runner 复用原 FileMapper、TransferCoordinator、handler 及作者 copy kernels。预留 128 MiB、每项 180 秒 + 20 秒收敛，off/shadow 两项共计划 400 秒。当前磁盘可满足这个小资格测试；GPU 身份、binary ABI 和真实 CUDA 尚未验证。

冻结确认并具备真实 scope 后的入口为：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py --mode off --name server08-p4-02-native-off-01 --launch --scope-record artifacts/prefix_io_v1/server08-p4-02-cpu/GPU_STAGE_AUTHORIZATION.json --source-lock artifacts/prefix_io_v1/server08-p4-02-cpu/gpu-source-lock.json
```

shadow 使用另一个未用过的名字和 --mode shadow。当前没有该 live GPU scope，实际执行必须拒绝，不能现在启动。失败时保留原错误、预算与 drain 证据；不得直接改驱动/系统或扩大阶段。

验收：新 Python 源路径和每项 SHA、锁定旧 binary fallback、真实 CUDA UUID、66-file store/full parent、age-store、exact SSD restore、两 shared-preload consumers、accepted/completed 与 shutdown 收敛一致。此 runner 的 off/shadow **不覆盖零普通额度研究模式**，也不授予模型成本或方法效果资格。

## G2：限定观测与真实单阶段测量

先在原作者执行器上验证已准备的 forward / owner 诊断接口。forward elapsed 只代表 model_forward，不能当完整 decode step；CUDA reference/fallback、host/GPU 时钟关联、实际 prepared batch、上下文增长、完整输出及全 accepted IO drain 均须实证。

V2 测量按预注册 ordinal 和 exact cell 从完整原始步轨迹选择窗口，完整输出数量与 selected 数量分开，warmup 排除统计。成本仅用明确有效的 GPU event duration；原 P3 host-only/12-pulse aggregate 和 scheduled-work-v1 不能转换填充新 cell。

SSD-only preload 和 retained H2D-only 有原接口可供限定诊断。原 store 自然耦合 D2H→SSD write；在合法独立窗口未证明前，这两个单阶段 cell 保持缺失，不能从耦合轨迹切片伪造。

正常持续 decode 的可信完整 step collector、GPU/owner 窗口关系以及表/ETA资格仍需要真实接口核验；当前工具不宣称生产闭环已完成。

## G3：小 P4 pilot 的前置条件

必须先完成真实来源/模型/布局/kernel/量子/load/误差资格和原生命周期安全。D/I/J 的未资格状态保持原路径；共同固定额度独立冻结，四臂 C00/C01/C10/C11 只改变依赖/干扰两因素，并保留独立 U 参照。影子、mock、forward-only 或 PreparedCostTable 不授予生产资格。

原模型轮次仍按 3 GiB 新存储预留和 8 GiB floor 执行。只读审查仅找到 AUX 7 MiB 潜在重复量，PRIMARY 无收益，不能解除阻塞；没有申请或执行合并。最终实际缺口以本轮最终 next-day plan 的 statvfs/primary-free 数为准。不能任意下调旧模型轮次预留；新限定无持久私有KV测量若使用更小预留，须先由具体 runner 证明真实写入上界并走同一存储门禁。

P4 完整实现/资格/效果未闭合；P5–P7 未启动。本轮不冻结 SLO、不声称吞吐、延迟或方法优势。
