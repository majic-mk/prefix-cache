# 真实 GPU 功能与精确成本验证交付

按用户最新要求“先完成功能和成本验证”，本轮已完成原U路径的模型输出、原生生命周期验证，并通过一个真实GPU精确SSD read成本单元的公开发行与独立检验。**有限条件下系统能够运行、测量与校验；I策略是否提速、是否达到论文要求仍未验证。**

实际操作在connect.westd.seetacloud.com:24828服务器的 /root/autodl-tmp/prefix-io-v1-handoff/project 执行。使用现有Qwen2.5-7B和RTX 5090，UUID为GPU-b2de2c25-cdc7-a350-267f-56e7763a287f。沿用用户持久GPU授权与原8小时累计预算。没有下载模型、修改系统/驱动/已安装包/旧冻结作者文件，也没有删除已有代码、实验数据或私有缓存。

## 真实结果与范围

| 项目 | 执行结果 | 可以证明什么 |
| --- | --- | --- |
| 原U baseline/off01 | 12/12受控请求，每个完整生成128 tokens，共1536；原guard exit0、原shutdown返回、OS会话排空 | 原模型、缓存及写入路径可以运行；不代表自然工作负载质量或策略收益 |
| 精确缓存与写入 | 9个请求报告448缓存tokens、3个为0；D2H与SSDwrite各216次、各198,180,864 bytes；216个accepted/completed/reaped | 原生写入和生命周期证据有效；off journal没有SSDread/H2D事件，不能算读取收益实验 |
| CAL05六窗 | AB/BA/AB顺序，6个fresh模型进程全部exit0；每窗128输出、128模型帧、128 CUDAEvent见证，共768帧/见证/输出；三对完整输出一致 | 同一冻结源码的受控完整配对成本测量 |
| 原生关闭 | 每窗原shutdown返回、handler关闭、worker/AIO停止、reactor关闭；6次完整窗口早验通过；原guard exit0、SID7931排空 | 自然关闭与随后OS排空分别有证据 |
| 原公开发行器 | PASS_ACTUAL_EXACT_GPU_COST_CELL，实际发行1个private CostTable；原估计器与独立heldout覆盖检查通过 | 当前GPU/source/model/layout/common-domain下的一个有限成本单元 |
| I策略与论文效果 | 新策略GPU运行0，formal SLO=false、strategy effect=false | 尚未证明提速、满足服务目标或论文收益 |

原U的整流观测仍为UNKNOWN/invalid（52frames、0CUDAEvent witnesses）；未补造缺失帧。CAL05受控六窗通过，不代替自然混合流量的观测资格。

CPU直接读取约21.4MB原始闭合记录，逐窗检查真实128帧/见证/输出、连续原生ordinal、六个独立PID、原shutdown、真实child退出及三对输出一致。摘要v1误把open_event_pair=false当作None，CPU摘要失败；v2按记录布尔类型校验通过。未重跑模型或再次发行成本。[六窗直接读回](CAL05_ACTUAL_SIX_WINDOW_RAW_SUMMARY.json)、[公开成本发行](CAL05_ACTUAL_COST_ISSUER_REPORT.json)、[独立只读复核](FINAL_EXACT_COST_RESULT_REVIEW.md)。

## 精确成本单元

条件固定为batch=1、active_decode=1、prefill=0、context=527，单次SSDread 917,504 bytes，四类existing I/O均为0，以及记录中的GPU、源码、模型、KV布局和eager运行域。模型为BF16/TP1、block16、28层、4个KV heads、head128。

| 分量 | 值 |
| --- | --- |
| baseline | 13.085264 ms |
| incremental_or_joint | 3.541552 ms |
| uncertainty | 0.006304 ms |
| total | **16.633120 ms** |

这是该成本单元定义下的总成本界，不能当成SSD单独延迟或方法加速比。前两对用于拟合，最后一对用于独立检验；heldout没有参与拟合，没有用于改阈值、预算或上界。一个独立检验配对不足以声明普遍概率保证或覆盖所有context和并发。

JSON仅保存证据，不能复用为进程内私有GPU能力。后续须以真实plan、闭合raw、guard、intent和源记录由原发行器重新发行，并严格继承校准叶子与运行域。V10完整继承V9已核验叶子；仅锁名变化不强制重新测量，实际校准来源/运行域改变须重新判断资格。resource_release_credit仍为false。

## 实际修改与原路径

共同兼容修复与I研究策略分开，新增候选文件、保留旧冻结版本。原py-kvcache、作者vLLM、精确Prefix Cache、成本准入、load planner、共享staging、preload、复制合并、异步流水线与原模型执行继续复用，没有重写引擎或执行器。原U off路径真实运行；本次未进行新I策略效果对照。

| 修改位置（相对于server12-gpu-prerental-preparation-20261004） | 改动 |
| --- | --- |
| runner/strong_native_cost_runner_v2..v7.py | 实际UUID/设备节点绑定、相对父子config、frontend cached512与真实initial execution496/prefill16分开、失败capture保存、逐child关闭后调用原严格早验。最终GPU运行用V7 |
| runner/bounded_native_full_step_collector_v2.py | 作者源码认证的初始零scheduled-work保存NO_FORWARD诊断；继续委托原collector，不伪造模型帧/Event、不改ordinal或原KV控制通知；未支持情况保持UNKNOWN |
| raw_*检查目录 | 设备、config、缓存语义、失败诊断、作者AST、两个collector叶子、实际初始执行契约的CPU针对检查 |
| runner/native_runtime_v2.py、runner/strong_trace_runner_v2.py、raw_runtime_collector_binding/ | 未来成本消费路径核验新collector与原delegate、校准来源继承；原资格/无成本shadow路径保留；仅CPU准备 |
| formal_trace_binding/ | 正式trace/tokenizer/namespace、family disjoint、独立SLO和真实control reserve的CPU来源契约；实际输入仍UNBOUND，尚未接入正式effect GPU入口 |
| 本交付目录prepare/supervise/close/summary/receipt/pack脚本 | 原guard管理与闭合证据、CPU归档；不能替代模型或私有资格 |

关键冻结源SHA-256：

- V7 wrapper：d07c54e330b0004a20fc380ab47a7e742f05736ad61180f0a42b38b9c635de55。
- 新collector：a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0；原delegate：9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d。
- 原native reactor：a2390db63df27366f60dce6b81e0affa0c1272727a3b6b60e86bff367e23db47。
- 原公开发行器：455abbd6391616fa49be323ba13daa4cdc3eda1b3642152b6f597c6ad681dcfe。
- 原配对估计器：3cd840c6dd388e3dcf9023e54dd5eb172ab59777c66f731a5d1e3f9afecd99ac。

原严格成本验证器、CostTable、配对数学与guard保留，全部实际路径/字节/哈希见plan、源锁和归档清单。

## 命令与验证

CAL05实际命令（原guard，900秒执行上限加20秒收尾）：

```text
.venv/bin/python -B experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/live-off01/EFFECTIVE_GPU_PERMISSION.json --label server12-strong-exact-cal05 --seconds 900 -- .venv/bin/python -B artifacts/prefix_io_v1/server12-gpu-prerental-preparation-20261004/runner/strong_native_cost_runner_v7.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --config artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/raw05/CONFIG.json --execute
```

模型与guard结束后，CPU隐藏GPU执行原公开发行闭合：

```text
CUDA_VISIBLE_DEVICES='' /root/autodl-tmp/prefix-io-v1-handoff/project/.venv/bin/python -B -I -S /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server12-strong-gpu-system-verification-20261005/close_actual_exact_cost_cpu_v1.py --project-root /root/autodl-tmp/prefix-io-v1-handoff/project --attempt 5 --wrapper strong_native_cost_runner_v7.py --expected-plan-sha256 1538d6529653a17bd1b8322b40b6b21589b5db79837544dd50f3d8981211776c --expected-guard-sha256 50505f549edece8b0972344e5a18ed104845083d46d0067699926311a1783e2a
```

实际argv、cwd、exit、STDOUT/STDERR均在归档COMMAND/RESULT保存。verify_actual_raw_source_bytes_cpu.py在GPU作业前后各核验4,876个冻结引用，两侧failure=[]，V9锁SHA为c70041b407438fa9c16cf151d59049ae12774309319ed52ede5527368de2a0f3。

最终freeze_prerental_sources_v10.py实际核验4,893个引用、16,072,097,972 bytes，PASS_FULL_CPU_SOURCE_BYTES；V10锁SHA为069f96134f950e2d1100fdc751a2449145f7d534a9e625030503462dfdd1273a。[真实CPU及命令收据](FINAL_ACTUAL_CPU_DELIVERY_RECEIPT.json)、[V10证明](PRERENT_SOURCE_PROOF_V10.json)。

| 针对CPU套件 | 实际服务器结果 |
| --- | --- |
| UUID/设备节点 | 8/8 |
| 相对父子config | 3/3 |
| frontend缓存语义 | 5/5 |
| 失败诊断保存 | 5/5 |
| 初始NO_FORWARD作者AST和拒绝路径 | 16/16 |
| 两个collector源叶子绑定 | 5/5 |
| 实际initial execution契约/早停 | 8/8 |
| 未来runtime collector绑定 | 6/6 |
| 正式trace输入契约 | 21/21 |

九个套件均exit0。按实际套件报告，没有与旧171项重复累加。模型输出/成本资格来自真实GPU记录，不由CPU测试替代。

## 失败、资源与预算

CAL01相对config、CAL02 frontend cached语义、CAL03完整collector资格真实失败，未发行成本。CAL03没有保存capture，无法倒推坏帧。CAL04前两个完整窗口与原I/O通过，但initial contract登记511、真实496/prefill16，原validator拒绝；第三进程启动后由原guard SIGTERM停止，不能称其自然shutdown或六窗成功。CAL04不纳入CAL05拟合。日志“Recovered from KV load failure16tokens”本身不能认定缓存代码故障，原break-even拒绝亦可能走该路径。全部旧记录保留。

| guard作业 | exit | 实际账本墙钟秒 |
| --- | --- | --- |
| server12-strong-u-qual-off01 | 0 | 143.513610497 |
| server12-strong-exact-cal01 | 1 | 15.607374024 |
| server12-strong-exact-cal02 | 1 | 144.246099111 |
| server12-strong-exact-cal03 | 1 | 146.638762113 |
| server12-strong-exact-cal04 | 143 | 286.433405299 |
| server12-strong-exact-cal05 | 0 | 807.293711927 |


上述6个guard实际新增账本墙钟合计1543.732962970秒，约25.73分钟，包含失败和停止。原8小时累计已用26,222.62820187537秒，剩余2,577.37179812463秒（约42.96分钟），active=None；账本SHA为774847ccdb7ec4c1785c19a87e718b2a91e15fba44248eb949dab559e08fdd6b。这不是AutoDL租赁费用，未估算人民币账单。

真实CAL05结束排空记录SID7931和GPU compute PIDs均为空；最终CPU收据再验SID为空、账本未变。最终CPU环境nvidia-smi不可执行，未修改权限，明确引用已保存的真实排空证据，没有冒充新查询。最新CPU收据PRIMARY可用約48.56GiB，本轮无需扩盘。此后只有CPU交付操作。

## 证据交付与下一允许阶段

GPU_DELIVERY_SOURCE_EVIDENCE.zip收录本轮修改源码及引用的小型验证/作者叶子、源锁和证明、真实成功/失败原始记录、plan/intent/guard与账本。逐文件SHA清单为GPU_DELIVERY_SOURCE_EVIDENCE_MANIFEST.json；服务器CRC/成员SHA收据为GPU_DELIVERY_ARCHIVE_VERIFICATION.json；下载后本地实际复核为LOCAL_GPU_DELIVERY_ARCHIVE_VERIFICATION.json，最终状态以这些独立收据为准。

这是源码与实验依据包，**不是完整模型/数据盘备份**。权重、SDK、编译缓存和SSD私有payload未复制进zip，仍保留服务器原路径。不要据此释放服务器而遗漏资产。旧交付包及既有数据未删。

当前指定的功能和成本阶段完成，不再需要GPU为本轮交付工作运行。下一步先在CPU准备真实自然trace/tokenizer收据、预先独立固定逐token服务目标及development control reserve协议，接入并检查正式U/I入口，再判断受校准覆盖的候选。超出覆盖或完整观测失败继续拒绝成本消费。当前缺少这些材料，因此没有执行策略GPU对照；未来按冻结协议先development shadow、再U/I比较，包含全部控制成本，以真实输出一致性、TTFT、逐token间隔和吞吐判断收益，不保证提速。
