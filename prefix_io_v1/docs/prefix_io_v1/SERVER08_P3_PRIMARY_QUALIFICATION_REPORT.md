# Server08 P314：主盘来源注册与新实例资格验收

本轮在 AutoDL 服务器 `connect.westd.seetacloud.com:20739` 上完成。现有作者 py-kvcache + 配套 vLLM 链路在这台实例上可以继续实验，无需为本轮验收更换平台、驱动或系统。CPU 注册设施、新硬件成本资格与关闭新增策略的原生模型回放均通过。本轮没有验证新增研究策略的收益，P3 尚未完整完成，P4–P7 仍关闭。

## 实际改动与接口边界

P313 后验发现原实验驱动将来源锁定在 AUX 系统盘，并用 `os.link` 创建私有副本。AUX 来源设备号 76，当前主盘设备号 2304，直接只改输出目录会产生 EXDEV。P314 保留旧默认来源和原生 hardlink 克隆函数，新增显式 `--storage-registration`，没有通用跨盘 fallback。

- `experiments/prefix_io_v1/scripts/storage_source_registration.py`：仅登记固定 P312 来源的逐字节相同 PRIMARY 副本。严格 schema、固定 origin manifest/SHA、批准 UUID/PRIMARY 路径、无 symlink/重叠、完整文件集合/大小/SHA、source/output 同设备；校验失败在 clone、runtime/model 导入前终止。
- `run_concurrent_pilot.py`：仅 opt-in 调用登记校验，并把来源绑定本轮成本资格 reference plan 的唯一 external group 与 GPU。未传选项仍使用原 AUX 来源，原 native clone、配置和请求循环保留。
- 新增 34 项注册设施测试、3 项关闭路径/启动前校验/原生配置 AST 回归。历史 readiness AST 测试兼容已知新增设施后仍保留原完整对比；新测试另断言默认分支仅有唯一原 guard，防止剥离隐藏新增动作。
- 复制、对照分析、打包与文档属于实验设施。没有改动作者 native worktrees、vLLM、模型、attention、缓存策略、LoadPlanner 准入或模型执行器；没有新增研究机制。

校准/模型使用 `third_party/work/py-kvcache-p3-order-cpu`。P313 DispatchShadow 新模块没有接入本轮完整模型回放。结果中的 `policy_mode="shadow"` 表示既有被动观测元数据；`ordinary_quota_installed=false`，新增研究调度策略未介入。

## 版本与授权

| 项目 | 版本/范围 |
|---|---|
| GPU | RTX 5090；GPU-83ac724c-6d18-6f08-ab96-fb871fb1f4a8 |
| 驱动 | 595.71.05 |
| Python / Torch / CUDA build | 3.12.3 / 2.11.0+cu130 / 13.0 |
| 作者 py-kvcache HEAD | 3abba7a502d553f6e7e2e58b92086487e3395d7e |
| 作者 vLLM HEAD | 817a7e3124f817cd6e549581d3e5483207a753a4 |
| 模型 | Qwen2.5-7B-Instruct，revision 16c174980d8a1492910551634b4969e69cdc2444 |
| GPU 总预算 | 延续累计 8 小时，未重置 |
| 系统/驱动、删除、租赁、远程 push | 没有执行；原授权边界保持 |

详细锁、命令与来源在 `artifacts/prefix_io_v1/server08-p3-14`。旧 GPU 曲线/许可保留原 UUID 与 SHA；没有把旧资格改标为新硬件。阶段输入锁与可变 GPU 扣账分开，最终交付锁包含结束后的账本快照。

## CPU 执行与结果

实际运行 `run_shadow_cpu_qualification.py --label server08-p3-14-registration-matrix[-02]`；该脚本在新 shadow worktree 中运行主矩阵，并将 3 项历史模型运行时 guard 单独路由到其原 order worktree。CUDA 初始化被禁止。

最终主矩阵 961 passed / 16 skipped，历史 guard 3 passed：**964 项唯一通过、16 跳过、0 失败**，新增 37 项。随后针对静态审查补强的 AST 唯一分支断言与注册/历史对比，38 项复核通过；这些是重复检查，未累加到唯一数量。所有 CPU guard 的 CUDA 初始化与 GPU workload 数为零。

保留的非最终成功记录：

1. 首轮矩阵 960 passed、16 skipped、1 fail；失败为历史 AST 比对尚未剥离新注册设施。保留原断言、增加明确的关闭路径完整 AST 对照后，全矩阵重跑通过。
2. AST 补强复核首轮使用未指定 qualified worktree 的 PYTHONPATH，发生 collection ImportError，未执行该项测试。补齐 `py-kvcache-p3-shadow-cpu:src:scripts` 后 38 项通过。
3. 首个 GPU 完成后预算账本自然改变；一次 populate 启动前不可变锁校验挡住启动，没有 GPU 子任务。检查账本仅增加已授权 80.033793966 秒，保留原账本 SHA，执行锁只排除这个可变账本。
4. 首版完整输出离线比较将 `ambiguous_events=[]` 错与整数 0 比较，误标失败。保留 v1，新建 v2 按列表 schema/零长度校验；raw GPU 结果未改、未重跑 GPU。

原始命令、stdout/stderr、JUnit、CUDA guard、v1/v2 分析均保留。

## PRIMARY 来源与生命周期

固定 origin manifest SHA：`c03381abb29e21b54a62a4815892553b58e4ef39cc8087761cc4d173cf75d274`。

CPU 复制 3,048 个文件 / 2,796,552,192 B 至已授权 PRIMARY `runs/server08-p3-14-source/heldout-long`。新来源与 origin 逐文件 SHA 相同，metadata 与 payload 树隔离；无删除、合并或原来源改写。注册 proof SHA 为 `bf4dce139c4843f5561b5283ab78ae672567f56ae633a21362fad6e553194a09`，注册 JSON SHA 为 `9bdc5995fee795df892c987531097717a773299a9259890ac4ef28ba32993c2c`。

实际链路为：固定来源复制 → CPU 注册证明 → 同来源新 GPU 数值/成本资格 → 原 native 私有 hardlink 克隆 → 原生请求/异步 pipeline → 原 native drain/shutdown → source 后验 SHA。模型私有来源的全部 3,048 条既有路径与 PRIMARY 源同 inode，额外 suffix 写入仅在私有树。

注册是**时点内容和文件系统证明**，不是并发 writer lock、资源所有权或释放依赖证明。`original_payloads_read=false` 仅表示注册函数没有读取 AUX origin payload；注册函数确实完整哈希 PRIMARY payload。整轮复制及后验保全检查读取过 origin。注册函数本身不复制或操作 GPU，但整轮实际执行了 GPU。

原 source/DMA/consumer 保护、共享 staging、预加载、fusion、异步 continuation、完整父工作保留。没有以 Future.done、cache hit、节点完成或排序成功奖励资源释放额度；硬 parent admission 和完整四阶段预算执行仍未接入。

## 新硬件成本资格：13 次真实 GPU

在 GPU 空闲、无计算进程、输入 SHA/新标签/预算/磁盘检查后，逐项使用原 `run_gpu_stage.py --seconds 240`。固定四 knot 2048/4096/8192/16384；reps=3、rep0 热身保留，每路径/点两个独立引擎、四个测量。KV 2 GiB、staging 1 GiB；基线 seq1/d4。没有并行测试或大文件复制干扰采集。

拟合顺序：native 冷/热 top5 参考、作者 populate、cached top5 参考、参考门禁、cold01/paired01/paired02/cold02、原 exporter。校准源实有 5,760 文件 / 5,284,823,040 B。

精确缓存参考 24 组全部通过，SSD/staging 与对应 native GPU-hot 参考输出和 top5 概率完全一致，容差 0。**不是独立冷重算与 cached 逐 bit 相同**：fit 冷/热诊断保留 1 处 selected-token 差异（4096/rep0），所有冷/热 top5 诊断不同；没有全词表相等主张。

原 exporter 核验 72 条记录，输出新候选 `calibration-candidate/curves-v2.json`，SHA：
`fd05683418e851d5130ba2ab646e9e33cbb0ea490584787dca3d019abff6d451`。

| prefix tokens | 冷算 f 中位秒 | SSD g_ssd 中位秒 | staging g_mem 中位秒 |
|---:|---:|---:|---:|
| 2048 | 0.142227 | 0.269058 | 0.066505 |
| 4096 | 0.297040 | 0.471041 | 0.061320 |
| 8192 | 0.668030 | 0.893740 | 0.084135 |
| 16384 | 1.649385 | 1.458012 | 0.161160 |

成本模型 SSD break-even=13,968 tokens、staging=2,048；这是原 LoadPlanner 路径成本交点，没有研究策略收益含义。staging 单点样本变动及各 session 原数值保留，不给置信区间或全域精度主张。

随后只验 C2/d8 的冻结 16,256 内容点，native/cached 两参考 + cold01/paired01/paired02/cold02 共 6 run。既见三来源族未参与本次拟合，不能称本轮新未见内容泛化。6 组 cached 输出/top5 精确比较通过，冷/热 selected 输出相同但 top5 诊断仍不同。最终原 `qualify_concurrent_pilot.py` 门禁：

| 路径 | 冻结预测秒 | 观测中位秒 | 绝对相对误差 |
|---|---:|---:|---:|
| f | 1.631600 | 1.616733 | 0.911% |
| g_ssd | 1.452064 | 1.305661 | 10.082% |
| g_mem | 0.159635 | 0.161911 | 1.426% |

全部低于原定 25%，`PASSED_C2_UNLOADED_POINT_GATE`。阈值未放宽、heldout 未回流拟合。新 workload 仅显式登记新 UUID/对应基线 engine，families/requests/到达日程原样保留并记录旧 SHA。许可仅覆盖单请求 unloaded point 支撑的有限 C2/d8 开发回放，full-domain/loaded prediction/formal evaluation 未验证。

## 追加一次原生模型迁移验收

成本资格通过后，新建并冻结 `model-off-plan.json`，增加一个 240 秒上限的原 native 回放，成本 13-job 包络保留原样。使用显式注册来源、新候选/新许可、C2/d8、既有 mixed workload 的五族/10 请求/每请求 128 tokens；原 LoadPlanner on，新研究调度策略关闭，被动观测保留。没有重置缓存或注入延迟。

实际 GPU 正常退出，`PASSED_NATIVE_C2_DEVELOPMENT_REPLAY`。完整输出 v2 与 P313 同 GPU 原生 cold/hot 完整参考分别对齐，**10/10 请求、1,280 token 完全一致**；每请求 128 时间戳/127 ITL，总 1,270 ITL，零歧义，时间有限且严格递增。读取 6,254,624,768 B、写入 1,896,480,768 B；final AIO accepted=reaped、outstanding=0、观测故障=0、engine shutdown 完成。source 3,048 文件全部保持。

单 baseline cohort 含 drain 为 31.378286 秒，只作迁移运行记录，不与旧硬件或旧策略样本比较，不作为研究增益。

## 保全、预算与下一允许阶段

本轮 **14 次真实 GPU run、1,137.714157 秒（约 18.96 分钟）**，全部 exit 0，无 timeout，进程 session 排空。累计 11,404.191354 秒，8 小时剩 **4.832169 小时**。没有新模型下载、安装、系统/驱动变更、删除、缓存合并、租赁或 push。最终 GPU 2 MiB、0%、无计算进程。

后验完整 SHA 核验：模型 11 文件 / 15,242,788,168 B；AUX 原来源和 PRIMARY 副本各 3,048 文件；385 个执行输入均未改变。校准源另建完整 SHA 清单。主盘余约 47.66 GiB，后续混合模型仍须保留原 3 GiB 预留与 8 GiB 下限；AUX 20 GiB 限制不变。

下一允许阶段为 P3：先 CPU 补足真正的 accepted-parent/排空状态和 unknown-aware owner 快照，正确涵盖 incoming、active、failed-but-draining；完整 admission 必须保留 jobs_to_flush 来源保护与待提交 parent，不能将 native worker 的 False 当新协议。之后对有限阶段入口预算/必需 continuation 逐级 qualification，再做有冻结计划的同硬件策略对照。未完成前不能宣布 full caps/admission 已安装，不能进入 P4–P7 或主张研究提升。
