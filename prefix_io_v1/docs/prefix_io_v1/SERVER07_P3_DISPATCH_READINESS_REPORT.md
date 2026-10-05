# P3 第 11 次交付：四阶段额度 CPU 合同与真实写回等待观测

## 当前结论
在现有 AutoDL、作者 py-kvcache/vLLM、授权 Linux AIO 上继续推进；P3 仍未完成，P4–P7 尚未进入。本轮没有更换执行器或改动 attention，也没有启用 dependency/interference/joint。

本轮最有用的新证据：冻结混合请求流复现一次 **0.616136868 秒**的 restore-destination fence 等待；3 个被保护的小 store（23、25、27）在首次 D2H 提交前仍排队。保留的等待期采样同时观察到大背景 store 22 和满 I/O 深度。该证据支持下一步检验 mandatory 优先的简单排序，但不是反事实加速证明，更不是可立即释放 GPU 容量的证据。

## 实际改动与边界
- 新增 `src/prefix_io_control/dispatch_budget.py`：四阶段累计操作/字节额度、独立在途额度、联合 SSD/复制额度、staging 预留、pre-intake parent 上限合同、epoch/expiry/owner 验证、单次 backend 接受凭据及明确 progress override。纯 CPU 合同，**尚未接入 native 提交或 intake**，不持有 Future/slot，不建立第二条 I/O 队列。
- 新增 `src/prefix_io_control/store_readiness.py`：在已有 post-pump observation hook 采样，最多 32 parents、64 copies、64 ring ops、96 状态组；相同状态合并并保留首末观测时刻。模型实验仅保留存在 mandatory store 的采样。没有 CUDA event query、全局同步或释放量猜测。
- 实验驱动 `run_concurrent_pilot.py` 增加显式 `--store-readiness-probe`，只能与既有 passive flush probe 组合，禁止与 start-budget 实验同时开启。新增 `store_readiness_worker_probe.py` 在热身排空后组合既有 sink；native shutdown 后导出。AST 测试证明移除该可选参数/分支后原驱动不变。
- 新增 `qualify_store_readiness_gpu.py`、`analyze_store_readiness.py` 和 67 项 CPU 测试。
- 所有 author native reactor、vLLM 调度/模型代码本轮未改；LoadPlanner、精确 Prefix、共享 staging、preload、fusion、D2H→SSD 续接及完整 parent completion 保持原样。
- 按用户对 SHA 清单的明确授权，合并 8,231 份已完成实验的私有重复缓存。普通权限文件未改变。

额度合同不能假装已经落实完整限流：原生 D2H→SSD write 直接续接，H2D fusion 的物理字节随共享使用者增加。后续接入必须解决整批计费、续接可进展和 native intake 的共享上限；不能把这些 CPU 检查写成已实施的 GPU 字节上限。

## CPU 验证
命令均在服务器项目根目录执行，环境差异存于对应 *-command.json，CUDA_VISIBLE_DEVICES 为空，run_aio_cpu_tests.py 禁止 CUDA 初始化。

```text
.venv/bin/python experiments/prefix_io_v1/scripts/run_start_budget_cpu_qualification.py --label server07-p3-11-first
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_start_budget/test_store_readiness.py --basetemp /root/prefix-io-v1-validation/cpu-evidence/server07-p3-11-readiness-final --junitxml artifacts/prefix_io_v1/server07-p3-11/readiness-final.xml
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_start_budget --basetemp /root/prefix-io-v1-validation/cpu-evidence/server07-p3-11-final-targeted --junitxml artifacts/prefix_io_v1/server07-p3-11/cpu-final-targeted.xml
```

首轮矩阵 779 passed / 16 skipped；最终相关套件 166 passed / 0 skipped。按 test identity 去重并取最新结果，合计 **795 passed / 16 skipped / 0 failed**，不把重复运行相加。新增模块为合同 45 项、观测/集成 10 项、证据分析 12 项。三个 CUDA guard 均为未初始化。16 项跳过中，13 项专测 no-vLLM fallback（当前加载真实 vLLM，故不适用），3 项原 io_uring setup 因 EPERM 跳过；不能计作通过。

证据：`cpu-summary.json`、`cpu-final-targeted.xml`；首轮完整 XML/日志/guard 在 `/root/prefix-io-v1-validation/cpu-evidence/server07-p3-11-first/`。

## 真实 GPU 执行
通过既有 run_gpu_stage.py 包装器逐次校验 UUID、剩余预算、独占进程、冻结源码、128 MiB 或 3 GiB 辅助盘预留，以及主盘 8 MiB 临时预留。未降低原有预留或 8 GiB 文件系统余量门槛。

| 运行 | 真实内容 | 结果 | 计入 GPU 预算 |
|---|---|---|---:|
| server07-p3-11-readiness-01 | off / readiness 各 store+restore，真实 CUDA/Linux AIO，每方向 8 MiB，原始同源字节比较 | 4 个阶段通过，排空，staging 16,650,239 B ≤16 MiB | 23.914699838 s |
| server07-p3-11-mixed-readiness-01 | Qwen2.5-7B BF16，冻结 depth8，2 GiB KV /1 GiB staging，10 请求×128 token | 10 个完整输出精确匹配；3,048 个原缓存文件保留；原生排空和 shutdown 完成 | 103.442915373 s |

前一个 GPU 检查使用观测器的早期逐样本版本；其后仅增加状态合并与 mandatory 过滤，分别经最终 CPU 测试及第二个真实模型运行验证。两个版本都有独立源码锁和前版本备份，不混用资格证据。

模型 cohort（含 drain）为 **30.470028398 s**。这是带被动观测的一次开发诊断，不与不同观测配置的历史结果作速度比较。没有新的性能收益、SLO goodput、置信区间或观测开销达标结论。

本轮 GPU 增量 **127.357615210 s**；累计 **9716.287493106 s / 2.698968748 h**；8 小时总授权剩 **5.301031252 h**。两次包装器均 exit 0、未超时、session_drained=true；执行结束 GPU 0 MiB / 0%、无活跃预算预约。无下载、驱动/系统修改、外部 push 或新租机。

## 等待证据与限制
- 唯一匹配的 native fence 保护 parents 23、25、27，各 917,504 B，合计 2,752,512 B。目的块属于新的 allocation generation，active_refs=1；parent 退休后仍非 free queue。保护解除不等于新释放 GPU 容量。
- 等待开始至第一个相关 D2H host enqueue 为 **0.598843761 s**，之后至 wait 返回约 **0.017293107 s**。CUDA event elapsed 不是与 host enqueue 同轴的设备执行起止时间。
- 520 次 bounded mandatory 采样形成 156 个不同状态组；保留最近 96 组，60 组被覆盖，无 parent/copy/ring 截断或 observer fault。
- parent 23 的保留 pre-enqueue 观测为 91 组 /280 次，覆盖的首末观测点是等待后 0.282580–0.597682 s；parents 25/27 各 92 组 /281 次，末观测点为 0.599110 s。这些点全部 data_inflight=iodepth=8、free staging slots=0，并观察到先于小 parent 的大背景 parent 22。临近大任务发完最后文件的部分点不再存在“尚待提交的大背景 parent”，因此不是全窗口每一点都满足该描述。
- free slots=0 不能证明 staging 耗尽：未查询可回收 cache；此次原生 foreground reserve 尝试 3,794 次、失败 0 次。不能把清洁缓存占用当成不可释放债务。
- 未记录每次原生选择时的拒绝原因，也未查询 compute event readiness。采样缺前段且有间隔，不能把首末点之间积分为确定的等待持续时间。没有估算策略收益。

证据：`readiness-analysis.json` 和真实运行 `details/result.json`、`native-cohort.trace.json`。脚本遇到身份不匹配、重复 parent、乱序时间、截断 witness 会拒绝结论。

## 已授权的存储合并
清单：`/root/prefix-io-v1-validation/audits/server07-p3-11-four-runs-audit.json`
SHA-256：`79dedf7944ba9a85848c75729b41583d053c85cbc4560655636125e69d51b211`

只覆盖 P310 的 off/fixed/pressure 三次 mixed start 实验及 P311 mixed readiness，一共 **8,231** 个私有文件；排除全局审计里另外 8 个目标。逐文件重新验证、保留可回滚旧 inode、替换为同字节 canonical hardlink，完成后复核 2,068 个 canonical SHA 与全部目标 inode/路径；没有临时残留。回收 **7,551,975,424 B（约 7.03 GiB）**。

```text
.venv/bin/python experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py --audit /root/prefix-io-v1-validation/audits/server07-p3-11-four-runs-audit.json --authorization experiments/prefix_io_v1/configs/authorizations/server07_p311_archive_dedup.json --journal /root/prefix-io-v1-validation/audits/server07-p3-11-approved-journal.jsonl --apply
.venv/bin/python experiments/prefix_io_v1/scripts/verify_private_cache_archive.py --audit /root/prefix-io-v1-validation/audits/server07-p3-11-four-runs-audit.json --output artifacts/prefix_io_v1/server07-p3-11/archive-verification.json
```

合并后辅助盘计入用量 11,911,929,856 B，下一轮固定 3 GiB 预留检查通过；打包仍需新检查。主盘仅约 12 MiB 高于 8 GiB floor，大日志/缓存/交付包继续放授权辅助盘。不得把本次授权扩展到后续新缓存。

## 锁定版本与下一允许动作
根项目 HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08；py-kvcache author 3abba7a502d553f6e7e2e58b92086487e3395d7e；vLLM author 817a7e3124f817cd6e549581d3e5483207a753a4。实际 dirty 状态与文件 SHA 见 version-lock.json、source-lock-before-gpu.json、source-lock-before-model.json 及最终 source-lock.json。公共兼容补丁未改，CPU 合同与 observer 增量分别出 patch。

下一阶段仍为 P3：
1. 在隔离工作树实现并验证有界 mandatory store 优先的强简单基线，只重排尚未提交的 native 工作，保留物理安全、已接受续接、合并与 parent completion；默认关闭回到原路径。
2. 将四阶段 CPU 合同逐步薄接入 native 接受点，解决整批大小、下游额度和共同 intake 边界；不能直接把本轮 prototype 配成完整限流。
3. CPU/真实 GPU 正确性门槛通过后，再按冻结同配置比较优先排序与原顺序，并补足观测开销及 P3 最强简单基线检验。没有结果前不宣布可行性收益或进入 P4。
