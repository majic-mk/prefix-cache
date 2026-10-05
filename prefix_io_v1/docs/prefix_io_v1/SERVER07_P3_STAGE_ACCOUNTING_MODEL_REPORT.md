# Server07 P3：四阶段计量、真实 GPU 与模型入口验证

本轮从 CPU 集成推进到了真实 CUDA／Linux AIO 和 Qwen2.5-7B 请求流验证。**实现能运行并保持本次输出一致，但当前固定／压力启动额度没有带来提速；依赖排序、干扰自适应及联合策略仍未启用。P3 尚未全部完成。**

## 实际结果

| 实验臂 | 含末尾排空的请求流耗时 | 相对关闭启动额度 | TTFT P95 | 请求内 ITL P95 的中位数 |
|---|---:|---:|---:|---:|
| off | 21.751 s | — | 12.320 s | 31.856 ms |
| fixed | 27.831 s | +27.95%（更慢） | 19.000 s | 45.875 ms |
| pressure | 27.334 s | +25.67%（更慢） | 17.946 s | 81.624 ms |

三组沿用 P306 的 all_hit 开发负载：同样 10 个请求、每请求 16,257 个输入 token、128 个输出 token、原到达时刻、两序列执行配置、2 GiB GPU KV 和 1 GiB staging 预算。每组新建原生 engine 和 GPU／CPU 缓存，初始 SSD 内容通过硬链接使用同一已发布语料。新增写回保留私有命名空间。LoadPlanner、共同成本表、预加载、共享 staging、复制合并与异步执行器不变。

每组只运行一次，固定顺序 off→fixed→pressure；没有置信区间或正式 SLO goodput，期间还执行了少量 CPU 合同测试。因此表格是开发／集成运行的描述性结果，不能当作稳定退化幅度或研究效果估计。没有按结果改变参数，也没有剔除慢请求。

30 个请求、3,840 个输出 token 逐项一致。三组都完成真实 SSD read/write 和末尾 drain，阶段计量无错误、无未回收在途字节，启动控制器没有进入 faulted fallback。三组 pending flush wait 都为 0；这次全命中负场景没有给出释放依赖优化的机会证据。

fixed/pressure 参数预先冻结为每 10 ms 共享 4 个文件链路启动单位，超龄阈值 100 ms；pressure 额外保留 1 个空闲 staging slot。fixed 记录 196,763 次额度拒绝，pressure 记录 72,868 次额度拒绝和 147,243 次水位拒绝；这些是重复 pump 的拒绝计数，不是不同请求数。两组分别发生 58、179 次 age override。限制确实生效，但没有在这次负载上换来收益。

## 真实 GPU 检查

先执行一项不加载模型的真实 GPU 往返任务，包含 13 个阶段：
off/shadow/fixed/pressure 各执行 store、重新创建 coordinator 后的真实 SSD load、共享预加载双消费者；另有 pressure shutdown 排空。

关键通过项：
- 合成 GPU KV 逐字节相同，真实 CUDA 与 Linux AIO，没有测试替身。
- fixed/pressure 普通额度耗尽后，native mandatory wait 独立完成后续文件。
- 单次 3,670,016 B SSD 读取服务两组互不重叠的 GPU 目标，H2D 实际复制为 7,340,032 B。
- shutdown 继续推进已接受链路，所有 Future、CUDA copy、AIO CQE 和 worker 正常排空。
- 此项使用合成 KV，不代表完整模型调度；之后的三个 Qwen 请求流任务单独提供模型集成证据。

本轮共 **4 次 GPU 任务，均 exit 0，session_drained=true**。新增 GPU 账本消耗 **308.2757 s**，累计 **8868.5650 s / 28800 s**，剩余约 **5.5365 小时**。模型下载为 0，未修改驱动或系统，未调整权限或 GPU 预算。

## 实际代码改动

所有开发与测试在服务器完成，本地仅收取和验证交付包。

1. 新增 src/prefix_io_control/stage_accounting.py：有界的被动四阶段记账，记录 SSD read/write、H2D/D2H 的已接受、已完成、在途和峰值数量／字节。无第二任务队列、无资源所有权、无资源释放收益推断。SSD 字节取实际对齐请求；GPU 字节取实际复制映射长度，包含合并复制的多个消费者。
2. 新增 src/prefix_io_control/start_options.py：严格验证 off/shadow/fixed/pressure 启动选项、有限参数、运行身份；每个 reactor 独立创建状态。off 不创建额度或计量对象。
3. 仅在隔离工作树 third_party/work/py-kvcache-p3-quota-cpu 修改 reactor.py、vllm.py，加入薄计量钩子与可选配置适配。旧 py-kvcache-p2-aio 与作者 vLLM 原生库未改；没有全局切换环境。工作树目录名仍保留 cpu，其本轮 GPU 资格仅以源文件哈希和本报告限定的测试为准。
4. 现有 run_concurrent_pilot.py 增加显式 --start-budget-config；不传此参数时保留原入口。新增 GPU 检查、模型合同、结束后读取计量、结果分析脚本。
5. 新增只读缓存重复审计和待授权的归档硬链接合并工具。本轮只有审计、预演、临时测试夹具验证，未合并旧实验缓存。

阶段计量的 accepted 边界是原后端 API 成功返回，包含后端尚待提交的队列；不声称测到 SSD 硬件瞬时带宽。CUDA 完成以原生 event query 为准，短 I/O 结果按实际返回字节记录。可选计量异常会使计量结果失效，不能阻断原生任务，也不能作为策略成功。

**仍未实现所有阶段的硬字节上限。** 当前启动额度与实际字节计量分开；已接受 D2H→write、read→H2D 续接继续保持原顺序和进展。计量不是新的传输引擎，不提前 complete_store。GPU qualification 没有把 full_per_stage_caps 改成 true。

## CPU 结果与实际命令

完整回归：677 通过、16 跳过。后续增加模型合同、分析防误报和归档工具故障恢复测试，最终专项 95 通过。按 classname/name 去重合并，**698 个不同通过项、16 跳过、0 最终失败**；其中本轮新增 49 项。不能把多次运行相加。

首次专项有 1 个测试失败：短写夹具把错误结果配置到了不存在的 CQE 0，实际首个标识是 1。修正测试夹具后，完整回归及最终专项均通过，保留初次失败日志。原作者 GPU e2e 文件没有作为 CPU 通过项；16 项跳过原因仍为 13 项无-vLLM fallback 场景与 3 项容器 io_uring EPERM。

实际主要命令（均在项目根目录执行，完整环境与 argv 位于 *-command.json）：

~~~bash
.venv/bin/python experiments/prefix_io_v1/scripts/run_start_budget_cpu_qualification.py --label server07-p3-09-cpu

.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py -q --import-mode=importlib tests/prefix_io_v1_start_budget --basetemp /root/prefix-io-v1-validation/cpu-evidence/server07-p3-09-final-targeted --junitxml artifacts/prefix_io_v1/server07-p3-09/final-targeted.xml

.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label server07-p3-09-start-budget --seconds 180 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_start_budget_gpu.py --output /root/prefix-io-v1-validation/runs/server07-p3-09-start-budget

# 三次模型运行的完整 argv 分别见 model-off/fixed/pressure-command.json；
# 均通过 run_gpu_stage.py，单次限时 600 s；run_concurrent_pilot.py 的原模型、
# 来源、成本表、manifest、qualification 参数保持一致，只增加对应 start-budget-config。

.venv/bin/python experiments/prefix_io_v1/scripts/analyze_start_budget_model.py --output artifacts/prefix_io_v1/server07-p3-09/model-analysis.json

.venv/bin/python experiments/prefix_io_v1/scripts/audit_private_cache_duplicates.py --output artifacts/prefix_io_v1/server07-p3-09/private-cache-duplicate-audit.json

.venv/bin/python experiments/prefix_io_v1/scripts/consolidate_private_cache_copies.py --audit artifacts/prefix_io_v1/server07-p3-09/private-cache-duplicate-audit.json --journal artifacts/prefix_io_v1/server07-p3-09/dedup-pending-journal.jsonl
~~~

最后一条没有 --apply，只验证计划，未写旧缓存，也未创建授权。

## 容量与下一允许阶段

CPU 主回归启动前预留 640 MiB，临时数据落在已授权辅助目录。本轮 GPU 合成检查预留 128 MiB；三个既有 all_hit 模型任务每次预留 512 MiB，并分别重查容量。所有任务保持既定 8 GiB 空闲底线与 20 GiB 辅助目录额度。

只读审计对 15 个已结束实验的缓存及已发布源进行 SHA-256 校验，覆盖 15,759 个独立 inode、14,458,945,536 B 数据。发现 10,457 个可去重的私有副本，内容归属 2,117 组，预计可释放 **9,594,339,328 B，约 8.94 GiB**。模型文件、已有共享文件、原始结果、日志和图表不在替换范围。清单与预演已保存，实际缓存变更为 0。

合并会让原本独立的归档文件共享 inode，属于旧实验数据组织方式的改变，因此尚未执行。具体作用、风险、回退与限定授权见 SERVER07_P3_ARCHIVE_CONSOLIDATION_PROPOSAL.md。一般权限仍为原值。

下一允许工作仍在 P3：
- 补齐实际阶段字节与在途上限的预算合同，保持原生已接受链路续接。
- 恢复混合实验所需容量后，先做预注册的真实问题复现与强简单基线筛选。
- 若正常负载没有可重复的目标阻塞，或合理简单配置已消除它，按交接包的 P3 停止条件报告当前场景不支持继续投入；不人为限速制造收益。
- 满足 P3 后才接入 P4 的 dependency_only/interference/joint 并进入 P5 效果评估。当前未证明研究方法优势，也未证明所有场景下不可行。

## 证据位置

服务器项目根目录 /root/autodl-tmp/prefix-io-v1-handoff/project。
本轮证据 artifacts/prefix_io_v1/server07-p3-09/，包含原始命令、CPU JUnit/guard、GPU 账本前后、GPU 13 阶段明细、三个模型原始结果／逐 token 时刻／原生 trace、模型分析、源文件锁及缓存审计清单。

实际 GPU 数据目录 /root/prefix-io-v1-validation/runs/server07-p3-09-*。
CPU 原始临时数据 /root/prefix-io-v1-validation/cpu-evidence/server07-p3-09-*。
本地交付包不重复装入模型、KV 缓存和 CPU 大临时文件；其服务器位置和校验清单保留。
