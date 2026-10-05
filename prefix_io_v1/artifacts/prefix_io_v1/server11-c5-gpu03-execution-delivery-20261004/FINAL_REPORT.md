# GPU03 真实六窗口验证交付（2026-10-04）
本轮已在指定 AutoDL 服务器 connect.westc.seetacloud.com:26909 的 RTX 5090（GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9）完成一次真实公共六窗口模型与原生 I/O 标定，并通过真实来源、原守卫、留出样本及 canonical 凭据核验。
当前阶段是 C5 公共成本资格，尚非正常 off/shadow/on 策略验证或 P4 性能对照。
## 实际执行与改动
本轮未修改算法、缓存引擎、模型执行器或此前已冻结的入口源码；没有再跑已通过的 74 项 CPU 检查。仅依据用户“批准授权，继续”记录了新的 gpu03 人类授权并生成现场权限、配置、计划、scope 和实际启动/验证产物。GPU02 的已消耗授权未复用。
固定入口位于 artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004；source revision SHA 4372cb6bb43399aededd3509305691dabb88cfbb16962ddecbb0887697c5a305，site lock SHA ec397301ccac4ab76e6e34ebc38207608649b1e32aef507a3a39ff3d3302afde。63 公共文件字节和 32 原函数 AST 继承保持。
实际命令顺序：control_native_cost_job.py bind → prepare_and_verify_native_cost.py --prepare → control ... scope → control ... launch → 原 guard 自然结束/排空 → control ... after → prepare_and_verify_native_cost.py --verify。完整 argv/cwd/stdout/stderr/exit 在 SERVER_LIVE_PATH_BIND/PLAN/SCOPE/AFTER/VERIFY_01 和 path delivery/GPU03_LAUNCH_CONTROLLER_*。
原 run_gpu_stage.py 实际运行唯一作业 server11-c5-native-common-cost-gpu03，固定 1200 秒执行+20 秒收尾；六个新模型子进程按 A0/B0/B1/A1/A2/B2 顺序，同一模型与生产 KV 来源，原 preload/shared staging/copy aggregation/异步执行路径、原工程预算和停止规则保留，bridge=None、LoadPlanner off。无重试、下载、删除、系统/驱动/安装包修改。
## 真实结果
六个子进程全部 exit0，各正常返回原 engine shutdown；父进程完整六窗 PASS，守卫 exit0/child_exit0，没有超时或中断，session 前后成员均空，reservation 已清除。
每窗完整测量输出 128 tokens、128 帧，总计 768 个测量输出 tokens；另有每窗 cold 资格请求，计数不混入六窗成本样本。真实配对 prompt/seed/output equality、原 native accounting/drain 与物理读取因果证据由 verifier 验证。
结束后 source after 重哈希 4741 个 site 源/资产引用全部通过，与 before 闭包一致。模型权重哈希在准备/启动前/结束后执行，没有在被测窗口内反复全量扫描。
原公式两标定对：
- pair0：A 12.957824 ms，B 20.456415 ms；
- pair1：A 12.644704 ms，B 16.047457 ms。
baseline=12.801264 ms，incremental_or_joint=5.450672 ms，uncertainty=2.204479 ms；条件成本上界 20.456415 ms。
独立留出 pair2：A 13.111712 ms，B 15.870112 ms；B 被上界覆盖，underprediction=0，signed_error=-4.586303 ms。未使用留出数据重新拟合。native_execution_verified=true、conditional_cost_cell_qualified=true、heldout_covered=true。
限定条件：当前 GPU/model/KV/kernel 来源，单 active decode、batch1、context144、选定 offset16、917504 字节 SSD-read 条件、已有 I/O 四阶段基线均为0；未知条件拒绝。这是有限工程标定，不是统计/SLO 保证或多负载通用上界。
## Canonical 实际凭据
真实 load_verified_single_file 重新序列化六窗 raw，并验证 guard/source/math/kernel，通过后返回 ExactSingleFileReceipt 原类型；未调用私有 issuer 或用 synthetic/CPU origin 伪造对象。
仅校准域 binding：CALIBRATION_DOMAIN_BINDING.json，SHA 821b1fddc8ab7da9caebc2b7ec4ab7030c7036e5f9c220689ec30752a08c1178。
actual cost_upper=20456415 ns，A-only 工程 step_budget=12957824 ns。后者仅来自两校准 A 最大值，不是用户 SLO，未用动作或留出样本扩大预算。
REAL_CANONICAL_RECEIPT_SUMMARY.json 只是实际对象的审查摘要，不能作为 typed receipt 传给 policy，也不提供新的 GPU 执行授权。normal_runtime_binding_verified=false、actual_on_installed=false、full_runtime_cost_qualified=false、production_qualified=false、strategy_effect_verified=false、resource_release_credit=false。
## GPU 预算与存储
本次唯一 guard 实际记账 748.744795425795 秒（约12分29秒）。原累计预算由 22380.692507382948 变为 23129.437302808743 秒，剩余 5670.562697191257 秒（约1小时34分31秒）。账本 SHA 4fd592aaecf9371ae050c65035e7cf037b56d876653aac2e49fa65f5fc5c3da8；唯一 reservation/event 与结果逐项对账，无活动 reservation，也没有 compute 进程残留。
PRIMARY 约56.4 GiB可用；无删除/扩容，后续没有获授权的新 GPU 作业在运行。证据备份不含模型权重或私有缓存副本，保留所有服务器原始路径。
## 结论与下一允许阶段
已证明当前原模型+原生异步 I/O 路径能够真实运行，并在该限定条件下得到可追溯的成本凭据；未证明新策略更快或论文收益。20.456 ms 的上界覆盖15.870 ms的留出样本不是提速结果。
下一允许工作是 CPU 准备真实正常 off/shadow/on 入口绑定与完整观测成本：StageD load_receipt/freeze_common/prepare 仍是 CPU 阻断 stub，不能直接删 block 或复用已消耗的 common GPU03 授权。应直接复用 G common/source 类型与正确模型/Event/runner refs，保持三模式同一 common lock，独立固定每模式 scope/config/前驱资格，补齐实际等待/通知与净 CPU 观测成本。
普通模型/on/完整观测成本通过后，才运行 P4 公平对照。没有实际 defer 时标记 NOT_EXERCISED，不调整负载、预算、SLO、样本或阈值制造阳性。后续 GPU 作业必须另具对应目标/源码/作业的明确授权，并在原剩余预算内。

