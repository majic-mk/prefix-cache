# Server09 下一允许阶段

当前结论：迁移与 CPU 集成通过；新 GPU 没有运行。G1 授权请求已提交，未经明确答复不得激活新卡许可或运行 CUDA/NVML。旧累计预算不得重置。

受限提案：GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2；off/shadow 各一次，每次 180 s＋20 s drain，计划总预留 400 s；16 MiB staging、iodepth 1、PRIMARY 128 MiB 新空间/轮、8 GiB floor。禁止模型下载、系统/驱动修改、既有数据删除合并、租用、付款和远端 push。详情 G1_GPU_SCOPE_PROPOSAL.json。

授权后部署只做以下已审查增量：

1. 从 gpu-scope-candidate/CANDIDATE_HASHES.json 检查实际两个原 canonical 脚本 SHA 未变，原字节另存 append-new 历史副本并写路径/bytes/SHA 映射。候选资格与预算函数 AST 复用证明在 CANDIDATE_AST_REUSE_PROOF.json。
2. 应用候选两个脚本：可选 --permissions-path，默认旧路径/argv 不变。保留原 permissions.yaml 和 CPU_ONLY_AUTHORIZATION.json；只生成单独 server09 G1 权限文件，并明确 allow_remote_push=false、allow_model_downloads=false、无 AUX。
3. 生成新的 server09 实际源锁，重新哈希全部原 2048 项当前内容，显式包含新 canonical qualify_p4_native_gpu.py、run_gpu_stage.py 和有效权限文件；候选/历史文件只作来源。旧 source-lock 是历史快照，不可声称版本演进后原路径仍匹配。
4. 按实际用户答复生成独立 GPU_STAGE_AUTHORIZATION.json，status=USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION；权限 path/bytes/SHA、最终 source_lock path/SHA、精确 UUID、qualification_context 必须匹配。临时测试 fixture 或 PROPOSED_NOT_AUTHORIZED 提案不授权限。
5. 实际跑纯 CPU scope/source/storage/budget 门禁，成功后只调用正式 qualifier 的外层 --launch。不得直调 --execute 或复制内部 guard。仍用原唯一累计账本与 session drain；检查前一作业结清才开后一项。

模式命令形状（路径均应替换为实际冻结的新文件，不可直接运行下列占位字符串）：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py --mode off --name server09-p4-native-off-01 --launch --permissions-path <有效server09权限文件> --scope-record <有效server09-G1授权记录> --source-lock <新server09实际源锁>
```

第二项 mode=shadow，name=server09-p4-native-shadow-01；两次失败与成功都保留完整 result/log/ledger 消耗，达到限定范围即停止新增 GPU 运行。若任何资格失败，关闭新策略、保留现场并如实报告，不修改驱动或系统。

G1 通过也不代表方法效果或 P4 完成。后续 G2 尚需在原作者 runtime 装入完整 execute_model→sample_tokens 观测、显式 async_scheduling=False、GPU 计时/时钟边界、实际 batch/context/load、真实逐阶段 I/O 和完整 drain；冷 prefill 的真实 pre_context=0 需要新版完整 trace 契约，不能伪改旧 context 维度。

合法单阶段成本、生产 ETA、真实 owner/witness 和上下文资格未完成，I/D/J 继续回退原 U。D2H→SSD 自然续接没有独立成本隔离证据时对应 cell 缺失；四臂与独立 U 必须同一共同配置。P5–P7 未启动，SLO 仍为 null。

PRIMARY 最新门禁通过不意味着 AUX 扩额；旧 AUX 20 GiB cap/8 GiB floor 下 3 GiB 轮次仍不能运行。
