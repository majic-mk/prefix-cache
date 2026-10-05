本轮已在 server09 完成 G3 原生 KV/SSD 资格验证的 CPU 准备：服务器实际 96/96 测试通过，前后 4,050 个 G2 源与资产引用无变化，旧成本曲线 5/5 被正确拒绝。此包尚无可启动的 GPU 实验入口，也没有实际 KV 捕获安装器；不能称为 SSD 恢复成功、P4 方法有效或 P5 性能结果。

本轮实际新增四个独立 CPU 模块及对应测试：
- g3_native_pilot_contract.py：字节锁定并执行原 PCHIP/成本表/LoadPlanner 标量代码，检查原 SSD 与 RAM 门禁、共享 staging/preload、布局和两进程合同；CPU 合成 layout/哈希不会变成生产事实。
- g3_curve_context_audit.py：只读核对真实旧表字节和预期 GPU、实际模型加载路径、密度、来源与支持域；不改表、别名、阈值或准入算法。即使 caller 元数据匹配也不授予 GPU/生产资格。
- g3_native_owner_observer.py：原注册链/源代码身份检查、有限 scalar transfer/results/tail 记录；off 直通，原方法只调用一次，保持返回与异常身份。无 hook 安装器。公开诊断记录可以由 caller 构造，单个 source_bound_runtime/closure 字段不能当作真实 GPU 或阶段完成证明。
- g3_kv_byte_evidence.py：有限完整文件、单 group、原 4096 对齐/零外层 padding，检查 layer-major 页序、块/哈希/job 关联、四阶段 bytes/SHA 一致及全局 restore-before-compute。只核 CPU fixture；没有真实 tensor/event/SSD 捕获来源。

原 py-kvcache、作者 vLLM、旧观测 helper、模型资产和研究策略均未修改。保留精确 Prefix Cache、原成本准入、共享 staging、preload、复制合并、异步路径；没有重新实现缓存引擎或模型执行器。本轮新增候选未接入实际模型执行，不据此宣称已完成 off/shadow 真机恢复对照。

服务器实际命令（完整原样命令与 exit/stdout 保存在 ACTUAL_SERVER_CPU_G3_EXECUTION_COMMAND.json；四个子命令及日志 SHA 在 ACTUAL_SERVER_CPU_G3_RUN_01.json）：

```sh
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S \
  artifacts/prefix_io_v1/server09-g3-production-kv-io-20261002/run_g3_cpu_checks.py \
  --root /root/autodl-tmp/prefix-io-v1-handoff/project \
  --lock artifacts/prefix_io_v1/server09-g2-normal-worker-site-cache-v4-final-20261002/gpu-source-lock-candidate.json \
  --drain-source artifacts/prefix_io_v1/server09-g2-native-drain-20261001/g2_native_drain_provider.py \
  --connector-source artifacts/prefix_io_v1/server09-runtime-connector-20261001/p4_runtime_scalar_connector.py \
  --adapter-source artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py \
  --inputs artifacts/prefix_io_v1/server09-g3-production-kv-io-20261002/G3_EXISTING_CURVE_CONTEXT_INPUT.json \
  --receipt artifacts/prefix_io_v1/server09-g3-production-kv-io-20261002/ACTUAL_SERVER_CPU_G3_RUN_01.json
```

上述相对路径展开后对应实际原样命令。实际服务器 Python 3.12.3，四组测试为 29 配置/原标量准入、12 曲线上下文、23 owner/tail、32 字节/顺序；全部 exit=0、无 skip，合计 96。本轮原始成功、失败和历史反例均保留。合同初次运行 28 项出现 2 failures+4 errors；字节初次测试因 AST future flags 出现错误；旧多文件实现曾错误接受文件0 compute=8 早于文件1 capture=15，已保留旧源码、真实反例和修复后日志。一次过大 CPU 曲线读取的 stdout 不是有效内层 JSON，原记录保留，仅以随后有效的有界查询和服务器实读 gate 为结论。

现有五份表的模型名都是 Qwen/Qwen2.5-7B-Instruct，而实际 ModelConfig 使用冻结的本地模型路径；全部 GPU UUID 来自旧服务器。最早两表支持至 1024 tokens，却要求 SSD 门槛 1040，无法在支持域内准入；后三表明确为 candidate-only、无独立内容验证、限制原预算诊断用途。真实 gate 5/5 拒绝；原 CPU 标量入口也按实际模型路径拒绝，不替换为逻辑别名骗过检查。后续应在当前环境按原校准路线取得有来源的新测量，不能把旧表改标签或将 G2 host/current-stream 诊断升级为 GPU 成本。

4,050 引用的原 G2 锁 SHA-256：
0325cb500f74051dedf6aefe6b51fd9d0b817f831ee81fe45e1c3942df13336b。
本轮新增 CPU 模块版本另列 G3_CPU_MODULE_SOURCE_LOCK.json；它仅为 CPU 候选锁，不能用于 GPU 授权。ORIGINAL_G3_MODIFICATION_SITE_AUDIT.json 实读并锁定 4 个原文件的 21 处接口，CURRENT_SOURCE_MIRROR_REFS 来源核验在相邻 g3_source_readonly 目录；前后整锁核验分别在 SERVER_SOURCE_4050_UNCHANGED.json 和 SERVER_POST_CPU_SOURCE_4050_UNCHANGED.json。

真实生命周期审计确认：prepare_store_kv 延迟目标 store 至下一次原引擎 step；OffloadingConnectorWorker.shutdown 会清除未提交 store，不能把 shutdown 等同已刷 SSD。下一有限正确性流程应为同私有 SSD 根的原 producer seed→不共前缀 flush→目标写入完成/原 shutdown/OS 会话排空→独立新 consumer。新进程防止刚写入的 shared staging 命中掩盖 SSD read。未来 off、shadow 各一对，共四个 GPU 作业，仅为计划形状，无授权。

本轮真实 GPU 运行：0。上一轮 off05/shadow05 两次 G2 已实际成功，总生成 512 tokens；那只证明完整输出与生命周期，不证明 SSD、真实 KV offload、GPU 计时、运行中释放额度、策略收益。G2 两次限定授权已消耗，不能再用来校准或重试。
本轮预算账本仍为 229 events / 347,318 bytes / SHA 7dd465ef868fb6cc4fbacddd8cc58045cc47b11aa0768765325613b454c6b744，累计 GPU guard wall 17,206.34363487875s，原 28,800s 剩余 11,593.65636512125s（约 3.22h），active=null。数据盘可用 12,789,006,336 bytes（约 11.91 GiB）。没有驱动、系统、安装包修改、下载或删除已有实验数据；SOURCE/预算实际收尾命令与 JSON 已保存。

下一允许阶段仍是 CPU：审核复用原成本校准入口，完成来源绑定的有限 GPU 校准运行包；实现并核验真实 canonical KV 在原 D2H/H2D 完成和消费者计算前的有限被动捕获接线、目标 flush 与 fresh-consumer 闭环。完成这些具体准备后才申请新的限定 GPU 阶段。当前 GPU 阶段因新校准/捕获接线未就绪及无新限定授权而未执行。P4 效果和 P5 性能仍未验证。

服务器证据目录：
/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g3-production-kv-io-20261002。
本地同字节交付目录：
C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server09_candidates/g3_production_kv_io。
交付逐文件清单与两侧核验回执以最终 DELIVERY_MANIFEST.json / REMOTE_DELIVERY_VERIFICATION.json / LOCAL_AND_REMOTE_DELIVERY_VERIFICATION.json 为准；不将未同步的其他本地诊断算作服务器实测。
