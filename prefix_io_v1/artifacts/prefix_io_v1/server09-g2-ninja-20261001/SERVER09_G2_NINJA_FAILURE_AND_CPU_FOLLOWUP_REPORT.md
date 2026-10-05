# Server09 G2 Ninja 修复版：实际失败与 CPU 后续
生成依据：本次真实 off03 作业、原字节下载、独立审核与服务器只读收尾；本次未启动 shadow03，未重试 off03。

## 实际执行与结果
- 2026-10-01，原预算守卫在 GPU UUID GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2 上执行 off03 一次，耗时 131.6455060718581 秒，exit/child_exit 均为 1；未超时、未收到中断信号，OS 作业会话最终无残留成员。
- 真实模型已加载四份 safetensors 分片（日志报告 14.29 GiB），CUDA 13 私有 SDK 与单个 Ninja 路径接线已实际生效。私有 FlashInfer 缓存中已有 3 个 ELF 对象文件与 sampling.so；服务器只读核对给出路径、字节数、SHA-256。此文件证据只证明该次生成编译/链接产物，不单独证明 GPU 内核或模型输出资格。
- 随后失败：ModuleNotFoundError: vllm.third_party.deep_gemm has no exact locked binary fallback。normal phases 为零，无 cold/repeat 完成文件，无 original_engine_shutdown_returned。
- result 中 source_lock_unchanged_after_original_shutdown 是 finally 内源码核验标签，不能证明原模型 shutdown。此次仅能证明预算守卫完成进程会话清理。
- GPU 后实际 CPU 源门禁核验了全部 4,040 个冻结引用；shadow03 的真实 CPU 门禁返回 78，原因 off must pass and drain before shadow。未运行 shadow GPU。
- 本轮仍未获得正常的 128-token 输出、跨模式一致性、SSD 生命周期资格或性能提升证据，P4 未完成。

## 授权、预算与存储
- 人类实际回复：“授权这 2 次 Ninja 修复版 G2 验证并继续”。授权绑定 source lock SHA-256 7b8445363e985a7dc4bff08960b31ee4cc325e3d309322f7f4b15d5b0f8aa423，以及 off03 成功后才可启动 shadow03 的先后条件。
- 本次增加一次真实 GPU 事件；累计 16802.03327032877 秒，原 8 小时预算剩余 11997.966729671229 秒，active_reservation 为 null。旧预算事件保持原序和值。
- 收尾 PRIMARY 可用 12889706496 字节，私有运行缓存实际独占分配 27893760 字节；维持 8 GiB 空间底线。
- 不下载、不删除、不修改系统、驱动或已安装包；没有新 GPU 实验、同名重试、研究策略启动或预算重置。

## 实际改动与 CPU 检查
本轮新增的是实际授权记录、启动回执、日志/结果与预算快照、只读源码/缺包/生成缓存审计、独立审核和本报告。执行代码使用已交付冻结 Ninja 候选，无新的研究策略改动。此前服务器 CPU 51/51 通过；该统计是 48 项回放加 3 项新增 Ninja 合同检查，不是 51 项全新测试。

服务器只读确认作者目录确实没有 deep_gemm 包、Python/bytecode/extension 各候选，原 PathFinder 对这个固定名称返回 None；七个已锁二进制也没有 DeepGEMM。原作者 import_utils.has_deep_gemm 查询外部包或这个可选 vendored 包，_has_module 使用 find_spec 的 None 语义。冻结加载器把不存在的 vllm 模块一律转为缺少锁定二进制的异常。CPU 已复现同一异常；当前 GPU 日志没有完整调用栈，因此直接抛错位置以已有真实错误与源码审计为界，不虚构 traceback。

下一允许操作是纯 CPU 准备并审核狭窄的可选能力查询适配，不改变作者引擎、数值路径或原严格加载器。任何改动后的 GPU 验证须有新的冻结版本及具体授权；本轮失败的 off03 与条件未满足的 shadow03 不能借用于重试。

## 执行命令与证据
实际 GPU 启动命令：
```sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode off --name server09-g2-normal-off-03 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-cuda13-ninja-20261001/GPU_STAGE_AUTHORIZATION.json --launch
```
其内部由 experiments/prefix_io_v1/scripts/run_gpu_stage.py --permissions-path experiments/prefix_io_v1/configs/permissions.server09.g1.yaml --label server09-g2-normal-off-03 --seconds 300 执行原 child，另保留 20 秒收尾。

CPU 收尾使用 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S，执行冻结 runner --preflight，以及 stdlib PathFinder/文件字节审计；全部原样命令与 stdout/stderr/exit 在 ACTUAL_POST_OFF_CPU_SOURCE_AND_SHADOW_GATE.json、ACTUAL_READONLY_OPTIONAL_ABSENCE_AND_JIT_CACHE.json 和 source-readonly/ACTUAL_SOURCE_INVENTORY.json。

核心真实文件：
- runs/off/result.json（1394B，2be39debcff271f09bf88e7aa4168d22f9aaa105c72df85350ad79cbeef346e8）
- runs/off/process.log（9774B，ec994b78ebb8db48f98df293d7ac73e42d38fdf77095efad569383c16cb981cc）
- runs/off/details/normal-model-lifecycle-result.json（8885B，b50191a1d9161972f15eb7ca729d32a3f2dd77426335edb3953dc10c68260dd7）
- POST_OFF_GPU_BUDGET_LEDGER.json（342596B，9a4c7bd1f030ed236d2461b9e3ba3521d4ded139c6a5fc818aa05332cd15f8b1）
- INDEPENDENT_OFF03_FAILURE_REVIEW.json（9025B，0d77a3a9dbfc5c2288082d3e88a9475c6f19e31679985069e8bb6d8ca5ba162e）

本地目录：C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server09_g2_ninja_20261001
服务器交付目录：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-ninja-20261001
原作业目录：/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/runs/server09-g2-normal-off-03

