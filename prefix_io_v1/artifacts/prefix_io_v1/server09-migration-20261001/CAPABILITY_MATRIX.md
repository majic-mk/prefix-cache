# Server09 能力矩阵（当前 CPU 证据）

| 能力 | 新服务器状态 | 实际证据与边界 |
|---|---|---|
| 克隆源码与 P3/P4 输入 | 通过 | 7,813 个文件实际 SHA，一共 3,283,686,731 字节；四份锁引用 2,724/70/2,198/2,048 |
| 注册精确 KV 来源 | 通过 | 3,048 份、2,796,552,192 字节，成员集合与 SHA 一致 |
| CPU 集成 | 通过 | 1,956 通过、16 跳过、0 失败；P4 目录 547 项通过，非额外累加 |
| Linux AIO | CPU 已验证 | 既有异步原执行路径测试通过；不是新服务器 CUDA/AIO 集成资格 |
| io_uring | 受限 | 3 项真实 io_uring_setup 返回 errno1，按实际结果跳过；本轮未修改系统 |
| 原 py-kvcache/作者 vLLM | 来源锁一致 | 固定原 commit＋已冻结增量；真实模型、binary ABI、GPU 缓存恢复尚未在新服务器运行 |
| GPU 分配身份 | CPU 元数据可读 | /dev 仅 nvidia6；proc 对应 GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2/RTX 5090；没有 NVML/CUDA/free-VRAM 验证 |
| 模型材料 | 元数据存在 | Qwen 四分片存在，config/index 可读；本轮未全量哈希权重，不构成模型加载资格 |
| G1 新权限路径候选 | CPU 通过、未应用 | 21 项服务器纯 CPU 测试；2,057 条当前及候选源真实校验；提案实际 --launch 在 GPU 前拒绝 |
| PRIMARY 模型储存门禁 | 通过 | 当前 13008269312 B；原 3 GiB reserve＋8 GiB floor 可满足 |
| AUX 模型储存门禁 | 阻塞 | 实际唯一 inode 20095844352 B；20 GiB cap/8 GiB floor 下 3 GiB 不满足；不扩大 AUX |
| 累计 GPU 预算 | 保持 | 已用 16,520.562887 s，剩余 12,279.437113 s，无活动预留、新增秒数 0 |
| 完整正常模型 step/output/drain collector | 尚未接通 | 作者 execute_model 返回 None 后还需 sample_tokens；原启动必须显式 async_scheduling=False；CPU 适配不授生产资格 |
| 生产 D/I/J 与方法收益 | 未验证 | 成本、ETA、actual load、真实释放及完整 GPU collector 门槛仍未闭合；维持原路径 |

历史 server07/server08 GPU 资格仅保留为历史结果，不自动转为 server09 资格。
