# 新服务器第二轮交付：P1 局部 GPU 资格通过，完整 P1 仍阻塞

执行服务器：connect.westc.seetacloud.com:24801。项目根目录：`/root/autodl-tmp/prefix-io-v1-handoff/project`。所有开发、构建和测试在服务器完成；本地副本仅作交付。当前 P0 完成，P1 CPU 基础及本轮局部 GPU 检查完成，完整缓存集成未验收；P2–P7 仍受前序门禁约束。

## 实际修改

1. 从固定作者 vLLM `817a7e3124f817cd6e549581d3e5483207a753a4` 构建 RTX 5090 所需原生扩展，安装固定 py-kvcache 与 simple-profiler。项目私有 CUDA 编译器/CRT/NVVM 固定 13.0.88，CCCL 13.0.85；没有修改系统 CUDA 或驱动，没有改用其他缓存/模型执行器。
2. 修复作者 NVML 平台不能解析完整 GPU UUID 的公共兼容问题。补丁只改 `vllm/platforms/cuda.py`，对所有实验臂一致；数字设备编号保持原路径，未知 UUID 不降级到另一 GPU。作者原始仓库的跟踪源码保持干净，修改仅在工作树。
3. 加固项目 GPU runner：启动前持久化预算预留，按单调时钟核算，退出时清理整个子会话，包括另建进程组的后代。中断、超时、未完成预留或无法确认清理时拒绝继续。此工具不提供网络隔离或模型下载配额实现。
4. 新增作者原生复制、接口身份和平台元数据资格检查，记录精确字节、符号、模块路径及哈希。更新版本锁、环境锁、能力矩阵、修改地图、生命周期说明和阶段状态。观察桥接仍单独保存，研究策略未激活。

5. 准备固定 Qwen 模型的离线清单验证脚本。它验证 revision、精确大小、配置/index 哈希与权重 LFS SHA256，并检查 20 GiB 上限；不含网络下载，不修改账本，不预留额度。当前真实清单仍 BLOCKED / UNEXECUTED，不能直接当作可下载授权。

6. 准备原生 GPU Prefix smoke：固定 BF16、64 MiB 配置 KV 预算、两次相同的 128-token 输入和 16-token 输出，要求原生缓存计数 0→112，并严格比较输出 token IDs。显式仅 safetensors、拒绝额外未验证分片、检查本地哈希与有效 KV 配置。仅 CPU 编译/API/CLI/拒绝启动检查通过，模型与 GPU 运行未执行，实际 KV tensor 分配仍未知。准备证据和未来命令位于 new-server-02/native-prefix-preparation/。

原有精确 Prefix 身份、成本准入、共享 staging、预加载、复制合并和异步流水线未被重写。普通干扰额度和有限候选策略尚未越过 P1–P3 门禁部署。

## 版本与构建证据

| 组件 | 锁定版本/来源 |
|---|---|
| py-kvcache | 3abba7a502d553f6e7e2e58b92086487e3395d7e + 已记录公共补丁 |
| 作者 vLLM | 817a7e3124f817cd6e549581d3e5483207a753a4 + 公共 UUID 补丁 |
| kvcache-experiments | 0e023a84a21246b9bbc06266fa8070397eccbdc9 |
| simple-profiler | ec0d563bf68856df83c5824ac579700ec076b9e2 |
| Python / Torch | 3.12.3 / 2.11.0+cu130 |
| vLLM 安装元数据 | 0.1.dev1+g817a7e312；以作者源码 SHA 为准，不冒充其他上游发行版 |

构建 build01/02 失败记录保留；build03 在调整并行度时停止，随后清理包括独立进程组在内的所有后代。build04 成功，uv 报告准备/安装约 43 分 28 秒；这是 CPU 编译过程，不计作 GPU 推理时长。保存了 CMake、Ninja 366 条已观测动作、外部构建源 commit 和子模块清单。214 个已安装包的依赖检查通过。

原生 `_C.abi3.so`：30045664 字节，SHA256 `e30616d12f903169493f73c28e707540be17916e89794a9214b141d6e4769b94`。
哈希依赖锁：`runtime-build-hashed.txt`，SHA256 `ae3e456822d4e7dc0c3d4794867bb640a641a869d5293270d4d72c4e450f0fa0`。
UUID 公共补丁 SHA256：`661805f65fde5ed80f8f81faf963ac103ac4bda00238268a10d90ba7d1227ac5`。

## 实际测试

| 检查 | 结果 | 证据 |
|---|---|---|
| GPU runner CPU 回归 | 21 passed | new-server-02/runner-review/session-final.xml |
| UUID 公共修复 CPU 回归 | 修复前 8 failed / 8 passed；修复后 16 passed | new-server-02/platform-fix/cpu-before.xml、cpu-after.xml |
| 公共补丁校验 | 正向/反向应用、空白检查、作者固定 Ruff 均通过 | new-server-02/platform-fix/verification.json 与检查日志 |
| 模型下载离线计划校验 | 26 passed；仅 synthetic 输入，不是实际模型资格 | new-server-02/model-review/plan-cpu-verified.xml |
| 原生复制 CPU ABI | 零长度调用和非法参数拒绝通过 | new-server-02/author-copy-abi.json |
| 基础 CUDA | 1024 元素算术及 D2H 精确比较通过 | runs/cuda-base-01 |
| 作者原生复制 GPU | 334 B H2D、1358 B D2H，guard 精确比较通过 | runs/author-copy-01/copy.json |
| 作者接口导入 | 22 项符号/类型/来源能力检查通过 | runs/author-import-02/qualification.json |
| 指定 GPU 平台 | 修复前实际失败；修复后 UUID、RTX 5090、12.0 能力元数据通过 | runs/author-platform-01、author-platform-02 |
| 原生 LiburingRing(2) | 真实失败：EPERM | new-server-02/io-uring-current.json |
| 部署预检 | exit 2，仍有 io_uring、真实 handler 两项阻塞 | new-server-02/preflight.json |

上述 new-server-02 相对路径基于 `artifacts/prefix_io_v1/`，runs 相对路径基于 `experiments/prefix_io_v1/`。历史完整 CPU 回归 388 passed / 8 skipped 在 new-server-01；本轮没有将它伪记为重新运行。

复制测试只验证记录的原始字节缓冲区。导入检查没有构造实际 handler/ring。二者都不证明同一生产 KV 的 store/restore、真实资源释放、mandatory 排空、模型输出、逐 token ITL 或流水线重叠。

## GPU 与下载预算

本轮授权为 8 GPU 小时、20 GiB 模型下载，仅 `GPU-8b500efe-1a50-0e8e-b21e-716807eebedf`。系统改动、租赁、付费和远程发布保持禁止。

| 实际作业 | 退出码 | 计费墙钟秒 |
|---|---:|---:|
| cuda-base-01 | 0 | 3.268902 |
| author-copy-01 | 0 | 3.436855 |
| author-import-01 | 0 | 12.987586 |
| author-platform-01 | 1 | 6.188231 |
| author-platform-02 | 0 | 6.390838 |
| author-import-02 | 0 | 13.000635 |

合计 **45.273048002272844 秒，即 0.0125758467 GPU 小时**，剩余约 **7.9874241533 小时**。失败作业已计入。当前无未结算预留；新 runner 作业均确认会话排空。机器可见设备并未被全部用于测试。

模型下载 **0 字节**，20 GiB 额度未消费。源码、工具链和 Python 依赖安装与模型权重下载分开记录。服务器限定模型/cache 位置未找到现成权重；官方 Hugging Face metadata 请求失败，IPv4 TCP 超时、IPv6 不可达，追加 HTTPS DNS 只读查询被重置，没有取得可信 revision/文件清单。未改 DNS、代理或路由，也未替换模型源。证据：new-server-02/model-review/hf-connectivity.json、hf-address-diagnostic.json、hf-doh-diagnostic.json。模型执行和 GPU Prefix 命中仍未验证。

## 命令、限制与下一允许阶段

主要入口和精确构建脚本的区别见 `REPRODUCE_NEW_SERVER_02.md`。GPU 实参/返回值见 `experiments/prefix_io_v1/gpu-budget-ledger.json`；主线程远程命令快照位于本轮 artifacts 的 commands.jsonl（截止时刻见 command-snapshot.json）；专项审查目录另存各自命令日志。失败和中间结果保留，不覆盖为成功。

目前存在两个独立环境阻塞：

- 官方模型源不可达，不能冻结并取得 Qwen/Qwen2.5-7B-Instruct 的真实权重清单。因此原生模型与 GPU Prefix smoke 尚未运行。
- 作者原生 io_uring 仍 EPERM，因此 CPU staging、SSD restore/store、真实 handler/Plan 调用链、生产 KV 往返和完整生命周期均未验收。O_DIRECT 通过不能替代此项。

下一允许动作仍是 **P1**：模型源恢复后，在剩余下载预算内冻结实际 revision/哈希并准备本地权重，再以预算 runner 执行原生 GPU Prefix smoke；平台提供 io_uring 能力后，先重跑原生 ring/open 测试，再验证 staging、SSD、生产 KV 与释放协议。平台需求见 `PLATFORM_IO_URING_REQUEST.md`。这些条件未满足前不能宣布 P1 完成或进入 P2–P7 的真实验收，不能绕过为另一套 I/O 或模型执行器，也不作性能/论文收益结论。
