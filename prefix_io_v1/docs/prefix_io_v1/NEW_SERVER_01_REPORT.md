# 新服务器交接与执行报告

本轮连接 connect.westc.seetacloud.com:24801，操作根目录 /root/autodl-tmp/prefix-io-v1-handoff/project。未保留登录密码到项目文件。

## 已实际执行

- 迁移核验：上一交付 manifest 的 120 个文件哈希全部一致；py-kvcache、作者 vLLM、experiments、simple-profiler 四个固定 SHA 一致且原始检出干净。
- 新硬件/容器审计：暴露设备 6，RTX 5090，UUID GPU-8b500efe-1a50-0e8e-b21e-716807eebedf；cgroup 内存上限 90 GiB、CPU 配额 25 核。设备存在已确认，当前可用显存/占用及计算能力尚未执行 GPU 查询或 kernel 验证。
- 新服重新运行 CPU 回归，初始 385 passed / 8 skipped。
- 只读审查发现 publisher owner 校验在异常失效范围外。先加入测试确认 3 个失败案例，再将校验纳入统一失效处理；修复后聚焦 37 通过，全套 **388 passed / 8 skipped**。同时修正文档对 observer patch 状态的陈旧描述。
- 独立 O_DIRECT 4096 字节往返通过；作者 io_uring 与独立 raw setup 均返回 EPERM。内核 io_uring_disabled=0，进程 Seccomp=2；原因仍需平台核实。
- 已重新生成 source/runtime capability 审计和只读 preflight；结果 BLOCKED。
- 已安装项目私有 uv 0.12.17，建立独立 .venv，保留 .venv-prefix 和系统 Torch 不变。作者 vLLM 要求 Torch 2.11.0，当前原 CPU 环境为 Torch 2.8.0+cu128，因此不能认定同步过来的环境已兼容。
- 官方 PyPI 候选解析在 45 秒限时内未完成（exit 124）；使用现有软件镜像的 HTTPS 地址成功解析 191 个依赖，并生成带哈希的候选清单。候选解析不等于作者 vLLM 已安装、编译或通过真实调用。191 项运行依赖随后在独立 .venv 中安装成功（exit 0），uv pip check 通过，Torch 实际导入为 2.11.0+cu130，项目内 nvcc 为 13.4。环境约 8 GiB；作者 vLLM 本体尚未编译/安装，Rust 1.95 等构建准备仍需完成。详见 runtime-install-summary.json；未将依赖安装冒充正式运行资格。

## 代码与证据

只改了项目内三处代码/文档：src/prefix_io_control/publication.py、tests/prefix_io_v1_observer/test_publication.py、patches/prefix_io_v1/observer/README.md。错线程、跨 reactor、缺少 worker 都会使旧快照永久不可用，不可借新鲜时间戳重新启用失效 publisher。

完整补丁、原件、先失败再通过的测试及执行命令位于：
artifacts/prefix_io_v1/new-server-01/review-fix/ 与 review-fix-before/。
补丁正/反向及 whitespace 检查通过。作者 py-kvcache 工作树、vLLM 源码、原生调度/资源所有权未改。新策略仍关闭。

其他证据均在 artifacts/prefix_io_v1/new-server-01/：

- migration-audit.json、environment-current.json；
- io-uring-diagnostic.json、odirect-probe.json；
- source-runtime-audit/、preflight.json；
- runtime-candidate-hashed.txt、runtime-resolution-mirror.txt、runtime-install.log；
- observation-overhead-cpu.json；
- commands.jsonl 与 review-fix/commands.jsonl。

CPU 微基准重新运行于新容器、修复后的代码：每轮 20,000 次、7 轮，合成 32 parent / 64 ring op / 64 copy。每次采集满窗口的轮均值中位数约 41.0 微秒，1 ms 采样诊断约 0.366 微秒/调用。它不运行设备 I/O、native pump 或模型，不能当作 GPU/端到端开销、干扰标定或相对旧容器的策略收益。

## 实际命令入口

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES="" RUN_E2E_TESTS="" \
PYTHONPATH=src:third_party/work/py-kvcache-observer \
.venv-prefix/bin/python -m pytest -q -ra --import-mode=importlib \
  -W error::pytest.PytestUnhandledThreadExceptionWarning \
  tests/prefix_io_v1 tests/prefix_io_v1_progress tests/prefix_io_v1_observer \
  third_party/work/py-kvcache-observer/tests
```

审计已执行：
```bash
.venv-prefix/bin/python experiments/prefix_io_v1/scripts/audit.py \
  --output artifacts/prefix_io_v1/new-server-01/source-runtime-audit
```
再次执行须选择新的证据目录，不覆盖本轮结果。GPU E2E 仍未启用；原作者 E2E 默认 80K prompt、32 GiB staging 且关闭 GPU Prefix，不能直接当成本合同的正式服务流设置。

## GPU 与仍需解决的门槛

**GPU 作业执行 0 次、0 GPU 小时；模型下载 0 GiB。** 软件依赖下载与模型下载分别记录。没有运行 nvidia-smi、CUDA kernel、推理或 GPU 模型测试；仅读取设备节点和 proc 驱动信息。

permissions.yaml 仍原样：GPU=false、模型下载=false，两个预算为空。已提出首轮“该卡最多 8 GPU 小时、模型下载最多 20 GiB、项目独立目录”的具体授权请求；未收到答复前不修改权限，也不把泛化的继续指令当成有限预算。

io_uring 阻塞独立于 GPU 预算。请平台按 PLATFORM_IO_URING_REQUEST.md 核查容器系统调用策略。当前没有修改系统/驱动授权，且现有进程的 seccomp 限制不能通过项目内补丁解除。

P1 仍未验收；P2 为 CPU 预备，P3–P7 未启动。下一允许动作：明确 GPU/下载权限和预算，并使原生 io_uring 可用；随后在独立环境完成锁定作者运行栈、copy/handler、原生模型及 GPU/CPU staging/SSD 各层 smoke。只有 P1 通过后才验收真实 mandatory/shadow 和 owner/fence/generation 适配。释放分析的 parent lifetime 身份约束必须在 live adapter 接入时明确，当前没有 live adapter，不宣称已支持真实释放 witness。
