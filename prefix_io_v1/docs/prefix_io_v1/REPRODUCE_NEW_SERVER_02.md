# 新服务器第二轮：已执行命令与复现入口

工作目录为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。真实命令、标准输出、退出码保存在本轮 artifacts 与 `commands.jsonl` 快照；下列为主要入口，不表示应重复消耗预算。所有重新运行使用新 label，保留旧证据。

## 源码构建与公共补丁

固定作者 vLLM 817a7e3124f817cd6e549581d3e5483207a753a4，工作树 `third_party/work/vllm-author-build`。原始 checkout 保持干净，不能换成上游预编译 wheel。锁见 `experiments/prefix_io_v1/locks/dependency-lock.json`、`runtime-build-hashed.txt` 与 `build-source-lock.json`。

实际成功构建的完整脚本是 `artifacts/prefix_io_v1/new-server-02/author-build-04-command.sh`；build04 exit 0。当前 `build_author_vllm.sh` 在其基础上增加唯一 label、私有临时目录和日志防覆盖，经过 bash 语法检查，但这份后续脚本不是 build04 的原始执行版本。

```bash
# 仅在需要重建时使用新名称
bash experiments/prefix_io_v1/scripts/build_author_vllm.sh author-build-replay-01
```

作者源码公共 UUID 补丁：`patches/prefix_io_v1/common/vllm-author/0001-full-gpu-uuid.patch`。已安装工作树中已应用，不要重复 apply。它对全部实验臂一致，不改变研究策略。

## 已执行的局部资格检查

所有 GPU 作业均通过下列 runner，批准 UUID 由 permissions.yaml 读取；完整实参和耗时在 gpu-budget-ledger.json。再次执行须使用新的 label。

```bash
.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label author-copy-replay-01 --seconds 120 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_author_copy.py --library third_party/work/vllm-author-build/vllm/_C.abi3.so --gpu-uuid GPU-8b500efe-1a50-0e8e-b21e-716807eebedf --output experiments/prefix_io_v1/runs/author-copy-replay-01/copy.json

.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label author-import-replay-01 --seconds 120 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_author_imports.py

.venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label author-platform-replay-01 --seconds 60 -- .venv/bin/python experiments/prefix_io_v1/scripts/qualify_author_platform.py --gpu-uuid GPU-8b500efe-1a50-0e8e-b21e-716807eebedf
```

这些检查只验证各自记录的算术、原始字节复制、符号/API身份和平台元数据，不验证生产 KV、真实 handler、模型推理或 SSD。

## CPU 检查与部署门禁

本轮 runner 21 项和 UUID 16 项 CPU 测试已通过。历史 388 passed / 8 skipped 是 new-server-01 的完整回归，不能记为本轮重新运行。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-prefix/bin/python -m pytest -q tests/prefix_io_v1_runner
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-prefix/bin/python -m pytest -q tests/prefix_io_v1_vllm_platform
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m prefix_io_control.preflight --controller docs/prefix_io_v1/templates/controller_spec.yaml --permissions experiments/prefix_io_v1/configs/permissions.yaml --capabilities artifacts/prefix_io_v1/new-server-02/capability-report.json
```

当前 preflight 真实 exit 2：io_uring unavailable、real handler unverified。模板是本项目合同，不是 vLLM 参数。P1 未验收，P2–P7 门禁保持关闭。后续先按真实可达性准备锁定模型；平台提供可用 io_uring 后再重跑原生 ring/open 检查以及 staging、SSD restore/store、生产 KV 与生命周期验证。不能以同步 I/O 或另一引擎替代验收。

模型预备校验补充：prepare_qwen_download.py 是离线计划校验器，不是下载器；26 项 synthetic CPU 测试证据在 new-server-02/model-review/plan-cpu-verified.xml。实际权重下载仍须先锁定账本、检查剩余额度并持久预留；未结算或不确定时停止。当前官方 revision/manifest 未取得，不能跳过这些前提执行 smoke。

原生 GPU Prefix 待执行命令及前提见 artifacts/prefix_io_v1/new-server-02/native-prefix-preparation/README.md。其 frozen-pending-plan.json 中 revision=null，不是可用于运行的模型 manifest。此准备脚本尚未在 GPU 或实际模型上验证。
