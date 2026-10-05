# P4 CPU 复现

项目为 `/root/autodl-tmp/prefix-io-v1-handoff/project`。使用已锁的 `.venv/bin/python`，不使用系统/base Miniconda pytest。完整实际 argv/env 在 `current-integration-01/plan.json`，源码锁在 `test-input-lock-v2.json`。GPU 继续关闭，HF/Transformers离线；不需要下载、安装或编译新依赖。

实际已执行：

```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH='/root/autodl-tmp/prefix-io-v1-handoff/project/third_party/work/prefix-io-p4-01-cpu/src:/root/autodl-tmp/prefix-io-v1-handoff/project/experiments/prefix_io_v1/scripts' \
.venv/bin/python experiments/prefix_io_v1/scripts/run_p4_cpu_qualification.py --name current-integration-01

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -I -S experiments/prefix_io_v1/scripts/qualify_p4_patch_roundtrip.py --name patch-roundtrip-01

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
.venv/bin/python -I -S experiments/prefix_io_v1/scripts/prepare_p4_gpu_stage.py \
--project . --output artifacts/prefix_io_v1/server08-p4-01-cpu/gpu-launch-denied-v3.json --check-launch
```

结果分别为统一测试退出0、patch退出0、GPU启动拒绝退出78。旧输出不可覆盖。以后有具体修复需要重跑时，先冻结新源码锁并改用新 receipt 名称；统一 CPU runner 的 `--name` 使用新值，scratch自动落入原批准的 `experiments/prefix_io_v1/runs`。当前70份源码必须与v2锁一致；若改变代码，旧测试资格不得借给新版本。

独立guard命令详见 `policy-production-contract-cpu-tests-02-command.json`、`independent-value-abi-cpu-tests-02-command.json`、`native-bridge/native-bridge-cpu-handoff-01.json`。它们证明单元与接口，已经包含在统一179项内，不重复加数。

GPU准备命令不授予执行权限。当前CostTable生产gate未开放，真实producer/verifier/表loader/native production应用未完成。本包只能复现CPU合同与fake事件行为，不能用CPU mock重建GPU资格或性能结果。
