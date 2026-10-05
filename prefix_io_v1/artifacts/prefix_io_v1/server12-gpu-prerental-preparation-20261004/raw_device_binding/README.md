本次仅追加 CPU 修复：原 raw binder 固定 `/dev/nvidia0`，当前容器实际节点为 `/dev/nvidia1`，GPU display index/CUDA logical index 不能作为物理 Device Minor。

`uuid_minor_device.py` 严格解析真实 `nvidia-smi -i UUID -q -x` 的 UUID/minor，仅接受同 UUID 的唯一记录，并检查对应 `/dev/nvidia{minor}` 非 symlink、真实 character device、NVIDIA major195 与 rdev minor 一致。保留完整有界 XML 原文、摘要及实际节点 metadata。备用 `/proc/driver/nvidia/gpus/*/information` parser 只供显式使用，不在查询失败时自动放宽。ctl/uvm、未知 UUID、重复字段、缺失 minor、普通文件和错误 major/minor 都拒绝。

`../runner/strong_native_cost_runner_v2.py` 与原 wrapper 只有 `bind_live_plan` 的 AST 不同：替换设备门槛并增加真实查询证据，完整原 UUID/source/permission/off/guard/budget/storage 门禁保留。仍在原 runner 目录读取 sealed `strong_trace_runner.py`，collector/activation 路径不迁移。原16文件和原 manifest 不修改。使用前 root 必须将新增 helper/v2 加入新的完整 source lock；CPU fixtures 不发 GPU 资格。

服务器 CPU 命令（project cwd，D 是新准备服务器目录）：

```
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S D/raw_device_binding/test_uuid_minor_device_cpu.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S D/runner/strong_native_cost_runner_v2.py --help
```

后续真实 plan 使用 v2 的 `--prepare-plan`，helper 也必须在其完整冻结源清单内。实际 `bind_live_plan` 仍只可在强 off 成功、原 shutdown/OS session 排空及源核验完成后执行。本轮未执行 RPC、GPU query、模型加载或 GPU 实验。
