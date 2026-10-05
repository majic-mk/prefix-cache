# Server07 P3 独立内容验证复现

所有 GPU 操作在 connect.westd.seetacloud.com:38819 的
/root/autodl-tmp/prefix-io-v1-handoff/project 进行。不要删除旧输出，也不要重用已存在的 run 标签。

本轮实际完整命令保存在 artifacts/prefix_io_v1/server07-p3-04/commands.json，
环境保存在同目录 environment-overrides.json。
GPU UUID 由 run_gpu_stage.py 从 permissions.yaml 读取，累计预算由 gpu-budget-ledger.json 串行维护。

## CPU 重检

在项目根目录设置 PYTHONPATH 为 third_party/work/py-kvcache-p2-aio:src，并清空 CUDA_VISIBLE_DEVICES。使用新的输出文件名：

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/third_party/work/py-kvcache-p2-aio:$PWD/src" \
.venv/bin/python experiments/prefix_io_v1/scripts/run_aio_cpu_tests.py \
-q --import-mode=importlib tests/prefix_io_v1_pilot tests/prefix_io_v1_native_costs \
--junitxml artifacts/prefix_io_v1/server07-p3-04/recheck-new.xml
```

最终本轮为 89 passed。CPU 测试禁止 torch.cuda 初始化。

离线重算分析时，同样设置以上 PYTHONPATH 和 CUDA_VISIBLE_DEVICES=""。
使用 validate_heldout_costs.py --reference-only --plan <reference-plan.json> --out <新文件> 校验数值；
使用同脚本 --plan <cost-plan.json> --out <新文件> 检查成本。
成本检查预期返回 exit=1、FAILED_HELDOUT_POINT_PREDICTION，不应为了通过而改变 25% 阈值。

## GPU 原始顺序

1. server07-p3-heldout-native-ref-01：cold，--native-hot-diagnostic --diagnostic-logprobs。
2. server07-p3-heldout-populate-01：populate，新建独立内容存储。
3. server07-p3-heldout-cached-ref-01：paired，--cached-reference-logprobs。
4. server07-p3-heldout-cold-cost-01：cold，不带数值诊断选项。
5. server07-p3-heldout-paired-cost-01：paired。
6. server07-p3-heldout-paired-cost-02：paired。
7. server07-p3-heldout-cold-cost-02：cold。
8. server07-p3-2k-drift-control-01：paired，原标定内容。
9. server07-p3-2k-drift-control-02：paired，原标定内容。

每次均由 .venv/bin/python experiments/prefix_io_v1/scripts/run_gpu_stage.py --label <标签> --seconds 180 -- 包装。

前七次公共参数：
--sizes 2048 --reps 3 --domain 16384
--prompt-manifest artifacts/prefix_io_v1/server07-p3-04/heldout-prompts.json
--storage experiments/prefix_io_v1/runs/server07-p3-heldout-storage-01
--model-dir models/Qwen2.5-7B-Instruct-ms-16c174980d8a1492910551634b4969e69cdc2444
--model-plan artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json
--output-dir experiments/prefix_io_v1/runs/<标签>/details

后两次不传 --prompt-manifest，使用 server07-p3-long-storage-01 现有标定缓存。
所有路径、标签和完整数组均以 commands.json 为准。

同一 manifest 的参考与测量必须使用同一模型、预算、家族和源缓存；分析不允许混用成本诊断 logprobs 时间。运行后始终检查 wrapper result.json、details/result.json、I/O 结算与会话清理。

## 新数据与安全边界

最终 prepare_heldout_validation.py 需要 --out <项目 artifacts 下的新目录>。
本轮最初生成清单时目录固定在源码内，之后仅将输出路径参数化；清单在所有 GPU 执行前已经冻结，字节 SHA256 未变。

当前数据盘不足以重新 populate 一份同等规模的完整缓存，已有 source 和原始证据保持不动。新增长前缀数据的系统盘目录方案仍待授权。不要为了复跑而降低 8 GiB 底线、删除历史数据或绕过预算包装器。不同物理存储位置的成本必须单独核验。

这是单 token 成本验证；不把任何数字用作持续 decode goodput、研究策略收益或最终论文结果。
