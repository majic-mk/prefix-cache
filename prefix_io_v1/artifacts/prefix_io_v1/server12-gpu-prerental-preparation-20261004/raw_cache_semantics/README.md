V4 只修正共同缓存语义，不修改 executor、collector、issuer、validator 或统计估计器。

原真实 scheduler 与 PrefillStats 源表明：前端 `num_cached_tokens` 记录 local+external cache 命中；完整512外部命中结束后，原 scheduler 为下一 token logits 将执行 `num_computed_tokens` 降为511，调度最后1个 prompt token。因此 `CACHED_TOKENS=512` 仅用于原暖启动/前端断言；`cached_prompt_tokens=511` 是兼容字段名，显式 semantics=`initial_execution_pre_context`，另外列出 `frontend_cached_prompt_tokens=512` 与 `initial_execution_pre_context=511`。不能解释成 GPU 本地缓存512。

512 prompt、offset16/context527、AB/BA/AB、2 calibration+1独立 heldout、完整128真实输出/帧门槛均保留。cal02 尚无 measured capture，只是语义失败证据，不能当成本或性能结果。source 推导的511/1仍须新的真实 cal03 全帧验证；不接受未知或少帧。

5项实际 CPU 回归通过：执行原 PrefillStats.set 与 scheduler 完成分支 AST；原 PreparedFrame 值拷贝显示首帧511/1、decode527；原 validator 的128项结构回放和512执行上下文/缺帧拒绝；descriptor 语义；所有原16、v2/v3/node helper/collector/issuer/validator 字节不变。明确 CPU fixture，GPU/模型/事件/qualification 均0。

服务器命令（project cwd）：

```
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S D/raw_cache_semantics/test_cache_semantics_cpu.py
CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S D/runner/strong_native_cost_runner_v4.py --help
```

root 需以 V4 和新增 CPU 文件重新冻结 V7、plan/config/intent 与新 cal03 标签，不改旧 cal02 记录。
