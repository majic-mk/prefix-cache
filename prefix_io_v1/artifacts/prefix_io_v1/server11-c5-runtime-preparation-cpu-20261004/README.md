# C5 通知启动接线：仅 CPU 准备

本目录从冻结 runtime_v4 的 8 个文件复制建立，只在新副本修改启动接线。没有修改 C5/C4 reactor、collector、policy、receipt、作者源码或旧实验记录。`BASELINE_INHERITANCE.json` 保存原 8 文件引用和继承状态。复制的旧 `test_*.py` 及 `V4_CALIBRATION_BINDING_NOTES.md` 是祖先材料；其 C4/v6 资格不属于本目录，也不作为新 C5 测试通过统计。

所有 GPU/原生资格入口无条件返回 `GPU_BLOCKED_NO_C5_RECEIPT_AND_CPU_ENVIRONMENT_QUALIFICATION`。三个 CLI 不传配置也返回可读 JSON 和 exit 2，在配置文件读取、预算保留、torch/vLLM 导入或设备访问之前停止。直接调用 execute_window、load_configuration、verify_guard、controller load_receipt/freeze/prepare/launch/after、verify_runtime 同样抛出此阻断。没有命令行开关、旧 scope、旧 receipt 或环境变量可以解除。新命名空间是 `server11-c5-notification-{off,shadow,on}-preparation01`，未生成 GPU 作业、预算授权或 SCOPE 文件。

## 实际准备的接线

1. 新 runtime 指向冻结 C5 reactor/collector；native bridge/capture `run_id=LABEL`，前端 request 仍用独立 `rid=LABEL-p0-B`。验证器分别核对 native owner run、前端 request 和原生 native_request_id，仍调用原 128 帧验证算法。没有删除冻结 C5 的同 run 身份检查。
2. 原 bridge.attach_single_file_capture 后调用共享 `prepare_notification_capture`。只有 on 执行真实 C5 `capture.attach_single_file_wait`；核对同 reactor/capture/bridge 的弱引用、实际源码 SHA、真实原队列和四个 invalidation hook。off/shadow 不安装新等待，也不包装 Queue.get。保留每个新 reactor 仅一次测量 capture 的限制。
3. adapter 在同一原 Queue 实例上透明观察原 `queue.Queue.get`，只在已有 armed token 时记录最多 16 条值记录。保持原参数/timeout、原函数一次调用、原错误传播、原 intake/pump 和资源规则。记录原 arrival/freshness 最早截止、token、匹配/旧 wake、其他原消息、原截止超时、end record 的可得 host 时间和清理/故障。通知类型没有原因字段，匹配 wake 不能被说成一定来自 end，更不能证明 CUDA 完成。真实 end 在 get 前清注册并先入队的合法竞态允许 nullable 注册，但必须返回匹配原 wake。
4. 原 capture.detach 后保留 audit/capture 引用，原 drain 完成后才关闭观察并导出/验证。异常收尾保留原 drain/shutdown，之后撤除观察。观察器只保存 weakref、原 unbound function 和有界标量，恢复时不覆盖后来安装者。任何观察溢出/故障或未清理注册均拒绝证据。只有 defer 计数、旧 wake 或其他消息不会被提升成“通知已验证”。

## 本轮 CPU 证据的界限

`test_notification_runtime_preparation.py` 实际调用共享 launcher helper、冻结 C5 的 source-extracted install/wait/intake/原 reserve-preview-drain 和真实 Python Queue/线程。事件、frame、native receipt/物理 I/O 元数据是明确的 SYNTHETIC fixture；四个 observer hook 在基础 fixture 中是 bound synthetic 方法。测试没有执行真实 vLLM `collector.install`、模型或 GPU。独立 review 另覆盖原 observer/128 帧契约。新测试不重跑既有 score，不提供 timing 资格，不声称改善吞吐/延迟。

off 保持的是“C5 内新等待未安装”：无 Queue.get 包装、无通知注册。C5 已有 `_run` 的未安装 early return 仍存在；不能据此声称对 C4/原作者完全零开销。on 的值观测也有尚未测量的开销，未来真实资格及成本校准须包含最终实际源码闭包。

冻结 C5 的 receipt 工厂仍严格绑定 C4/native-v6；本目录故意不修改、不绕过此事实。未来若具备足够 CPU 环境并另获限定 GPU 授权，需要独立新冻结 overlay/receipt 工厂与六窗口校准，绑定实际 C5 reactor、同 collector、model/Event/runner 和最终观测接线。collector 仍需在 receipt overlay refs 中满足原 bridge 校验，并另严格等于新 calibration plan 的 collector ref。可以继承原完整 128 帧算法和成本公式，但不能用旧 C4/v6 数字或本 CPU 时间充当 C5 成本。不得修改预算/阈值来强制产生延期。

## CPU 执行（服务器由 root 执行）

先只上传本目录顶层 11 个 `.py` 和 2 个 `.md` 文件。不要上传本机生成的三个 source-lock/manifest/inheritance JSON：服务器 C5 包含从 C4 继承的额外原生 Python 文件，必须在服务器上传完成后执行下面的 CPU freezer，绑定服务器实际全部 C5 `.py`。freezer 只读取父 runtime 的明确 8 个祖先文件，忽略其原 GPU CONFIG/SCOPE/日志/结果。它不生成 GPU 资格锁；已有生成文件时拒绝覆盖。本机早期 d1 锁及其完整源码保留在 `LOCAL_FREEZE_V1_PROVENANCE`，对应 `LOCAL_CPU_RESULT_01`；不可当作服务器最终闭包。

```sh
ROOT=/root/autodl-tmp/prefix-io-v1-handoff/project
D="$ROOT/artifacts/prefix_io_v1/server11-c5-runtime-preparation-cpu-20261004"
C5="$ROOT/artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003"
V4="$ROOT/artifacts/prefix_io_v1/server11-p4-single-file-runtime-v4-20261003"
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$D/freeze_cpu_preparation.py" \
  --candidate-root "$C5" --runtime-v4-root "$V4"
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$D/run_cpu_preparation.py" \
  --candidate-root "$C5" --author-root "$ROOT" --output-dir "$D/SERVER_CPU_RESULT_01"
```

输出目录必须全新。runner 在前后核对 `PREPARATION_SOURCE_LOCK.json`，保留实际测试日志、命令、解释器、源码锁 SHA 和测试统计；始终 `actual_gpu_runs=0`、`native_execution_verified=false`、`cpu_timing_qualification=false`。本机命令使用同 runner，显式将 author-root 指向原 CPU 依赖 source 目录；不能将本机结果冒充服务器执行。

本轮允许下一步是服务器 CPU 语义/独立审查交付；GPU 阶段仍阻塞，不自动申请资源、不修改系统、不删除数据。
