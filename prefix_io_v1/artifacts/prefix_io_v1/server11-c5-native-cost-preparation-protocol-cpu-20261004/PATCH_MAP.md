# C5 common 校准接线只读审计

本目录只检查新 `notification_v5_native_cost_preparation_cpu`，不修改该实现或任何冻结源码。结果是 CPU API/source 边界验证，不是模型、CUDA Event、Linux AIO、原生成本或 on 观测成本资格。common 校准维持 bridge=None，因而不依赖可消费 policy receipt，也不执行 on 通知。

| 边界 | 原路径与最小新接线 | 保持条件/检查 |
|---|---|---|
| 校准来源 | 原 v6 runner → 新独立 preparation runner；新 common_candidate 接纳 C5 reactor/collector 与 canonical receipt 模块 | 未重写引擎、owner、queue、pump 或事件执行器。GPU_UUID=None，所有 native/GPU入口仍 blocked |
| collector refs | source lock 的 files list → 按真实 row.path 建立 refs → `collector_source_view` 的局部副本 | 只有 historical legacy reactor lookup key 指向实际新 canonical row。row.path 不能改成旧路径；原 refs/list 不可修改；legacy-only/wrong canonical row 拒绝。唯一 source list 仍由父冻结/serializer约束，不把alias map当新的source lock |
| capture身份 | common journal owner=LABEL；原六窗每次 capture.run_id=外部 rid | 原 scalar observer 也接收 rid，原 native_request_id 继续用于每帧prepared/output。不能照搬 on runtime 的 capture=LABEL 契约，也不删除原serializer的 capture/front-end RID 相等检查 |
| 原128帧 | C5 collector.install → 原 G2 connect_worker_observation → 原prepare/execute/sample observer | max_pending/max_steps=128，源哈希由实际C5 alias传入；原同adapter身份/前后reactor SHA/128 frames检查保持。无新增GPU query/wait/synchronize；CPU测试中的Event与connector全部synthetic |
| 生命周期 | 原 collector.install、attach_prepare、detach | 原prepare返回值与一次调用保持；安装失败正常detach；detach只恢复自己的wrapper，不能覆盖后来安装者。common安装不设置wait reactor/registration，不安装通知Queue observer |
| 数学与类型 | 新canonical `prefix_io_control.p4_single_file_receipt` 由factory负责独立接线；common校准本身无receipt消费 | 原 original_estimator、validate_capture、validate_io、original_post_shutdown_drain、analyze_paired AST保持；A-only预算和holdout规则不变。canonical typed identity不能由第二个同名class/dict替代。此typed身份由factory及独立review核验，本审计不私构issuer |
| source group与128帧 | 实际新 serializer 的 validate_common_source_binding；新 common_candidate 的 reactor/collector逐字节继承C5 | synthetic ref reader 返回实际两文件的bytes/SHA；检查新job/overlay/CPU origin、owner bridge=None/parent8、实际module来源以及前后native SHA和全部128帧。旧job/hash/collector、错误owner、未验证原方法、无reactor、127帧/bool帧数/adapter改绑拒绝。原 verify_raw_window 的 capture/native-request/journal三个实际调用AST保持 |

这组公共接线只改变候选/collector的真实来源归属和安全的 CPU-only 入口包装。source view 的 legacy key 只供既有 collector.install 查 native_source_sha256；不允许它成为存储路径替换、全局模块别名、原始帧篡改或新I/O实现。没有证据表明需要修改原128帧算法或原数值算法；若独立检查发现二者改变，应先停下解释差分，不能默认算作兼容性修复。

真实 C5 collector API 的 CPU 测试使用 `origin=cpu_fixture`，替身实现仅记录 common.load_ref/load_pinned/connect_worker_observation 的调用及参数，原 collector.install/prepare-wrapper/detach 的实际源码执行。源检查另外抽取新 runner 的实际 collector_source_view，source-group检查抽取新serializer的实际validate_common_source_binding并使用明确synthetic ref reader；其引用由真实C5源文件计算，不代表任何GPU/raw native资格。synthetic Event 在 native_gpu_recording 来源检查中必须失败。这些测试没有调用真实 worker connector、模型或 torch.cuda.Event，也没有颁发成本/receipt。

共同成本只覆盖 bridge=None 下的共同原生路径，包括 C5 未安装等待时的分支；不能因完整source闭包列出新receipt/collector而宣称 active policy、通知或Queue observer成本已测。后续仍需真实设备来源、六窗raw、原公式和holdout、新授权与完整CPU环境资格，再按 off→shadow→on 有限阶段推进。所有预算、129/128/offset16/917504B/parent8和阈值保持不变；本审计不发作业/scope，不重新运行性能score。

## CPU执行

runner固定本目录三个源文件，不把运行时 COMMAND/RESULT/日志列入源码。服务器必须给外层 `c5_native_cost_preparation_cpu_source_lock_v1`，其中各资格标志=false、GPU UUID=null。前后检查全部动态files，结果保存相同source_before/source_after及实际hash/count。

```sh
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" -B -I -S "$PROTOCOL/run_cpu_common_protocol.py" \
  --source-lock "$LOCK" --project-root "$ROOT" \
  --preparation-root "$ROOT/artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004" \
  --candidate-root "$ROOT/artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003" \
  --native-root "$ROOT/artifacts/prefix_io_v1/server11-native-cost-v6-20261003" \
  --output-dir "$PROTOCOL/SERVER_CPU_PROTOCOL_01"
```

ROOT/PROTOCOL/LOCK由父流程绑定实际新目录和锁；不是GPU命令。输出目录必须新建。本机未带外层锁的结果明确记录null，不能冒充服务器完整闭包。最终测试数与结果以CPU_PROTOCOL_RESULT.json和完整日志为准。
