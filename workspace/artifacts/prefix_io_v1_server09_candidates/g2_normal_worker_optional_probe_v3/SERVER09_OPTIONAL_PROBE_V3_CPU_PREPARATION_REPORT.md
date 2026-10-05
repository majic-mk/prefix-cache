# Server09 最终可选能力查询修复：CPU 准备与下一阶段
本交付只完成 CPU 准备及冻结，未创建 off04/shadow04 的人类授权或 GPU 作业。现有 py-kvcache、作者 vLLM、原预算守卫、模型与研究策略范围保持原路线。

## 结论和当前阶段
真实 off03 已失败：131.6455060718581 秒、exit/child_exit 1、OS 会话排空、正常 phases 为零。权重加载、CUDA 13/Ninja 接线、采样器生成 ELF 产物与 KV Cache 初始化已进入真实路径；错误为 vllm.third_party.deep_gemm has no exact locked binary fallback。没有完整输出或方法提升证据，shadow03 因先决条件未满足未启动。

此次在服务器完成最终修复版 66/66 CPU 测试、0 跳过，以及全部 4,045 个冻结引用核验。预算仍为原累计 8 小时：已使用 16802.03327032877 秒、剩余 11997.966729671229 秒、226 个事件、active_reservation=null；修复开发增加 GPU 作业 0。PRIMARY 可用 12884910080 字节，下一阶段每次仅预留 128 MiB，保持 8 GiB 底线。

P4 的四组性能对照仍未开始。先要实际 off/shadow 两模式分别完成 cold/repeat 各 128 tokens、四份数组一致、缓存计数 0/112、原引擎正常 shutdown 和会话清理；随后还要真实 I/O、资源释放及计时资格，才能进入四组效果对照。本交付不能证明 GPU 完整模型、SSD、计时映射或性能提升。

## 实际修改
- 新增 optional_capability_probe.py，位于本次独立目录，不修改作者文件、原 BoundAuthorFinder、已安装包、系统或驱动。
- 仅在该私有进程内包装作者 import_utils._has_module：先调用原缓存函数。只有精确 vllm.third_party.deep_gemm 查询抛出原加载器的真正缺包异常，才在异常类型/名称/字符串参数、三个原始代码帧、同一原 finder、原始源码 SHA、实际缺包、唯一父路径及标准 PathFinder 校验后返回 False。
- 其余正常返回、缓存成功、外部 deep_gemm、未知异常、必需模块导入保持原行为。原 finder 仍严格拒绝未锁来源；不存在向后续未知 finder 放行 None 的路径。
- 新入口在原框架导入成功后、原 LLM 构造前安装适配；原 shutdown 后恢复，并保留后续外部覆盖。恢复失败记录失败且继续既有收尾。
- 增加受限 traceback 证据（24 层、最多 12,000 字符）；CPU 门禁不安装运行适配、导入框架、加载模型或编译器。
- 原 worker 整份字节不变：13090B SHA096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b。原 Prefix Cache、成本准入、共享 staging、预加载、复制合并、异步 I/O 路线与研究策略分离；本次仍 strategies=off。

## 测试与版本冻结
服务器实际命令使用 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S：
1. test_g2_worker_observation.py --source-root 项目目录 --scalar-source 固定 scalar --frame-source 固定 frame：22/22，0.046 秒。
2. test_g2_normal_model_lifecycle.py --source-root 项目目录：31/31，0.091 秒。
3. test_optional_capability_probe.py --source-root 项目目录：13/13，0.219 秒。
4. 冻结入口 --mode off --name server09-g2-normal-off-04 --source-lock 当前锁 --preflight：4,045 个引用通过、GPU0、运行 hook 未安装。
5. stdlib 原字节分块装配、原缺包查询复现、预算/磁盘/原源码收尾：全部通过。

66 是原 51 项回放加 15 项新增合同（13 个 helper 合同与 2 个入口接线合同），不能计为 66 个全新测试。本机 Python3.12 为 65 通过加 1 个明确 Linux symlink 跳过，服务器实际 Linux 已验证该项。
初期本机误用了 Python3.8，相关失败原样保留于未用于服务器最终入口的第一份 CPU 草稿；切换现有 Python3.12 后通过。独审发现未知错误字符串转换、异常子类读取 traceback 的传播边界，前两版 CPU 候选和证据保留，未对它们启动 GPU。本次 final helper 已收口这些问题，独审通过 13 项合同及 4 项必要异常反例。修复不保证接下来所有 GPU 初始化和内核都通过；这部分必须真实运行才能判断。

当前 lock：897904B / 4,045 refs / SHA-256 678aa560117eae06e9e8fd4fc1cb9f3cdf483766bfbf1658c70dcc55b3b5ea15。原 4,040 引用保持原序和原值，只追加新 runner/worker/plan/helper 四文件，以及旧 4,040 锁本身。声明仅覆盖这些冻结引用，不声称整份虚拟环境的每个文件都已锁定。

## 证据位置与下一允许阶段
- 实际 CPU 命令、stdout/stderr/exit：ACTUAL_SERVER_CPU_NORMAL_MODEL_TESTS.json、ACTUAL_SERVER_CPU_COMPLETE_SOURCE_PREFLIGHT.json、ACTUAL_SERVER_CPU_SOURCE_LOCK_ASSEMBLY.json、ACTUAL_SERVER_CPU_BUDGET_STORAGE_CLOSURE.json。
- 独审：INDEPENDENT_BOUNDARY_REVIEW.json、INDEPENDENT_OPTIONAL_PROBE_REVIEW.json、INDEPENDENT_FINAL_EXCEPTION_EDGE_RESULT.json；原外层预热 guard 的 CPU 阴阳性验证在 INDEPENDENT_ORIGINAL_OUTER_GUARD_CPU_RESULT.json。
- 上传返回结果在 ACTUAL_SERVER_CPU_SOURCE_UPLOAD_RESULTS.json。较大的全部原样上传请求/返回另外保留本地 ACTUAL_SERVER_CPU_INITIAL_UPLOAD_RECEIPTS.json；交付清单不复制这份重复编码原字节，CPU/GPU执行命令与结果均有正式独立记录。
- 真实 GPU off03 证据双侧冻结于 server09-g2-ninja-20261001（manifest21文件 SHA e6872ca21865561603635dd421fdfb681d733ff59cbccfcf086b58f5b4b13629）。
- 本地最终目录：C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server09_candidates/g2_normal_worker_optional_probe_v3
- 服务器最终目录：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001

下一允许 GPU 阶段须新限定授权：先 off04，成功且原 shutdown/会话清理合格后再 shadow04，最多 2 次、每次 300 秒加 20 秒收尾、总预留 640 秒。仅现有 Qwen2.5-7B、CUDA 13、Ninja 和本冻结代码，不下载、不修改系统/驱动、不删除已有数据。原 off03 已失败，shadow03 的条件未满足；旧授权绑定旧源码和名称，不能用于这次重试或修复版。

拟定实际启动命令（目前未执行）：
```sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode off --name server09-g2-normal-off-04 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/GPU_STAGE_AUTHORIZATION.json --launch
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode shadow --name server09-g2-normal-shadow-04 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/GPU_STAGE_AUTHORIZATION.json --launch
```
具体子命令继续使用原 run_gpu_stage.py --permissions-path permissions.server09.g1.yaml --seconds 300。所有模板 allow_gpu_runs/initialization=False、无新的 HUMAN_AUTHORIZATION_RECORD_G2.json 和 GPU_STAGE_AUTHORIZATION.json；此报告与提案不是授权。后续 P4 I/O/计时/四臂效果阶段不借用 G2 资格授权。

