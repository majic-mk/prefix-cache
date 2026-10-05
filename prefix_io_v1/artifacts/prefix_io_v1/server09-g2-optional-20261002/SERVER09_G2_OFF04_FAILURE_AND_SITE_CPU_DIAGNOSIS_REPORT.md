# Server09 G2 off04 真实失败与 CPU 启动环境诊断

本交付记录已获直接人类授权的 off04 实际运行及失败。shadow04 未启动，未借其名额重试。真实结果不改写为通过；CPU 修正另存独立候选，不修改此负结果。

## 真实 GPU 运行与预算

- 作业：server09-g2-normal-off-04；GPU：GPU-54ace1eb-dc9d-3fee-d720-2051f5d337a2。
- 原守卫限时 300 秒、收尾预留 20 秒。实际 GPU 守卫记账 127.51177137065679 秒；exit=child_exit=1；没有 timeout、signal 或守卫 error。
- reservation_id=9f37659840394c668e72b4a5238dcb88；session_id=11928；cleanup 前后 OS 会话成员均为空，session_drained=true。
- 正常模型结果 FAILED_NORMAL_MODEL_LIFECYCLE，phases=[]。四片权重真实加载（日志 2.25 秒、14.29 GiB），原采样器真实 JIT 与 GPU KV 初始化已到达；没有完整 frontend token 输出。
- 错误：ValueError: unknown path hook。原可选缺包异常进入修复 wrapper，随后在 helper._validate_search_path 中被全局 hooks 数量约束拒绝。进程内可选查询包装已恢复 restored=true。
- LLM 构造尚未返回，没有 original_engine_shutdown_returned=true。OS 会话排空是实际证据；source_lock_unchanged_after_original_shutdown=true 是现有 finally 的标签，只证明源码收尾，不能证明构造失败时正常原引擎 shutdown。
- 新账本227事件，原226事件逐项完全不变；唯一新 event 与原守卫 result.json 相同。累计16929.545041699428秒、剩余11870.454958300572秒（约3.30小时），active_reservation=null。原8小时预算未重置。
- 本轮 GPU 作业1，shadow作业0；后续 CPU 诊断新增GPU作业0。PRIMARY 可用12856795136字节；私有缓存实际分配27893760字节。不删除已有数据，不改系统、驱动、已安装包或作者源文件。

## 实际 CPU 诊断与根因

先以 CUDA_VISIBLE_DEVICES='' .venv/bin/python -B -I -S 只读列出六份 .pth 及其来源源码。随后使用与实际 GPU 入口相同的正常 .venv/bin/python -B 启动，仅导入 stdlib，确认没有 torch/vllm/py_kvcache 框架导入、模型运行或编译。

真实正常 site 启动具有三个 sys.path_hooks：zipimporter、标准 FileFinder，以及 __editable___vllm_0_1_dev1_g817a7e312_finder._EditableNamespaceFinder._path_hook。第三项来自已安装的 editable vLLM finder，源码4717B，SHA-256 11446ee6db26a2901e8df4e15096cc393a2e819426cdd397efe803aa6d2f2753。它处理 editable namespace 的 placeholder，不负责当前唯一实际作者目录。

在正常 site 的 CPU 进程中，标准 PathFinder 对真实 canonical third_party 父目录查询 None，实际 parent cache 已是标准 FileFinder；旧 helper 仍仅因 len(sys.path_hooks)==3 而抛出完全相同的 unknown path hook。该失败已在 CPU 复现，不需要再消耗 GPU 来定位。

此前66项CPU合同在 -I -S 下通过，-S跳过site/.pth，未覆盖这个真实启动差异。纯 source-only preflight 也不会执行 runtime parent-cache 校验。这是新增共同修复的预检缺口，不能归咎为缺失 DeepGEMM 必须安装、驱动问题、GPU资源不足或方法本身无效。

最小修正方向：在原 query 已通过原 finder 对 sole canonical parent 查询之后，仅验证那个必须存在的标准 cached FileFinder、路径、loader 与无实例覆盖，并在二次 PathFinder 查询前后确认同一缓存实例。保留所有现有 site hooks，不放行未知 parent importer，不改变原 strict finder/作者源，也不执行未知的后续 meta finder。具体修复与测试另行冻结，不能使用旧04授权运行。

## 已执行命令

真实 launch：
~~~sh
cd '/root/autodl-tmp/prefix-io-v1-handoff/project' && .venv/bin/python -B artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/run_g2_normal_model_lifecycle.py --project '/root/autodl-tmp/prefix-io-v1-handoff/project' --mode off --name server09-g2-normal-off-04 --source-lock artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/gpu-source-lock-candidate.json --scope-record artifacts/prefix_io_v1/server09-g2-normal-worker-optional-probe-v3-20261001/GPU_STAGE_AUTHORIZATION.json --launch
~~~

外层仍调用原 run_gpu_stage.py --permissions-path experiments/prefix_io_v1/configs/permissions.server09.g1.yaml --label server09-g2-normal-off-04 --seconds 300，再运行固定 --execute 子命令。完整 argv 在原 result.json 与 ACTUAL_OFF_LAUNCH_RECEIPT.json。

实际 CPU 命令及 stdout/stderr/exit 原样保留在 ACTUAL_AUTHORIZED_OFF_CPU_PREFLIGHT.json、ACTUAL_CPU_SITE_PTH_INVENTORY.json、ACTUAL_CPU_SITE_HOOK_SOURCES.json、ACTUAL_CPU_NORMAL_SITE_PATH_HOOKS.json、ACTUAL_CPU_OLD_V3_SITE_FAILURE_REPRODUCTION.json 和 ACTUAL_POST_OFF_CPU_SOURCE_AND_SHADOW_GATE.json。off source-only 门禁在真实运行后通过全部4045 refs；shadow04 的新 scope preflight 实际exit78，原因 off must pass and drain before shadow。

## 版本与证据

本次冻结源码678aa560117eae06e9e8fd4fc1cb9f3cdf483766bfbf1658c70dcc55b3b5ea15（897904B/4045refs）；直接人类回复来自 call_hb3C9ZGsu8wzfuX5FbeJH9Yt，答复“授权这 2 次能力查询修复版 G2 验证并继续”。human2459B/6b0501f8f849590b950be9808f7f1a86bd03696615195d21e9a0ce4b366f00e5；scope9603B/a1401a3b63e98030efa4da4e06bf89d26e3f3670c89a6e415e9ad84403638fdf。独立复核确认19个scope字段与当前runtime template逐项、逐类型一致，temperature保留float0.0。

原始 run/result1410B/SHA821ff0d80b558414de7bc642c22f7c16974c057b9a090618190083d0022f4340；process.log9774B/SHAeb03eda6399f1aed9db362e2e4a968d0b609bae8c199784d73d04c95e83369d5；normal结果16768B/SHA3e5007f9fe594ab6a6b4726a7447bec27acd9bb411985c020fac2a6210fb85fc。下载实际字节、账本与唯一事件核对在 RAW_OFF_EVIDENCE_DOWNLOAD_VERIFY.json、LOCAL_RAW_OFF04_FAILURE_CLOSURE.json。

本地：C:/Users/mamengkui/OneDrive/文档/论文2/artifacts/prefix_io_v1_server09_g2_optional_20261002
服务器：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server09-g2-optional-20261002

## 下一允许阶段及阶段口径

当前只允许继续CPU共同修正、真实正常site启动回放、独审及冻结交付。off04已失败，shadow04条件未满足；旧源码/名称绑定的授权不能用于任何重试或修复版。

按用户交接包03：P4是 dependency_only/interference/joint 最小新增策略、shadow/fake-backend与授权小规模真机验证；完整单模型请求流与四臂性能比较属于P5。此前将四臂性能对照简称“P4性能对照”不够准确，在此更正。G2正常模型/生命周期仍未通过，P4 GPU策略资格与P5正式收益均未完成。

CPU修正通过后，需要单独的新正常模型有限GPU资格授权；之后才能推进P4实际I/O/释放/计时/策略真机资格。成本、SSD往返、GPU完整模型、GPU计时映射、真实运行中释放收益及方法性能提升目前都不能宣称通过。

