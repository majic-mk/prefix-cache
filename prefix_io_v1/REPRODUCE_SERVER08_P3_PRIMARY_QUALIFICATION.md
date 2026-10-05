# P314 复现与证据定位

服务器项目根为 /root/autodl-tmp/prefix-io-v1-handoff/project。代码与命令在服务器执行；本机仅 SSH 传输与 ZIP/SHA 验证。所有已完成 GPU/CPU label、result 与交付文件均 append-only，不直接复跑旧 label 或覆盖证据。

完整实跑命令、环境、退出码及 stdout/stderr 在 artifacts/prefix_io_v1/server08-p3-14/*-command.json；executed-command-index.json 为总索引。GPU 冻结 argv 在 execution-plan.json、jobs/*.json、model-off-plan.json，CPU 具体子进程 argv 在各 PRIMARY CPU evidence/plan.json。GPU 只经 unchanged run_gpu_stage.py 扣累计账，新的 run 必须取新 label/output，并重新冻结计划/输入 SHA、核 UUID、空闲/无进程/预算/磁盘余量。

CPU 全矩阵实跑：

    PYTHONPATH=src:experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/prefix_io_v1/scripts/run_shadow_cpu_qualification.py --label server08-p3-14-registration-matrix-02

复跑 CPU 时只替换为未用 label。该 wrapper 生成完整命令与新 temp root，禁止 CUDA 初始化，分别路由 qualified shadow 与历史 order worktree。最终新设施/历史 AST 子集实跑环境包含 third_party/work/py-kvcache-p3-shadow-cpu:src:experiments/prefix_io_v1/scripts，38 passed；错误的初始环境/collection-error 已留存，不计新唯一 case。

13 次采集顺序与 gate 位于 execution-plan.json。acquire_native_aio_costs.py 参数由对应 jobs plan 完整记录：four knots 2048,4096,8192,16384，reps3/domain16384/seq1/depth4；随后 heldout 16256/seq2/depth8，只读取旧 frozen content manifest。native-hot/logprobs 与 cached-reference-logprobs 只用于单独数值参考，未混入性能拟合。fit-cold-01/paired-01/paired-02/cold-02 四独立引擎 fit 共72 raw rows。

实际离线 gate（以下输出已存在，重算应另选新文件，不能覆盖）：

    .venv/bin/python experiments/prefix_io_v1/scripts/analyze_cached_references.py --plan artifacts/prefix_io_v1/server08-p3-14/fit-reference-plan.json --output artifacts/prefix_io_v1/server08-p3-14/fit-reference-result.json
    .venv/bin/python experiments/prefix_io_v1/scripts/export_uniform_calibration_candidate.py --plan artifacts/prefix_io_v1/server08-p3-14/fit-cost-plan.json --out artifacts/prefix_io_v1/server08-p3-14/calibration-candidate
    .venv/bin/python experiments/prefix_io_v1/scripts/validate_heldout_costs.py --reference-only --plan artifacts/prefix_io_v1/server08-p3-14/d8/reference-plan.json --out artifacts/prefix_io_v1/server08-p3-14/d8/reference-result.json
    .venv/bin/python experiments/prefix_io_v1/scripts/qualify_concurrent_pilot.py --plan artifacts/prefix_io_v1/server08-p3-14/d8/cost-plan.json --result artifacts/prefix_io_v1/server08-p3-14/d8/cost-result.json --manifests artifacts/prefix_io_v1/server08-p3-14/d8/manifest-index.json --output artifacts/prefix_io_v1/server08-p3-14/d8/permit.json

离线环境：CUDA_VISIBLE_DEVICES=，PYTHONDONTWRITEBYTECODE=1，PYTHONPATH=third_party/work/py-kvcache-p3-order-cpu:src:experiments/prefix_io_v1/scripts。GPU 环境在 gpu-environment.json，wrapper 设置批准 UUID/HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE。

成本 gate 通过后唯一追加 model-off-plan.json：原 native model route，显式 --storage-registration 与新 permit 同 source，3 GiB reservation，LoadPlanner on，新增研究策略未启用，既有被动 observation 保留。10请求×128 token按P313 same-GPU full native cold/hot reference分别核对。compare_model_off_v2.py和其已记录的command验证输出/schema，不重跑GPU。首版错误分析 model-off-comparison.json 保留。

来源 CPU 复制脚本 copy_frozen_primary_source.py 的一次实跑及校验位于 primary-source-copy-command.json；原AUX origin未改。注册 JSON/source-manifest/copy-proof 位于 PRIMARY runs/server08-p3-14-source。不要复跑一次性复制脚本到已有目录，也不使用EXDEV通用fallback。校准来源 complete SHA清单为 calibration-source-manifest.json。权重与缓存 payload 留在服务器，不打包下载。

最终模型/两来源/执行输入保全、idle GPU和预算在 post-GPU-preservation.json；model-timeline-drain-audit.json核逐token与原native AIO排空；gpu-events.json是14个真实wrapper事件。source-lock-final.json和version-lock.json为结束后的交付版本快照；阶段执行锁记录各次冻结输入，预算账本自然逐次变化。

补丁 patches/prefix_io_v1/0009-primary-source-registration.patch 相对冻结before原脚本/历史test增量。patch-roundtrip.json包含真实 git apply check/apply 与全部目标SHA对照。旧作者工作树补丁/HEAD沿P313，不重新实现引擎。

交付打包仅执行一次：

    PYTHONPATH=src:experiments/prefix_io_v1/scripts PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= .venv/bin/python experiments/prefix_io_v1/scripts/package_primary_qualification_delivery.py

P3仍未完整完成。下一阶段先CPU accepted parent/failed-draining/unknown-aware owner快照与真正admission协议，再逐级原native阶段预算qualification。P4–P7保持关闭，不把此成本曲线/单baseline迁移验收当研究收益。
