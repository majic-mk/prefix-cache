# P4-02 无卡阶段实际交付（2026-10-01）

本轮在 connect.westd.seetacloud.com:20739 的 /root/autodl-tmp/prefix-io-v1-handoff/project 实际执行，保持 CPU_ONLY。当前可独立用 CPU 闭合的接口准备、语义合同、反例验证与交付工作已完成。真实 GPU 运行0次；完整 P4、生产实现和方法提升均未验证。正常运行时的完整 step/output/drain 采集和生产资格仍未完成。

继续增量修改现成 py-kvcache 与作者 vLLM，保留精确 Prefix Cache、原成本准入、共享 staging、预加载、复制融合、原异步流水线及原 owner/执行器。共同修复、观测、研究建议、证据工具分开。新策略关闭后回到原路径；I/J 未资格仍保持原 U。P3 原有限 pilot 闭合结论保留、U 为原阶段最强独立参照；正向研究优势未证明，P5–P7 未启动。

实际改动位置（项目相对路径）：

| 范围 | 修改位置 | 资格边界 |
|---|---|---|
| 有界 ETA/候选咨询 | third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_eta.py、p4_bridge.py；native reactor.py | 原 whole-parent 成功 drain 才形成历史，无提前释放信用，production ETA=None |
| 原负载消息/启动 | p4_load_observation.py、p4_startup_evidence.py、p4_options.py；native vllm.py；作者 offloading/common.py、scheduler.py、worker.py | scheduled-work-v1 不能冒充 actual GPU decode，原 budget/FD/slot/iodepth 决策保留 |
| 配对语义/离线 CLI | p4_paired_measurement_verifier.py、p4_verified_cost_loader.py、p4_raw_pair_recorder.py；prepare_p4_raw_pair.py | V1/V2 字节/源/阶段/量子/ABBA/拆分绑定，完整轨迹与选择窗口分开，prepared only，production lookup=None |
| 可选观测接口 | p4_gpu_step_observation.py、p4_native_window_journal.py | CPU 注入接点，未安装真实 worker；model_forward 不等于完整 decode step |
| 次日入口 | prepare_p4_gpu_next_day.py、qualify_p4_native_gpu.py、prepare_p4_calibration_startup.py | 纯 CPU 外层门禁在旧唯一 GPU guard 前；当前 --launch 实际拒绝，exit78 |
| 原版本补全 | vllm-author-p4-02-cpu 的92份缺失 Python 源、七个旧 .so 精确字节锁 | 从相同 HEAD 现有 build 复制 Python，未安装/编译/下载；binary 只构造 spec，CPU 未执行 .so，ABI 未验证 |
| CPU 回归/交付 | tests/prefix_io_v1_p4_* 与 scripts 内 run/freeze/audit/storage/version/package_p4_02_* | 实际命令、失败尝试和原始证据保留，重复运行不累计 |

独立审查保留并关闭实际反例：边界 I/O 双计；原生 invalidate 后旧帧有效；host 时间倒退仍归属窗口；完整128输出虚假集中单步。未知状态不授成本/资源释放/生产资格。所有明确纯 decode 步（含非选中步）逐请求最多一token、输出等于 active/batch、batch 不超过完整 run 请求数。unknown/mixed/spec/async/dummy 拒绝；明确 prefill 前置记录只保留完整轨迹，不入选成本窗。native/full-decode 标签本身不证明真实 GPU 来源。

最终统一回归：1956通过、16跳过、0失败，70.01秒。按唯一(classname,name)计数，当前 P4 目录547项通过，单独 agent 重跑及历史12个旧 P3 fixture 不累加。跳过为13项 no-vLLM fallback 专用测试、3项 io_uring_setup errno1。真实 Linux AIO 已测试；CUDA 未初始化、GPU=0、无活动 reservation、进程 session 已完整收敛，最终冻结源与 GPU ledger 前后字节相同。

标量 owner CPU fixture 微基准（200次/轮、7轮）：off 的1/32 parent 中位约0.157/0.167微秒；shadow 的1 parent/1 work约57.296微秒、32 parent/64 work约1.874毫秒。较大 fixture 的成本是明天需要测量的风险；未证明2%开销目标、端到端吞吐/延迟/goodput或方法优势。

保护审计：P3 2724锁定输入、原21个 control及原 native、P4-01 70项输入保持一致。3048份注册缓存共2796552192字节逐项实际 SHA 保持一致，无额外注册bin。当前 CPU冻结2198项、未来GPU来源冻结2048项，源锁不授GPU资格。补丁将共同 CUDA UUID/旧flush、92份源补全、观测胶水、研究建议、测量准备分开；P4-01+旧author build 与 P3+author HEAD 两条准确base均实际正逆字节通过，CLI 另有准确旧源补丁证明。

实际版本：project HEAD=cc7898b1ba59d89ce7fdbb186ded21880f1adf08；py-kvcache=3abba7a502d553f6e7e2e58b92086487e3395d7e；author vLLM=817a7e3124f817cd6e549581d3e5483207a753a4。Python3.12.3；安装 metadata Torch=2.11.0、pytest=9.1.1、numpy=2.3.5。metadata 查询不加载 GPU backend，不代表 CUDA/ABI 资格。完整 SHA 在 P4_02_VERSION_LOCK.json、test-input-lock.json、gpu-source-lock.json。

空间只读快照：PRIMARY free=10256891904 bytes，128MiB原生资格预留通过；原模型3GiB预留在8GiB floor下缺1554268160 bytes（约1.448GiB）。批准 AUX 实际唯一inode占用20095848448 bytes；20GiB cap下新3GiB预留缺1842237440 bytes（约1.716GiB），floor亦失败。候选审查只找到7MiB潜在收益、PRIMARY收益0，不足解除模型阻塞；未申请或执行合并。磁盘会变化，明天用实时同一门禁，不任意降低原轮次3GiB预留。

新增GPU秒数/预留均0；账本累计16520.562887秒，8小时上限内余12279.437113秒（约3.411小时）。无GPU探测、模型权重读取、下载、安装、构建、驱动/系统修改、缓存合并/删除/移动。

实际执行主命令见 EXECUTED_COMMANDS.json（含精确命令、返回码和agent记录）。在上述项目 cwd、CUDA_VISIBLE_DEVICES=''、PYTHONDONTWRITEBYTECODE=1下执行：

    .venv/bin/python -I -S experiments/prefix_io_v1/scripts/freeze_p4_02_sources.py
    PYTHONPATH='third_party/work/prefix-io-p4-02-cpu/src:experiments/prefix_io_v1/scripts' .venv/bin/python experiments/prefix_io_v1/scripts/run_p4_02_cpu_qualification.py --name current-integration-01
    .venv/bin/python -I -S experiments/prefix_io_v1/scripts/audit_p4_02_cpu_preservation.py
    .venv/bin/python -I experiments/prefix_io_v1/scripts/record_p4_02_version_lock.py
    .venv/bin/python -I experiments/prefix_io_v1/scripts/check_p4_02_storage.py
    .venv/bin/python -I -S artifacts/prefix_io_v1/server08-p4-02-cpu/EXECUTE_FINAL_CPU_PREVIEWS.py

证据根目录：/root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/server08-p4-02-cpu。最终统一 XML、guard、result 在 current-integration-01/；版本/保护/存储在本目录对应JSON。成本最终源及原始CLI证据在 cost-verifier/source-lock-v2-final.json 与 fixtures-v2-cli-01/；独审在 eta/review-cost/v2-independent-02/、v2-cli-readonly-01/，eta/review-integration/observer-counterexamples-03/、nested-binary-spec-01/、patch-roundtrip-06/、FINAL_INDEPENDENT_CLOSEOUT_V2.json。真实5条预览/拒绝命令在 gpu-next-day/FINAL_CPU_CLI_RESULTS.json；空间候选在 storage-next-day/。失败尝试和修复前反例均保留，最终测试数量只来自统一 XML。

明天下一允许阶段：有卡并具备匹配 permissions/批准UUID/源锁/context 的 GPU scope 后，先 G1 新原生off/shadow（各180秒+20秒收敛，共计划400秒、每项128MiB），验证真实来源、binary ABI、CUDA/AIO、完整parent/共享预加载/内容/shutdown。随后在原作者 runtime 补接并资格验证完整step/output/drain、时钟、actual load及合法单阶段成本。自然耦合 D2H→write 不能伪造两种单阶段cell；成本/ETA/load/真实释放安全资格完整后，才可激活研究建议及同一共同固定预算四臂加独立U的小pilot。原模型轮次空间仍阻塞。详细入口与门禁在 GPU_NEXT_DAY_RUNBOOK.md；P4完成及方法提升保持开放。
