# GPU stage — native model / GPU Prefix passed; complete P1 blocked

Current server connect.westc.seetacloud.com:24801; only approved GPU-8b500efe-1a50-0e8e-b21e-716807eebedf. Authorization stays 8 GPU hours /20 GiB model downloads. System/driver changes, rentals, payment and remote publishing remain disabled.

Native-prefix-04 passed with pinned author vLLM 817a7e3124f817cd6e549581d3e5483207a753a4 and verified official ModelScope Qwen2.5-7B-Instruct. Same engine, identical128-token input, two16-token greedy outputs: cached0→112 and exact output equality. Actual model and KV are BF16. Native worker metadata measured28 unique storages totaling66,977,792B under67,108,864B configuration,73 blocks/1,168tokens. These are backing bytes, not allocator reserved or release witness.

Three earlier failures remain recorded: long IPC path; missing Ninja PATH during sampler JIT; explicit BF16 cache string rejected by the author's Triton entry. Minimal project runtime fixes now use short IPC, native supported chunked prefill, auto KV with BF16 assertions, project Ninja/private NVCC and FlashInfer workspace/no-download setting. No model/cache/kernel implementation replaced. Run02's default FlashInfer cache outside the project is disclosed and retained; later runs use the private location.

Cumulative GPU budget285.24748319387436seconds =0.07923541199829844hours; remaining7.920764588001702hours. This delivery239.97443519160151seconds. Failures,JIT,startup/shutdown and five NVML-only checks count toward this delivery's budget. All new sessions drained, active reservation null, postrun approved GPU has no compute process. Conservative download charge19,422,798,722B; application payload returned15,242,868,376B; remaining2,052,037,758B.

Native io_uring still EPERM, real handler/staging/SSD/production-KV lifecycle unverified. Final preflight exit2. The passed native short synthetic diagnostic is not full P1, SSD/ITL/throughput or research-benefit evidence. Staging backing and live release witness remain unknown. Observer inactive, policy off, P2–P7 closed.

See NEW_SERVER_03_REPORT.md and REPRODUCE_NEW_SERVER_03.md. Earlier reports are historical snapshots. Next permitted phase remains P1 after platform io_uring support and native integration prerequisites.

## 第四轮续办（2026-09-27）

本轮GPU作业0、下载0；权限与预算账本字节未变。io_uring_setup再次返回EPERM，preflight仍BLOCKED，P1未验收。32项CPU测试通过并生成候选校准计划，未执行实际校准。GPU累计仍285.247483194秒/8小时；下载保守扣费仍19,422,798,722 B/20 GiB。详情与命令见 NEW_SERVER_04_REPORT.md、REPRODUCE_NEW_SERVER_04.md。

## 第五轮环境可行性裁定（2026-09-27）

NO_GO_CURRENT_CONTAINER：标准C与固定作者LiburingRing均在setup阶段EPERM；当前可访问SSH环境没有发现平台支持的容器管理入口。已记录用户有限io_uring授权，未进行系统变更。该结论仅限当前容器配置/管理权限，不否定架构在其他环境的可行性。P1仍未验收，后续阶段关闭。本轮GPU/下载为0，预算账本未变；详见NEW_SERVER_05_FEASIBILITY_REPORT.md。


## 2026-09-29 新服务器 07 GPU 验证

新指定 RTX5090 上作者真实拷贝、原 handler/reactor + AIO SSD、shared preload、pinned staging、真实模型 KV 字节往返及正常 wait/shutdown 通过。最终证据为 runs/server07-native-aio-kv-02，成功后不重写 CPU 备份，Prefix 0/112 与输出 token 一致。当前模型 scheduler 未接入外部 connector，AIO 成本曲线与 GPU 源块释放见证尚待验证；P1 仍 partial，不进入策略验收。完整范围、预算、命令见 NEW_SERVER_07_GPU_REPORT.md。


## 2026-09-29 server07 原生标定与规划器验证

详见 SERVER07_CALIBRATION_REPORT.md 和 REPRODUCE_SERVER07_CALIBRATION.md。本轮完成有界 P1 AIO 原生路径、实测成本、原 LoadPlanner 调用与自然 GPU 复用。短前缀 SSD 成本不利；未改变原准入或缓存执行器，未启用研究策略。P2 mandatory/release witness 和观察开销仍待验证。


## Server07 P3 两并发与 I/O 深度验证（2026-09-29）

本轮 19 个真实 GPU 作业全部完成，CPU 测试 182 passed/0 failed/0 skipped；另有一次离线成本门槛失败，阻止深度 2 两次计划重放。新增 GPU 记账 1580.150959 s，累计 8195.989781 s / 8 h，新增下载 0。收尾 GPU 计算进程为空、active_reservation=null。GPU 环境可行；当前阻塞为 P3 研究/集成证据未齐。 详见 [本轮报告](SERVER07_P3_CONCURRENT_REPORT.md)。
