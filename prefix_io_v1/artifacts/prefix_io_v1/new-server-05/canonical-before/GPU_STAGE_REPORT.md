# GPU stage — native model / GPU Prefix passed; complete P1 blocked

Current server connect.westc.seetacloud.com:24801; only approved GPU-8b500efe-1a50-0e8e-b21e-716807eebedf. Authorization stays 8 GPU hours /20 GiB model downloads. System/driver changes, rentals, payment and remote publishing remain disabled.

Native-prefix-04 passed with pinned author vLLM 817a7e3124f817cd6e549581d3e5483207a753a4 and verified official ModelScope Qwen2.5-7B-Instruct. Same engine, identical128-token input, two16-token greedy outputs: cached0→112 and exact output equality. Actual model and KV are BF16. Native worker metadata measured28 unique storages totaling66,977,792B under67,108,864B configuration,73 blocks/1,168tokens. These are backing bytes, not allocator reserved or release witness.

Three earlier failures remain recorded: long IPC path; missing Ninja PATH during sampler JIT; explicit BF16 cache string rejected by the author's Triton entry. Minimal project runtime fixes now use short IPC, native supported chunked prefill, auto KV with BF16 assertions, project Ninja/private NVCC and FlashInfer workspace/no-download setting. No model/cache/kernel implementation replaced. Run02's default FlashInfer cache outside the project is disclosed and retained; later runs use the private location.

Cumulative GPU budget285.24748319387436seconds =0.07923541199829844hours; remaining7.920764588001702hours. This delivery239.97443519160151seconds. Failures,JIT,startup/shutdown and five NVML-only checks count toward this delivery's budget. All new sessions drained, active reservation null, postrun approved GPU has no compute process. Conservative download charge19,422,798,722B; application payload returned15,242,868,376B; remaining2,052,037,758B.

Native io_uring still EPERM, real handler/staging/SSD/production-KV lifecycle unverified. Final preflight exit2. The passed native short synthetic diagnostic is not full P1, SSD/ITL/throughput or research-benefit evidence. Staging backing and live release witness remain unknown. Observer inactive, policy off, P2–P7 closed.

See NEW_SERVER_03_REPORT.md and REPRODUCE_NEW_SERVER_03.md. Earlier reports are historical snapshots. Next permitted phase remains P1 after platform io_uring support and native integration prerequisites.

## 第四轮续办（2026-09-27）

本轮GPU作业0、下载0；权限与预算账本字节未变。io_uring_setup再次返回EPERM，preflight仍BLOCKED，P1未验收。32项CPU测试通过并生成候选校准计划，未执行实际校准。GPU累计仍285.247483194秒/8小时；下载保守扣费仍19,422,798,722 B/20 GiB。详情与命令见 NEW_SERVER_04_REPORT.md、REPRODUCE_NEW_SERVER_04.md。
