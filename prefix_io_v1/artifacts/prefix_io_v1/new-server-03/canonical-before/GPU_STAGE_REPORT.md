# GPU stage — P1 partial, model and SSD integration blocked

Current server: connect.westc.seetacloud.com:24801. The first round is authorized for 8 GPU hours, 20 GiB model downloads, only GPU-8b500efe-1a50-0e8e-b21e-716807eebedf and project-private dependency/experiment roots. System/driver changes, rentals, payments and remote publishing remain disabled. See permissions.yaml and new-server-02/authorization.json.

The fixed author vLLM source 817a7e3124f817cd6e549581d3e5483207a753a4 was successfully built for SM 12.0, with Torch 2.11.0+cu130 and project-private CUDA 13.0.88 compiler/CRT/NVVM. Original tracked sources remain clean. The common full-GPU-UUID compatibility patch is applied only to the build worktree and must be shared by every experiment arm.

Actual qualifications: base CUDA arithmetic/D2H, CPU copy ABI, guarded author native byte copies (334 B H2D, 1358 B D2H), 22 import/API/symbol identity checks, and specified-GPU platform metadata passed. Platform-01 failed before the common UUID fix; platform-02 and import-02 passed after it. These checks do not qualify production KV, model execution, native handler/Plan execution, SSD integration, lifecycle release or overlap.

Six GPU jobs including one failure consumed 45.273048002272844 seconds = 0.0125758467 GPU hours. About 7.9874241533 hours remain. Current active_reservation is null; revised-runner sessions were drained. Model weights downloaded: 0 bytes. The complete ledger and each command/result are in experiments/prefix_io_v1/gpu-budget-ledger.json and runs/.

The original LiburingRing(2) still fails EPERM; current preflight exits 2 for io_uring unavailable and real handler unverified. O_DIRECT success is not a replacement. Model cache checks found no local weights, and official Hugging Face metadata access failed; IPv4/IPv6 reachability and a read-only HTTPS DNS diagnostic also failed. No revision or weight hashes were invented, and no system networking was changed. Model execution / GPU Prefix smoke remain unexecuted.

The next permitted work is still P1: restore model-source access and prepare pinned weights within the existing budget, then execute the native GPU-only Prefix smoke; obtain platform io_uring support and pass native ring/open tests before real staging/SSD/store/restore and production-KV lifecycle acceptance. P1 is incomplete; P2–P7 are unaccepted and research policy is off.

See NEW_SERVER_02_REPORT.md for changes, commands, tests, evidence and remaining limits. Earlier delivery reports and before-GPU_STAGE_REPORT.md are historical snapshots, including previous zero-GPU or build-pending statements.
