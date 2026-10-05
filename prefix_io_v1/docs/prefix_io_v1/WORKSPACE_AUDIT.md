# P0 workspace audit — 2026-09-26
Server workspace: /root/autodl-tmp/prefix-io-v1-handoff/project. No existing server project, permissions or ancestor AGENTS.md were found. The isolated checkout starts from the user's current committed HEAD cc7898b1ba59d89ce7fdbb186ded21880f1adf08 on codex/prefix-io-v1. Local dirty/untracked work is neither imported nor edited; inventory: artifacts/prefix_io_v1/p0/local-status-before.txt. No reset to the historical SHA.

All 12 handoff hashes match PACKAGE_CHECK.json. All normative documents and templates were read. vLLM AGENTS.md was read; vLLM remains unmodified and no upstream PR is proposed.

The permissions template is copied unchanged to experiments/prefix_io_v1/configs/permissions.yaml. GPU/model-download authorization remains false, budgets/approved roots null. SSH access authorizes this isolated project workspace, not GPU use.

Python 3.12.3, preinstalled Torch 2.8.0+cu128, CUDA toolkit directory 12.8, driver metadata 595.71.05. No /dev/nvidia devices; cgroup memory limit is 2 GiB. Host MemTotal is not usable capacity. No models or GPU pools were instantiated.

Server GitHub transfers failed/time-out, including HTTP/2 framing errors. Fixed sources were transported through the local machine; audits, implementation and tests execute on the server. Partial download directories are preserved and excluded from builds. Canonical selected source: third_party/upstream/vllm-author.

Candidate preload@d6eadf416bb5234047760bf55d532f2f038cf697 lacks PlanCandidate/PlanDecision/PlanOutcome. Author with_profiling@817a7e3124f817cd6e549581d3e5483207a753a4 provides them and manager.plan_candidates call sites. This is a source-qualified candidate, not an installed/GPU-certified combination.

Untouched upstream tests: 258 passed, 2 failed, 9 skipped. Two CPU staging tests fail because missing vLLM discards the Torch import. Five E2E GPU tests are disabled, three io_uring tests skip with EPERM, one needs SciPy. Evidence: upstream-baseline.txt/xml.

P0 audit is delivered before production changes. P1 CPU repairs/tests are allowed. P1 integration/GPU acceptance is BLOCKED; later phase gates stay closed. See capability-report.json and dependency-lock.json.
