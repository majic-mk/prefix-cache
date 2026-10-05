# Server12 finite native recalibration preparation

Current server: connect.westd.seetacloud.com:24828.
Project: /root/autodl-tmp/prefix-io-v1-handoff/project.
Observed GPU: GPU-a67d2f38-8e1b-29c3-7e5b-375e1410f8ac; driver 580.95.05.

This directory changes only the site binding and its source-bound SDK metadata.
Original G/common, author model execution, sampling, cache ownership, SSD inputs,
shared staging, asynchronous copies, drain and shutdown remain inherited.

The server12 migration audit actually checked all 4,779 source/asset hashes.
Existing server11 normal CPU results (100 tests) are retained as historical CPU
evidence; they were not rerun on this server. Existing real 580 CPU SDK evidence
was revalidated with the unchanged SDK helper; no compiler, shared library or
GPU framework was loaded.

There is no fresh GPU execution authorization in this directory.
The old GPU receipt remains valid historical evidence only; its UUID and driver
differ from the current hardware. Old numerical cost and budget are not reused.

Proposed next job: server12-c5-native-common-cost-gpu01.
Exactly one guarded job, six fresh original model processes/windows,
A0/B0/B1/A1/A2/B2, 128 complete generated tokens in each window.
Two calibration pairs and one independent held-out pair; no held-out refit.
Execution limit 1,200 seconds plus 20 seconds cleanup; reserve 1,220 seconds.
Existing Qwen2.5-7B and 24 production KV inputs are reused offline.
Primary storage reserve 2 GiB, minimum remaining free space 8 GiB.
Stop at any failure; no retry under the same grant.

The original eight-hour cumulative GPU ledger remains authoritative.
No model download, rental, payment, system/driver/package edit, author source
edit, deletion, budget reset, new scheduler or performance claim is permitted.

After targeted CPU checks and the live source/resource context are frozen,
the human reviews that exact context and grants this job explicitly.
Only after a real new calibration and lifecycle validation can normal
off/shadow/on qualification be separately bound. P4/P5 improvement remains
unverified; a CPU fixture cannot qualify a native receipt.

