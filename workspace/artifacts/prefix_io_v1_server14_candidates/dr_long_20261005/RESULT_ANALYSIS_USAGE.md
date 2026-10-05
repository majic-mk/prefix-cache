This CPU script analyzes the existing original P316 output files. It performs
no experiment, GPU operation, native submission, model import, download or fit.

The run index is a JSON list. Each row supplies `label`, `arm`, and local paths
for `config`, `result`, `adapter`, `guard`, `frozen_config`, and `source_lock`.
Paths may be relative to `--project`, or absolute. Preserve the original file
bytes when downloading. A missing result or frozen configuration may be `null`;
the row remains a failure, with its actual guard/error information retained.

Effect labels are `U1`, `DR1`, `DR2`, `U2`; arms are `U` or `F8+D_R`.
Calibration labels may be `CAL01`, `CAL02`, etc., with arm `CAL`. Only the
predeclared adjacent U1/DR1 and DR2/U2 pairs are compared, with actual guard
execution order verified. Calibration is excluded from effect estimates.
The original calibration `all_hit` pattern is checked separately from the
original effect `mixed_readwrite` pattern. The F8 configuration is the exact
preserved P316 `fixed8-control`: cumulative SSD/D2H are 8Q, cumulative H2D
64Q, inflight SSD 64Q, and inflight copies 1152Q. Shared limits are also
checked; the original limits are not changed by analysis.

`ACTUAL_DOWNLOADED_RUN_INDEX_01.json` contains only the four downloaded
CAL01..04 failures. `PLANNED_RUN_INDEX_TEMPLATE_01.json` additionally names
future U1/DR1/DR2/U2 paths under the actual `u01/dr01/dr02/u02` directories;
those template entries are not experiment evidence. Save each actual
effect config unchanged as its directory's `config.json` when downloading.
Do not include a future template row in the actual index until its run
exists. A failed run with no output can be included with missing leaves.

```text
python -B -I -S analyze_dr_long_results.py --project PROJECT --runs RUN_INDEX.json --gpu-uuid GPU-UUID --common-domain-sha256 SHA256 --clock-proof MONOTONIC_TOKEN_CLOCK_SOURCE_01.json --output NEW_ANALYSIS.json
```

The primary metric is the unchanged original `response_seconds`, from first
scheduled arrival to all complete outputs. Other reported values are the
original cohort duration including drain, tail drain, generated-token
throughput, SSD read/write bytes, native H2D accepted-byte delta, and each
request's latency and original within-request ITL p95. TTFT subtracts the
scheduled arrival from the first monotonic engine token event only after the
frozen source-clock refs and actual same-process worker binding pass. It never
subtracts the wall-clock `metrics.arrival_time`.

Validity requires full matching 128-token outputs, the same frozen workload,
sampling, initial cache, model and common resource controls, successful native
startup, original shutdown, zero native/AIO work, ended workers, and an empty
original guard OS session. Failure rows remain visible and excluded from valid
pairs. Runtime source and configuration refs are checked using downloaded
metadata and raw file bytes; model weight files are not rehashed.
The guard argv must exactly match the known server project, interpreter,
frozen runner and config; only the frozen runner's relative path or its
exact known-server absolute path is accepted. Shutdown counter summaries
must equal their saved original snapshot; closed/drained/fatal are read
from that full shutdown snapshot.

A proposed permutation or enabled flag cannot prove D_R ran. The analysis
requires an inverted pair of real ready work IDs, actual restore dependencies,
the original queue-read receipts, and matching `user_data`/FD identities in
the successful original Linux `io_submit` prefix. Zero proven changes is
`NOT_EXERCISED`; proposed changes lacking a complete join remain explicit.
Even with a faster combined arm and proven ordering, U versus F8+D_R cannot
isolate D_R from the fixed-stage contribution. No service SLO, statistical
significance, production prediction qualification or allocator release credit
is claimed.

The six local CPU tests passed. They exercise exact frozen guard metadata,
exact frozen configuration,
arithmetic and minimal scalar
join logic and preserve missing-result failures; they contain no passing
current GPU model/guard/lifecycle fixture. Actual current GPU effect files
must still be supplied before this script can report an effect comparison.
`ACTUAL_U_DESCRIPTIVE_ANALYSIS_02.json` now contains one verified downloaded
U01 baseline (U1) and the four failed CAL attempts, with no valid effect
pair and no improvement claim.
