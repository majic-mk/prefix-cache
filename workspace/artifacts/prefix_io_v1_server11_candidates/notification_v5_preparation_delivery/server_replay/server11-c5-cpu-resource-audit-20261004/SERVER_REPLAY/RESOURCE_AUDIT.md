# Existing CPU comparison: resource audit

This is a read-only replay of 132 SHA-bound historical results (48 warmups, 84 scored trials, 42 pairs). It creates no new benchmark sample. The frozen qualification remains unchanged.

| Scenario | Pairs | A throttled | B throttled | Both unthrottled | Required | A/B total CPU median (ms) | Paired CPU B-A (ms) |
|---|---:|---:|---:|---:|---:|---|---:|
| step_end_40ms | 12 | 0/12 | 0/12 | 12 | 9 | 33.081 / 8.747 | -24.310 |
| step_end_20ms | 6 | 0/6 | 0/6 | 6 | 4 | 17.839 / 8.552 | -9.261 |
| step_end_80ms | 6 | 5/6 | 0/6 | 1 | 4 | 51.810 / 8.514 | -43.264 |
| mandatory_20ms | 6 | 0/6 | 0/6 | 6 | 4 | 18.008 / 9.009 | -9.028 |
| stop_20ms | 6 | 0/6 | 0/6 | 6 | 4 | 17.670 / 8.471 | -9.147 |
| original_deadline_100ms | 6 | 4/6 | 0/6 | 2 | 4 | 63.876 / 8.867 | -55.073 |

## Longer scenarios: absolute distributions

| Scenario | Arm | Wall min/median/max (ms) | Submit from ready min/median/max (ms) |
|---|---|---|---|
| step_end_80ms | A | 81.780 / 83.435 / 95.724 | 80.507 / 82.129 / 93.727 |
| step_end_80ms | B | 81.718 / 81.846 / 81.935 | 80.505 / 80.573 / 80.683 |
| original_deadline_100ms | A | 120.304 / 120.350 / 120.386 | 100.066 / 100.244 / 100.732 |
| original_deadline_100ms | B | 120.307 / 120.321 / 120.390 | 100.548 / 100.686 / 100.752 |

## Interpretation and limits

The main 40 ms scenario has CPU savings in all 12/12 pairs; paired worker-plus-producer median delta is -24.310 ms. Producer CPU increases by 0.067 ms and is included. Wall paired median changes by +0.150 ms. This supports reduced synthetic control-path polling CPU, not inference acceleration.

B has no new throttling in its 42 scored trials. The two resource-limited scenarios are limited by A samples under the frozen requirement that both arms in a pair be unthrottled. This is insufficient upgrade evidence under the preregistration, not evidence that B lost native progress. The original gate is not relaxed and throttled trials are not discarded.

For the deadline scenario, subtract 100 ms from the submission column to obtain deadline lateness. The approximately 120 ms wall includes a producer deliberately holding the external step until 120 ms; it is not the deadline response latency. No scored submission precedes the original deadline.

Cgroup throttle duration is an aggregate counter and cannot be assigned one-for-one to a thread, request or removable waiting time. No additional causal claim is inferred from it.

The physical backend, event completion, payload and pool remain CPU fixtures. Transparent counter overhead remains measured. No true CUDA/DMA lifecycle, D/J release, real queue controllability or end-to-end model gain is established.

Keep the original off default and failed upgrade qualification. Do not repeat this closed score, adjust start phase/settling to obtain favorable quota observations, or automatically launch GPU. Separate bounded CPU launcher preparation belongs to another task.

## Evidence

Original analysis SHA-256: `be79e7b1d9c539804d4b8a141a8e69f3219a6a6fd97cbb41efc6681a69512fe6`.
Frozen protocol SHA-256: `6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218`.
The accompanying JSON stores every raw SHA, all scored pair details, all warmup records, input hashes, execution command and script hash. No source input is written.
