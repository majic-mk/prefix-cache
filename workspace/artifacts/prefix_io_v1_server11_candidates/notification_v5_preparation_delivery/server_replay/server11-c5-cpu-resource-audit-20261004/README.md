# Historical CPU resource audit

This is an independent read-only calculation over the single completed `server-score01` run. It is not another benchmark, a protocol revision, or a GPU qualification. It imports only the Python standard library, reads existing JSON, verifies all 132 result hashes, preserves all 48 warmups and 84 scored samples, and derives the 42 fixed A/B pairs. No candidate, frozen protocol or historical result is changed.

The source analysis is pinned to SHA-256 `be79e7b1d9c539804d4b8a141a8e69f3219a6a6fd97cbb41efc6681a69512fe6`; the original protocol remains `6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218`. PLAN and SUITE bind through the original analysis. Per-arm reactor/collector identities, raw CPU sums, preparation inclusion, cgroup deltas, complete fixed order, paired medians, deadline arithmetic and resource gate are checked. Missing quota counters are not interpreted as zero. The script refuses mismatches and output overwrite.

Local replay:

```text
python -B -I -S artifacts/prefix_io_v1_server11_candidates/notification_v5_cpu_resource_audit/audit_resource_results.py --benchmark-root artifacts/prefix_io_v1_server11_candidates/notification_v5_cpu_delivery/server_replay/benchmark --output-dir artifacts/prefix_io_v1_server11_candidates/notification_v5_cpu_resource_audit/local_replay01
```

Server replay, using the already existing benchmark directory and an existing Python interpreter:

```text
ROOT/.venv/bin/python -B -I -S AUDIT/audit_resource_results.py --benchmark-root ROOT/artifacts/prefix_io_v1/server11-p4-notification-benchmark-v5-cpu-20261003 --output-dir NEW_SERVER_AUDIT_OUTPUT
```

The script creates `RESOURCE_AUDIT.json` and `RESOURCE_AUDIT.md` only in the new output directory. JSON preserves every source SHA, full per-pair details and all warmup records. It records its own SHA and actual command. Standard output reports validation status without granting any new permission.

Scope of the result: C5 reduced CPU used by the instrumented synthetic control path; B had no additional quota throttling in its 42 scored trials. Two longer scenarios lack enough pairs with **both** arms unthrottled, so the existing CPU upgrade qualification remains false. The original 100 ms deadline is measured at first submission; the 120 ms total wall contains the deliberately held producer step. These metrics cannot demonstrate GPU speedup, true CUDA/DMA lifecycle, D/J credit, controllable native I/O queueing, end-to-end model improvement or paper readiness.

No rerun, threshold change, new policy or automatic GPU experiment is proposed. Subsequent CPU launcher preparation is handled separately. Preserve the off default and this closed result.
