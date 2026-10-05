# Copy-time delivery redaction (CPU only)

Status: **PREPARED / REAL DELIVERY COPY UNEXECUTED**. No archive was created. Original evidence, canonical files, permissions, budget ledgers, GPU state, network state and model files were not changed.

The bounded scanner covers the task-specific docs/src/tests/patches directories, experiment scripts/configs/locks/runs, and artifacts/prefix_io_v1. The exact roots are the script's SCOPES constant and every plan's source_scopes. It explicitly includes the server parent commands.jsonl at delivery path server-root/commands.jsonl. It excludes the three raw redirect evidence files specified in redirect-review/README.md and generated Python bytecode. The generated plan itself is excluded to avoid a self-hash cycle. Other unreadable/unknown binary/symlink sources block the plan; existing CPU .bin probes receive byte screening and are preserved if clean. Per-file read ceiling is 64 MiB.

Secret vocabulary remains in process memory. It is derived by comparing the raw redirect-review.json with its existing public copy; all three raw files must match the public copy's recorded SHA256 and byte sizes. Complete Set-Cookie values are replaced exactly, including JSON-escaped variants. auth_key values are replaced exactly plus a bounded URL-query regular expression for additional values. Nested JSON string escaping is covered through eight encoding levels (the actual raw cookie strings need no added escaping). No token values are printed or written to the plan.

Do not split cookie assignments and globally replace their component values: public tenant/domain text can legitimately appear in both a cookie and ordinary provenance. The initial diagnostic exposed that overbroad approach; the final code redacts complete cookie headers. A regression test preserves ordinary public tenant/domain text.

The final recorded scan found **469 files / 13,149,231 source bytes**, **one affected file**, **eight substitutions**: server-root/commands.jsonl (five complete Set-Cookie values, three auth_key values). All remaining planned files were unchanged; the three raw redirect files remain excluded. These counts are a point-in-time observation and will change as ongoing work adds files/log records.

CPU validation: **15 passed**, including nested JSON, query replacement/idempotence, malformed provenance refusal, binary refusal, source-hash changes, source preservation, excluded raw evidence, symlink refusal and staging-directory reuse refusal. Test copies contain synthetic fixtures only; no real delivery copy was executed.

## Commands

Run from /root/autodl-tmp/prefix-io-v1-handoff/project:

```sh
CUDA_VISIBLE_DEVICES='' RUN_E2E_TESTS='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:third_party/work/py-kvcache-observer .venv-prefix/bin/python -m pytest artifacts/prefix_io_v1/new-server-03/delivery-redaction/test_redact_delivery.py -q --import-mode=importlib -W error::pytest.PytestUnhandledThreadExceptionWarning

/root/miniconda3/bin/python -I -S artifacts/prefix_io_v1/new-server-03/delivery-redaction/redact_delivery.py --project /root/autodl-tmp/prefix-io-v1-handoff/project --plan /root/autodl-tmp/prefix-io-v1-handoff/project/artifacts/prefix_io_v1/new-server-03/delivery-redaction/delivery-plan.json
```

The default only writes the dedicated delivery-plan.json, never source files. The plan lists every source/copy SHA256, byte count, encoding, per-file redaction counts and excluded paths. Exit 0 means PLAN_READY; exit 2 means blocked/failure. Regenerate after any source/log changes.

The later, explicitly selected copy action is the same command plus `--copy-to /root/autodl-tmp/prefix-io-v1-delivery-sanitized-NEW`. That destination must not exist and must be outside the original project, with an existing nonsymlink parent. Copy validates all original source hashes and transformations before creating the destination, then writes a DELIVERY_REDACTION_MANIFEST.json. It does not create an archive, delete anything, change originals or overwrite an existing staging tree.

The caller should run plan and copy sequentially within **one outer logged invocation**, without appending to source logs between them. Otherwise the copy correctly rejects the changed source SHA. If output/log mutation occurs afterward, the staged tree remains a snapshot of exactly the recorded source bytes. Do not archive the original project directly: archive only the successfully created sanitized staging tree and its copy manifest.

Evidence: commands.jsonl, cpu-tests-final.txt, scan-plan-final.txt and delivery-plan.json. The plan will become stale during ongoing work; regenerate at the final snapshot boundary.
