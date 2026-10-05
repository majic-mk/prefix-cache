# Completed CPU receipt-reuse prototype

This append-only prototype changes no frozen controller, runner, public receipt,
engine, notification, model, estimator or migration threshold. Off01 completed
128 tokens, but its selected step was 16,893,473 ns, above the frozen 16,238,752 ns
upper by 654,721 ns (about 4.03%). Its qualification remains false and shadow/on
remain blocked. Startup metadata work cannot explain or repair that selected-step
gate failure without separate evidence.

The actual frozen runner invokes the public receipt loader seven times in one
process: `main.load_configuration` once, `main.verify_guard` twice through nested
configuration verification; `execute_window` repeats those three calls; the
registered public canonical loader is called once more before bridge creation.
All seven calls precede the measured request. No claim about their individual
elapsed time is possible from the old record because they were not timed.

`entry_receipt_session.py` is an independent completed-record CPU harness. Its
first call uses the unchanged official public loader once and retains that exact
`prefix_io_control.p4_single_file_receipt.ExactSingleFileReceipt` object. Each get
rehashes every original COMMON leaf and separately rechecks mode metadata bytes,
config/authority, before/launch/after proofs, original calibration guard validation,
completed normal guard fields, first-appended-event and ledger prefix accounting,
full original SDK/current driver audit and actual PID/SID/process group. It caches
only the actual typed object; it never caches or promotes PASS or qualification.
The active GPU entry intentionally rejects. No original callable or module global
is patched. It is not integrated into the runner and is not a demonstrated GPU
startup improvement.

Run the targeted CPU rejection tests locally:

```powershell
python -B -I artifacts/prefix_io_v1_server12_candidates/normal_entry_dedup_cpu/test_entry_receipt_session.py
```

After the root agent uploads these source-only files into a new project directory,
the real server can time two completed CPU reuse entries using the existing assets:

```sh
.venv/bin/python -B -I artifacts/prefix_io_v1/server12-c5-normal-entry-dedup-cpu-20261004/entry_receipt_session.py \
  --root /root/autodl-tmp/prefix-io-v1-handoff/project --requests 2 \
  --output artifacts/prefix_io_v1/server12-c5-normal-entry-dedup-cpu-20261004/COMPLETED_CPU_REUSE_RESULT.json
```

Do not use `-S`, because the original permissions parser needs real PyYAML. This
command loads no model, CUDA library or GPU executor and changes no GPU ledger.
The report measures actual public-loader and subsequent CPU metadata time; it
does not compare against seven freshly replayed calls or claim performance gain.
The local machine lacks the complete server asset root, so local tests never
construct a synthetic native receipt or assert an actual session success.

Integrating this metadata change into a future GPU revision would still require
an explicit active-entry design and tests, source-pinned single public class shared
with the original policy, fresh per-entry current guard validation, and retention
of the failed scientific migration gate. This prototype does not do that work.
