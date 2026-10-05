# Session cleanup follow-up — current runner qualification

This supersedes the group-only cleanup described in REVIEW.md. The previous
19-test implementation and its original patch are preserved as pre-session
evidence. The current patch is 0002-session-aware-gpu-budget-runner.patch, a
complete cumulative patch against run_gpu_stage.before.py; do not apply it on
top of 0001.

A legitimate child can call setpgid()/setpgrp() while remaining in the same
session. GNU timeout/ninja may do this. The previous process-group enumeration
incorrectly ignored those descendants and could report exit 0 with live work.

The runner already creates a new session. Cleanup now enumerates non-zombie
members by that session ID, verifies each current SID before signalling, and
sends TERM/KILL to every session member. It refuses to signal its own session.
The KILL phase repeats scans so descendants created during the prior sweep are
also handled. Separate process groups are supported; a deliberate new setsid()
still escapes containment and remains outside the trusted-command contract.

Tests:
- session-before.txt/xml: 1 expected failure against the preserved group-only
  implementation; it reports exit 0 while an os.setpgrp() sleeper remains alive.
  The proof test forcibly cleans its own CPU sleeper in finally.
- session-final.txt/xml: **21 CPU tests passed**, 5.18 seconds, including normal
  parent exit with a nested group and timeout of a TERM-resistant nested group.
- session-verification.json: final patch forward/reverse checks passed; real GPU
  ledger, permissions.yaml and cuda_base_smoke.py are unchanged.

Current result fields are session_id, session_members_before_cleanup,
session_members_after_cleanup and session_drained. Historical group-based
events and existing GPU results are not rewritten.

Final command:
```bash
cd /root/autodl-tmp/prefix-io-v1-handoff/project
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv-prefix/bin/python -m pytest -q -ra tests/prefix_io_v1_runner --junitxml=artifacts/prefix_io_v1/new-server-02/runner-review/session-final.xml
```

No GPU operation, model download, real-ledger write, or permission edit occurred
in this review. Uncatchable termination/host failure and unresolved cleanup
retain reservations and require explicit reconciliation; no auto-reset exists.
