# Server09 G1 permissions-path CPU candidate

This directory contains a CPU-tested candidate, not an activated GPU grant.
Both baseline scripts were downloaded from the currently connected server09.
The server's original project files have not been changed by this agent.

The only runtime source edits are the two scripts in candidate/. Their patches,
byte counts and exact SHA-256 values are recorded in *.diff and CANDIDATE_HASHES.json.
The original native qualification sequence and the budget runner's accounting,
session, cleanup and atomic-write functions are unchanged in AST; see
CANDIDATE_AST_REUSE_PROOF.json.

The optional --permissions-path defaults to the original canonical path.
Default guarded child/launcher arguments are compared against the actual baseline.
An explicit path must stay within ROOT, reject symlinks, bind actual permission
file bytes in scope and source-lock, propagate to outer guard and child, and
match the real active reservation's permission reference. It cannot increase the
base GPU budget or change the base roots. Explicit G1 scope prohibits model
downloads, system/driver changes, shared deletion, payments, cloud rental and AUX
authorization. The common single existing GPU ledger is retained.

Explicit G1 storage preflight uses the same validated effective permissions and
PRIMARY root with the original 128 MiB reserve and 8 GiB free-space floor.
Default storage still calls experiment_storage.preflight. No AUX grant for an
old GPU is borrowed.

Local Windows pure CPU tests: 19 passed, 2 skipped, 0 failed. The two skips are
actual file/ancestor symlink creation because Windows denies that privilege.
On server Linux, run the whole directory's test_permissions_adapter.py with
CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1 to verify those two tests
against real symlinks. Unit scopes are marked fixture_only inside temporary
directories, backend imports are prohibited, subprocess launch is mocked, and
no actual GPU budget is changed.

Canonical source edits require a documented version transition: save both
original scripts and an exact old path -> historical snapshot SHA map first,
retain all server08 manifests and permissions/CPU scope bytes, then create a
new server09 source lock. Do not claim the old path-based source lock continues
to match the edited current canonical scripts.

To inspect a prepared candidate without launching:
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=experiments/prefix_io_v1/scripts .venv/bin/python experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py --mode off --name server09-p4-02-native-off-01 --dry-run --permissions-path experiments/prefix_io_v1/configs/permissions.server09.yaml --scope-record artifacts/prefix_io_v1/server09-p4-02/GPU_STAGE_AUTHORIZATION.json --source-lock artifacts/prefix_io_v1/server09-p4-02/gpu-source-lock.json

--check-launch always reports CPU_ONLY; it is not proof that real scope, storage,
budget or source gates passed. The pure CPU launch_gates function can validate
those gates without invoking invoke_existing_guard.

Only after the human explicitly authorizes the new exact UUID and G1 stage,
create an activated scope and source-bound effective permissions, then use the
normal outer --launch entry. Each off/shadow command is bounded to 180 seconds
plus 20 seconds of termination reserve; two commands reserve up to 400 seconds,
with actual wall time charged in the old cumulative 8-hour ledger. No allowance
is reset by migration. G1 does not verify complete P4 or method performance.

