"""Private standing-budget adapter for the original bounded GPU command guard.

Only a SHA-bound direct user grant for U versus F8+D_R may remove the cumulative
time ceiling. Every job remains bounded, and all original ledger/session rules
remain in force. The default and ordinary finite permission paths are unchanged.

Only trusted project commands belong here. A process that calls setsid() escapes
the session; SIGKILL or host failure cannot run this process's cleanup.
An unfinished durable reservation therefore always blocks the next invocation.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import uuid

import yaml

TERM_GRACE_SECONDS = 15.0
KILL_GRACE_SECONDS = 5.0
RESERVE_SECONDS = 20
PERMISSION = "experiments/prefix_io_v1/configs/permissions.yaml"
STANDING_SCOPE = "U_VS_F8_PLUS_D_R_ONLY"
STANDING_JOB_SECONDS = 300
STANDING_LABEL = r"server14-dr-[A-Za-z0-9_-]+"


def standing_authorization(root, permission):
    """Read one frozen human-authority record; never reset prior usage."""
    if "max_gpu_hours" not in permission or permission["max_gpu_hours"] is not None:
        raise RuntimeError("standing GPU budget must be explicit null")
    reference = permission.get("standing_gpu_authorization_ref")
    if (type(reference) is not dict or set(reference) != {"path", "bytes", "sha256"}
            or type(reference["path"]) is not str or not reference["path"]
            or "\\" in reference["path"] or Path(reference["path"]).is_absolute()
            or any(part in ("", ".", "..") for part in reference["path"].split("/"))
            or type(reference["bytes"]) is not int or not 0 < reference["bytes"] <= 65536
            or type(reference["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None):
        raise RuntimeError("exact project-relative standing authorization reference required")
    path = root / reference["path"]
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise RuntimeError("standing authorization path symlink rejected")
        cursor = cursor.parent
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("standing authorization outside project")
    raw = path.read_bytes()
    if len(raw) != reference["bytes"] or hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise RuntimeError("standing authorization bytes differ from frozen reference")
    grant = json.loads(raw.decode("utf-8"))
    fields = {"schema_version", "authority", "user_literal", "issued_utc", "approved_gpu_ids",
              "scope", "max_gpu_hours", "completion_condition", "per_job_seconds_max",
              "cleanup_reserve_seconds"}
    preservation = {"historical_usage_preserved", "budget_not_reset"}
    if (type(grant) is not dict or not fields.issubset(grant)
            or not set(grant).issubset(fields | preservation)
            or any(grant[key] is not True for key in preservation if key in grant)
            or type(grant["schema_version"]) is not int or grant["schema_version"] != 1
            or grant["authority"] != "direct_user_instruction"
            or type(grant["user_literal"]) is not str or not grant["user_literal"].strip()
            or type(grant["issued_utc"]) is not str
            or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", grant["issued_utc"]) is None
            or type(grant["approved_gpu_ids"]) is not list or len(grant["approved_gpu_ids"]) != 1
            or grant["approved_gpu_ids"] != permission.get("approved_gpu_ids")
            or type(grant["approved_gpu_ids"][0]) is not str
            or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", grant["approved_gpu_ids"][0]) is None
            or grant["scope"] != STANDING_SCOPE or grant["max_gpu_hours"] is not None
            or grant["completion_condition"] != "narrow_experiment_complete"
            or type(grant["per_job_seconds_max"]) is not int or grant["per_job_seconds_max"] != STANDING_JOB_SECONDS
            or type(grant["cleanup_reserve_seconds"]) is not int or grant["cleanup_reserve_seconds"] != RESERVE_SECONDS):
        raise RuntimeError("standing authorization must match the direct user scope, GPU and bounded job contract")
    return dict(reference)


def gpu_budget_hours(permission, relative, seconds, label):
    if permission.get("max_gpu_hours") is not None:
        return positive_number(permission["max_gpu_hours"], "max_gpu_hours")
    if relative == PERMISSION:
        raise RuntimeError("standing budget requires an explicit effective permission path")
    if not 1 <= seconds <= STANDING_JOB_SECONDS or re.fullmatch(STANDING_LABEL, label) is None:
        raise RuntimeError("standing authorization permits only bounded server14 U/F8+D_R jobs")
    return None


def require_cumulative_capacity(used, seconds, hours):
    if hours is not None and used + seconds + RESERVE_SECONDS > hours * 3600:
        raise RuntimeError("insufficient cumulative GPU budget including termination reserve")

def permission_source(root,relative=PERMISSION):
    if (not isinstance(relative,str) or not relative or "\\" in relative
            or Path(relative).is_absolute()
            or any(part in ("",".","..") for part in relative.split("/"))):
        raise RuntimeError("project-relative POSIX permission path required")
    path=root/relative
    cursor=path
    while cursor!=root:
        if cursor.is_symlink():
            raise RuntimeError("permission path symlink rejected")
        cursor=cursor.parent
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise RuntimeError("permission path outside project")
    raw=path.read_bytes()
    permission=yaml.safe_load(raw.decode())
    if relative!=PERMISSION:
        # A new device-specific G1 grant cannot enlarge the original contract.
        base=yaml.safe_load((root/PERMISSION).read_text())
        if permission.get("max_gpu_hours") is None:
            standing_authorization(root, permission)
        elif positive_number(permission["max_gpu_hours"],"max_gpu_hours")>positive_number(base["max_gpu_hours"],"base max_gpu_hours"):
            raise RuntimeError("effective GPU budget exceeds base contract")
        for key in ("approved_experiment_root","approved_dependency_root"):
            if permission.get(key)!=base.get(key):
                raise RuntimeError("effective permission root differs from base: "+key)
        for key in ("allow_model_downloads","allow_driver_or_system_changes",
                    "allow_shared_data_deletion","allow_payment","allow_new_cloud_rental"):
            if permission.get(key) is not False:
                raise RuntimeError("effective G1 scope prohibits "+key)
        if permission.get("approved_auxiliary_storage") is not None:
            raise RuntimeError("effective G1 scope is PRIMARY-only")
    return permission,dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())



class Interrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def positive_number(value, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise RuntimeError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite")
    return value


def session_members(session_id):
    """Read the complete session, including nested process groups; no GPU probes."""
    result = []
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / "stat").read_text()
        except FileNotFoundError:
            continue
        fields = stat[stat.rfind(")") + 2:].split()
        if int(fields[3]) == session_id and fields[0] not in ("Z", "X"):
            result.append(int(directory.name))
    return sorted(result)


def signal_session(session_id, signum):
    if session_id == os.getsid(0):
        raise RuntimeError("refusing to signal the runner's own session")
    for pid in session_members(session_id):
        try:
            # setpgid() descendants remain in this session. Check again after the
            # /proc scan before signalling; a vanished process is already drained.
            if os.getsid(pid) == session_id:
                os.kill(pid, signum)
        except ProcessLookupError:
            pass


def cleanup_session(proc):
    """Drain the whole launched session, including nested PGIDs and descendants."""
    before = session_members(proc.pid)
    if before:
        signal_session(proc.pid, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while session_members(proc.pid) and time.monotonic() < deadline:
        proc.poll()  # Reap the direct child; grandchildren may be adopted by init.
        time.sleep(0.02)
    if session_members(proc.pid):
        signal_session(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + KILL_GRACE_SECONDS
        while session_members(proc.pid) and time.monotonic() < deadline:
            signal_session(proc.pid, signal.SIGKILL)
            proc.poll()
            time.sleep(0.02)
    proc.poll()
    after = session_members(proc.pid)
    return {"session_members_before_cleanup": before, "session_members_after_cleanup": after,
            "session_drained": not after}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--seconds", type=int, required=True)
    p.add_argument("--permissions-path",default=PERMISSION)
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    if not a.label.replace("-", "").replace("_", "").isalnum():
        p.error("invalid label")
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not command or not 1 <= a.seconds <= 3600:
        p.error("command and per-job seconds 1..3600 required")
    permission,permission_ref = permission_source(root,a.permissions_path)
    if permission["allow_gpu_runs"] is not True:
        raise RuntimeError("GPU operation not authorized")
    hours = gpu_budget_hours(permission, a.permissions_path, a.seconds, a.label)
    approved = permission["approved_gpu_ids"]
    if (not isinstance(approved, list) or len(approved) != 1
            or not isinstance(approved[0], str)
            or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", approved[0])):
        raise RuntimeError("first-round runner requires exactly one approved GPU UUID")
    runroot = Path(permission["approved_experiment_root"]).resolve()
    if not runroot.is_relative_to(root):
        raise RuntimeError("experiment root outside project")
    runroot.mkdir(exist_ok=True, parents=True)
    ledger = root / "experiments/prefix_io_v1/gpu-budget-ledger.json"
    with ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget = json.loads(ledger.read_text())
        used = positive_number(budget["gpu_wall_seconds"], "gpu_wall_seconds", allow_zero=True)
        if not isinstance(budget["events"], list):
            raise RuntimeError("budget events must be a list")
        if budget.get("active_reservation") is not None:
            raise RuntimeError("unfinished GPU reservation: inspect prior process session and reconcile usage before retry")
        require_cumulative_capacity(used, a.seconds, hours)
        run = runroot / a.label
        run.mkdir(exist_ok=False)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=approved[0], PYTHONDONTWRITEBYTECODE="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        started = time.time()  # Wall time is evidence only, never budget arithmetic.
        started_mono = time.monotonic()
        reservation = {"id": uuid.uuid4().hex, "label": a.label, "command": command,
                       "started_unix": started, "gpu_uuid": approved[0],
                       "seconds_limit": a.seconds, "reserved_seconds": a.seconds + RESERVE_SECONDS,
                       "runner_pid": os.getpid(), "process_group": None, "session_id": None, "evidence": str(run)}
        if a.permissions_path != PERMISSION:
            reservation["permissions"] = permission_ref
        if hours is None:
            reservation["standing_gpu_authorization"] = permission["standing_gpu_authorization_ref"]
            reservation["cumulative_gpu_hours_limit"] = None
        budget["active_reservation"] = reservation
        atomic_json(ledger, budget)  # Commit before any child can acquire resources.
        proc = None
        child_code = None
        timed_out = False
        interrupted_signal = None
        error = None
        cleanup = {"session_members_before_cleanup": [], "session_members_after_cleanup": [],
                   "session_drained": True}
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)}
        launch_in_progress = False
        pending_launch_signal = None
        def on_signal(signum, frame):
            nonlocal pending_launch_signal
            if launch_in_progress:
                # Do not raise between OS process creation and storing its handle.
                # Unlike masking signals, this does not alter the child signal mask.
                pending_launch_signal = signum
                return
            raise Interrupted(signum)
        for sig in old_handlers:
            signal.signal(sig, on_signal)
        try:
            with (run / "process.log").open("wb") as log:
                launch_in_progress = True
                try:
                    proc = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                finally:
                    launch_in_progress = False
                if pending_launch_signal is not None:
                    raise Interrupted(pending_launch_signal)
                reservation["process_group"] = proc.pid
                reservation["session_id"] = proc.pid
                atomic_json(ledger, budget)
                remaining = max(0.0, a.seconds - (time.monotonic() - started_mono))
                child_code = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
        except Interrupted as exc:
            interrupted_signal = exc.signum
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # A repeated Ctrl-C / SIGTERM must not interrupt cleanup or accounting.
            for sig in old_handlers:
                signal.signal(sig, signal.SIG_IGN)
            if proc is not None:
                try:
                    cleanup = cleanup_session(proc)
                    child_code = proc.poll()
                except BaseException as exc:
                    cleanup = {"session_drained": False, "cleanup_error": f"{type(exc).__name__}: {exc}"}
        elapsed = max(0.0, time.monotonic() - started_mono)
        if timed_out:
            code = 124
        elif interrupted_signal is not None:
            code = 128 + interrupted_signal
        elif error is not None:
            code = 1
        elif child_code is None:
            code = 70
        else:
            code = child_code if child_code >= 0 else 128 - child_code
            # A successful parent leaving live children is an incomplete command.
            if code == 0 and cleanup.get("session_members_before_cleanup"):
                code = 70
        if not cleanup["session_drained"]:
            code = 70
        event = {"label": a.label, "command": command, "gpu_uuid": approved[0],
                 "started_unix": started, "elapsed_seconds": elapsed,
                 "exit": code, "child_exit": child_code, "timed_out": timed_out,
                 "interrupted_signal": interrupted_signal, "error": error,
                 "evidence": str(run), "gpu_job_attempted": proc is not None,
                 "reservation_id": reservation["id"], "session_id": proc.pid if proc is not None else None, **cleanup}
        if a.permissions_path != PERMISSION:
            event["permissions"] = permission_ref
        if hours is None:
            event["standing_gpu_authorization"] = permission["standing_gpu_authorization_ref"]
            event["cumulative_gpu_hours_limit"] = None
        budget["gpu_wall_seconds"] = used + elapsed
        budget["events"].append(event)
        if cleanup["session_drained"]:
            budget["active_reservation"] = None
        else:
            reservation["state"] = "cleanup_unresolved"
            reservation["accounted_seconds"] = elapsed
        try:
            # Failure to persist retains the previously durable reservation.
            atomic_json(ledger, budget)
            atomic_json(run / "result.json", event)
            print(json.dumps(event, indent=2))
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
