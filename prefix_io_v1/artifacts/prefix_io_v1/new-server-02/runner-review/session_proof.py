"""CPU-only process tests; each fixture has private fake permissions and ledger.

No torch/NVML imports, no real GPU ledger writes and no device operations.
The copied runner uses shortened termination grace periods for these tests.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
import yaml

SOURCE = Path(__file__).parent / "run_gpu_stage.pre-session.py"
UUID = "GPU-00000000-0000-0000-0000-000000000000"


@pytest.fixture
def sandbox(tmp_path):
    script = tmp_path / "experiments/prefix_io_v1/scripts/run_gpu_stage.py"
    script.parent.mkdir(parents=True)
    text = SOURCE.read_text().replace("TERM_GRACE_SECONDS = 15.0", "TERM_GRACE_SECONDS = 0.2")
    text = text.replace("KILL_GRACE_SECONDS = 5.0", "KILL_GRACE_SECONDS = 1.0")
    script.write_text(text)
    config = script.parent.parent / "configs/permissions.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({
        "allow_gpu_runs": True, "max_gpu_hours": 1,
        "approved_gpu_ids": [UUID], "approved_experiment_root": str(tmp_path / "runs"),
    }))
    ledger = script.parent.parent / "gpu-budget-ledger.json"
    ledger.write_text(json.dumps({"gpu_wall_seconds": 0, "model_download_bytes": 0, "events": []}))
    return tmp_path, script, ledger


def argv(sandbox, code, seconds=3, label="cpu-case"):
    return [sys.executable, str(sandbox[1]), "--label", label, "--seconds", str(seconds),
            "--", sys.executable, "-c", code]


def run(sandbox, code, **kwargs):
    result = subprocess.run(argv(sandbox, code, **kwargs), capture_output=True, text=True, timeout=10)
    return result, json.loads(sandbox[2].read_text())


def wait_path(path, proc, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        if proc.poll() is not None:
            raise AssertionError(f"runner stopped before child readiness: {proc.communicate()}")
        time.sleep(0.01)
    raise AssertionError(f"child did not create {path}")


def live(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat[stat.rfind(")") + 2:].split()[0] not in ("Z", "X")


def test_cpu_success_records_visible_uuid_and_offline_settings(sandbox):
    code = ("import os,json,time; time.sleep(.06); "
            "print(json.dumps({k:os.environ[k] for k in "
            "['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE']}))")
    result, budget = run(sandbox, code)
    assert result.returncode == 0, result.stderr
    assert budget["active_reservation"] is None
    event = budget["events"][-1]
    assert event["session_drained"] is True
    assert event["elapsed_seconds"] >= .06
    log = (sandbox[0] / "runs/cpu-case/process.log").read_text()
    assert UUID in log and '"HF_HUB_OFFLINE": "1"' in log
    assert budget["model_download_bytes"] == 0


def test_monotonic_charge_survives_backwards_wall_clock(sandbox):
    # Only this copied CPU-test script is instrumented; production stays unchanged.
    text = sandbox[1].read_text()
    prefix = "import time\n_wall_clock=iter([1000., -10000., -20000.])\ntime.time=lambda: next(_wall_clock, -30000.)\n"
    sandbox[1].write_text(prefix + text)
    result, budget = run(sandbox, "import time; time.sleep(.08)")
    assert result.returncode == 0, result.stderr
    assert budget["events"][0]["started_unix"] == 1000
    assert budget["gpu_wall_seconds"] >= .08


def test_unfinished_reservation_blocks_before_launch(sandbox):
    budget = json.loads(sandbox[2].read_text())
    budget["active_reservation"] = {"id": "previous-unresolved"}
    sandbox[2].write_text(json.dumps(budget))
    target = sandbox[0] / "should-not-exist"
    result, after = run(sandbox, f"from pathlib import Path; Path({str(target)!r}).touch()")
    assert result.returncode != 0
    assert "unfinished GPU reservation" in result.stderr
    assert not target.exists()
    assert after == budget


def test_launch_exception_is_recorded_without_gpu_attempt(sandbox):
    command = argv(sandbox, "pass")
    command[-3:] = ["/no/such/prefix-io-test-executable"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    budget = json.loads(sandbox[2].read_text())
    assert result.returncode == 1, result.stderr
    event = budget["events"][-1]
    assert "FileNotFoundError" in event["error"]
    assert event["gpu_job_attempted"] is False
    assert budget["active_reservation"] is None


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_termination_signal_cleans_child_and_commits_charge(sandbox, stop_signal):
    ready = sandbox[0] / "child.pid"
    code = f"import os,time; open({str(ready)!r},'w').write(str(os.getpid())); time.sleep(60)"
    proc = subprocess.Popen(argv(sandbox, code), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_path(ready, proc)
        pid = int(ready.read_text())
        reserved = json.loads(sandbox[2].read_text())["active_reservation"]
        assert reserved["process_group"] == pid
        proc.send_signal(stop_signal)
        out, err = proc.communicate(timeout=5)
        assert proc.returncode == 128 + stop_signal, (out, err)
        budget = json.loads(sandbox[2].read_text())
        assert budget["active_reservation"] is None
        assert budget["events"][-1]["interrupted_signal"] == stop_signal
        assert budget["gpu_wall_seconds"] > 0
        assert not live(pid)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_successful_parent_cannot_leave_background_group_running(sandbox):
    childpid = sandbox[0] / "background.pid"
    code = ("import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
            f"open({str(childpid)!r},'w').write(str(p.pid))")
    result, budget = run(sandbox, code)
    assert result.returncode == 70, result.stderr
    event = budget["events"][-1]
    pid = int(childpid.read_text())
    assert pid in event["session_members_before_cleanup"]
    assert event["child_exit"] == 0 and event["session_drained"]
    assert not live(pid)
    assert budget["active_reservation"] is None


def test_timeout_kills_sigterm_resistant_child(sandbox):
    childpid = sandbox[0] / "resistant.pid"
    code = ("import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"open({str(childpid)!r},'w').write(str(os.getpid())); time.sleep(60)")
    result, budget = run(sandbox, code, seconds=1)
    assert result.returncode == 124, result.stderr
    event = budget["events"][-1]
    assert event["timed_out"] and event["child_exit"] == -signal.SIGKILL
    assert 1 <= event["elapsed_seconds"] < 5
    assert not live(int(childpid.read_text()))
    assert budget["active_reservation"] is None


def test_exception_after_launch_still_drains_group(sandbox):
    childpid = sandbox[0] / "exception.pid"
    # Force a runner-side exception after the real CPU child has started.
    injected = """
_ORIGINAL_WAIT = subprocess.Popen.wait
def _raise_once(self, timeout=None):
    if not getattr(self, "_raised_once", False):
        self._raised_once = True
        time.sleep(.15)
        raise RuntimeError("injected CPU runner failure")
    return _ORIGINAL_WAIT(self, timeout=timeout)
subprocess.Popen.wait = _raise_once
"""
    sandbox[1].write_text(sandbox[1].read_text().replace('if __name__ == "__main__":', injected + '\nif __name__ == "__main__":'))
    code = f"import os,time; open({str(childpid)!r},'w').write(str(os.getpid())); time.sleep(60)"
    result, budget = run(sandbox, code)
    assert result.returncode == 1, result.stderr
    event = budget["events"][-1]
    assert event["error"] == "RuntimeError: injected CPU runner failure"
    assert event["session_drained"] and not live(int(childpid.read_text()))
    assert budget["active_reservation"] is None


def test_reservation_exists_before_popen_and_survives_killed_runner(sandbox):
    # The abrupt interruption is simulated before child launch: no orphan CPU
    # process and no device operation. It exercises the durable crash boundary.
    injected = """
def _terminate_before_child(*args, **kwargs):
    os.kill(os.getpid(), signal.SIGKILL)
subprocess.Popen = _terminate_before_child
"""
    original = sandbox[1].read_text()
    sandbox[1].write_text(original.replace('if __name__ == "__main__":', injected + '\nif __name__ == "__main__":'))
    result, budget = run(sandbox, "raise AssertionError('must not launch')")
    assert result.returncode == -signal.SIGKILL
    assert budget["active_reservation"]["process_group"] is None
    sandbox[1].write_text(original)
    retry, after = run(sandbox, "raise AssertionError('must not launch')", label="retry")
    assert retry.returncode != 0 and "unfinished GPU reservation" in retry.stderr
    assert after == budget


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, None])
def test_invalid_hours_fail_closed_before_child(sandbox, value):
    config = sandbox[1].parent.parent / "configs/permissions.yaml"
    permission = yaml.safe_load(config.read_text())
    permission["max_gpu_hours"] = value
    config.write_text(yaml.safe_dump(permission))
    result, budget = run(sandbox, "raise AssertionError('must not launch')")
    assert result.returncode != 0
    assert budget["events"] == [] and budget["gpu_wall_seconds"] == 0


def test_budget_rejection_reserves_termination_time(sandbox):
    budget = json.loads(sandbox[2].read_text())
    budget["gpu_wall_seconds"] = 3590
    sandbox[2].write_text(json.dumps(budget))
    result, after = run(sandbox, "raise AssertionError('must not launch')", seconds=1)
    assert result.returncode != 0 and "termination reserve" in result.stderr
    assert after == budget

def test_failed_group_verification_keeps_reservation_and_blocks_retry(sandbox):
    # The CPU child has already exited. Simulate unavailable /proc verification
    # without leaving a real process running in this test.
    injected = """
def _cannot_verify_group(pgid):
    raise OSError("injected CPU group inspection failure")
session_members = _cannot_verify_group
"""
    original = sandbox[1].read_text()
    sandbox[1].write_text(original.replace('if __name__ == "__main__":', injected + '\nif __name__ == "__main__":'))
    result, budget = run(sandbox, "pass")
    assert result.returncode == 70
    assert budget["active_reservation"]["state"] == "cleanup_unresolved"
    assert budget["active_reservation"]["accounted_seconds"] > 0
    assert budget["events"][-1]["session_drained"] is False
    sandbox[1].write_text(original)
    retry, after = run(sandbox, "raise AssertionError('must not launch')", label="retry")
    assert retry.returncode != 0 and "unfinished GPU reservation" in retry.stderr
    assert after == budget

def test_signal_between_os_spawn_and_handle_assignment_cleans_real_child(sandbox):
    pidfile = sandbox[0] / "spawn-race.pid"
    injected = f"""
_ORIGINAL_POPEN = subprocess.Popen
def _interrupt_before_assignment(*args, **kwargs):
    child = _ORIGINAL_POPEN(*args, **kwargs)
    Path({str(pidfile)!r}).write_text(str(child.pid))
    os.kill(os.getpid(), signal.SIGTERM)
    return child
subprocess.Popen = _interrupt_before_assignment
"""
    sandbox[1].write_text(sandbox[1].read_text().replace('if __name__ == "__main__":', injected + '\nif __name__ == "__main__":'))
    try:
        result, budget = run(sandbox, "import time;time.sleep(60)")
        assert result.returncode == 143, result.stderr
        event = budget["events"][-1]
        assert event["gpu_job_attempted"] is True
        assert event["interrupted_signal"] == signal.SIGTERM
        assert event["session_drained"] is True
        assert not live(int(pidfile.read_text()))
        assert budget["active_reservation"] is None
    finally:
        if pidfile.exists() and live(int(pidfile.read_text())):
            # Prevent a test regression from leaving a CPU sleeper behind.
            os.killpg(int(pidfile.read_text()), signal.SIGKILL)

def test_nested_process_group_is_drained_with_session(sandbox):
    childpid = sandbox[0] / "nested-pgid.pid"
    nested_code = "import os,time;os.setpgrp();time.sleep(60)"
    code = ("import subprocess,sys,time,os; "
            f"p=subprocess.Popen([sys.executable,'-c',{nested_code!r}]); "
            "time.sleep(.1); "
            f"open({str(childpid)!r},'w').write(str(p.pid)); "
            "assert os.getpgid(p.pid)==p.pid; assert os.getsid(p.pid)==os.getsid(0)")
    try:
        result, budget = run(sandbox, code)
        pid = int(childpid.read_text())
        assert result.returncode == 70, result.stderr
        event = budget["events"][-1]
        assert event["child_exit"] == 0
        assert pid in event["session_members_before_cleanup"]
        assert event["session_drained"] and not live(pid)
        assert budget["active_reservation"] is None
    finally:
        if childpid.exists() and live(int(childpid.read_text())):
            os.kill(int(childpid.read_text()), signal.SIGKILL)


def test_timeout_kills_resistant_nested_process_group(sandbox):
    childpid = sandbox[0] / "resistant-nested-pgid.pid"
    nested_code = ("import os,signal,time;os.setpgrp();"
                   "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)")
    code = ("import subprocess,sys,time,signal; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            f"p=subprocess.Popen([sys.executable,'-c',{nested_code!r}]); "
            f"open({str(childpid)!r},'w').write(str(p.pid));time.sleep(60)")
    try:
        result, budget = run(sandbox, code, seconds=1)
        event = budget["events"][-1]
        assert result.returncode == 124, result.stderr
        assert event["timed_out"] and event["session_drained"]
        assert not live(int(childpid.read_text()))
        assert budget["active_reservation"] is None
    finally:
        if childpid.exists() and live(int(childpid.read_text())):
            os.kill(int(childpid.read_text()), signal.SIGKILL)
