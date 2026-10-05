#!/usr/bin/env python3
"""Linux CPU-only io_uring_setup(entries=2, flags=0), then immediate fd close.

Python standard library only. No mmap, io_uring_enter, I/O submissions, GPU,
network, sysctl writes, privilege changes, or seccomp changes. Run once with:
    python3 io_uring_setup_probe.py
Exit codes: 0 setup succeeded and fd closed; 2 setup syscall failed;
3 unsupported OS/ABI; 4 diagnostic/probe or close error.
"""
import ctypes
import datetime
import errno
import json
import os
from pathlib import Path
import platform
import resource


class SqOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32), ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32), ("ring_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("dropped", ctypes.c_uint32),
        ("array", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("user_addr", ctypes.c_uint64),
    ]


class CqOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32), ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32), ("ring_entries", ctypes.c_uint32),
        ("overflow", ctypes.c_uint32), ("cqes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("reserved", ctypes.c_uint32),
        ("user_addr", ctypes.c_uint64),
    ]


class Params(ctypes.Structure):
    _fields_ = [
        ("sq_entries", ctypes.c_uint32), ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32), ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32), ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 3),
        ("sq_off", SqOffsets), ("cq_off", CqOffsets),
    ]


def read_optional(path):
    try:
        return {"value": Path(path).read_text().strip(), "error": None}
    except OSError as exc:
        return {"value": None, "error": f"{type(exc).__name__}: {exc}"}


def main():
    result = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "CPU-only setup-and-close; not an io_uring transfer qualification",
        "kernel": platform.release(), "machine": platform.machine(),
        "python_version": platform.python_version(), "effective_uid": os.geteuid(),
        "syscall_number": 425, "depth": 2, "flags": 0,
        "params_bytes": ctypes.sizeof(Params), "setup_attempted": False,
        "mmap_attempted": False, "enter_attempted": False,
        "io_submissions": 0, "gpu_operations": 0, "network_operations": 0,
        "sysctl_changed": False, "privileges_changed": False, "seccomp_changed": False,
    }
    result["io_uring_disabled"] = read_optional("/proc/sys/kernel/io_uring_disabled")
    result["io_uring_group"] = read_optional("/proc/sys/kernel/io_uring_group")
    status = read_optional("/proc/self/status")
    wanted = ("CapEff", "NoNewPrivs", "Seccomp", "Seccomp_filters")
    result["process_security"] = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in (status["value"] or "").splitlines()
        if line.split(":", 1)[0] in wanted
    }
    result["process_status_read_error"] = status["error"]
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        result["rlimit_memlock"] = {
            "soft": "unlimited" if soft == resource.RLIM_INFINITY else soft,
            "hard": "unlimited" if hard == resource.RLIM_INFINITY else hard,
            "unit": "bytes",
        }
    except (OSError, ValueError) as exc:
        result["rlimit_memlock"] = {"error": f"{type(exc).__name__}: {exc}"}
    supported = platform.machine().lower() in (
        "x86_64", "amd64", "aarch64", "arm64", "riscv64")
    if (platform.system() != "Linux" or not supported or
            ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(Params) != 120):
        result.update(status="UNSUPPORTED_OS_OR_ABI", exit_code=3)
        print(json.dumps(result, indent=2))
        return 3
    fd = -1
    code = 4
    try:
        params = Params()  # ctypes zero-initializes all fields, including flags.
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        ctypes.set_errno(0)
        result["setup_attempted"] = True
        fd = int(libc.syscall(ctypes.c_long(425), ctypes.c_uint32(2),
                              ctypes.byref(params)))
        saved_errno = ctypes.get_errno()  # Capture immediately, before any other call.
        result["return_value"] = fd
        if fd < 0:
            result.update(status="SETUP_FAILED", errno=saved_errno,
                          errno_name=errno.errorcode.get(saved_errno, "UNKNOWN"),
                          error=os.strerror(saved_errno), fd_closed=None)
            code = 2
        else:
            result.update(status="SETUP_SUCCEEDED",
                          errno=None, returned_sq_entries=params.sq_entries,
                          returned_cq_entries=params.cq_entries,
                          returned_flags=params.flags, returned_features=params.features)
            code = 0
    except Exception as exc:
        result.update(status="PROBE_ERROR", error=f"{type(exc).__name__}: {exc}")
        code = 4
    finally:
        if fd >= 0:
            try:
                os.close(fd)
                result["fd_closed"] = True
            except OSError as exc:
                result.update(status="CLOSE_ERROR", fd_closed=False,
                              close_error=f"{type(exc).__name__}: {exc}")
                code = 4
    result["exit_code"] = code
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
