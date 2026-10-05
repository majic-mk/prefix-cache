"""Probe fixed author's ring constructor and close it; no file I/O/GPU."""
import datetime
import errno
import hashlib
import json
from pathlib import Path
import sys
import py_kvcache.liburing_file as native
expected=Path("third_party/work/py-kvcache/py_kvcache/liburing_file.py").resolve()
actual=Path(native.__file__).resolve()
if actual!=expected:
    raise RuntimeError("unexpected native library source: "+str(actual))
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            implementation="fixed author LiburingRing(2)",
            source=str(actual),source_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
            depth=2,gpu_operations=0,io_submissions=0,model_loaded=False)
try:
    ring=native.LiburingRing(2)
    ring.close()
    result.update(status="RING_SETUP_MMAP_CLOSED_ONLY",available=True,exit_code=0)
except OSError as exc:
    result.update(status="NATIVE_RING_UNAVAILABLE",available=False,errno=exc.errno,
                  errno_name=errno.errorcode.get(exc.errno),error=str(exc),exit_code=2)
print(json.dumps(result,indent=2))
raise SystemExit(result["exit_code"])
