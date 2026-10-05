#!/usr/bin/env python3
"""CPU-only byte identity audit of the 16 files in an exact metadata audit."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import shutil
import stat
import sys
import time

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
OUT = ROOT / "artifacts/prefix_io_v1/server08-p3-16"
VERSION = OUT / "current-environment-version-lock.json"
VERSION_SHA = "0da4fe0d988bae381e8f1b821bb3df60e28dfd5474aab435a36ca0f58e1ade67"
RESULT = OUT / "compiled-backend-byte-identities.json"
RECEIPT = OUT / "compiled-backend-byte-identities-command-receipt.json"
MAX_TOTAL = 4 * 1024**3
BLOCK = 8 * 1024**2

def identity(s):
    return {"device":s.st_dev,"inode":s.st_ino,
            "size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns}

def expected(row):
    return {k:row[k] for k in ("device","inode","size_bytes","mtime_ns")}

def write_new(path,value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

started = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = time.monotonic()
if RESULT.exists() or RECEIPT.exists():
    raise SystemExit("exclusive outputs already exist")
if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
    raise SystemExit("must invoke with -I -S -B")
if any(n in sys.modules for n in ("torch","vllm","py_kvcache")):
    raise SystemExit("unexpected framework import")
if shutil.disk_usage(OUT).free < 8589934592 + 1048576:
    raise SystemExit("minimum free storage plus bounded evidence reserve unavailable")
vb = VERSION.read_bytes()
if hashlib.sha256(vb).hexdigest() != VERSION_SHA:
    raise SystemExit("exact metadata audit SHA mismatch")
version = json.loads(vb)
if version["status"] != "ACTUAL_VERSION_AUDIT_NOT_GPU_QUALIFICATION":
    raise SystemExit("wrong metadata audit status")
inventory = version["compiled_backends"]
if len(inventory) != 16 or len({r["path"] for r in inventory}) != 16:
    raise SystemExit("exact 16-file inventory required")
pre = []
for row in inventory:
    p = Path(row["path"])
    if not p.is_absolute() or str(p.resolve()) != row["realpath"]:
        raise SystemExit("backend path changed")
    s = p.stat()
    if not stat.S_ISREG(s.st_mode) or identity(s) != expected(row):
        raise SystemExit("metadata audit stat changed: "+str(p))
    if p.is_symlink() != row["is_symlink"]:
        raise SystemExit("symlink state changed")
    pre.append(identity(s))
total = sum(s["size_bytes"] for s in pre)
if total > MAX_TOTAL:
    print(json.dumps({"status":"STOPPED_SIZE_LIMIT_NO_FULL_READ",
                      "total_bytes":total,"maximum_bytes":MAX_TOTAL}))
    raise SystemExit(2)

rows = []
errors = []
total_read = 0
for original in inventory:
    p = Path(original["path"])
    row = {"path":str(p),"realpath":str(p.resolve()),
           "metadata_audit_stat":expected(original),
           "sha256":None,"bytes_read":0,"stat_stable":False,
           "explicit_binary_loading_by_this_audit":False}
    file_start = time.monotonic()
    fd = None
    try:
        before = p.stat()
        row["stat_before"] = identity(before)
        if identity(before) != expected(original):
            raise RuntimeError("pre-read metadata audit identity mismatch")
        if str(p.resolve()) != original["realpath"]:
            raise RuntimeError("resolved path changed")
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os,"O_NOFOLLOW") and not original["is_symlink"]:
            flags |= os.O_NOFOLLOW
        fd = os.open(p,flags)
        row["fstat_before"] = identity(os.fstat(fd))
        if row["fstat_before"] != row["stat_before"]:
            raise RuntimeError("open-file identity mismatch")
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RuntimeError("opened path is not a regular backend file")
        h = hashlib.sha256()
        first = True
        while True:
            block = os.read(fd,BLOCK)
            if not block:
                break
            if first:
                if block[:4] != b"\x7fELF":
                    raise RuntimeError("opened file has no ELF magic")
                first = False
            h.update(block)
            row["bytes_read"] += len(block)
        row["fstat_after"] = identity(os.fstat(fd))
        row["stat_after"] = identity(p.stat())
        if first:
            raise RuntimeError("empty ELF file")
        if row["bytes_read"] != before.st_size:
            raise RuntimeError("full-file byte count mismatch")
        if not (row["stat_before"] == row["fstat_before"] ==
                row["fstat_after"] == row["stat_after"] == expected(original)):
            raise RuntimeError("dev/inode/size/mtime changed during streaming read")
        if str(p.resolve()) != original["realpath"]:
            raise RuntimeError("resolved path changed after streaming read")
        row["stat_stable"] = True
        row["sha256"] = h.hexdigest()
    except (OSError,RuntimeError) as e:
        row["error"] = type(e).__name__+": "+str(e)
        errors.append({"path":str(p),"error":row["error"]})
    finally:
        if fd is not None:
            os.close(fd)
        row["elapsed_seconds"] = time.monotonic()-file_start
        total_read += row["bytes_read"]
        rows.append(row)

version_after = hashlib.sha256(VERSION.read_bytes()).hexdigest()
if version_after != VERSION_SHA:
    errors.append({"path":str(VERSION),"error":"version audit changed during byte identity audit"})
if total_read != total:
    errors.append({"error":"aggregate byte count differs from 16-file inventory"})
unexpected = [n for n in ("torch","vllm","py_kvcache") if n in sys.modules]
if unexpected:
    errors.append({"error":"unexpected framework modules: "+repr(unexpected)})
passed = (not errors and len(rows)==16 and
          all(r["stat_stable"] and r["sha256"] and
              r["bytes_read"]==r["stat_before"]["size_bytes"] for r in rows))
source = Path(__file__).read_bytes()
result = {
 "schema_version":1,
 "status":"PASSED_COMPILED_BACKEND_BYTE_IDENTITIES_CPU_ONLY" if passed else "FAILED_COMPILED_BACKEND_BYTE_IDENTITIES_CPU_ONLY",
 "created_utc":started,
 "finished_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "version_audit":{"path":str(VERSION),"expected_sha256":VERSION_SHA,
                  "actual_sha256_before":hashlib.sha256(vb).hexdigest(),
                  "actual_sha256_after":version_after},
 "audit_command":shlex.join([sys.executable,"-I","-S","-B",str(Path(__file__))]),
 "audit_source":{"path":str(Path(__file__)),"sha256":hashlib.sha256(source).hexdigest()},
 "actual_command_receipt":str(RECEIPT),
 "inventory_file_count":16,"audited_file_count":len(rows),
 "total_bytes":total,"total_bytes_read":total_read,"maximum_authorized_bytes":MAX_TOTAL,
 "block_bytes":BLOCK,"elapsed_seconds":time.monotonic()-start,
 "files":rows,"errors":errors,
 "scope":{"cpu_streaming_sha256_only":True,"gpu_workloads_executed":False,
          "framework_imports":unexpected,"compiled_backends_loaded_by_this_audit":False,
          "tests_executed":False,"model_or_private_cache_payload_read":False,
          "binary_payload_packaged":False,"system_or_permission_changes":False,
          "abi_or_runtime_qualification":False,"complete_reproducible_snapshot":False},
 "limitations":[
  "This receipt establishes exact full-file SHA identities for the existing 16-file metadata inventory and stable dev/inode/size/mtime during each read.",
  "No compiled backend is explicitly loaded; no Python/C++/CUDA ABI compatibility or GPU runtime qualification is asserted.",
  "Binary bodies are excluded from delivery; only JSON identities and the CPU audit command are added.",
  "The existing version audit remains unchanged; this supplemental identity record does not replace root's source snapshot."
 ]
}
write_new(RESULT,result)
print(json.dumps({"status":result["status"],"path":str(RESULT),
 "sha256":hashlib.sha256(RESULT.read_bytes()).hexdigest(),
 "file_count":len(rows),"total_bytes":total,"total_bytes_read":total_read,
 "elapsed_seconds":result["elapsed_seconds"],"errors":errors,
 "gpu_workloads_executed":False},ensure_ascii=False))
raise SystemExit(0 if passed else 1)
