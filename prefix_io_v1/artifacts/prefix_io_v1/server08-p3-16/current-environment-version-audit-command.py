#!/usr/bin/env python3
"""Read-only metadata/ELF inventory. Does not import framework or native modules."""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import sysconfig
import time
import urllib.parse

ROOT = Path("/root/autodl-tmp/prefix-io-v1-handoff/project")
OUT = ROOT / "artifacts/prefix_io_v1/server08-p3-16"
LOCK = OUT / "current-environment-version-lock.json"
RECEIPT = OUT / "current-environment-version-audit-command-receipt.json"
SITE = ROOT / ".venv/lib/python3.12/site-packages"
START = datetime.datetime.now(datetime.timezone.utc).isoformat()
commands = []
errors = []

def small_file(path, limit=2*1024*1024):
    p = Path(path)
    if not p.is_file():
        return {"path":str(p),"exists":False}
    s = p.stat()
    if s.st_size > limit:
        raise RuntimeError("bounded text file exceeded: " + str(p))
    b = p.read_bytes()
    return {"path":str(p),"realpath":str(p.resolve()),"exists":True,
            "size_bytes":len(b),"sha256":hashlib.sha256(b).hexdigest(),
            "text":b.decode("utf-8")}

def redacted_url(value):
    if not isinstance(value,str):
        return value
    if "://" not in value:
        return value
    p = urllib.parse.urlsplit(value)
    netloc = p.netloc.rsplit("@",1)[-1]
    if "@" in p.netloc:
        netloc = "[userinfo-redacted]@" + netloc
    return urllib.parse.urlunsplit((p.scheme,netloc,p.path,"",""))

def cmd(argv):
    before = time.monotonic()
    try:
        p = subprocess.run(argv,stdin=subprocess.DEVNULL,capture_output=True,
                           text=True,timeout=12,
                           env=dict(os.environ,GIT_OPTIONAL_LOCKS="0",
                                    PYTHONDONTWRITEBYTECODE="1"))
        stdout = p.stdout
        if argv[:1] == ["git"] and "get-url" in argv:
            stdout = redacted_url(stdout.strip()) + ("\n" if stdout else "")
        result = {"argv":argv,"shell_command":shlex.join(argv),"exit":p.returncode,
                  "stdout":stdout,"stderr":p.stderr,
                  "wall_seconds":time.monotonic()-before}
    except (OSError,subprocess.TimeoutExpired) as e:
        result = {"argv":argv,"shell_command":shlex.join(argv),"exit":None,
                  "error":type(e).__name__+": "+str(e),
                  "wall_seconds":time.monotonic()-before}
    commands.append(result)
    return result

def literal_assignments(record):
    values = {}
    if not record.get("exists"):
        return values
    for n in ast.parse(record["text"]).body:
        if isinstance(n,ast.Assign):
            names = [t.id for t in n.targets if isinstance(t,ast.Name)]
        elif isinstance(n,ast.AnnAssign):
            names = [n.target.id] if isinstance(n.target,ast.Name) else []
        else:
            continue
        if getattr(n,"value",None) is None:
            continue
        try:
            v = ast.literal_eval(n.value)
        except (ValueError,TypeError):
            continue
        for name in names:
            values[name] = v
    return values

def permission_fields(text):
    result = {}
    section = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("- "):
            if section != "approved_gpu_ids":
                raise ValueError("unexpected YAML list")
            result[section].append(line[2:].strip())
            continue
        indent = len(line)-len(line.lstrip())
        key,sep,value = line.strip().partition(":")
        if not sep:
            raise ValueError("unsupported permission YAML")
        value = value.strip()
        if not value:
            section = key
            result[key] = [] if key == "approved_gpu_ids" else {}
            continue
        scalar = True if value=="true" else False if value=="false" else int(value) if value.isdigit() else value
        if indent:
            if section != "approved_auxiliary_storage":
                raise ValueError("unexpected YAML nesting")
            result[section][key] = scalar
        else:
            result[key] = scalar
            section = None
    return result

def elf_info(path):
    p = Path(path)
    s = p.stat()
    with p.open("rb") as f:
        header = f.read(64)
    record = {"path":str(p),"realpath":str(p.resolve()),"is_symlink":p.is_symlink(),
              "size_bytes":s.st_size,"mtime_ns":s.st_mtime_ns,
              "device":s.st_dev,"inode":s.st_ino,
              "header_bytes_read":len(header),
              "header_sha256":hashlib.sha256(header).hexdigest(),
              "full_binary_sha256":None,"binary_loaded":False}
    if len(header)<20 or header[:4]!=b"\x7fELF":
        record["elf"] = False
        return record
    endian = "<" if header[5]==1 else ">" if header[5]==2 else None
    if endian is None:
        raise ValueError("invalid ELF encoding")
    etype,machine = struct.unpack(endian+"HH",header[16:20])
    record.update(elf=True,elf_class_bits={1:32,2:64}.get(header[4]),
                  byteorder={1:"little",2:"big"}.get(header[5]),
                  elf_osabi=header[7],elf_abi_version=header[8],
                  elf_type=etype,elf_machine=machine,
                  elf_machine_label={62:"x86_64",183:"aarch64"}.get(machine,"unmapped"),
                  filename_python_abi_tag=("abi3" if ".abi3." in p.name else
                    re.search(r"cpython-[^.]+",p.name).group(0) if re.search(r"cpython-[^.]+",p.name) else None))
    if shutil.which("readelf"):
        d = cmd(["readelf","-d","--wide",str(p)])
        record["readelf_dynamic_exit"] = d["exit"]
        record["dynamic_needed"] = re.findall(r"\(NEEDED\).*?\[(.*?)\]",d.get("stdout",""))
        record["dynamic_soname"] = re.findall(r"\(SONAME\).*?\[(.*?)\]",d.get("stdout",""))
        record["dynamic_rpath_runpath"] = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]",d.get("stdout",""))
        if d["exit"] != 0:
            errors.append("readelf did not complete: "+str(p))
    else:
        record["readelf_dynamic_exit"] = None
        errors.append("readelf unavailable")
    return record

def write_new(path, value):
    with path.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())

if LOCK.exists() or RECEIPT.exists():
    raise SystemExit("exclusive output already exists; no overwrite")
if shutil.disk_usage(OUT).free < 8589934592 + 1048576:
    raise SystemExit("minimum 8GiB free plus bounded metadata reserve not available")
if not sys.flags.isolated or not sys.flags.no_site:
    raise SystemExit("audit requires -I -S to avoid .pth/import hooks")
if any(n in sys.modules for n in ("torch","vllm","py_kvcache")):
    raise SystemExit("framework module unexpectedly imported")

packages = []
for d in sorted(metadata.distributions(path=[str(SITE)]),
                key=lambda x:(x.metadata.get("Name","").lower(),x.version)):
    meta = d.metadata
    path = Path(d._path)
    entry = {"name":meta.get("Name"),"version":d.version,
             "metadata_path":str(path),"site_packages_root":str(SITE),
             "requires_python":meta.get("Requires-Python"),
             "requires_dist":meta.get_all("Requires-Dist",[])}
    for name in ("METADATA","WHEEL","INSTALLER","direct_url.json"):
        file = small_file(path/name)
        if name=="direct_url.json" and file.get("exists"):
            parsed = json.loads(file["text"])
            if isinstance(parsed,dict) and "url" in parsed:
                parsed["url"] = redacted_url(parsed["url"])
            entry["direct_url"] = parsed
        if name=="WHEEL" and file.get("exists"):
            entry["wheel_tags"] = [s[5:] for s in file["text"].splitlines() if s.startswith("Tag: ")]
            entry["wheel_generator"] = [s[11:] for s in file["text"].splitlines() if s.startswith("Generator: ")]
        if name=="INSTALLER" and file.get("exists"):
            entry["installer"] = file["text"].strip()
        file.pop("text",None)
        entry.setdefault("metadata_files",[]).append(file)
    packages.append(entry)

editable = []
for p in sorted(SITE.glob("__editable__*finder.py")):
    file = small_file(p)
    values = literal_assignments(file)
    editable.append(dict(path=str(p),sha256=file["sha256"],
                         mapping=values.get("MAPPING"),executed=False))
pth = []
for p in sorted(SITE.glob("*.pth")):
    f = small_file(p)
    f["executed"] = False
    pth.append(f)

repos = []
for relative in ("third_party/upstream/py-kvcache",
                 "third_party/upstream/vllm-author",
                 "third_party/upstream/simple-profiler",
                 "third_party/work/py-kvcache",
                 "third_party/work/py-kvcache-p3-16-cpu",
                 "third_party/work/vllm-author-build"):
    p = ROOT/relative
    head = cmd(["git","-C",str(p),"rev-parse","--verify","HEAD"])
    commit = cmd(["git","-C",str(p),"show","-s","--format=%H%n%ct%n%s","HEAD"])
    origin = cmd(["git","-C",str(p),"remote","get-url","origin"])
    repos.append({"path":str(p),"head":head.get("stdout","").strip() if head["exit"]==0 else None,
                  "head_exit":head["exit"],"commit":commit.get("stdout") if commit["exit"]==0 else None,
                  "origin":origin.get("stdout","").strip() if origin["exit"]==0 else None,
                  "origin_exit":origin["exit"],
                  "working_tree_content_snapshot_performed":False})
    if head["exit"]!=0:
        errors.append("Git HEAD unavailable: "+str(p))

backend_roots = [
 ROOT/"third_party/work/vllm-author-build/vllm",
 ROOT/"third_party/work/py-kvcache-p3-16-cpu/py_kvcache",
 ROOT/"third_party/upstream/simple-profiler/python",
]
backend_scans = []
backend_paths = set()
for p in backend_roots:
    paths = sorted(p.rglob("*.so")) if p.exists() else []
    if len(paths)>64:
        raise RuntimeError("compiled backend inventory bound exceeded")
    backend_scans.append({"root":str(p),"exists":p.exists(),
                          "shared_object_count":len(paths),"paths":[str(x) for x in paths]})
    backend_paths.update(paths)
backend_paths.update(SITE.glob("torch/_C*.so"))
for name in ("libtorch.so","libtorch_cpu.so","libtorch_cuda.so","libtorch_python.so",
             "libtorch_global_deps.so","libc10.so","libc10_cuda.so"):
    p = SITE/"torch/lib"/name
    if p.exists():
        backend_paths.add(p)
libc = Path("/lib/x86_64-linux-gnu/libc.so.6")
if libc.exists():
    backend_paths.add(libc)
backends = [elf_info(p) for p in sorted(backend_paths)]

version_sources = []
for p in (SITE/"torch/version.py",ROOT/"third_party/work/vllm-author-build/vllm/_version.py"):
    f = small_file(p)
    f["literal_values"] = literal_assignments(f)
    version_sources.append(f)
io_backend = []
for name in ("linux_aio.py","liburing_file.py"):
    f = small_file(ROOT/"third_party/work/py-kvcache-p3-16-cpu/py_kvcache"/name)
    f["static_abi_lines"] = [s for s in f.get("text","").splitlines()
                            if any(x in s for x in ("x86_64","amd64","byteorder","CDLL","_NR_IO_","_NR_","206","207","208","209"))][:40]
    f.pop("text",None)
    io_backend.append(f)

permission = small_file(ROOT/"experiments/prefix_io_v1/configs/permissions.yaml")
fields = permission_fields(permission["text"])
authorization_refs = []
for relative in ("experiments/prefix_io_v1/configs/authorizations/new_server_08.json",
                 fields["approved_auxiliary_storage"]["authorization_record"],
                 "experiments/prefix_io_v1/configs/authorizations/io_uring_20260927.json"):
    f = small_file(ROOT/relative)
    f.pop("text",None)
    authorization_refs.append(f)

driver = small_file("/proc/driver/nvidia/version")
cuda_version = small_file("/usr/local/cuda/version.json")
os_release = small_file("/etc/os-release")
source = small_file(__file__)
source.pop("text",None)
unexpected = [n for n in ("torch","vllm","py_kvcache") if n in sys.modules]
if unexpected:
    errors.append("unexpected framework imports: "+repr(unexpected))
finished = datetime.datetime.now(datetime.timezone.utc).isoformat()
lock = {
 "schema_version":1,"status":"ACTUAL_VERSION_AUDIT_NOT_GPU_QUALIFICATION",
 "created_utc":START,"finished_utc":finished,"project_root":str(ROOT),
 "scope":{"metadata_only":True,"framework_imports":unexpected,
          "gpu_workload_executed":False,"gpu_qualification":False,
          "nvidia_smi_executed":False,"tests_executed":False,
          "model_or_private_cache_read":False,"downloads":False,
          "system_changes":False,"existing_files_modified":False,
          "complete_reproducible_snapshot":False,
          "compiled_backends_loaded":False,
          "full_compiled_binary_hashes_performed":False},
 "interpreter":{"argv":sys.argv,"executable":sys.executable,
                "executable_realpath":str(Path(sys.executable).resolve()),
                "version":sys.version,"version_info":list(sys.version_info),
                "audit_no_site_prefix":sys.prefix,"base_prefix":sys.base_prefix,
                "explicit_venv_root":str(ROOT/".venv"),
                "explicit_metadata_path":str(SITE),
                "venv_cfg":small_file(ROOT/".venv/pyvenv.cfg"),
                "isolated":bool(sys.flags.isolated),"no_site":bool(sys.flags.no_site),
                "sys_path":sys.path,"soabi":sysconfig.get_config_var("SOABI"),
                "extension_suffix":sysconfig.get_config_var("EXT_SUFFIX"),
                "multiarch":sysconfig.get_config_var("MULTIARCH")},
 "host":{"uname":list(platform.uname()),"platform":platform.platform(),
         "machine":platform.machine(),"byteorder":sys.byteorder,
         "pointer_bits":struct.calcsize("P")*8,
         "glibc_confstr":os.confstr("CS_GNU_LIBC_VERSION"),
         "os_release":os_release,"nvidia_driver_proc_text":driver,
         "cuda_toolkit_version_file":cuda_version},
 "installed_distributions":packages,"distribution_count":len(packages),
 "editable_finders":editable,"pth_inventory":pth,"repository_heads":repos,
 "compiled_backend_scans":backend_scans,"compiled_backends":backends,
 "compiled_backend_count":len(backends),"literal_version_source_files":version_sources,
 "native_io_backend_source_inventory":io_backend,
 "permissions":{"file":permission,"fields":fields,
                "authorization_record_files":authorization_refs,
                "new_authorization_granted_by_this_audit":False},
 "audit_source":source,"command_receipt":str(RECEIPT),"audit_errors":errors,
 "limitations":[
  "Version fields are actual installed metadata, source literals and Git HEADs; working-tree source snapshot is separately maintained by root.",
  "The P316 native overlay differs from the installed editable py-kvcache mapping; this audit does not execute a model or establish runtime import resolution.",
  "ELF headers, filename ABI tags and dynamic dependencies are recorded without loading binaries; Python/C++/CUDA ABI compatibility is not qualified here.",
  "Compiled binary full content SHA and all system packages are outside this bounded inventory; this is not a complete reproducible environment snapshot.",
  "Torch METADATA version and torch/version.py CUDA build suffix are distinct evidence fields.",
  "No inference is made about GPU workload success, method efficacy, cache merge authorization, or remaining measured GPU wall budget.",
 ],
}
receipt = {"schema_version":1,"status":"COMPLETED_METADATA_ONLY_COMMAND",
           "started_utc":START,"finished_utc":finished,
           "main_argv":[sys.executable,"-I","-S",str(Path(__file__))],
           "main_command":shlex.join([sys.executable,"-I","-S",str(Path(__file__))]),
           "source":source,"subprocess_commands":commands,
           "internal_read_operations":["explicit venv importlib.metadata distribution records",
              "small permissions/provenance/version text",
              "Git HEAD and origin without index locks",
              "bounded ELF 64-byte headers and readelf dynamic records"],
           "gpu_workload_executed":False,"tests_executed":False,
           "frameworks_imported":unexpected,"all_reads_exclude_model_and_private_cache":True,
           "main_result":"audit JSON and receipt exclusively created",
           "audit_errors":errors}
write_new(RECEIPT,receipt)
write_new(LOCK,lock)
print(json.dumps({"status":lock["status"],"lock":str(LOCK),"lock_sha256":hashlib.sha256(LOCK.read_bytes()).hexdigest(),
                  "receipt":str(RECEIPT),"receipt_sha256":hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
                  "audit_source_sha256":source["sha256"],"distributions":len(packages),
                  "compiled_backend_count":len(backends),"audit_errors":errors,
                  "gpu_workload_executed":False},ensure_ascii=False))
