"""No-CUDA environment/asset binding, kept separate from execution approval."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil

from .exact_key import digest
from .plan import pilot_plan, plan_for_protocol


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def code_files(root):
    root = Path(root).resolve()
    paths = list((root / "src/pacekv").glob("*.py"))
    paths += list((root / "scripts/pacekv").glob("*.py"))
    paths += list((root / "tests").glob("test_pacekv*.py"))
    if not paths:
        raise ValueError("missing pilot code")
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)}


def host_available_bytes():
    """Use the stricter host/cgroup limit. None means unobserved, not infinity."""
    observations = []
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                observations.append(int(line.split()[1]) * 1024)
    for maximum, current in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        if Path(maximum).is_file() and Path(current).is_file():
            limit = Path(maximum).read_text().strip()
            if limit != "max":
                observations.append(max(0, int(limit) - int(Path(current).read_text())))
    return min(observations) if observations else None


def audit_environment(root, model_path, stack="torch27"):
    model = Path(model_path).resolve()
    problems = []
    dependencies = {}
    for name in ("torch", "transformers", "numpy", "safetensors"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
            problems.append("missing dependency: " + name)
    plan = pilot_plan(stack)
    if dependencies.get("torch") != plan["torch_version"] + "+cu128":
        problems.append("RTX 5090 pilot requires Torch %s+cu128 wheel" % plan["torch_version"])
    if dependencies.get("transformers") != plan["engine"].removeprefix("transformers_"):
        problems.append("pilot adapter requires Transformers 4.40.2")
    if model.name != plan["revision"]:
        problems.append("model snapshot revision differs")
    candidates = sorted(p for p in model.iterdir() if p.is_file() and
                        p.suffix in (".json", ".model", ".safetensors")) if model.is_dir() else []
    assets = {p.name: dict(sha256=file_sha(p), bytes=p.stat().st_size) for p in candidates}
    if "config.json" not in assets or not any(n.endswith(".safetensors") for n in assets):
        problems.append("missing config or safetensors weights")
    if not any(n.startswith("tokenizer") for n in assets):
        problems.append("missing tokenizer")
    if "config.json" in assets:
        cfg = json.loads((model / "config.json").read_text())
        if cfg.get("architectures") != ["MistralForCausalLM"]:
            problems.append("model architecture differs")
    host_free = host_available_bytes()
    disk_free = shutil.disk_usage(root).free
    capacity_pending = []
    if host_free is None or host_free < plan["minimum_host_headroom_bytes"]:
        capacity_pending.append("GPU-time model loader needs observed >=32 GiB host/cgroup headroom")
    if disk_free < 2 * 1024**3:
        problems.append("need >=2 GiB output space without deleting historical artifacts")
    code = code_files(root)
    body = dict(protocol=plan["protocol"], code_files=code, code_digest=digest(code),
                plan_sha256=plan["plan_sha256"], model_path=str(model), assets=assets,
                dependencies=dependencies, python=platform.python_version(),
                capacity=dict(host_available_bytes=host_free, output_disk_free_bytes=disk_free),
                runtime_capacity_pending=capacity_pending,
                runtime_capacity_recheck_required=True,
                cpu_asset_preflight_passed=not problems, pending=problems,
                cuda_initialized=False, model_loaded=False, gpu_validated=False,
                native_serving_ready=False)
    body["audit_sha256"] = digest(body)
    return body


def verify_audit(audit, root):
    body = dict(audit)
    sha = body.pop("audit_sha256")
    if digest(body) != sha or not body["cpu_asset_preflight_passed"]:
        raise ValueError("invalid/failed environment audit")
    try:
        plan = plan_for_protocol(body.get("protocol"))
    except ValueError as exc:
        raise ValueError("environment audit belongs to another hardware/protocol plan") from exc
    if body.get("protocol") != plan["protocol"] or body.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("environment audit belongs to another hardware/protocol plan")
    if code_files(root) != body["code_files"]:
        raise ValueError("code changed since preflight")
    for name, info in body["assets"].items():
        p = Path(body["model_path"]) / name
        if p.stat().st_size != info["bytes"] or file_sha(p) != info["sha256"]:
            raise ValueError("model asset changed: " + name)
    for name, version in body["dependencies"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("dependency changed: " + name)


def write_new_json(path, value):
    """Never overwrite evidence, including a failed result."""
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
