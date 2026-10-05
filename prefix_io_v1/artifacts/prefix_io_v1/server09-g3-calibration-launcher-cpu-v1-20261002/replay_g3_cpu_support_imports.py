"""Import real original acquisition support under normal site, with GPU blocked.

No original main, model import, runtime cache preparation, native owner creation,
or authorization is performed. This isolates support-path and PRIMARY adapter
compatibility before the next paid GPU run.
"""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys


class NoGPUImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "vllm", "py_kvcache", "simple_profiler", "flashinfer"):
            raise AssertionError("CPU support replay forbids backend import: " + fullname)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--source-lock", required=True)
    a = p.parse_args()
    root = a.project.resolve(strict=True)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    blocker = NoGPUImports()
    sys.meta_path.insert(0, blocker)
    rows = {r["path"]: r for r in json.loads((root / a.source_lock).read_text())["files"]}
    def load(relative, name):
        path = root / relative
        raw = path.read_bytes()
        assert len(raw) == rows[relative]["bytes"] and hashlib.sha256(raw).hexdigest() == rows[relative]["sha256"]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        return module
    scripts = root / "experiments/prefix_io_v1/scripts"
    sys.path[:0] = [str(scripts), str(root / "third_party/work/prefix-io-p4-02-cpu/src")]
    base = load("experiments/prefix_io_v1/scripts/native_gpu_prefix_smoke.py", "native_gpu_prefix_smoke")
    acquire = load("experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py", "_real_cpu_acquirer")
    prepare = load("experiments/prefix_io_v1/scripts/prepare_p4_gpu_next_day.py", "_real_cpu_prepare")
    storage = load("experiments/prefix_io_v1/scripts/experiment_storage.py", "experiment_storage")
    contract = load("experiments/prefix_io_v1/scripts/concurrent_pilot_contract.py", "concurrent_pilot_contract")
    heldout = load("experiments/prefix_io_v1/scripts/heldout_manifest.py", "heldout_manifest")
    original = storage.permission
    permit = prepare.permission_fields((root / "experiments/prefix_io_v1/configs/permissions.server09.g1.yaml").read_text())
    permit["approved_auxiliary_storage"] = None
    def permission(project=storage.ROOT):
        assert Path(project).resolve() == root
        return dict(permit)
    storage.permission = permission
    try:
        estimate = heldout.disk_requirement([128], 3, True)
        candidate = root / "experiments/prefix_io_v1/runs/server09-g3-calibration-shared-01/storage"
        assert not candidate.exists()
        validated, auxiliary = storage.authorized_path(candidate)
        preflight = storage.preflight(candidate, estimate["required_free_bytes"] - estimate["floor_bytes"])
        assert validated == candidate and auxiliary is None
        delta = contract.acquisition_delta(1024, 1)
        depth = contract.acquisition_io_depth(1024, 1, 4)
        assert depth == 4 and delta == {}
        assert [len(acquire.prompt(128, rep)) for rep in range(3)] == [129, 129, 129]
        assert base.MODEL_ID == "Qwen/Qwen2.5-7B-Instruct"
    finally:
        storage.permission = original
    assert not candidate.exists()
    assert storage.permission is original
    assert not any(n.split(".")[0] in ("torch", "vllm", "py_kvcache", "simple_profiler", "flashinfer") for n in sys.modules)
    print(json.dumps(dict(status="PASS_REAL_ORIGINAL_CPU_SUPPORT_IMPORTS", GPU_operations=0,
        model_loaded=False, original_main_called=False, cache_directories_created=0,
        normal_site=True, effective_PRIMARY_only=True, permission_helper_restored=True,
        original_acquisition_source=rows["experiments/prefix_io_v1/scripts/acquire_native_aio_costs.py"],
        native_layout_not_created=True, original_disk_estimate=estimate, actual_cpu_storage_preflight=preflight,
        raw_path_qualified=False, production_qualified=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
