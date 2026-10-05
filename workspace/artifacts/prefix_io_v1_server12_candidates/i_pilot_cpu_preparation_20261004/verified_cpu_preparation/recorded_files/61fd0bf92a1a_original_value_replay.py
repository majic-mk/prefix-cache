"""Load SHA-bound original pure values/functions for CPU replay only.

No backend package __init__, CUDA, ctypes.CDLL, model or GPU launch is imported.
Dependency imports of selected arithmetic modules are replaced by their exact
original pure symbols; no function/class AST is edited. Alignment-only module
is clearly a CPU value namespace, never an installed native backend.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import __future__

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
INPUTS = HERE.parent / "source_inputs"
PREFIXES = (
    PROJECT / "artifacts/prefix_io_v1_server11_candidates/notification_v5_gpu_entry_path_revision/common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache",
    PROJECT / "artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004/common_candidate/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache",
)
HASHES = {
    "fs_config": "7a2429d15f6d6009da5f7965ba6b15279a9ef25b348cadd29927f61b9c488698",
    "load_planner": "92581e373d7e5c0903520417fb0ce80141f2f65506bf16c7ecbde94ab1ca5fe2",
    "cost_model": "e85922054fe96dce5c0221f1cc17cc3f1b2d7addd7eae452cabad305fc412397",
    "staging": "18d2b634b894fec32d7abbf767132afbd4fc5ec66fef738ffef724f3e95e615b",
    "liburing_file": "6a8995ca6e5f49ae470e8c49dbf3d98caf2d09dd091cb6f08f9e892c051414f5",
    "profiling": "b6025d8afb9256be2a2d35df01e4bf96f3926a2948ecc88a32c2c63336b333d3",
    "_pchip": "d3f186b52ae296181ae3e547cb8516ba1538a20062bb76152c6042b13ff5d397",
    "break_even": "2419c9470df0e43558ccc729aa3927ebd4d1381e00276d4caaa57b3cf19c42a6",
    "vllm": "901cc5b9a20245e60d2200ef190979b80adb77ca5a049624c6bc4ec171a712d1",
}


def source(name):
    candidates = (INPUTS / ("py_kvcache_" + name + ".py"),) + tuple(p / (name + ".py") for p in PREFIXES)
    path = next((p for p in candidates if p.is_file()), None)
    if path is None or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError("exact original pure source missing or drifted: " + name)
    return path


def manifest_entries():
    entries = []
    for filename in ("STRONG_BASELINE_SOURCE_INPUTS.json", "AUTHOR_BENCHMARK_SOURCE_INPUTS.json"):
        rows = json.loads((INPUTS / filename).read_text(encoding="utf-8"))
        if type(rows) is not list:
            raise ValueError("actual server source manifest must be a list")
        for row in rows:
            path = INPUTS / row["local_basename"]
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError("actual author input drift: " + row["server_relative"])
            entries.append((row, path))
    return entries


def _module(name, path, *, env=None, nodes=None):
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
    if env:
        module.__dict__.update(env)
    sys.modules[name] = module
    code = path.read_text(encoding="utf-8") if nodes is None else ast.Module(body=nodes, type_ignores=[])
    exec(compile(code, str(path), "exec", flags=__future__.annotations.compiler_flag), module.__dict__)
    return module


def load_original_values():
    package = ModuleType("_strong_cpu_value_namespace")
    package.__path__ = []
    sys.modules[package.__name__] = package
    prefix = package.__name__ + "."
    fs = _module(prefix + "fs_config", source("fs_config"))  # complete original module
    be = _module(prefix + "break_even", source("break_even"))
    pchip = _module(prefix + "_pchip", source("_pchip"))
    profile = _module(prefix + "profiling", source("profiling"))
    cost_tree = ast.parse(source("cost_model").read_text(encoding="utf-8"))
    cost_nodes = [n for n in cost_tree.body if not (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("py_kvcache."))]
    cost = _module(prefix + "cost_model", source("cost_model"),
                   env=dict(Pchip=pchip.Pchip, Curve=be.Curve, CurveData=be.CurveData), nodes=cost_nodes)
    planner_tree = ast.parse(source("load_planner").read_text(encoding="utf-8"))
    planner_nodes = [n for n in planner_tree.body if not (isinstance(n, ast.ImportFrom) and n.level)]
    planner = _module(prefix + "load_planner", source("load_planner"),
                      env=dict(CostTables=cost.CostTables, add_event=profile.add_event, now_ns=profile.now_ns), nodes=planner_nodes)
    alignment_tree = ast.parse(source("liburing_file").read_text(encoding="utf-8"))
    alignment_nodes = [n for n in alignment_tree.body if (isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == "DIRECT_IO_ALIGNMENT" for t in n.targets)) or
                       (isinstance(n, ast.FunctionDef) and n.name == "align_up")]
    alignment = _module(prefix + "liburing_file", source("liburing_file"), nodes=alignment_nodes)
    staging = _module(prefix + "staging", source("staging"))
    native_tree = ast.parse(source("vllm").read_text(encoding="utf-8"))
    helpers = [n for n in native_tree.body if isinstance(n, ast.FunctionDef) and n.name in
               ("_get_nested_attr", "_first_int", "_parse_non_negative_int")]
    spec_class = next(n for n in native_tree.body if isinstance(n, ast.ClassDef) and n.name == "PyKvCacheOffloadingSpec")
    build = next(n for n in spec_class.body if isinstance(n, ast.FunctionDef) and n.name == "_build_planner")
    builder = _module(prefix + "builder", source("vllm"), nodes=helpers + [build],
                      env=dict(load_curves=be.load_curves, build_cost_tables=cost.build_cost_tables,
                               LoadPlanner=planner.LoadPlanner, StagingPool=staging.StagingPool,
                               align_up=alignment.align_up, logger=__import__("logging").getLogger("CPU-original-builder"),
                               VLLM_AVAILABLE=True, PLAN_API_AVAILABLE=True))
    return SimpleNamespace(fs=fs, break_even=be, cost=cost, planner=planner, staging=staging,
                           builder=builder, alignment=alignment,
                           cpu_fixture_capability_flags=True, gpu_runtime_qualified=False)


def load_author_values():
    manifest_entries()
    lock = json.loads((INPUTS / "dependency-lock.json").read_text(encoding="utf-8"))
    if lock["repositories"]["kvcache-experiments"]["commit"] != "0e023a84a21246b9bbc06266fa8070397eccbdc9":
        raise ValueError("author experiment version differs from retained lock")
    common = _module("_strong_author_common_cpu", INPUTS / "prefix_cache_common.py")
    replay_path = INPUTS / "shared_storage_trace_replay.py"
    tree = ast.parse(replay_path.read_text(encoding="utf-8"))
    names = {"prompt_from_sharegpt_item", "prompt_from_json_item", "load_trace_prompts", "build_global_specs", "build_parser"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    replay = _module("_strong_author_trace_cpu", replay_path, nodes=nodes,
                     env=dict(argparse=argparse, json=json, Path=Path, random=__import__("random"), RequestSpec=common.RequestSpec))
    return SimpleNamespace(common=common, replay=replay, original_functions_executed=sorted(names),
                           cluster_launcher_imported=False, network_client_imported=False)
