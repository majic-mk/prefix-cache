"""Stdlib asset/layout tests and exact existing FlashInfer AST replay, no GPU."""
import argparse
import ast
import contextlib
import copy
import functools
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--selection-receipt", type=Path, default=Path(__file__).parents[2] /
                    "prefix_io_v1_server09_g2_20261001/ACTUAL_CPU_CUDA13_LAYOUT_AND_SOURCE.json")
parser.add_argument("--source-root", type=Path)
args, rest = parser.parse_known_args()
sys.argv = [sys.argv[0], *rest]
spec = importlib.util.spec_from_file_location("cuda13_overlay_cpu_test", Path(__file__).with_name("cuda13_sdk_overlay.py"))
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def ref(path):
    raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def tree_ref(path):
    files = []
    for p in sorted(path.rglob("*")):
        if p.is_file():
            row = ref(p); row["path"] = p.relative_to(path).as_posix(); files.append(row)
    files.sort(key=lambda row: row["path"])
    return dict(path=str(path), bytes=sum(row["bytes"] for row in files),
                sha256=M.tree_manifest_digest(files), files=files)


def fixture(root):
    toolkit = root / "cu13"
    for relative in ("bin/nvcc", "bin/ptxas", "bin/nvlink", "nvvm/bin/cicc",
                     "include/cuda.h", "include/cuda_runtime.h", "include/crt/host_config.h",
                     "lib/libcudart.so.13"):
        p = toolkit / relative; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(relative.encode())
    driver = root / "actual-libcuda.so.580.0"; driver.write_bytes(b"CPU fixture driver bytes, never loaded")
    proof_path = root / "compiler-cpu-proof.json"
    trees = {name: tree_ref(toolkit / name) for name in ("bin", "include", "nvvm")}
    directories = {}
    for name, tree in trees.items():
        files = [dict(path=str(Path(tree["path"]) / f["path"]), relative=f["path"], bytes=f["bytes"], sha256=f["sha256"])
                 for f in tree["files"]]
        directories[name] = dict(root=tree["path"], files=files, file_count=len(files),
                                 total_bytes=tree["bytes"], inventory_sha256=M.tree_manifest_digest(files))
    manifest_path = root / "CUDA13_SOURCE_INVENTORY.json"
    manifest_path.write_text(json.dumps(dict(schema_version=1, status="CPU_ONLY_EXISTING_CUDA13_SOURCE_INVENTORY",
        toolkit_root=str(toolkit), directories=directories, cudart=ref(toolkit / "lib/libcudart.so.13"),
        driver=ref(driver), GPU_operations=0, framework_imports=0, system_or_driver_modified=False, downloads=False)))
    outputs = []
    for name in ("probe.cu", "probe.o", "probe.so"):
        p = root / name; p.write_bytes(b"CPU fixture bytes, no compiler execution or shared-object load")
        outputs.append(ref(p))
    nvcc = str(root / "sdk-overlay/bin/nvcc")
    proof_path.write_text(json.dumps(dict(schema_version=1, status="PASS_CPU_CUDA13_SM120F_COMPILE_AND_HOST_LINK_ONLY",
        manifest_ref=ref(manifest_path), source_inventory_counts={k: len(t["files"]) for k, t in trees.items()},
        source_inventory_bytes=sum(t["bytes"] for t in trees.values()), driver_ref=ref(driver),
        cudart_ref=ref(toolkit / "lib/libcudart.so.13"), compiler_commands=[
            dict(command=[nvcc, "--version"], exit=0, stdout="release 13.0, V13.0.88"),
            dict(command=[nvcc, "-gencode=arch=compute_120f,code=sm_120f", "-c", str(root / "probe.cu")], exit=0),
            dict(command=["c++", "-shared", "-lcudart", "-lcuda"], exit=0),
            dict(command=["readelf", "-d", str(root / "probe.so")], exit=0, stdout="[libcudart.so.13]")],
        output_refs=outputs, framework_imports=0, GPU_operations=0, GPU_kernels_executed=0,
        shared_object_loaded=False, system_or_driver_modified=False, downloads=False, stubs_on_runtime_library_path=False)))
    return dict(schema_version=1, status="CPU_VERIFIED_EXISTING_CUDA13_ASSETS", toolkit_root=str(toolkit),
                trees=trees, cudart=ref(toolkit / "lib/libcudart.so.13"), driver=ref(driver),
                compiler_cpu_proof=ref(proof_path), inventory_ref=ref(manifest_path))


class AssetContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="prefix-io-cuda13-cpu-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.pin = fixture(self.root)

    def test_all_existing_members_bytes_sha_and_total_inventory_are_verified(self):
        self.assertEqual(M.verify_assets(self.pin), self.root / "cu13")
        loaded = M.load_audited_assets(self.pin["inventory_ref"], self.pin["compiler_cpu_proof"])
        self.assertEqual(loaded, self.pin)

    def test_source_drift_and_unexpected_file_reject_before_overlay_creation(self):
        (self.root / "cu13/bin/nvcc").write_bytes(b"changed!")
        with self.assertRaises(ValueError): M.verify_assets(self.pin)
        self.pin = fixture(self.root)
        (self.root / "cu13/include/unknown.h").write_bytes(b"extra")
        with self.assertRaises(ValueError): M.verify_assets(self.pin)

    def test_inventory_totals_order_duplicates_and_bool_bytes_reject(self):
        for mutate in (lambda t: t.update(bytes=True), lambda t: t.update(sha256="0" * 64),
                       lambda t: t["files"].reverse(), lambda t: t["files"].append(t["files"][0]),
                       lambda t: t["files"][0].update(bytes=False)):
            tree = copy.deepcopy(self.pin["trees"]["include"]); mutate(tree)
            with self.assertRaises(ValueError): M.verify_tree(tree)

    def test_relative_paths_bound_and_expected_members_required(self):
        for value in ("../x", "/x", "a//b", "a/./b", "C:/x", "a\\b", ""):
            with self.assertRaises(ValueError): M.relative_file(value)
        bad = copy.deepcopy(self.pin)
        bad["trees"]["bin"]["files"] = bad["trees"]["bin"]["files"][1:]
        with self.assertRaises(ValueError): M.verify_assets(bad)

    def test_driver_cudart_proof_sha_and_compiler_semantics_fail_closed(self):
        (self.root / "actual-libcuda.so.580.0").write_bytes(b"changed")
        with self.assertRaises(ValueError): M.verify_assets(self.pin)
        self.pin = fixture(self.root)
        p = Path(self.pin["compiler_cpu_proof"]["path"])
        proof = json.loads(p.read_text())
        wrong_commands = copy.deepcopy(proof["compiler_commands"]); wrong_commands[1]["exit"] = 1
        for key, value in (("framework_imports", False), ("GPU_operations", 1), ("compiler_commands", wrong_commands),
                           ("shared_object_loaded", True), ("source_inventory_bytes", True), ("manifest_ref", {})):
            bad = dict(proof, **{key: value}); p.write_text(json.dumps(bad))
            self.pin["compiler_cpu_proof"] = ref(p)
            with self.assertRaises(ValueError): M.verify_assets(self.pin)

    def test_environment_is_process_copy_without_gpu_or_arch_mutations(self):
        old = dict(PATH="old/bin", LD_LIBRARY_PATH="old/lib", CUDA_VISIBLE_DEVICES="uuid", OTHER="keep")
        before = dict(old); process_before = dict(os.environ)
        overlay = self.root / "run/details/runtime-cache/cuda13-sdk"
        env = M.planned_environment(overlay, old)
        self.assertEqual(old, before); self.assertEqual(dict(os.environ), process_before)
        self.assertEqual(env["CUDA_HOME"], str(overlay)); self.assertEqual(env["CUDA_PATH"], str(overlay))
        self.assertEqual(env["FLASHINFER_NVCC"], str(overlay / "bin/nvcc"))
        self.assertEqual(env["CUDACXX"], env["FLASHINFER_NVCC"])
        self.assertEqual(env["PATH"], str(overlay / "bin") + os.pathsep + "old/bin")
        self.assertEqual(env["LD_LIBRARY_PATH"], str(overlay / "lib64") + os.pathsep + "old/lib")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "uuid")

    def test_arch_spoof_or_disabled_jit_environment_rejected(self):
        for key in M.FORBIDDEN:
            with self.assertRaises(ValueError): M.planned_environment(self.root, {key: ""})
        with self.assertRaises(ValueError): M.planned_environment(self.root, {"PATH": object()})
        for key in ("FLASHINFER_EXTRA_CFLAGS", "FLASHINFER_EXTRA_CUDAFLAGS", "FLASHINFER_EXTRA_LDFLAGS",
                    "FLASHINFER_NVCC_LAUNCHER", "FLASHINFER_CXX_LAUNCHER", "FLASHINFER_NVCC", "CUDACXX", "CUDA_HOME"):
            with self.assertRaises(ValueError): M.planned_environment(self.root, {key: "foreign"})
        with self.assertRaises(ValueError): M.planned_environment(self.root, {"LD_LIBRARY_PATH": str(self.root / "stubs")})

    @unittest.skipUnless(os.name == "posix", "Actual Linux symlinks exercised on server; Windows cannot create them here")
    def test_real_linux_overlay_new_only_exact_links_linker_only_actual_driver(self):
        run = self.root / "run"; cache = run / "details/runtime-cache"; cache.mkdir(parents=True)
        before = dict(os.environ)
        result = M.prepare_overlay(approved_run_root=run, runtime_cache=cache, pin=self.pin,
                                   inherited_environment={"PATH": "/usr/bin"})
        overlay = Path(result.overlay)
        self.assertEqual(len(result.links), 6)
        for link, target in result.links:
            self.assertTrue(Path(link).is_symlink()); self.assertEqual(Path(link).resolve(), Path(target))
        self.assertTrue((overlay / "lib64").is_dir()); self.assertFalse((overlay / "lib64").is_symlink())
        self.assertEqual([p.name for p in (overlay / "lib64/stubs").iterdir()], ["libcuda.so"])
        self.assertNotIn(str(overlay / "lib64/stubs"), dict(result.environment)["LD_LIBRARY_PATH"])
        self.assertFalse(result.GPU_collector_verified); self.assertFalse(result.production_qualified)
        self.assertEqual(dict(os.environ), before)
        with self.assertRaises(FileExistsError):
            M.prepare_overlay(approved_run_root=run, runtime_cache=cache, pin=self.pin, inherited_environment={})

    @unittest.skipUnless(os.name == "posix", "Server-only Linux containment checks")
    def test_containment_wrong_location_and_symlinked_tree_reject(self):
        run = self.root / "run"; cache = run / "details/runtime-cache"; cache.mkdir(parents=True)
        with self.assertRaises(ValueError):
            M.prepare_overlay(approved_run_root=run, runtime_cache=self.root, pin=self.pin, inherited_environment={})
        os.symlink(self.root / "cu13/include", self.root / "linked-include", target_is_directory=True)
        bad = copy.deepcopy(self.pin["trees"]["include"]); bad["path"] = str(self.root / "linked-include")
        with self.assertRaises(ValueError): M.verify_tree(bad)


class SourceSelectionContracts(unittest.TestCase):
    def setUp(self):
        if args.source_root is None:
            receipt = json.loads(args.selection_receipt.read_text(encoding="utf-8"))
            source = json.loads(receipt["result"]["stdout"])
            self.files = {row["path"].split("site-packages/", 1)[1]: row for row in source["files"]}
        else:
            pins = [
                ("flashinfer/compilation_context.py", 3909, "e55c7f58a83810590e18686954527cc62735a6e91c3fa67286a5e7f00afc9be5"),
                ("flashinfer/jit/cpp_ext.py", 12445, "9d20f28baed969411220456b140a2786a081a2b7195d2433a70daec3c8ae7403"),
                ("flashinfer/jit/utils.py", 2644, "a32b738a91bafbe392c69e10fd11533ff98adbecbba9d769a19f2c5e920f5f31"),
                ("torch/version.py", 317, "323d35171ef1184f1d7db3bbd1f3d3e227e0e826be8fd52200778346e17c873f"),
            ]
            site = args.source_root / ".venv/lib/python3.12/site-packages"
            self.files = {name: dict(text=(site / name).read_bytes().decode("utf-8"), bytes=nbytes, sha256=sha)
                          for name, nbytes, sha in pins}
        for row in self.files.values():
            raw = row["text"].encode("utf-8")
            self.assertEqual(len(raw), row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["sha256"])
        self.cpp_tree = ast.parse(self.files["flashinfer/jit/cpp_ext.py"]["text"])
        self.ctx_tree = ast.parse(self.files["flashinfer/compilation_context.py"]["text"])

    def functions(self, names, namespace):
        body = [copy.deepcopy(n) for n in self.cpp_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in body}, set(names))
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "actual_cpp_ext.py", "exec"), namespace)

    def test_torch_actual_annotated_cuda_version_and_nvcc_priority_explain_failure(self):
        tree = ast.parse(self.files["torch/version.py"]["text"])
        cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.AnnAssign) and n.target.id == "cuda")
        self.assertEqual(cuda, "13.0")
        checks = []
        class Version:
            def __init__(self, value): self.v = tuple(map(int, value.split(".")))
            def __ge__(self, other): return self.v >= other.v
        ns = dict(functools=functools, os=os, re=re, Version=Version, torch=NS(version=NS(cuda=cuda)),
                  subprocess=NS(check_output=lambda argv, text: checks.append(argv) or "Cuda release 12.8, V12.8.0",
                                CalledProcessError=RuntimeError))
        self.functions({"get_cuda_path", "get_cuda_version", "is_cuda_version_at_least"}, ns)
        with patch.dict(os.environ, {"CUDA_HOME": "/usr/local/cuda-12.8"}):
            self.assertFalse(ns["is_cuda_version_at_least"]("12.9"))
        self.assertEqual(checks, [[os.path.join("/usr/local/cuda-12.8", "bin/nvcc"), "--version"]])
        # Existing torch cu130 does not override a successfully detected 12.8 nvcc.

    def test_original_get_cuda_path_env_priority_and_normalized_sm120f(self):
        class Version:
            def __init__(self, value): self.v = tuple(map(int, value.split(".")))
            def __ge__(self, other): return self.v >= other.v
        queried = []
        ns = dict(functools=functools, os=os, re=re, Version=Version, torch=NS(version=NS(cuda="13.0")),
                  subprocess=NS(check_output=lambda argv, text: queried.append(argv) or "release 13.0, V13.0.88",
                                CalledProcessError=RuntimeError))
        self.functions({"get_cuda_path", "get_cuda_version", "is_cuda_version_at_least"}, ns)
        fake = ModuleType("flashinfer.jit.cpp_ext"); fake.is_cuda_version_at_least = ns["is_cuda_version_at_least"]
        klass = copy.deepcopy(next(n for n in self.ctx_tree.body if isinstance(n, ast.ClassDef)))
        ns.update(torch=NS(cuda=NS(device_count=lambda: 1, get_device_capability=lambda _: (12, 0))),
                  logger=logging.getLogger("cpu-fixture"))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[klass], type_ignores=[])), "actual_context.py", "exec"), ns)
        with patch.dict(sys.modules, {"flashinfer.jit.cpp_ext": fake}), \
             patch.dict(os.environ, {"CUDA_HOME": "/new/runtime-cache/cuda13-sdk", "CUDA_PATH": "/wrong"}):
            ns["get_cuda_path"].cache_clear()
            self.assertEqual(ns["get_cuda_path"](), "/new/runtime-cache/cuda13-sdk")
            context = ns["CompilationContext"]()
            self.assertEqual(context.TARGET_CUDA_ARCHS, {(12, "0f")})
            self.assertIn("-gencode=arch=compute_120f,code=sm_120f", context.get_nvcc_flags_list())
        self.assertEqual(queried, [[os.path.join("/new/runtime-cache/cuda13-sdk", "bin/nvcc"), "--version"]])

    def test_original_generated_ninja_link_flags_require_real_lib64_aliases(self):
        ns = dict(List=list, Optional=__import__("typing").Optional, Path=Path, os=os,
                  get_cuda_path=lambda: "/new/runtime-cache/cuda13-sdk",
                  build_common_cflags=lambda home, extra: ["-isystem $cuda_home/include"],
                  build_cflags=lambda common, extra: ["$common_cflags"],
                  build_cuda_cflags=lambda common, extra: ["-gencode=arch=compute_120f,code=sm_120f"],
                  parse_env_flags=lambda name: [], join_multiline=lambda values: " ".join(values),
                  jit_env=NS(FLASHINFER_JIT_DIR=Path("/new/cached_ops")))
        self.functions({"generate_ninja_build_for_op"}, ns)
        with patch.dict(os.environ, {"FLASHINFER_NVCC": "/new/runtime-cache/cuda13-sdk/bin/nvcc"}):
            ninja = ns["generate_ninja_build_for_op"]("cpu_probe", [Path("/input/probe.cu")], None, None, None, None)
        for value in ("cuda_home = /new/runtime-cache/cuda13-sdk", "-L$cuda_home/lib64",
                      "-L$cuda_home/lib64/stubs", "-lcudart", "-lcuda",
                      "nvcc = /new/runtime-cache/cuda13-sdk/bin/nvcc"):
            self.assertIn(value, ninja)
        self.assertNotIn("/usr/local/cuda", ninja)


if __name__ == "__main__": unittest.main(verbosity=2)
