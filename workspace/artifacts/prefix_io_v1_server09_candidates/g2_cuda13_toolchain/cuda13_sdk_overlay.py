"""CPU-only, append-new SDK layout for already installed CUDA 13 assets.

This helper does not import a framework, execute a compiler or initialize CUDA.
The caller must separately enforce the new run's source/scope/budget guard.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

MAX_FILES = 8192
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TREE_BYTES = 2 * 1024 * 1024 * 1024
FORBIDDEN = ("FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST", "FLASHINFER_DISABLE_JIT")
COMPILER_OVERRIDES = ("FLASHINFER_NVCC", "CUDACXX")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest_file(path, nbytes, expected):
    require(type(nbytes) is int and 0 <= nbytes <= MAX_FILE_BYTES, "bounded exact file bytes")
    require(type(expected) is str and len(expected) == 64 and
            all(c in "0123456789abcdef" for c in expected), "exact lowercase SHA256")
    require(path.is_absolute() and not path.is_symlink() and path.is_file() and
            path.resolve(strict=True) == path, "canonical existing real file")
    require(path.stat().st_size == nbytes, "existing file byte drift")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    require(h.hexdigest() == expected and path.stat().st_size == nbytes, "existing file SHA drift")


def relative_file(value):
    require(type(value) is str and value and "\\" not in value, "POSIX relative source path")
    pieces = value.split("/")
    require(all(p not in ("", ".", "..") for p in pieces), "no source traversal")
    require(not Path(value).is_absolute() and ":" not in value, "relative tree file only")
    return value


def tree_manifest_digest(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode("ascii")).hexdigest()


def verify_tree(tree):
    require(type(tree) is dict and set(tree) == {"path", "bytes", "sha256", "files"},
            "exact tree pin fields")
    root = Path(tree["path"])
    require(root.is_absolute() and root.is_dir() and not root.is_symlink() and
            root.resolve(strict=True) == root, "canonical existing real tree")
    files = tree["files"]
    require(type(files) is list and 1 <= len(files) <= MAX_FILES, "bounded full tree inventory")
    names = []
    total = 0
    for row in files:
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "exact file pin fields")
        names.append(relative_file(row["path"]))
        require(type(row["bytes"]) is int and 0 <= row["bytes"] <= MAX_FILE_BYTES, "bounded tree member")
        total += row["bytes"]
    require(names == sorted(set(names)), "sorted unique complete tree names")
    require(type(tree["bytes"]) is int and total == tree["bytes"] <= MAX_TREE_BYTES,
            "tree total byte accounting")
    require(type(tree["sha256"]) is str and tree_manifest_digest(files) == tree["sha256"],
            "tree inventory digest")
    observed = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            require(not (Path(directory) / name).is_symlink(), "no unpinned directory symlink")
        for name in filenames:
            path = Path(directory) / name
            require(not path.is_symlink() and path.is_file(), "no symlink or special tree member")
            observed.append(path.relative_to(root).as_posix())
            require(len(observed) <= MAX_FILES, "observed tree file bound")
    require(sorted(observed) == names, "complete inventory including unexpected files")
    for row in files:
        digest_file(root / row["path"], row["bytes"], row["sha256"])
    return root


def verify_assets(pin):
    require(type(pin) is dict and set(pin) == {"schema_version", "status", "toolkit_root",
            "trees", "cudart", "driver", "compiler_cpu_proof", "inventory_ref"}, "exact toolchain pin fields")
    require(type(pin["schema_version"]) is int and pin["schema_version"] == 1,
            "exact toolchain schema")
    require(pin["status"] == "CPU_VERIFIED_EXISTING_CUDA13_ASSETS", "CPU audited existing asset status")
    toolkit = Path(pin["toolkit_root"])
    require(toolkit.is_absolute() and toolkit.is_dir() and not toolkit.is_symlink() and
            toolkit.resolve(strict=True) == toolkit, "canonical installed toolkit")
    require(type(pin["trees"]) is dict and set(pin["trees"]) == {"bin", "include", "nvvm"},
            "exact linked toolkit tree set")
    roots = {name: verify_tree(pin["trees"][name]) for name in ("bin", "include", "nvvm")}
    require(all(root == toolkit / name for name, root in roots.items()), "single coherent toolkit root")
    for required in ("nvcc", "ptxas", "nvlink"):
        require((roots["bin"] / required).is_file(), "required existing compiler asset")
    require((roots["nvvm"] / "bin/cicc").is_file(), "required existing cicc")
    for required in ("cuda.h", "cuda_runtime.h", "crt/host_config.h"):
        require((roots["include"] / required).is_file(), "required installed CUDA header")
    for name in ("cudart", "driver", "compiler_cpu_proof", "inventory_ref"):
        ref = pin[name]
        require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "exact scalar file ref")
        digest_file(Path(ref["path"]), ref["bytes"], ref["sha256"])
    require(Path(pin["cudart"]["path"]) == toolkit / "lib/libcudart.so.13", "actual CUDA13 runtime SONAME")
    proof = read_json(Path(pin["compiler_cpu_proof"]["path"]))
    require(type(proof) is dict and type(proof.get("schema_version")) is int and proof["schema_version"] == 1 and
            proof.get("status") == "PASS_CPU_CUDA13_SM120F_COMPILE_AND_HOST_LINK_ONLY" and
            proof.get("manifest_ref") == pin["inventory_ref"] and
            proof.get("driver_ref") == pin["driver"] and proof.get("cudart_ref") == pin["cudart"] and
            all(type(proof.get(name)) is int and proof[name] == 0 for name in
                ("framework_imports", "GPU_operations", "GPU_kernels_executed")) and
            all(proof.get(name) is False for name in ("shared_object_loaded", "system_or_driver_modified",
                                                      "downloads", "stubs_on_runtime_library_path")),
            "actual compiler-only CPU proof semantics")
    expected_counts = {name: len(pin["trees"][name]["files"]) for name in ("bin", "include", "nvvm")}
    require(type(proof.get("source_inventory_counts")) is dict and
            all(type(v) is int for v in proof["source_inventory_counts"].values()) and
            proof["source_inventory_counts"] == expected_counts and
            type(proof.get("source_inventory_bytes")) is int and
            proof["source_inventory_bytes"] == sum(pin["trees"][n]["bytes"] for n in expected_counts),
            "compiler proof full inventory accounting")
    commands = proof.get("compiler_commands")
    require(type(commands) is list and len(commands) == 4 and
            all(type(row) is dict and type(row.get("exit")) is int and row["exit"] == 0 and
                type(row.get("command")) is list and len(row["command"]) >= 2 and
                all(type(a) is str for a in row["command"])
                for row in commands), "four successful compiler/link/readelf commands")
    require(commands[0]["command"][1:] == ["--version"] and
            "release 13.0, V13.0.88" in commands[0].get("stdout", "") and
            commands[1]["command"][0] == commands[0]["command"][0] and
            "-gencode=arch=compute_120f,code=sm_120f" in commands[1]["command"] and
            "-c" in commands[1]["command"] and commands[2]["command"][0] == "c++" and
            all(flag in commands[2]["command"] for flag in ("-shared", "-lcudart", "-lcuda")) and
            commands[3]["command"][:2] == ["readelf", "-d"] and
            "[libcudart.so.13]" in commands[3].get("stdout", ""), "actual CPU SM120f/link evidence")
    outputs = proof.get("output_refs")
    require(type(outputs) is list and len(outputs) == 3, "source/object/shared-object byte refs")
    for ref in outputs:
        require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "exact CPU output ref")
        digest_file(Path(ref["path"]), ref["bytes"], ref["sha256"])
    return toolkit


def read_json(path):
    require(path.stat().st_size <= 4 * 1024 * 1024, "bounded audit JSON")
    def exact_pairs(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "duplicate audit JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=exact_pairs)


def load_audited_assets(inventory_ref, compiler_proof_ref):
    """Read exact existing server receipts and verify assets; no mkdir/import."""
    for ref in (inventory_ref, compiler_proof_ref):
        require(type(ref) is dict and set(ref) == {"path", "bytes", "sha256"}, "exact existing audit ref")
        digest_file(Path(ref["path"]), ref["bytes"], ref["sha256"])
    document = read_json(Path(inventory_ref["path"]))
    require(type(document) is dict and type(document.get("schema_version")) is int and
            document["schema_version"] == 1 and document.get("status") == "CPU_ONLY_EXISTING_CUDA13_SOURCE_INVENTORY" and
            type(document.get("framework_imports")) is int and document["framework_imports"] == 0 and
            type(document.get("GPU_operations")) is int and document["GPU_operations"] == 0 and
            document.get("system_or_driver_modified") is False and document.get("downloads") is False,
            "original CPU-only source inventory semantics")
    directories = document.get("directories")
    require(type(directories) is dict and set(directories) == {"bin", "include", "nvvm"}, "audited directory set")
    trees = {}
    for name in ("bin", "include", "nvvm"):
        row = directories[name]
        require(type(row) is dict and set(row) == {"root", "files", "file_count", "total_bytes", "inventory_sha256"},
                "original directory inventory fields")
        require(type(row["files"]) is list and type(row["file_count"]) is int and
                row["file_count"] == len(row["files"]) and tree_manifest_digest(row["files"]) == row["inventory_sha256"],
                "original full directory inventory digest")
        converted = []
        for file in row["files"]:
            require(type(file) is dict and set(file) == {"path", "relative", "bytes", "sha256"}, "original full file ref")
            require(file["path"] == str(Path(row["root"]) / relative_file(file["relative"])), "absolute/relative source binding")
            converted.append(dict(path=file["relative"], bytes=file["bytes"], sha256=file["sha256"]))
        converted.sort(key=lambda file: file["path"])
        trees[name] = dict(path=row["root"], bytes=row["total_bytes"],
                           sha256=tree_manifest_digest(converted), files=converted)
    pin = dict(schema_version=1, status="CPU_VERIFIED_EXISTING_CUDA13_ASSETS", toolkit_root=document["toolkit_root"],
               trees=trees, cudart=document["cudart"], driver=document["driver"],
               compiler_cpu_proof=compiler_proof_ref, inventory_ref=inventory_ref)
    verify_assets(pin)
    return pin


@dataclass(frozen=True)
class OverlayEvidence:
    overlay: str
    links: tuple
    environment: tuple

    @property
    def GPU_collector_verified(self): return False

    @property
    def production_qualified(self): return False


def planned_environment(overlay, inherited):
    require(type(inherited) is dict and all(type(k) is str and type(v) is str
            for k, v in inherited.items()), "exact process environment strings")
    require(not any(key in inherited for key in FORBIDDEN), "no arch spoof or disabled JIT")
    result = dict(inherited)
    compiler = str(overlay / "bin/nvcc")
    require(all(not inherited.get(k) or inherited[k] == compiler for k in COMPILER_OVERRIDES),
            "no foreign compiler override")
    require(all(not inherited.get(k) or inherited[k] == str(overlay) for k in ("CUDA_HOME", "CUDA_PATH")),
            "no foreign CUDA root override")
    require(not any(k.startswith("FLASHINFER_EXTRA_") or k in
                    ("FLASHINFER_CXX_LAUNCHER", "FLASHINFER_NVCC_LAUNCHER") for k in inherited),
            "no compiler flag or launcher injection")
    require(not any(Path(p).name == "stubs" for p in inherited.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p),
            "linker-only stubs never enter runtime library path")
    result.update(CUDA_HOME=str(overlay), CUDA_PATH=str(overlay), FLASHINFER_NVCC=compiler,
                  CUDACXX=compiler,
                  PATH=str(overlay / "bin") + (os.pathsep + inherited["PATH"] if inherited.get("PATH") else ""),
                  LD_LIBRARY_PATH=str(overlay / "lib64") +
                      (os.pathsep + inherited["LD_LIBRARY_PATH"] if inherited.get("LD_LIBRARY_PATH") else ""))
    return result


def prepare_overlay(*, approved_run_root, runtime_cache, pin, inherited_environment):
    """Create only a new local layout; return env for this child process only.

    No rollback/deletion modifies any pre-existing path. Any failure leaves
    partial new evidence and the caller must stop before framework imports.
    """
    require(os.name == "posix", "server Linux layout required")
    run = Path(approved_run_root)
    cache = Path(runtime_cache)
    require(run.is_absolute() and run.is_dir() and not run.is_symlink() and
            run.resolve(strict=True) == run, "canonical approved existing new run root")
    require(cache == run / "details/runtime-cache" and cache.is_dir() and
            cache.resolve(strict=True) == cache, "exact real approved runtime-cache location")
    toolkit = verify_assets(pin)
    overlay = cache / "cuda13-sdk"
    environment = planned_environment(overlay, inherited_environment)
    links = [(overlay / name, toolkit / name) for name in ("bin", "include", "nvvm")]
    links += [(overlay / "lib64" / name, Path(pin["cudart"]["path"]))
              for name in ("libcudart.so", "libcudart.so.13")]
    links += [(overlay / "lib64/stubs/libcuda.so", Path(pin["driver"]["path"]))]
    overlay.mkdir(mode=0o755, exist_ok=False)
    (overlay / "lib64").mkdir(mode=0o755, exist_ok=False)
    (overlay / "lib64/stubs").mkdir(mode=0o755, exist_ok=False)
    for link, target in links:
        os.symlink(target, link, target_is_directory=target.is_dir())
    verify_assets(pin)
    for link, target in links:
        require(link.is_symlink() and Path(os.readlink(link)) == target and
                link.resolve(strict=True) == target, "created exact audited link target")
    require(not (overlay / "lib64/stubs").is_symlink() and
            [p.name for p in (overlay / "lib64/stubs").iterdir()] == ["libcuda.so"],
            "real linker-only directory to actual audited driver, no fake driver")
    return OverlayEvidence(str(overlay), tuple((str(a), str(b)) for a, b in links),
                           tuple(sorted(environment.items())))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Read-only existing CUDA13 source/link proof preflight; never runs GPU/compiler.")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-bytes", type=int, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--compiler-proof", type=Path, required=True)
    parser.add_argument("--compiler-proof-bytes", type=int, required=True)
    parser.add_argument("--compiler-proof-sha256", required=True)
    args = parser.parse_args(argv)
    inventory = dict(path=str(args.inventory.absolute()), bytes=args.inventory_bytes, sha256=args.inventory_sha256)
    proof = dict(path=str(args.compiler_proof.absolute()), bytes=args.compiler_proof_bytes, sha256=args.compiler_proof_sha256)
    try:
        pin = load_audited_assets(inventory, proof)
        print(json.dumps(dict(status="PASS_CPU_EXISTING_CUDA13_ASSET_PREFLIGHT_NO_OVERLAY_CREATED",
            source_files=sum(len(t["files"]) for t in pin["trees"].values()),
            source_bytes=sum(t["bytes"] for t in pin["trees"].values()), actual_gpu_runs=0,
            compiler_executions=0, framework_imports=0, overlay_created=False,
            GPU_collector_verified=False, production_qualified=False)))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps(dict(status="BLOCKED_CPU_CUDA13_ASSET_PREFLIGHT", reason=str(exc), actual_gpu_runs=0,
                              compiler_executions=0, framework_imports=0, overlay_created=False)))
        return 78


if __name__ == "__main__": raise SystemExit(main())
