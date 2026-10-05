"""CPU candidate: adapt one absent optional capability query, never imports.

The original BoundAuthorFinder and all author source bytes remain unchanged.
Installation is explicit/process-local; no GPU/framework operation is performed.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import importlib.machinery as machinery
from pathlib import Path
import sys
import types
import weakref
import zipimport

AUTHOR = "third_party/work/vllm-author-p4-02-cpu"
QUALIFY = "experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py"
OPTIONAL_NAME = "vllm.third_party.deep_gemm"
PARENT_RELATIVE = AUTHOR + "/vllm/third_party"
IMPORT_UTILS = AUTHOR + "/vllm/utils/import_utils.py"
SOURCE_PINS = (
    (QUALIFY, 29692, "5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d"),
    (PARENT_RELATIVE + "/__init__.py", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (IMPORT_UTILS, 14589, "f14b07131a4d49955150016dbfa34f7d8d5f586b1545a1173e0d4f904578b18b"),
    (AUTHOR + "/vllm/utils/deep_gemm.py", 19901, "99e316620e3cdda2db7d9fd7a4ae5968e9c9ef98d7399d62a2c4a5c506627854"),
    (AUTHOR + "/vllm/model_executor/warmup/deep_gemm_warmup.py", 13044, "b05db0127b19fc38da78e96c74df279e2cd53e0c262e9a7cad53486cf21f3414"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(root, relative):
    path = root / relative
    for cursor in (path, *path.parents):
        if cursor == root:
            break
        require(not cursor.is_symlink(), "pinned source/parent symlink rejected")
    require(path.resolve(strict=True) == path and path.is_relative_to(root),
            "pinned source path not canonical")
    return path


def verify_source_pins(root, refs):
    """Read only: no directory creation, environment change or import hook."""
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink() and
            root.resolve(strict=True) == root, "canonical existing project root")
    require(type(refs) is dict, "source refs must be exact dict")
    source_bytes = {}
    for relative, nbytes, sha in SOURCE_PINS:
        row = refs.get(relative)
        require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and
                type(row["path"]) is str and type(row["bytes"]) is int and
                type(row["sha256"]) is str and
                row == dict(path=relative, bytes=nbytes, sha256=sha),
                "exact pinned original source ref required")
        path = _canonical(root, relative)
        require(path.is_file() and path.stat().st_size == nbytes,
                "pinned original source size drift")
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == sha, "pinned original source SHA drift")
        source_bytes[relative] = raw
    return root, source_bytes


def _standard_loaders():
    return ((machinery.ExtensionFileLoader, tuple(machinery.EXTENSION_SUFFIXES)),
            (machinery.SourceFileLoader, tuple(machinery.SOURCE_SUFFIXES)),
            (machinery.SourcelessFileLoader, tuple(machinery.BYTECODE_SUFFIXES)))


def _validate_search_path(parent):
    hooks = sys.path_hooks
    require(type(hooks) is list and len(hooks) == 2 and hooks[0] is zipimport.zipimporter,
            "unknown path hook")
    expected = machinery.FileFinder.path_hook(*_standard_loaders())
    hook = hooks[1]
    require(type(hook) is types.FunctionType and hook.__code__ is expected.__code__ and
            hook.__closure__ is not None and len(hook.__closure__) == 2 and
            hook.__closure__[0].cell_contents is machinery.FileFinder,
            "nonstandard FileFinder path hook")
    details = hook.__closure__[1].cell_contents
    require(type(details) is tuple and len(details) == 3 and
            all(type(row) is tuple and len(row) == 2 and
                type(row[1]) in (list, tuple) and
                all(type(suffix) is str for suffix in row[1]) for row in details),
            "nonstandard FileFinder loader fields")
    require(tuple((loader, tuple(suffixes)) for loader, suffixes in details) == _standard_loaders(),
            "nonstandard FileFinder loaders")
    cached = sys.path_importer_cache.get(str(parent))
    if cached is not None:
        expected_loaders = [(suffix, loader) for loader, suffixes in _standard_loaders()
                            for suffix in suffixes]
        require(type(cached) is machinery.FileFinder and cached.path == str(parent) and
                cached._loaders == expected_loaders and
                "find_spec" not in vars(cached) and "find_loader" not in vars(cached),
                "unknown or altered author parent importer")


def _verify_tree_absence(root, refs):
    parent = _canonical(root, PARENT_RELATIVE)
    require(parent.is_dir(), "real locked author parent directory required")
    init = _canonical(root, PARENT_RELATIVE + "/__init__.py")
    require(init.is_file() and init.stat().st_size == 0 and
            hashlib.sha256(init.read_bytes()).hexdigest() == SOURCE_PINS[1][2],
            "empty locked author parent source drift")
    for row_path in refs:
        require(type(row_path) is str, "invalid source ref path")
        for prefix in (PARENT_RELATIVE + "/deep_gemm",
                       "third_party/work/vllm-author-build/vllm/third_party/deep_gemm"):
            require(row_path != prefix and not row_path.startswith(prefix + "/") and
                    not row_path.startswith(prefix + "."),
                    "optional source/binary binding already present")
    for entry in parent.iterdir():
        require(entry.name != "deep_gemm" and not entry.name.startswith("deep_gemm."),
                "optional source/namespace/extension/link is present")
    return parent


def verify_optional_absence(root, refs):
    """Pure source/physical-absence gate; it never installs or imports author code."""
    root, _ = verify_source_pins(root, refs)
    parent = _verify_tree_absence(root, refs)
    return dict(optional_fullname=OPTIONAL_NAME, parent=str(parent),
                source_pins_verified=True, physical_absence_verified=True,
                runtime_query_exception_verified=False, original_finder_modified=False,
                GPU_collector_verified=False, production_qualified=False,
                effect_verified=False)


def _verify_absence(root, refs, finder, *, require_loaded_parent):
    require(sys.meta_path and sys.meta_path[0] is finder,
            "original bound finder must remain first")
    require(OPTIONAL_NAME not in sys.modules, "preloaded optional module rejected")
    parent = _verify_tree_absence(root, refs)
    if require_loaded_parent:
        module = sys.modules.get("vllm.third_party")
        require(type(module) is types.ModuleType and
                module.__dict__.get("__file__") == str(parent / "__init__.py") and
                type(module.__dict__.get("__path__")) is list and
                module.__path__ == [str(parent)] and
                type(module.__path__[0]) is str,
                "sole canonical locked runtime parent path required")
    _validate_search_path(parent)
    require(machinery.PathFinder.find_spec(OPTIONAL_NAME, [str(parent)]) is None,
            "standard PathFinder discovered optional module")
    _validate_search_path(parent)


def _expected_functions(raw, filename):
    nodes = [node for node in ast.parse(raw, filename=filename).body
             if isinstance(node, ast.FunctionDef) and
             node.name in ("_has_module", "has_deep_gemm")]
    require({node.name for node in nodes} == {"_has_module", "has_deep_gemm"},
            "original optional capability functions absent")
    namespace = dict(cache=functools.cache, importlib=importlib)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec",
                 dont_inherit=True), namespace)
    return namespace


class CapabilityProbeInstallation:
    """Restore only our own process-local wrapper; preserve foreign overrides."""
    def __init__(self, module, original, wrapper):
        self._module = weakref.ref(module)
        self._original = original
        self._wrapper = wrapper

    def restore(self):
        module = self._module()
        if module is not None and module.__dict__.get("_has_module") is self._wrapper:
            module._has_module = self._original
            return True
        return False

    detach = restore

    def evidence(self):
        return dict(scope="EXACT_OPTIONAL_CAPABILITY_QUERY_ONLY",
                    optional_fullname=OPTIONAL_NAME, original_finder_modified=False,
                    original_source_modified=False, required_import_unchanged=True,
                    GPU_collector_verified=False, production_qualified=False,
                    effect_verified=False)


def install_optional_capability_probe(root, refs, base_module, finder, import_utils):
    """Explicit, reversible adapter for the original _has_module query only.

The runtime caller must already have loaded the original modules after its GPU
guard. CPU fixtures may use extracted original AST; neither is native evidence.
"""
    root, raw = verify_source_pins(root, refs)
    require(type(base_module) is types.ModuleType and
            base_module.__dict__.get("__file__") == str(root / QUALIFY) and
            type(finder) is base_module.BoundAuthorFinder and finder.root == root and
            finder.refs is refs and "find_spec" not in vars(finder) and
            "locked_binary_spec" not in vars(finder), "exact original bound finder required")
    frozen = types.ModuleType("_capability_base_identity_check")
    frozen.__file__ = str(root / QUALIFY)
    exec(compile(raw[QUALIFY], frozen.__file__, "exec", dont_inherit=True), frozen.__dict__)
    require(base_module.BoundAuthorFinder.find_spec.__code__ ==
            frozen.BoundAuthorFinder.find_spec.__code__ and
            base_module.BoundAuthorFinder.locked_binary_spec.__code__ ==
            frozen.BoundAuthorFinder.locked_binary_spec.__code__,
            "original finder code identity drift")
    require(type(import_utils) is types.ModuleType and
            import_utils.__name__ == "vllm.utils.import_utils" and
            import_utils.__dict__.get("__file__") == str(root / IMPORT_UTILS) and
            sys.modules.get(import_utils.__name__) is import_utils,
            "exact loaded author import_utils identity required")
    expected = _expected_functions(raw[IMPORT_UTILS], str(root / IMPORT_UTILS))
    original = import_utils.__dict__.get("_has_module")
    has = import_utils.__dict__.get("has_deep_gemm")
    cache_type = type(expected["_has_module"])
    require(type(original) is cache_type and type(original.__wrapped__) is types.FunctionType and
            original.__wrapped__.__code__ == expected["_has_module"].__wrapped__.__code__ and
            original.__wrapped__.__globals__ is import_utils.__dict__ and
            import_utils.__dict__.get("importlib") is importlib and
            type(has) is types.FunctionType and has.__code__ == expected["has_deep_gemm"].__code__ and
            has.__globals__ is import_utils.__dict__, "original cached capability function drift")
    find_code = base_module.BoundAuthorFinder.find_spec.__code__
    binary_code = base_module.BoundAuthorFinder.locked_binary_spec.__code__
    has_code = original.__wrapped__.__code__

    @functools.wraps(original)
    def wrapped(module_name):
        exact = type(module_name) is str and module_name == OPTIONAL_NAME
        try:
            return original(module_name)
        except ModuleNotFoundError as error:
            if not exact:
                raise
            # Message alone cannot prove origin; require these exact original
            # frames and this exact finder instance. Never retain the traceback.
            frames = []
            matched_finder = True
            trace = error.__traceback__
            while trace is not None:
                if (trace.tb_frame.f_code is has_code or trace.tb_frame.f_code is find_code or
                        trace.tb_frame.f_code is binary_code):
                    if trace.tb_frame.f_code is not has_code:
                        matched_finder = matched_finder and (
                            trace.tb_frame.f_locals.get("self") is finder and
                            trace.tb_frame.f_locals.get("fullname") == OPTIONAL_NAME)
                    frames.append(trace.tb_frame.f_code)
                trace = trace.tb_next
            if not (type(error) is ModuleNotFoundError and error.name == OPTIONAL_NAME and
                    str(error) == OPTIONAL_NAME + " has no exact locked binary fallback" and
                    matched_finder and len(frames) == 3 and
                    frames[0] is has_code and frames[1] is find_code and frames[2] is binary_code):
                raise
            verify_source_pins(root, refs)
            _verify_absence(root, refs, finder, require_loaded_parent=True)
            return False

    import_utils._has_module = wrapped
    return CapabilityProbeInstallation(import_utils, original, wrapped)


def install_capability_probe(module, finder, root, refs):
    """Convenience runtime protocol; original loader must register its module."""
    base = sys.modules.get(type(finder).__module__)
    require(type(base) is types.ModuleType, "original bound finder module unavailable")
    return install_optional_capability_probe(root, refs, base, finder, module)
