"""CPU candidate: preserve one factually absent author optional-module probe.

No module is invented, no live hook is installed and no framework is imported.
All names except the exact audited optional fullname retain the original finder.
"""
from __future__ import annotations

import hashlib
import importlib.machinery as machinery
import os
from pathlib import Path
import sys
import types
import zipimport

AUTHOR = "third_party/work/vllm-author-p4-02-cpu"
QUALIFY = "experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py"
OPTIONAL_NAME = "vllm.third_party.deep_gemm"
PARENT_RELATIVE = AUTHOR + "/vllm/third_party"
SOURCE_PINS = (
    (QUALIFY, 29692, "5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d"),
    (PARENT_RELATIVE + "/__init__.py", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (AUTHOR + "/vllm/utils/import_utils.py", 14589, "f14b07131a4d49955150016dbfa34f7d8d5f586b1545a1173e0d4f904578b18b"),
    (AUTHOR + "/vllm/utils/deep_gemm.py", 19901, "99e316620e3cdda2db7d9fd7a4ae5968e9c9ef98d7399d62a2c4a5c506627854"),
    (AUTHOR + "/vllm/model_executor/warmup/deep_gemm_warmup.py", 13044, "b05db0127b19fc38da78e96c74df279e2cd53e0c262e9a7cad53486cf21f3414"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(root, relative):
    require(type(relative) is str and relative and "\\" not in relative,
            "exact POSIX relative source path")
    parts = relative.split("/")
    require(not Path(relative).is_absolute() and all(x not in ("", ".", "..") for x in parts),
            "source path escape")
    path = root / relative
    for cursor in (path, *path.parents):
        if cursor == root:
            break
        require(not cursor.is_symlink(), "source/parent symlink rejected")
    require(path.resolve(strict=True) == path and path.is_relative_to(root),
            "source path not canonical")
    return path


def verify_source_pins(root, refs):
    root = Path(root)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink() and
            root.resolve(strict=True) == root, "canonical existing project root")
    require(type(refs) is dict, "source refs must be exact dict")
    source_bytes = {}
    for relative, nbytes, sha in SOURCE_PINS:
        require(refs.get(relative) == dict(path=relative, bytes=nbytes, sha256=sha),
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


def _check_search_chain(finder, parent):
    # Returning None delegates to later meta finders. Unknown finders cannot be
    # probed safely and are rejected rather than allowed to invent an unlocked spec.
    require(sys.meta_path == [finder, machinery.BuiltinImporter,
                              machinery.FrozenImporter, machinery.PathFinder],
            "unknown later meta finder; optional absence not authoritative")
    hooks = sys.path_hooks
    require(type(hooks) is list and len(hooks) == 2 and hooks[0] is zipimport.zipimporter,
            "unknown path hook")
    hook = hooks[1]
    expected = machinery.FileFinder.path_hook(*_standard_loaders())
    require(type(hook) is types.FunctionType and hook.__code__ is expected.__code__ and
            hook.__closure__ is not None and len(hook.__closure__) == 2,
            "unknown FileFinder path hook")
    values = tuple(cell.cell_contents for cell in hook.__closure__)
    require(values[0] is machinery.FileFinder and values[1] == _standard_loaders(),
            "nonstandard FileFinder loader contract")
    cached = sys.path_importer_cache.get(str(parent))
    if cached is not None:
        expected_loaders = [(suffix, loader) for loader, suffixes in _standard_loaders()
                            for suffix in suffixes]
        require(type(cached) is machinery.FileFinder and cached.path == str(parent) and
                cached._loaders == expected_loaders and
                "find_spec" not in vars(cached) and "find_loader" not in vars(cached),
                "unknown or altered parent path importer")


def make_optional_absence_finder(root, refs):
    """Build an uninstalled subclass from exact frozen original source bytes.

This verifies current files only. It grants no native/GPU/runtime qualification.
The caller must separately bind this helper and explicitly install a future hook.
"""
    root, raw = verify_source_pins(root, refs)
    module = types.ModuleType("_pinned_original_optional_absence_base")
    module.__file__ = str(root / QUALIFY)
    exec(compile(raw[QUALIFY], module.__file__, "exec", dont_inherit=True), module.__dict__)

    class OptionalAbsenceFinder(module.BoundAuthorFinder):
        def find_spec(self, fullname, path=None, target=None):
            if type(fullname) is not str or fullname != OPTIONAL_NAME:
                return super().find_spec(fullname, path, target)
            require(target is None and OPTIONAL_NAME not in sys.modules,
                    "optional reload/preloaded module rejected")
            parent = _canonical(self.root, PARENT_RELATIVE)
            require(parent.is_dir() and type(path) is list and len(path) == 1 and
                    type(path[0]) is str and path[0] == str(parent),
                    "sole canonical locked author parent path required")
            # Revalidate the empty original parent init at every absence decision.
            init = _canonical(self.root, PARENT_RELATIVE + "/__init__.py")
            require(init.is_file() and init.stat().st_size == 0 and
                    hashlib.sha256(init.read_bytes()).hexdigest() == SOURCE_PINS[1][2],
                    "locked parent source drift")
            for row_path in self.refs:
                require(type(row_path) is str, "invalid source ref path")
                for prefix in (PARENT_RELATIVE + "/deep_gemm",
                               module.BINARY + "/third_party/deep_gemm"):
                    require(row_path != prefix and not row_path.startswith(prefix + "/") and
                            not row_path.startswith(prefix + "."),
                            "optional module already has a source/binary binding")
            for entry in parent.iterdir():
                require(entry.name != "deep_gemm" and not entry.name.startswith("deep_gemm."),
                        "optional module/source/namespace/link is present")
            _check_search_chain(self, parent)
            require(machinery.PathFinder.find_spec(fullname, [str(parent)]) is None,
                    "optional PathFinder discovered a module")
            _check_search_chain(self, parent)
            return None

    return OptionalAbsenceFinder(root, refs)
