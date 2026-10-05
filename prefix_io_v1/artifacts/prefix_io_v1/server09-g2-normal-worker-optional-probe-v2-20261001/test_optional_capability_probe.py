"""Stdlib CPU contracts using exact frozen author capability/finder source."""
import argparse
import ast
from contextlib import contextmanager
import functools
import hashlib
import importlib
import importlib.machinery as machinery
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path)
parser.add_argument("--readonly-root", type=Path, default=Path(__file__).resolve().parents[2] /
                    "prefix_io_v1_server09_g2_ninja_20261001/source-readonly")
args, remaining = parser.parse_known_args()
spec = importlib.util.spec_from_file_location("optional_cpu_contract", Path(__file__).with_name("optional_capability_probe.py"))
H = importlib.util.module_from_spec(spec); sys.modules[spec.name] = H; spec.loader.exec_module(H)
SOURCE = {}
for relative, nbytes, sha in H.SOURCE_PINS:
    path = args.source_root / relative if args.source_root else args.readonly_root / Path(relative).name
    raw = path.read_bytes()
    assert len(raw) == nbytes and hashlib.sha256(raw).hexdigest() == sha, str(path)
    SOURCE[relative] = raw


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix="optional-probe-cpu-") as folder:
        root = Path(folder).resolve()
        refs = {}
        for relative, nbytes, sha in H.SOURCE_PINS:
            path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(SOURCE[relative])
            refs[relative] = dict(path=relative, bytes=nbytes, sha256=sha)
        base = types.ModuleType("_original_optional_probe_cpu_fixture")
        base.__file__ = str(root / H.QUALIFY)
        exec(compile(SOURCE[H.QUALIFY], base.__file__, "exec", dont_inherit=True), base.__dict__)
        finder = base.BoundAuthorFinder(root, refs)
        parent = types.ModuleType("vllm.third_party")
        parent.__file__ = str(root / H.PARENT_RELATIVE / "__init__.py")
        parent.__path__ = [str(root / H.PARENT_RELATIVE)]
        top = types.ModuleType("vllm"); top.__path__ = [str(root / H.AUTHOR / "vllm")]
        module = types.ModuleType("vllm.utils.import_utils")
        module.__file__ = str(root / H.IMPORT_UTILS)
        module.__dict__.update(cache=functools.cache, importlib=importlib)
        # Independent AST extraction preserves both exact original bodies/decorators.
        nodes = [x for x in ast.parse(SOURCE[H.IMPORT_UTILS], filename=module.__file__).body
                 if isinstance(x, ast.FunctionDef) and x.name in ("_has_module", "has_deep_gemm")]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), module.__file__, "exec",
                     dont_inherit=True), module.__dict__)
        entries = {base.__name__: base, "vllm": top, "vllm.third_party": parent,
                   module.__name__: module}
        with patch.dict(sys.modules, entries), patch.object(sys, "meta_path", [finder, machinery.BuiltinImporter,
                              machinery.FrozenImporter, machinery.PathFinder]):
            try:
                yield root, refs, base, finder, module, parent
            finally:
                for key in list(sys.path_importer_cache):
                    if type(key) is str and key.startswith(str(root)):
                        sys.path_importer_cache.pop(key, None)


class CapabilityContracts(unittest.TestCase):
    def test_original_missing_reproduced_patched_original_ast_false_required_import_still_raises(self):
        with fixture() as (root, refs, base, finder, module, parent):
            original = module._has_module
            with self.assertRaises(ModuleNotFoundError) as original_error: module.has_deep_gemm()
            self.assertEqual(original_error.exception.name, H.OPTIONAL_NAME)
            ids = (base.BoundAuthorFinder.find_spec, base.BoundAuthorFinder.locked_binary_spec)
            handle = H.install_capability_probe(module, finder, root, refs)
            self.assertIs(module.has_deep_gemm(), False)
            self.assertIs(module._has_module(H.OPTIONAL_NAME), False)
            with self.assertRaises(ModuleNotFoundError): finder.find_spec(H.OPTIONAL_NAME, parent.__path__)
            self.assertEqual(ids, (base.BoundAuthorFinder.find_spec, base.BoundAuthorFinder.locked_binary_spec))
            self.assertTrue(handle.detach()); self.assertIs(module._has_module, original)

    def test_external_available_is_preserved_without_optional_absence_validation(self):
        with fixture() as (root, refs, base, finder, module, parent):
            external = types.ModuleType("deep_gemm"); external.__spec__ = machinery.ModuleSpec("deep_gemm", None)
            handle = H.install_capability_probe(module, finder, root, refs)
            with patch.dict(sys.modules, {"deep_gemm": external}), patch.object(H, "_verify_absence", side_effect=AssertionError("must not inspect")):
                self.assertIs(module.has_deep_gemm(), True)
            self.assertTrue(handle.restore())

    def test_cached_success_and_unrelated_error_delegate_original_once_identity(self):
        with fixture() as (root, refs, base, finder, module, parent):
            original = module._has_module
            # Cache a successful original capability result without importing anything.
            with patch.object(importlib.util, "find_spec", return_value=object()): self.assertIs(original(H.OPTIONAL_NAME), True)
            handle = H.install_capability_probe(module, finder, root, refs)
            with patch.object(H, "_verify_absence", side_effect=AssertionError("cached success changed")):
                self.assertIs(module._has_module(H.OPTIONAL_NAME), True)
            error = ModuleNotFoundError("unrelated", name="vllm.unrelated")
            with patch.object(importlib.util, "find_spec", side_effect=error) as call:
                with self.assertRaises(ModuleNotFoundError) as caught: module._has_module("vllm.unrelated")
                self.assertIs(caught.exception, error); self.assertEqual(call.call_count, 1)
            handle.restore()

    def test_spoofed_name_message_or_origin_errors_propagate_same_object(self):
        with fixture() as (root, refs, base, finder, module, parent):
            H.install_capability_probe(module, finder, root, refs)
            for message, name in ((H.OPTIONAL_NAME + " has no exact locked binary fallback", H.OPTIONAL_NAME),
                                  ("different message", H.OPTIONAL_NAME), ("other", "another.module")):
                error = ModuleNotFoundError(message, name=name)
                with self.subTest(name=name, message=message), patch.object(importlib.util, "find_spec", side_effect=error):
                    with self.assertRaises(ModuleNotFoundError) as caught: module._has_module(H.OPTIONAL_NAME)
                    self.assertIs(caught.exception, error)
            class PoisonArgument:
                calls = 0
                def __str__(self): self.calls += 1; raise AssertionError("observer coerced exception argument")
            poison = PoisonArgument(); error = ModuleNotFoundError(poison, name=H.OPTIONAL_NAME)
            with patch.object(importlib.util, "find_spec", side_effect=error):
                with self.assertRaises(ModuleNotFoundError) as caught: module._has_module(H.OPTIONAL_NAME)
                self.assertIs(caught.exception, error); self.assertEqual(poison.calls, 0)

    def test_exact_original_codes_from_different_finder_cannot_forge_absence(self):
        with fixture() as (root, refs, base, finder, module, parent):
            other = base.BoundAuthorFinder(root, refs); seen = []
            def wrong_finder(name):
                try: return other.find_spec(name, parent.__path__)
                except ModuleNotFoundError as error: seen.append(error); raise
            H.install_capability_probe(module, finder, root, refs)
            with patch.object(importlib.util, "find_spec", side_effect=wrong_finder):
                with self.assertRaises(ModuleNotFoundError) as caught: module._has_module(H.OPTIONAL_NAME)
            self.assertIs(caught.exception, seen[0])

    def test_present_variants_and_optional_binary_refs_are_never_hidden(self):
        with fixture() as (root, refs, base, finder, module, parent):
            directory = root / H.PARENT_RELATIVE
            for name in ("deep_gemm", "deep_gemm.py", "deep_gemm.pyc", "deep_gemm.abi3.so", "deep_gemm.pyd"):
                with self.subTest(name=name):
                    item = directory / name
                    if name == "deep_gemm": item.mkdir()
                    else: item.write_bytes(b"present")
                    try:
                        with self.assertRaises(ValueError): H.verify_optional_absence(root, refs)
                    finally:
                        item.rmdir() if item.is_dir() else item.unlink()
            refs["third_party/work/vllm-author-build/vllm/third_party/deep_gemm.abi3.so"] = dict(path="unused", bytes=0, sha256="0"*64)
            with self.assertRaises(ValueError): H.verify_optional_absence(root, refs)

    def test_parent_source_or_symlink_drift_refuses_without_new_files(self):
        with fixture() as (root, refs, base, finder, module, parent):
            names = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
            init = root / H.PARENT_RELATIVE / "__init__.py"
            init.write_bytes(b"changed")
            with self.assertRaises(ValueError): H.verify_optional_absence(root, refs)
            init.write_bytes(b"")
            real_is_link = Path.is_symlink
            with patch.object(Path, "is_symlink", lambda p: p == init or real_is_link(p)):
                with self.assertRaises(ValueError): H.verify_optional_absence(root, refs)
            self.assertEqual(names, sorted(str(p.relative_to(root)) for p in root.rglob("*")))

    def test_runtime_parent_or_finder_position_or_preload_drift_never_returns_false(self):
        with fixture() as (root, refs, base, finder, module, parent):
            H.install_capability_probe(module, finder, root, refs)
            with patch.object(parent, "__file__", str(root / "wrong.py")):
                with self.assertRaises(ValueError): module._has_module(H.OPTIONAL_NAME)
            with patch.object(parent, "__path__", parent.__path__ + parent.__path__):
                with self.assertRaises(ValueError): module._has_module(H.OPTIONAL_NAME)
            with patch.object(sys, "meta_path", [machinery.BuiltinImporter, finder, machinery.FrozenImporter, machinery.PathFinder]):
                with self.assertRaises(ValueError): module._has_module(H.OPTIONAL_NAME)
            # An actual original positive discovery must pass through unchanged.
            loaded = types.ModuleType(H.OPTIONAL_NAME); loaded.__spec__ = machinery.ModuleSpec(H.OPTIONAL_NAME, None)
            with patch.dict(sys.modules, {H.OPTIONAL_NAME: loaded}): self.assertIs(module._has_module(H.OPTIONAL_NAME), True)

    def test_unknown_downstream_meta_finder_never_runs_required_import_stays_rejected(self):
        with fixture() as (root, refs, base, finder, module, parent):
            class Unknown:
                calls = 0
                def find_spec(self, *a, **k): self.calls += 1; raise AssertionError("unknown meta finder called")
            unknown = Unknown(); sys.meta_path.insert(1, unknown)
            H.install_capability_probe(module, finder, root, refs)
            self.assertIs(module._has_module(H.OPTIONAL_NAME), False)
            with self.assertRaises(ModuleNotFoundError): importlib.util.find_spec(H.OPTIONAL_NAME)
            self.assertEqual(unknown.calls, 0)

    def test_unknown_path_importer_or_hooks_rejected_after_original_missing(self):
        with fixture() as (root, refs, base, finder, module, parent):
            H.install_capability_probe(module, finder, root, refs)
            key = str(root / H.PARENT_RELATIVE)
            class Unknown:
                def find_spec(self, *a, **k): return None
            with patch.dict(sys.path_importer_cache, {key: Unknown()}):
                with self.assertRaises(ValueError): module._has_module(H.OPTIONAL_NAME)
            sys.path_importer_cache.pop(key, None)
            module._has_module(H.OPTIONAL_NAME)
            with patch.object(sys, "path_hooks", sys.path_hooks + [lambda p: None]):
                with self.assertRaises(ValueError): module._has_module(H.OPTIONAL_NAME)

    def test_original_function_code_or_source_pin_drift_block_install(self):
        with fixture() as (root, refs, base, finder, module, parent):
            original = module._has_module
            module._has_module = functools.cache(lambda name: False)
            with self.assertRaises(ValueError): H.install_capability_probe(module, finder, root, refs)
            module._has_module = original
            refs[H.IMPORT_UTILS] = dict(refs[H.IMPORT_UTILS], sha256="0"*64)
            with self.assertRaises(ValueError): H.install_capability_probe(module, finder, root, refs)

    def test_source_preflight_readonly_detach_preserves_foreign_override_and_false_qualification(self):
        with fixture() as (root, refs, base, finder, module, parent):
            original = module._has_module; before = set(sys.modules)
            names = sorted(str(p.relative_to(root)) for p in root.rglob("*")); environment = dict(__import__("os").environ)
            witness = H.verify_optional_absence(root, refs)
            self.assertFalse(witness["GPU_collector_verified"]); self.assertFalse(witness["runtime_query_exception_verified"])
            self.assertIs(module._has_module, original); self.assertEqual(before, set(sys.modules))
            self.assertEqual(names, sorted(str(p.relative_to(root)) for p in root.rglob("*")))
            self.assertEqual(environment, dict(__import__("os").environ))
            handle = H.install_capability_probe(module, finder, root, refs)
            self.assertFalse(handle.evidence()["production_qualified"])
            foreign = lambda name: "foreign"; module._has_module = foreign
            self.assertFalse(handle.detach()); self.assertIs(module._has_module, foreign)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
