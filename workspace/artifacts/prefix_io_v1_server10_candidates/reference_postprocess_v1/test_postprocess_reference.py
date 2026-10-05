"""Small CPU regressions for the Python 3.12 import-aware bytecode mismatch."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_postprocess_under_test", HERE / "postprocess_reference.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
RAW = b'import math\n\ndef analyze(value):\n    return math.isfinite(value)\n'


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.path = (Path(tempfile.gettempdir()) / "cpu-analyzer-regression-not-written.py").resolve()
        self.namespace = dict(__name__="_original_cpu_fixture")
        exec(compile(RAW, str(self.path), "exec", dont_inherit=True), self.namespace)
        self.function = self.namespace["analyze"]

    def test_real_module_function_passes_even_when_isolated_AST_differs(self):
        node = next(n for n in ast.parse(RAW).body if isinstance(n, ast.FunctionDef))
        isolated = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(self.path), "exec", dont_inherit=True), isolated)
        # 3.12's import-aware LOAD_GLOBAL/LOAD_ATTR forms differ, not semantics.
        import sys
        if sys.version_info[:2] == (3, 12):
            self.assertNotEqual(self.function.__code__, isolated["analyze"].__code__)
        P.original_function_from_module(self.function, self.path, RAW, "analyze")
        self.assertTrue(self.function(1.0))

    def test_tampered_code_and_wrong_filename_still_fail(self):
        changed = {}
        exec(compile(RAW.replace(b'math.isfinite(value)', b'True'), str(self.path), "exec", dont_inherit=True), changed)
        with self.assertRaises(ValueError):
            P.original_function_from_module(changed["analyze"], self.path, RAW, "analyze")
        with self.assertRaises(ValueError):
            P.original_function_from_module(self.function, self.path.with_name("other.py"), RAW, "analyze")

    def test_reference_compilation_never_executes_module_top_level(self):
        raw = RAW + b'\nraise RuntimeError("must never execute")\n'
        code = compile(raw, str(self.path), "exec", dont_inherit=True)
        original_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == "analyze")
        function = types.FunctionType(original_code, self.namespace)
        P.original_function_from_module(function, self.path, raw, "analyze")

    def test_private_checker_restored_after_success_and_exception(self):
        old = object()
        checker = types.SimpleNamespace(_original_function=old)
        with P.corrected_callable_check(checker):
            self.assertIs(checker._original_function, P.original_function_from_module)
        self.assertIs(checker._original_function, old)
        with self.assertRaisesRegex(RuntimeError, "expected"):
            with P.corrected_callable_check(checker):
                raise RuntimeError("expected")
        self.assertIs(checker._original_function, old)


if __name__ == "__main__":
    unittest.main()
