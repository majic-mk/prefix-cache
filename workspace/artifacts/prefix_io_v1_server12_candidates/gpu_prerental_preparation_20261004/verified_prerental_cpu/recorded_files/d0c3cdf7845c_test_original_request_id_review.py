"""CPU-only replay of pinned author ID mapping bodies; no runtime qualification."""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent
INPUTS = HERE.parent / "source_inputs"
PROVENANCE = INPUTS / "PRERENT_ID_API_SOURCE_INPUTS.json"


def source_refs():
    rows = json.loads(PROVENANCE.read_bytes())["files"]
    result = []
    for row in rows:
        path = INPUTS / Path(row["path"]).name
        raw = path.read_bytes()
        if path.is_symlink() or len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("pinned author request-ID source drift")
        result.append(dict(original_ref=row, local_path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    if {Path(row["original_ref"]["path"]).name for row in result} != {"input_processor.py", "output_processor.py"}:
        raise ValueError("complete original ID interface source required")
    return result


def body(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("unique pinned original function required")
    node = matches[0]
    node.decorator_list = []  # Extract the original method body without importing its enclosing GPU package.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class PoolingOutput:
    pass


KINDS = SimpleNamespace(FINAL_ONLY="final", DELTA="delta", CUMULATIVE="cumulative")


class OriginalRequestIDReplay(unittest.TestCase):
    def setUp(self):
        self.logger = SimpleNamespace(warning_once=lambda *args: None)
        self.envs = SimpleNamespace(VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False)
        self.assign = body(INPUTS / "input_processor.py", "assign_request_id", {
            "envs": self.envs, "logger": self.logger, "random_uuid": lambda: "12345678abcdef00",
        })
        namespace = {"RequestOutputKind": KINDS, "RequestOutput": lambda **kw: SimpleNamespace(**kw),
                     "PoolingRequestOutput": lambda **kw: SimpleNamespace(**kw), "PoolingOutput": PoolingOutput,
                     "CompletionOutput": SimpleNamespace, "cast": lambda kind, value: value}
        self.new = body(INPUTS / "output_processor.py", "_new_request_output", namespace)
        self.make = body(INPUTS / "output_processor.py", "make_request_output", namespace)

    def state(self, *, external="front-0", internal="front-0-12345678", parent=None, kind="cumulative"):
        state = SimpleNamespace(request_id=internal, external_req_id=external, parent_req=parent,
            prompt_token_ids=[11, 12], prompt_embeds=None, logprobs_processor=SimpleNamespace(prompt_logprobs=None),
            output_kind=kind, lora_request=None, prompt=None, num_cached_tokens=1, stats=None,
            stream_interval=1, detokenizer=None, sent_tokens_offset=0)
        state._new_request_output = MethodType(self.new, state)
        state._new_completion_output = lambda tokens, finish, stop: SimpleNamespace(token_ids=tokens)
        return state

    def test_external_kept_and_native_uuid_assigned(self):
        request = SimpleNamespace(request_id="front-0", external_req_id=None)
        self.assign(request)
        self.assertEqual(request.external_req_id, "front-0")
        self.assertEqual(request.request_id, "front-0-12345678")

    def test_prepopulated_external_id_rejected(self):
        with self.assertRaises(ValueError):
            self.assign(SimpleNamespace(request_id="front-0", external_req_id="already-set"))

    def test_disabled_randomization_retains_front_id(self):
        self.envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = True
        request = SimpleNamespace(request_id="front-0", external_req_id=None)
        self.assign(request)
        self.assertEqual((request.request_id, request.external_req_id), ("front-0", "front-0"))

    def test_completion_uses_external_id(self):
        state = self.state()
        output = self.make(state, [99], None, None, None)
        self.assertEqual(output.request_id, "front-0")
        self.assertNotEqual(output.request_id, state.request_id)
        self.assertEqual(output.prompt_token_ids, [11, 12])
        self.assertEqual(output.outputs[0].token_ids, [99])

    def test_parent_completion_uses_parent_external_id(self):
        parent = SimpleNamespace(external_req_id="parent-front", get_outputs=lambda native, output: ([output], True))
        output = self.make(self.state(parent=parent), [99], None, "length", None)
        self.assertEqual(output.request_id, "parent-front")
        self.assertTrue(output.finished)

    def test_final_only_not_finished_output_is_absent(self):
        self.assertIsNone(self.make(self.state(kind="final"), [99], None, None, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    before = source_refs()
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OriginalRequestIDReplay))
    after = source_refs()
    forbidden = [name for name in sys.modules if name == "torch" or name.startswith(("torch.", "vllm.", "py_kvcache.", "prefix_io_control."))]
    document = dict(schema="independent_original_request_id_cpu_replay_v1",
        status="PASS_CPU_ONLY_ORIGINAL_ID_MAPPING" if result.wasSuccessful() and before == after and not forbidden else "FAIL",
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        original_sources_before=before, original_sources_after=after, source_bytes_unchanged=before == after,
        body_source="unique original pinned functions extracted with ast; original package not imported",
        CPU_fixture_only=True, actual_gpu_runs=0, actual_RPC_calls=0, qualified_runtime=False,
        forbidden_runtime_imports=forbidden, test_log=log.getvalue())
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: document[key] for key in ("status", "tests_run", "failures", "errors", "skipped", "actual_gpu_runs", "source_bytes_unchanged")}))
    return 0 if document["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
