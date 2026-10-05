"""Standard-library, in-memory/temporary metadata fixtures; no actual receipts.

These tests establish rejection and structural preservation only. No fixture
is a formal manifest, actual tokenizer receipt, live private table or GPU run.
"""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from types import ModuleType
import sys
_ADAPTER_PATH=Path(__file__).with_name('development_diagnostic_contract.py')
adapter=ModuleType('development_diagnostic_contract')
adapter.__file__=str(_ADAPTER_PATH)
sys.modules[adapter.__name__]=adapter
exec(compile(_ADAPTER_PATH.read_bytes(),str(_ADAPTER_PATH),'exec'),adapter.__dict__)


BASE = Path(__file__).resolve().parents[2] / "gpu_prerental_preparation_20261004"
if not BASE.is_dir():
    BASE=Path(__file__).resolve().parents[2] / "server12-gpu-prerental-preparation-20261004"
SOURCE = BASE / "formal_trace_binding/formal_trace_binding.py"
RAW = SOURCE.read_bytes()
CONFIG = dict(phase="development", mode="off", arm="U", service_SLO=None)


def pair_fixture(value=100):
    # An existing-pair-shaped engineering setting, explicitly a fixture.
    return dict(I=dict(engine=dict(kv_transfer_config=dict(kv_connector_extra_config={
        "prefix_io_p4_policy": {"internal_step_budget_ns": value}}))))


def input_function(source=RAW):
    return next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "validate_formal_workload")


def metadata_fixture(namespace):
    # Shape-only fixture. Deliberately creates NO actual tokenizer receipt and
    # is used exclusively to reach the original real-file rejection gates.
    records = []
    for index, split in enumerate(namespace["PARTITIONS"]):
        records.append(dict(request_id=index,
            prompt_sha256=hashlib.sha256(("fixture-prompt-" + str(index)).encode()).hexdigest(),
            prompt_token_ids=[index + 1], scheduled_ns=index, max_tokens=128,
            min_tokens=128, seed=0, split=split, prefix_family="fixture-family-" + str(index)))
    doc = dict(schema="natural_trace_workload_v1",
        origin="frozen_actual_existing_trace_no_new_gpu_outcomes",
        dataset_sha256="1" * 64, ordered_prompt_digest="2" * 64,
        author_trace_sha256=namespace["AUTHOR_TRACE_SHA"], author_common_sha256=namespace["AUTHOR_COMMON_SHA"],
        tokenizer_receipt_digest="3" * 64, declaration_digest="4" * 64,
        model_manifest_sha256="5" * 64, records=records,
        partition_counts={key: 1 for key in namespace["PARTITIONS"]}, max_concurrency=1,
        arrival_rate=None, schedule_seed=0, schedule_origin="unchanged_author_build_global_specs",
        recorded_natural_arrival_claim=False, no_prefix_injection=True,
        no_per_request_cache_reset=True, no_request_drops_or_reorder=True,
        initial_cache_state=namespace["INITIAL"], cost_domain_covered=False,
        gpu_effect_qualified=False, gpu_operations=0)
    doc["workload_sha256"] = namespace["digest"](doc)
    return doc


def write_leaf(root, relative, value):
    path = root / relative
    path.write_bytes(json.dumps(value, sort_keys=True).encode())
    return dict(path=relative, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class DevelopmentDiagnosticTests(unittest.TestCase):
    def build(self, config=None):
        return adapter.build_consumer(RAW, config=CONFIG if config is None else config)

    def test_exact_frozen_source_and_new_nonformal_schema(self):
        consumer, audit = self.build()
        self.assertEqual(hashlib.sha256(RAW).hexdigest(), adapter.FORMAL_SHA)
        self.assertEqual(audit["retained_original_statements"], 34)
        self.assertEqual(audit["removed_external_deadline_statements"], 9)
        self.assertFalse(audit["evaluation_or_effect_contract_modified"])
        self.assertFalse(audit["GPU_launch_allowed"])
        self.assertFalse(audit["actual_runtime_namespace_adapter_integrated"])
        self.assertIn(adapter.DEVELOPMENT_SCHEMA, consumer.__code__.co_consts)
        self.assertNotIn("formal_natural_trace_binding_v1", consumer.__code__.co_consts)

    def test_all_original_non_deadline_statements_retained_exactly(self):
        recorded = []
        original_compile = builtins.compile
        def capture(source, filename, mode, *args, **kwargs):
            if isinstance(source, ast.Module) and filename == "<byte_pinned_development_input_consumer>":
                recorded.append(deepcopy(source))
            return original_compile(source, filename, mode, *args, **kwargs)
        with mock.patch("builtins.compile", side_effect=capture):
            self.build()
        self.assertEqual(len(recorded), 1)
        retained = recorded[0].body[0].body
        original = input_function().body
        expected = original[:26] + original[35:]
        self.assertEqual(len(retained), len(expected))
        # Only descriptor schema (statement 6) and output metadata (last) differ.
        for index, (actual, prior) in enumerate(zip(retained, expected)):
            if index not in (6, len(expected) - 1):
                self.assertEqual(ast.dump(actual), ast.dump(prior), "retained statement " + str(index))
        calls = {node.func.id for node in ast.walk(recorded[0])
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"validate_manifest_shape", "json_leaf", "closed", "load_pure",
                         "validate_namespace", "common_domain_sha", "exact"} <= calls)
        self.assertIn("freeze_trace", ast.unparse(recorded[0]))
        self.assertNotIn("independent_requirement_before_development", ast.unparse(recorded[0]))

    def test_original_helpers_derived_from_exact_source_not_caller_namespace(self):
        consumer, _ = self.build()
        self.assertEqual(consumer.__globals__["closed"].__code__.co_filename,
                         "<byte_pinned_original_formal_input_helpers>")
        self.assertIn("source leaf outside frozen closure", consumer.__globals__["closed"].__code__.co_consts)
        self.assertIn("actual source bytes changed: ", consumer.__globals__["closed"].__code__.co_consts)

    def test_changed_original_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact original formal input source"):
            adapter.build_consumer(RAW + b"\n", config=CONFIG)

    def test_effect_U_rejected_before_source_loading(self):
        with mock.patch("builtins.compile", side_effect=AssertionError("should not load source")):
            with self.assertRaisesRegex(ValueError, "development Uoff/Ishadow only"):
                adapter.build_consumer(None, config=dict(phase="effect", mode="off", arm="U"))

    def test_effect_I_rejected_before_source_loading(self):
        with self.assertRaisesRegex(ValueError, "development Uoff/Ishadow only"):
            adapter.build_consumer(None, config=dict(phase="effect", mode="on", arm="I"))

    def test_development_on_and_calibration_not_allowed(self):
        for cfg in (dict(phase="development", mode="on", arm="I"),
                    dict(phase="calibration", mode="off", arm="U"),
                    dict(phase="qualification", mode="off", arm="U")):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                adapter.build_consumer(None, config=cfg)

    def test_service_SLO_cannot_be_relabeled_as_diagnostic(self):
        with self.assertRaisesRegex(ValueError, "carries no service SLO"):
            self.build(dict(CONFIG, service_SLO={"TTFT_ns": 1}))

    def test_external_authority_or_deadline_not_constructed(self):
        for key in ("independent_deadline_ref", "authority_ref", "full_control_window_deadline_ns"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not construct or relabel"):
                self.build(dict(CONFIG, **{key: 1}))

    def test_unbound_actual_inputs_do_not_compile_or_import_backend(self):
        with mock.patch("builtins.compile", side_effect=AssertionError("must not compile")):
            result = adapter.validate_development_inputs(None, root=None,
                workload_ref=None, binding_ref=None, refs=None, pair=None,
                config=CONFIG, original_source=None)
        self.assertEqual(result["status"], "UNBOUND_NO_ACTUAL_DEVELOPMENT_INPUT")
        self.assertFalse(result["gpu_eligible"])
        self.assertFalse(result["formal_goodput_allowed"])
        self.assertIsNone(result["independent_deadline_ref"])

    def test_existing_pair_setting_observation_only_not_service_authority(self):
        cfg = dict(phase="development", mode="shadow", arm="I", service_SLO=None)
        result = adapter.preview_budget_from_existing_pair(cfg, pair_fixture(123))
        self.assertEqual(result["internal_step_budget_ns"], 123)
        self.assertTrue(result["observation_only"])
        self.assertTrue(result["reserve_not_measured_yet"])
        for key in ("ordinary_I_authorized", "preview_is_service_safety_claim",
                    "formal_goodput_allowed", "effect_budget_reuse_allowed", "gpu_eligible"):
            self.assertFalse(result[key])
        self.assertIsNone(result["service_SLO"])

    def test_missing_preview_number_is_not_invented(self):
        with self.assertRaises((KeyError, ValueError, TypeError)):
            adapter.preview_budget_from_existing_pair(CONFIG, {})

    def test_invalid_preview_numbers_rejected(self):
        for number in (True, False, 0, -1, 1.5, "123", None):
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "existing fixed engineering"):
                adapter.preview_budget_from_existing_pair(CONFIG, pair_fixture(number))

    def test_old_qualification_shape_never_becomes_natural_input(self):
        consumer, _ = self.build()
        with self.assertRaisesRegex(ValueError, "original formal freeze_trace schema"):
            consumer(dict(schema="strong_trace_workload_v1"), root=None,
                     workload_ref=None, binding_ref=None, refs=None, pair=None, partition="development")

    def test_old_formal_descriptor_not_accepted_by_new_consumer(self):
        consumer, _ = self.build()
        document = metadata_fixture(consumer.__globals__)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mref = write_leaf(root, "fixture_manifest.json", document)
            descriptor = {key: mref for key in consumer.__globals__["BINDING_FIELDS"] - {"schema"}}
            descriptor["schema"] = "formal_natural_trace_binding_v1"
            bref = write_leaf(root, "fixture_binding.json", descriptor)
            with self.assertRaisesRegex(ValueError, "actual closed formal input descriptor"):
                consumer(document, root=root, workload_ref=mref, binding_ref=bref,
                         refs={mref["path"]: mref, bref["path"]: bref}, pair=None, partition="development")

    def test_retained_real_leaf_closure_rejects_missing_source(self):
        consumer, _ = self.build()
        document = metadata_fixture(consumer.__globals__)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mref = write_leaf(root, "fixture_manifest.json", document)
            absent = dict(path="missing_fixture_leaf.json", bytes=1, sha256="0" * 64)
            descriptor = {key: absent for key in consumer.__globals__["BINDING_FIELDS"] - {"schema"}}
            descriptor.update(schema=adapter.DEVELOPMENT_SCHEMA, manifest_ref=mref)
            bref = write_leaf(root, "fixture_binding.json", descriptor)
            with self.assertRaisesRegex(ValueError, "source leaf outside frozen closure"):
                consumer(document, root=root, workload_ref=mref, binding_ref=bref,
                         refs={mref["path"]: mref, bref["path"]: bref}, pair=None, partition="development")

    def test_retained_real_leaf_closure_rejects_byte_drift(self):
        consumer, _ = self.build()
        document = metadata_fixture(consumer.__globals__)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mref = write_leaf(root, "fixture_manifest.json", document)
            data = write_leaf(root, "fixture_data.json", {"synthetic_fixture": True})
            descriptor = {key: data for key in consumer.__globals__["BINDING_FIELDS"] - {"schema"}}
            descriptor.update(schema=adapter.DEVELOPMENT_SCHEMA, manifest_ref=mref)
            bref = write_leaf(root, "fixture_binding.json", descriptor)
            (root / data["path"]).write_bytes(b"changed fixture source")
            with self.assertRaisesRegex(ValueError, "actual source bytes changed"):
                consumer(document, root=root, workload_ref=mref, binding_ref=bref,
                         refs={row["path"]: row for row in (mref, bref, data)}, pair=None, partition="development")

    def test_no_model_or_native_or_tokenizer_import_during_builder(self):
        original_import = builtins.__import__
        forbidden = {"torch", "vllm", "py_kvcache", "prefix_io_control", "tokenizers", "transformers"}
        def scoped_import(name, *args, **kwargs):
            if name.split(".")[0] in forbidden:
                raise AssertionError("forbidden actual backend import: " + name)
            return original_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=scoped_import):
            self.build()


if __name__ == "__main__":
    unittest.main()
