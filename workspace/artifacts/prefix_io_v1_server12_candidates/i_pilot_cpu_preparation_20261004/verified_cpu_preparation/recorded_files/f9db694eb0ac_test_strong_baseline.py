"""Actual source/parser/planner CPU contracts; fixture timings are not GPU data."""
from __future__ import annotations
import ast
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


B = load("strong_config_cpu", HERE / "strong_baseline_config.py")
R = load("strong_replay_cpu", HERE / "original_value_replay.py")


@contextmanager
def fixture_directory():
    temporary = tempfile.TemporaryDirectory(prefix="cpu_fixture_", dir=HERE)
    path = Path(temporary.name).resolve()
    assert path.is_relative_to(HERE.resolve())
    try:
        yield path
    finally:
        assert path.is_relative_to(HERE.resolve())
        temporary.cleanup()


def original_common_inputs():
    pin = json.loads((R.INPUTS / "STRONG_COMMON_ENGINE_SOURCE_INPUT.json").read_text(encoding="utf-8"))
    path = R.INPUTS / pin["local_basename"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in ("ENGINE", "SAMPLING")}
    engine, sampling = deepcopy(constants["ENGINE"]), deepcopy(constants["SAMPLING"])
    # Retain exact original P3 development-domain override from its source.
    prepare_tree = ast.parse((R.INPUTS / "prepare_p3_pilot.py").read_text(encoding="utf-8"))
    original_engine = next(n for n in ast.walk(prepare_tree) if isinstance(n, ast.keyword) and n.arg == "engine")
    engine.update({k.arg: ast.literal_eval(k.value) for k in original_engine.value.keywords})
    sampling.update(min_tokens=128, max_tokens=128)
    source_tree = ast.parse((R.INPUTS / "run_p3_native_pilot.py").read_text(encoding="utf-8"))
    extra_assignment = next(n for n in ast.walk(source_tree) if isinstance(n, ast.Assign) and
                            any(isinstance(t, ast.Name) and t.id == "extra" for t in n.targets))
    extra = {}
    def literal_arithmetic(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left, right = ast.literal_eval(node.left), ast.literal_eval(node.right)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError("only original numeric staging literal division")
            return left / right
        return ast.literal_eval(node)
    for keyword in extra_assignment.value.keywords:
        try:
            extra[keyword.arg] = literal_arithmetic(keyword.value)
        except ValueError:
            pass
    extra.pop("shared_storage_path", None)
    extra["prefix_cache_break_even_path"] = "/root/cpu-fixture/curves.json"
    if extra.get("prefix_io_observation_mode") == "shadow":
        extra["prefix_io_observation_run_id"] = "builder-binds-this"
    return engine, sampling, extra


def inputs():
    engine, sampling, extra = original_common_inputs()
    policy = dict(schema_version=1, run_id="cpu-I", mode="interference", sample_max_age_ns=100_000_000,
                  max_wait_ns=100_000_000, internal_step_budget_ns=1, candidate_batches=[1],
                  fixed_stage_policy=None, cost_table=None)
    return dict(engine_common=engine, sampling_common=sampling, connector_common=extra,
                storage_paths={"U": "/root/cpu-fixture/U/storage", "I": "/root/cpu-fixture/I/storage"},
                run_ids={"U": "cpu-U", "I": "cpu-I"}, i_policy=policy, max_accepted_parents=8)


def cpu_curve(path):
    # Only arithmetic/parser fixture; never saved as performance/cost evidence.
    curve = dict(model_name="cpu-fixture", kv_dtype="bfloat16", kv_bytes_per_token=57344,
                 curves={name: dict(floor=floor, knots={"16": low, "1024": high})
                         for name, floor, low, high in (("f", .02, .02, .1), ("g_ssd", .001, .002, .04),
                                                        ("g_mem", .0001, .0002, .004))}, golden=[])
    path.write_text(json.dumps(curve), encoding="utf-8")
    return curve


class StrongPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = R.load_original_values()
        cls.author = R.load_author_values()

    def test_actual_server_input_pin_set(self):
        self.assertEqual(len(R.manifest_entries()), 7)
        self.assertEqual(len(R.HASHES), 9)
        for name in R.HASHES:
            self.assertEqual(hashlib.sha256(R.source(name).read_bytes()).hexdigest(), R.HASHES[name])

    def test_pair_runs_original_config_parser_on_both_arms(self):
        pair = B.build_runtime_pair(**inputs())
        normalized = []
        for arm in ("U", "I"):
            extra = pair[arm]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
            parsed = self.values.fs.SharedFileConfig.from_extra_config(extra)
            self.assertEqual(parsed.load_planner, "on")
            self.assertTrue(parsed.enable_preload)
            self.assertTrue(parsed.preload_share_staging)
            row = asdict(parsed)
            row.pop("root_dir")
            normalized.append(row)
        self.assertEqual(normalized[0], normalized[1])
        self.assertTrue(B.validate_runtime_pair(pair)["common_runtime_equal"])
        self.assertFalse(B.validate_runtime_pair(pair)["gpu_effect_qualified"])

    def test_pair_runs_actual_off_interference_options_parser(self):
        # Source from the already fixed G; only pure control modules are loaded.
        py_source = R.source("vllm")
        control = py_source.parents[2] / "prefix-io-p4-02-cpu/src/prefix_io_control"
        package = ModuleType("_strong_control_values_cpu")
        package.__path__ = [str(control)]
        sys.modules[package.__name__] = package
        options = load(package.__name__ + ".p4_options", control / "p4_options.py")
        pair = B.build_runtime_pair(**inputs())
        parsed = {}
        for arm in ("U", "I"):
            extra = pair[arm]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
            parsed[arm] = options.parse_p4_options(extra)
        self.assertIsNone(parsed["U"].config)
        self.assertEqual(parsed["I"].config.mode, "interference")
        self.assertEqual(parsed["U"].original.parent_cap, parsed["I"].original.parent_cap)
        self.assertIsNone(parsed["I"].fixed)
        self.assertIsNone(parsed["I"].prepared_cost_request)

    def test_original_parser_rejects_removed_step_deadline(self):
        extra = dict(inputs()["connector_common"], shared_storage_path="/root/cpu-fixture/storage",
                     load_planner_defer_deadline_steps=2)
        with self.assertRaisesRegex(ValueError, "removed"):
            self.values.fs.SharedFileConfig.from_extra_config(extra)

    def test_weak_planner_off_is_rejected_for_effect_pair(self):
        value = inputs()
        value["connector_common"]["load_planner"] = "off"
        with self.assertRaisesRegex(ValueError, "LoadPlanner on"):
            B.build_runtime_pair(**value)

    def test_cannot_disable_prefix_preload_sharing_or_pipeline(self):
        changes = (("engine_common", "enable_prefix_caching", False),
                   ("connector_common", "enable_preload", False),
                   ("connector_common", "preload_share_staging", False),
                   ("connector_common", "disable_fusion", True))
        for group, key, val in changes:
            value = inputs()
            value[group][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                B.build_runtime_pair(**value)

    def test_arm_resource_or_sampling_drift_rejected(self):
        for location in ("sampling", "engine"):
            pair = B.build_runtime_pair(**inputs())
            if location == "sampling":
                pair["I"]["sampling"]["max_tokens"] += 1
            else:
                pair["I"]["engine"]["max_num_seqs"] += 1
            with self.subTest(location=location), self.assertRaises(ValueError):
                B.validate_runtime_pair(pair)

    def test_shared_namespace_or_wrong_i_identity_rejected(self):
        pair = B.build_runtime_pair(**inputs())
        u = pair["U"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
        i = pair["I"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]
        i["shared_storage_path"] = u["shared_storage_path"]
        with self.assertRaises(ValueError):
            B.validate_runtime_pair(pair)
        value = inputs()
        value["i_policy"]["run_id"] = "other"
        with self.assertRaises(ValueError):
            B.build_runtime_pair(**value)

    def test_original_build_planner_full_ast_creates_real_value_planner(self):
        with fixture_directory() as temporary:
            path = temporary / "CPU_FIXTURE_CURVES.json"
            cpu_curve(path)
            pair = B.build_runtime_pair(**inputs())
            normalized = []
            for arm in ("U", "I"):
                extra = deepcopy(pair[arm]["engine"]["kv_transfer_config"]["kv_connector_extra_config"])
                extra["prefix_cache_break_even_path"] = str(path)
                spec = SimpleNamespace(extra_config=extra, shared_file_config=self.values.fs.SharedFileConfig.from_extra_config(extra),
                                       storage_block_tokens=16)
                config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True),
                                         model_config=SimpleNamespace(max_model_len=1024))
                planner = self.values.builder._build_planner(spec, config, model_name="cpu-fixture", dtype="bfloat16")
                self.assertIsInstance(planner, self.values.planner.LoadPlanner)
                self.assertGreater(planner._max_preload_slots, 0)
                normalized.append((planner._tables, planner._max_preload_slots, planner._defer_tolerance))
                candidate = self.values.planner.CandidateCostInput(0, "fixture", 128, 8,
                                                                   tuple(bytes([n]) for n in range(8)))
                result = planner.plan([candidate])
                self.assertIn(result["fixture"].decision, tuple(self.values.planner.LoadDecision))
            self.assertEqual(normalized[0], normalized[1])

    def test_original_planner_missing_capability_or_v2_curve_rejects(self):
        with fixture_directory() as temporary:
            path = temporary / "CPU_FIXTURE_CURVES.json"
            cpu_curve(path)
            extra = dict(inputs()["connector_common"], shared_storage_path="/root/cpu-fixture/storage",
                         prefix_cache_break_even_path=str(path))
            spec = SimpleNamespace(extra_config=extra, shared_file_config=self.values.fs.SharedFileConfig.from_extra_config(extra),
                                   storage_block_tokens=16)
            config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True),
                                     model_config=SimpleNamespace(max_model_len=1024))
            self.values.builder.PLAN_API_AVAILABLE = False
            try:
                with self.assertRaisesRegex(ValueError, "planned-defer"):
                    self.values.builder._build_planner(spec, config, model_name="cpu-fixture", dtype="bfloat16")
            finally:
                self.values.builder.PLAN_API_AVAILABLE = True
            path.write_text(json.dumps(dict(model_name="cpu-fixture", break_even_tokens=0)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "v2"):
                self.values.builder._build_planner(spec, config, model_name="cpu-fixture", dtype="bfloat16")

    def test_original_mid_write_planner_preserves_defer_fence(self):
        tables = self.values.cost.CostTables(16, (0., .01), (0., .001), (0., .0001), (0., .001), 1)
        planner = self.values.planner.LoadPlanner(tables, max_preload_slots=8, defer_tolerance=2., time_source=lambda: 0.)
        candidate = self.values.planner.CandidateCostInput(0, "cpu-mid-write", 16, 1, (b"prefix",), blocked_on_store=True)
        result = planner.plan([candidate])
        self.assertIs(result[candidate.req_id].decision, self.values.planner.LoadDecision.DEFER)

    def test_original_author_trace_parser_and_schedule_actual_function(self):
        args = B.author_trace_arguments(shared_storage_path="/root/cpu-fixture/storage", dataset_path="/root/cpu-fixture/data.json",
                                        max_prompts=3, arrival_rate=2., max_concurrency=4, seed=17)
        parsed = self.author.replay.build_parser().parse_args(args)
        self.assertEqual(parsed.output_len, 128)
        self.assertFalse(parsed.wipe_shared_storage)
        self.assertEqual(parsed.inject_prefix_fraction, 0.)
        self.assertTrue(parsed.no_launch)
        first = self.author.replay.build_global_specs(3, parsed.arrival_rate, random.Random(parsed.seed))
        second = self.author.replay.build_global_specs(3, parsed.arrival_rate, random.Random(parsed.seed))
        self.assertEqual([asdict(x) for x in first], [asdict(x) for x in second])
        self.assertTrue(all(first[n].scheduled_time < first[n+1].scheduled_time for n in range(2)))

    def test_original_author_trace_input_parsing_is_executed(self):
        with fixture_directory() as temporary:
            path = temporary / "CPU_FIXTURE_TRACE.json"
            path.write_text(json.dumps(["first prompt", {"prompt": "second prompt"},
                                        {"conversations": [{"from": "human", "value": "third prompt"}]}]), encoding="utf-8")
            self.assertEqual(self.author.replay.load_trace_prompts(str(path), 3),
                             ["first prompt", "second prompt", "third prompt"])

    def test_cpu_cli_creates_config_without_effect_receipt(self):
        with fixture_directory() as temporary:
            inp, out = temporary / "CPU_FIXTURE_INPUT.json", temporary / "CPU_FIXTURE_OUTPUT.json"
            inp.write_text(json.dumps(inputs()), encoding="utf-8")
            result = subprocess.run([sys.executable, "-B", "-I", "-S", str(HERE / "strong_baseline_config.py"),
                                     "--input", str(inp), "--output", str(out)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertFalse(data["gpu_effect_qualified"])
            self.assertFalse(data["native_execution_verified"])
            self.assertIsNone(data["cost_receipt"])
            self.assertIn("new_strong_domain_calibration_and_holdout", data["required_external_gates"])

    def test_no_gpu_network_client_or_backend_imported(self):
        self.assertFalse(any(n == "torch" or n.startswith("torch.") or n == "vllm" or n.startswith("vllm.") or
                             n == "py_kvcache" or n.startswith("py_kvcache.") or n == "openai" or n.startswith("openai.")
                             for n in sys.modules))


if __name__ == "__main__":
    unittest.main()
