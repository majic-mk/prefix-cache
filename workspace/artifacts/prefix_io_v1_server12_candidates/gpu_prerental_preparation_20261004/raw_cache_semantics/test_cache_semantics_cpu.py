"""CPU source/value replays only; no GPU events, model or private table issued."""
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

HERE = Path(__file__).resolve().parent
PREP = HERE.parent
ROOT = HERE.parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def pinned_source(server_relative, local_relative, expected_sha):
    selected = ROOT / server_relative
    if not selected.is_file():
        selected = ROOT / local_relative
    raw = selected.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("CPU source replay rejected: original source pin")
    return raw


SCHEDULER = pinned_source("third_party/work/vllm-author-p4-02-cpu/vllm/v1/core/sched/scheduler.py",
    "artifacts/prefix_io_v1_server12_candidates/gpu_prerental_preparation_20261004/source_inputs/scheduler.py",
    "1dfad7e969d3a689a4e38212c089ef98edf437cd5c1c9f904a2774cd08505698")
STATS = pinned_source("third_party/work/vllm-author-p4-02-cpu/vllm/v1/metrics/stats.py",
    "artifacts/prefix_io_v1_server09_candidates/g3_current_context_calibration_cpu/source_readonly/third_party/work/vllm-author-p4-02-cpu/vllm/v1/metrics/stats.py",
    "08eb6bb969ad47e4207ba5568beaec35bdb09192b6db046f046dab4892ac1bf3")
FRAME_PATH_SERVER = "artifacts/prefix_io_v1/server09-migration-20261001/collector-candidate/p4_full_step_frame_adapter.py"
FRAME_PATH_LOCAL = "artifacts/prefix_io_v1_server09_candidates/collector/p4_full_step_frame_adapter.py"
FRAME_RAW = pinned_source(FRAME_PATH_SERVER, FRAME_PATH_LOCAL,
    "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582")
FRAME_PATH = ROOT / (FRAME_PATH_SERVER if (ROOT / FRAME_PATH_SERVER).is_file() else FRAME_PATH_LOCAL)
F = load("_cpu_original_full_frame_cache_values", FRAME_PATH)
V = load("_cpu_original_capture_cache_validation", PREP / "activation/native_conditional_cost.py")
W = load("_cpu_semantic_raw_v4", PREP / "runner/strong_native_cost_runner_v4.py")


def original_method(raw, owner_name, method_name):
    tree = ast.parse(raw)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner_name)
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = dict(__name__="CPU_actual_source_AST_replay", Request=NS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "CPU_actual_source_AST_replay", "exec", dont_inherit=True), namespace)
    return namespace[method_name]


def prepared(offset, *, first_context=511):
    context = first_context if offset == 0 else 512 + offset - 1
    runner = NS(speculative_config=None, use_async_scheduling=False, is_pooling_model=False,
        parallel_config=NS(pipeline_parallel_size=1, data_parallel_size=1, tensor_parallel_size=1),
        _profile_step=501 + offset,
        input_batch=NS(num_reqs=1, req_ids=["CPU-fixture-internal-request"],
                       num_computed_tokens_cpu=[context], num_prompt_tokens=[512]))
    scheduler = NS(num_scheduled_tokens={"CPU-fixture-internal-request": 1},
                   total_num_scheduled_tokens=1, scheduled_spec_decode_tokens={})
    return F.capture_prepared_frame(runner, scheduler, 500 + offset)


def CPU_capture_fixture():
    frames, witnesses, output = [], [], list(range(1000, 1128))
    for offset in range(128):
        begin = 10000 + offset * 1000
        frame = F.ClosedFrame(500 + offset, begin, begin + 100, prepared(offset),
                             (("CPU-fixture-internal-request", (output[offset],)),))
        value = asdict(frame)
        value["prepared"]["rows"] = list(value["prepared"]["rows"])
        value["prepared"]["input_seq_lens_from_cpu_inputs"] = list(value["prepared"]["input_seq_lens_from_cpu_inputs"])
        value["outputs"] = [["CPU-fixture-internal-request", [output[offset]]]]
        frames.append(value)
        witnesses.append(dict(native_step_ordinal=500 + offset,
            start_record_before_ns=begin-20, start_record_after_ns=begin-10,
            start_completed_query_ns=begin-5, end_record_before_ns=begin+110,
            end_record_after_ns=begin+120, end_completed_query_ns=begin+130,
            event_elapsed_source="torch.cuda.Event.elapsed_time", gpu_elapsed_ns=1))
    return dict(scope="server11_full_step_native_capture_v1", run_id="CPU-fixture-external-run",
        origin="native_gpu_recording", valid=True, frames=frames, event_witnesses=witnesses,
        failures=[], pending_event_pairs=0, open_event_pair=False, no_added_synchronization=True,
        cross_clock_absolute_mapping=False, selected_offsets=[16],
        CPU_fixture_only=True, actual_GPU_events=0, production_qualified=False), output


def validate_fixture(capture, output, *, cached=511):
    return V.validate_capture(capture, run_id="CPU-fixture-external-run", request_id="CPU-fixture-internal-request",
        output_ids=output, prompt_tokens=512, measured_offset=16, warmup_offsets=[1], cached_tokens=cached)


class OriginalCacheSemanticsCPU(unittest.TestCase):
    def test_original_frontend_stats512_and_full_remote_execution511_are_separate(self):
        stats = NS()
        original_method(STATS, "PrefillStats", "set")(stats, 512, 496, 16)
        self.assertEqual(stats.num_cached_tokens, 512)
        self.assertEqual((stats.num_local_cached_tokens, stats.num_external_cached_tokens), (496, 16))
        request = NS(request_id="CPU-fixture", num_tokens=512, num_computed_tokens=512)
        cached_before_recompute = []
        owner = NS(connector=object(), failed_recving_kv_req_ids=set(),
            finished_recving_kv_req_ids={request.request_id},
            kv_cache_manager=NS(cache_blocks=lambda req,n: cached_before_recompute.append(n)))
        original_method(SCHEDULER, "Scheduler", "_update_waiting_for_remote_kv")(owner, request)
        self.assertEqual(cached_before_recompute, [512])
        self.assertEqual(request.num_computed_tokens, 511)
        self.assertEqual(request.num_tokens-request.num_computed_tokens, 1)
        self.assertEqual(stats.num_cached_tokens, 512)
        self.assertEqual(owner.finished_recving_kv_req_ids, set())

    def test_original_prepared_first_prefill511_one_token_and_decode_context527(self):
        first, decoded, selected = prepared(0), prepared(1), prepared(16)
        self.assertEqual((first.context_length,first.prefill_tokens,first.active_decode,first.step_kind), (511,1,0,"prefill"))
        self.assertEqual(first.input_seq_lens_from_cpu_inputs, (512,))
        self.assertEqual((decoded.context_length,decoded.prefill_tokens,decoded.active_decode), (512,0,1))
        self.assertEqual((selected.context_length,selected.prefill_tokens,selected.active_decode), (527,0,1))
        self.assertFalse(F.ClosedFrame(500, 1, 2, first, ()).production_qualified)

    def test_unchanged_validator_demands_all128_frames_and_rejects512_execution_context(self):
        capture, output = CPU_capture_fixture()
        rows = validate_fixture(capture, output)
        self.assertEqual(len(rows), 128)
        self.assertFalse(capture["production_qualified"])
        with self.assertRaisesRegex(ValueError, "frozen cached prompt amount"):
            validate_fixture(capture, output, cached=512)
        wrong = deepcopy(capture)
        wrong["frames"][0]["prepared"]["context_length"] = 512
        with self.assertRaisesRegex(ValueError, "actual complete cold/decode load differs"):
            validate_fixture(wrong, output)
        missing = deepcopy(capture)
        missing["frames"].pop()
        with self.assertRaisesRegex(ValueError, "all 128"):
            validate_fixture(missing, output)

    def test_descriptor_names_execution511_and_frontend512_without_changing_cell_or_split(self):
        descriptor = W.cell_descriptor()
        self.assertEqual(W.CACHED_TOKENS, 512)
        self.assertEqual(descriptor["cached_prompt_tokens"], 511)
        self.assertEqual(descriptor["cached_prompt_tokens_semantics"], "initial_execution_pre_context")
        self.assertEqual(descriptor["initial_execution_pre_context"], 511)
        self.assertEqual(descriptor["frontend_cached_prompt_tokens"], 512)
        self.assertEqual((descriptor["prompt_tokens"],descriptor["measured_offset"]), (512,16))
        self.assertEqual(descriptor["prompt_tokens"]+descriptor["measured_offset"]-1, 527)
        self.assertEqual([r["arm_order"] for r in descriptor["entries"]], ["AB","BA","AB"])
        self.assertEqual([r["split"] for r in descriptor["entries"]], ["calibration","calibration","validation"])

    def test_only_common_semantic_wrapper_changes_and_all_previous_sealed_bytes_hold(self):
        old_raw = (PREP / "runner/strong_native_cost_runner_v3.py").read_bytes()
        new_raw = (PREP / "runner/strong_native_cost_runner_v4.py").read_bytes()
        old = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old_raw).body if isinstance(n,ast.FunctionDef)}
        new = {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(new_raw).body if isinstance(n,ast.FunctionDef)}
        self.assertEqual(old.keys(),new.keys())
        self.assertEqual([name for name in old if old[name]!=new[name]], ["cell_descriptor","source_patch"])
        pins = {"runner/strong_native_cost_runner_v3.py":"d0eca2e77dc1fb2cab5f95fac7f31a41978ea5c560f45f1ead8a8e1b79166240",
            "runner/strong_native_cost_runner_v2.py":"36b1f20a106a823cef372795b9e65323cce6f7944ddc756d792878d74cbe0d7b",
            "raw_device_binding/uuid_minor_device.py":"f43769e132a1fe28c5d1c244f93785e63cc895c8a991b89ae6e4205c4308d427",
            "activation/native_conditional_cost.py":"675be4f91821863e23cc82128aadc040d8a9513c3aaf9b407d6c3fccd978ccb8",
            "activation/source/prefix_io_control/gpu_cell_issuer.py":"455abbd6391616fa49be323ba13daa4cdc3eda1b3642152b6f597c6ad681dcfe",
            "runner/bounded_native_full_step_collector.py":"9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"}
        for path,sha in pins.items():
            self.assertEqual(hashlib.sha256((PREP/path).read_bytes()).hexdigest(),sha)
        manifest = json.loads((PREP/"runner/RUNNER_SOURCE_MANIFEST.json").read_bytes())
        for row in manifest["files"]:
            raw = (PREP/row["path"]).read_bytes()
            self.assertEqual(len(raw),row["bytes"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),row["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
