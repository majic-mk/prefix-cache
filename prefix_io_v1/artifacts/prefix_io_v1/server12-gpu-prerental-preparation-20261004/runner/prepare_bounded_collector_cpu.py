"""Generate a bounded observation-only derivative of the original full-step probe."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "i_pilot_cpu_preparation_20261004/i_bridge/frozen/native_full_step_collector.py"
OLD_SHA = "bcb58a9c812c2f7dc846013348cdca2de3a641e8c8a0515b4d7307e0fc8676bf"


def generate(source=OLD, output=HERE / "bounded_native_full_step_collector.py"):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != OLD_SHA:
        raise ValueError("original full-step collector source drift")
    text = raw.decode("utf-8")
    changes = [
        ('def __init__(self, *, run_id, origin, selected_offsets, action, clock=time.monotonic_ns):',
         'def __init__(self, *, run_id, origin, selected_offsets, action, clock=time.monotonic_ns, max_steps=128):'),
        ('require(type(selected_offsets) is tuple and 0 < len(selected_offsets) <= 8 and\n'
         '                all(type(item) is int and 1 <= item < 128 for item in selected_offsets) and',
         'require(type(max_steps) is int and 128 <= max_steps <= 4096, "bounded complete stream observation limit")\n'
         '        require(type(selected_offsets) is tuple and 0 <= len(selected_offsets) <= 8 and\n'
         '                (selected_offsets or action is None) and\n'
         '                all(type(item) is int and 1 <= item < max_steps for item in selected_offsets) and'),
        ('self.run_id, self.origin, self.selected_offsets = run_id, origin, selected_offsets',
         'self.max_steps = max_steps\n        self.run_id, self.origin, self.selected_offsets = run_id, origin, selected_offsets'),
        ('require(len(frames) == 128, "exactly 128 complete actual original steps")',
         'require((len(frames) == 128 if self.max_steps == 128 else 1 <= len(frames) <= self.max_steps),\n'
         '                    "complete bounded actual original steps; calibration default remains exactly128")'),
        ('ordinals == list(range(ordinals[0], ordinals[0] + 128))',
         'ordinals == list(range(ordinals[0], ordinals[0] + len(frames)))'),
        ('require(len(observer.events.diagnostics) == 128 and all(',
         'require(len(observer.events.diagnostics) == len(frames) and all('),
        ('scope="server11_full_step_native_capture_v1",',
         'scope=("server11_full_step_native_capture_v1" if self.max_steps == 128 else "bounded_original_full_step_stream_v1"),'),
        ('action=None, origin="native_gpu_recording"):', 'action=None, origin="native_gpu_recording", max_steps=128):'),
        ('selected_offsets=selected_offsets, action=action)',
         'selected_offsets=selected_offsets, action=action, max_steps=max_steps)'),
        ('capture.factory = BoundedEventFactory(event_class)',
         'capture.factory = BoundedEventFactory(event_class, max_steps=max_steps)'),
        ('event_factory=capture.factory, enabled=True, max_pending=128, max_steps=128)',
         'event_factory=capture.factory, enabled=True, max_pending=min(128, max_steps), max_steps=max_steps)'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError("single explicit collector patch preimage required: " + before)
        text = text.replace(before, after)
    old_tree, new_tree = ast.parse(raw), ast.parse(text)
    def methods(tree):
        return {(owner.name, member.name): ast.dump(member, include_attributes=False)
                for owner in tree.body if isinstance(owner, ast.ClassDef)
                for member in owner.body if isinstance(member, ast.FunctionDef)}
    a, b = methods(old_tree), methods(new_tree)
    changed = [key for key in a if a[key] != b[key]]
    if set(changed) != {("FullStepCapture", "__init__"), ("FullStepCapture", "export")}:
        raise ValueError("only complete observation bounds/export may change")
    for method in ("after_prepare", "current_single_file_step", "attach_prepare", "detach", "arm_single_file_wait"):
        if a[("FullStepCapture", method)] != b[("FullStepCapture", method)]:
            raise ValueError("original preparation/CUDA causal/wait boundary changed")
    output_raw = text.encode("utf-8")
    with output.open("xb") as stream:
        stream.write(output_raw)
    proof = dict(status="PASS_CPU_THIN_OBSERVATION_BOUND_DERIVATIVE", original_sha256=OLD_SHA,
                 original_bytes=len(raw), derivative_sha256=hashlib.sha256(output_raw).hexdigest(), derivative_bytes=len(output_raw),
                 original_default_calibration_steps=128, optional_bounded_stream_steps_max=4096,
                 changed_methods=[list(key) for key in changed], parameterized_install=True,
                 unchanged_prepare_current_step_wait_and_detach_AST=True, numerical_formulas_changed=False,
                 original_model_executor_and_native_cache_unchanged=True, added_gpu_synchronization=False,
                 actual_gpu_runs=0, qualified_cost_cells_created=0)
    with (HERE / "BOUNDED_COLLECTOR_CPU_SOURCE_PROOF.json").open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return proof


if __name__ == "__main__":
    print(json.dumps(generate()))
