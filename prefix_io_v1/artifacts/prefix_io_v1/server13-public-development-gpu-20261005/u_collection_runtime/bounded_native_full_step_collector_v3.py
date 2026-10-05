"""Source-bound heterogeneous observation of the unchanged original U engine.

Only copied CPU geometry changes. Original CUDA EventProxy/enclosures, method
authentication, scalar lifecycle and initial no-forward handling are retained.
No cache/controller, table, executor, synchronization or cost qualification is
implemented here. This source is usable only by the separate U collection gate.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

PINS = {
    "bounded_native_full_step_collector.py": (25070, "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d"),
    "bounded_native_full_step_collector_v2.py": (13016, "a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0"),
    "heterogeneous_full_step_frame_adapter.py": (24311, "c4417482d82b057556ae3357c0711d72ec07a3cb4419b22917c28f7ce6712e7c"),
    "worker": (13090, "096f2fffe91de7132f45997be51fecceec57d65fdf86caab6f8bbe5c71748f7b"),
    "scalar": (14795, "347fb842989e8e14c73968528396721b546a11d7312206aae2efbb3739b0f3cb"),
    "frame": (16067, "bea850fd60831010c2ac0bf88c1f5f121f3ffca50d63ab56759319ed6e512582"),
    "reserve_join": (32772, "92996cb217bb11a85e237991cb82b37c1b1d10e26f65889338f030a0d46342ff"),
}


def require(value, reason):
    if not value:
        raise ValueError("HETEROGENEOUS_U_OBSERVATION_REJECTED: " + reason)


def _source(root, path, refs, pin):
    root, path = Path(root).resolve(), Path(path).absolute()
    require(not path.is_symlink() and path.is_file(), "regular actual source")
    path = path.resolve()
    require(path.is_relative_to(root), "source inside actual project")
    key = path.relative_to(root).as_posix()
    row = refs.get(key)
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"} and row["path"] == key,
            "exact current frozen source leaf")
    raw = path.read_bytes()
    require((len(raw), hashlib.sha256(raw).hexdigest()) == pin == (row["bytes"], row["sha256"]),
            "exact frozen source bytes, never a cached executable")
    return raw, dict(row), path


def _load(raw, path):
    name = "_collection_source_" + str(time.monotonic_ns())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _derive(module, raw, name, replacements):
    """Replace exact source-authenticated private observation load sites only."""
    tree = ast.parse(raw)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(functions) == 1, "one exact original observation function")
    node = functions[0]
    for old, new in replacements:
        expected = ast.dump(ast.parse(old, mode="eval").body, include_attributes=False)
        replacement = ast.parse(new, mode="eval").body
        class Replace(ast.NodeTransformer):
            def __init__(self):
                self.count = 0
            def visit(self, item):
                if isinstance(item, ast.expr) and ast.dump(item, include_attributes=False) == expected:
                    self.count += 1
                    return ast.copy_location(deepcopy(replacement), item)
                return super().visit(item)
        edit = Replace()
        node = edit.visit(node)
        require(edit.count == 1, "one exact private loader expression: " + old)
    unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(unit, str(module.__file__), "exec", dont_inherit=True), vars(module))


def load_frame_adapter(root, refs):
    path = Path(__file__).with_name("heterogeneous_full_step_frame_adapter.py")
    raw, row, actual = _source(root, path, refs, PINS[path.name])
    return _load(raw, actual), row


def preflight_observation_sources(root, refs, *, common):
    """Pure bytes only; no framework import, event creation or runtime owner."""
    directory = Path(__file__).parent
    rows = {}
    for name in ("bounded_native_full_step_collector.py", "bounded_native_full_step_collector_v2.py",
                 "heterogeneous_full_step_frame_adapter.py"):
        _, row, _ = _source(root, directory / name, refs, PINS[name])
        rows[name] = row
    for name, relative in (("worker", common.WORKER), ("scalar", common.SCALAR), ("frame", common.FRAME)):
        _, row, _ = _source(root, common.safe(root, relative), refs, PINS[name])
        rows[name] = row
    return rows


class HeterogeneousCapture:
    def __init__(self, original, source_refs):
        self._capture = original
        self._source_refs = source_refs

    def __getattr__(self, name):
        return getattr(self._capture, name)

    def resolve_after_original_step(self):
        """Single bounded query/elapsed pass at an existing engine return boundary."""
        # Original observe catches diagnostics errors and fails closed. It does
        # not wait, spin, synchronize, change a request or drop pending frames.
        return self._capture.observer.resolve_ready()

    def export(self):
        result = self._capture.export()
        frames = result.get("frames", [])
        result["heterogeneous_observation"] = dict(
            schema="source_bound_heterogeneous_original_U_observation_v1",
            source_refs=deepcopy(self._source_refs),
            heterogeneous_frame_count=sum(frame["prepared"]["exact_cell_eligible"] is False for frame in frames),
            mixed_frames_preserved=True, exact_finite_cost_cells_qualified=False,
            pending_event_bound=128, full_frame_bound=4096,
            progress_retirement="original_query_only_resolve_ready_after_original_step",
            no_added_synchronization=True, table_issued=False, ordinary_I_authorized=False)
        # The original dataclass capture contains tuples; the real runtime
        # validates this in-memory document immediately after writing its JSON.
        # Canonicalize only immutable primitive tuples to JSON lists, without
        # replacing values, omitting frames or inventing measurement fields.
        return json.loads(json.dumps(result, allow_nan=False))

    def detach(self):
        return self._capture.detach()


def install(worker, *, common, root, refs, run_id, event_class, selected_offsets=(),
            action=None, origin="native_gpu_recording", max_steps=4096):
    require(origin == "native_gpu_recording" and selected_offsets == () and action is None and
            type(max_steps) is int and max_steps == 4096,
            "only unchanged U full-stream observation, no finite policy or action")
    source_refs = preflight_observation_sources(root, refs, common=common)
    directory = Path(__file__).parent
    modules, raws = {}, {}
    for key, path in (("old", directory / "bounded_native_full_step_collector.py"),
                      ("passive", directory / "bounded_native_full_step_collector_v2.py"),
                      ("worker", common.safe(root, common.WORKER)),
                      ("scalar", common.safe(root, common.SCALAR))):
        pin = PINS[path.name] if key in ("old", "passive") else PINS[key]
        raw, _, path = _source(root, path, refs, pin)
        modules[key], raws[key] = _load(raw, path), raw
    frame, _ = load_frame_adapter(root, refs)
    scalar, observed_worker, old, passive = (modules[name] for name in ("scalar", "worker", "old", "passive"))
    scalar._collection_frame = frame
    _derive(scalar, raws["scalar"], "connect_runtime_scalar_observer", [
        ("load_frozen_adapter(adapter_source)", "_collection_frame")])
    observed_worker._collection_scalar, observed_worker._collection_frame = scalar, frame
    _derive(observed_worker, raws["worker"], "connect_worker_observation", [
        ('load_pinned(scalar_source, *FROZEN["scalar"])', "_collection_scalar"),
        ("scalar.load_frozen_adapter(frame_source)", "_collection_frame")])
    old._collection_worker, old._collection_scalar = observed_worker, scalar
    old._collection_frame_path = directory / "heterogeneous_full_step_frame_adapter.py"
    _derive(old, raws["old"], "install", [
        ("common.load_ref(root, refs[common.WORKER])", "_collection_worker"),
        ('worker_module.load_pinned(common.safe(root, common.SCALAR), *worker_module.FROZEN["scalar"])', "_collection_scalar"),
        ("common.safe(root, common.FRAME)", "_collection_frame_path")])
    metadata_types, metadata_ref = passive._metadata_types(root, refs)
    inner = old.install(worker, common=common, root=root, refs=refs, run_id=run_id,
        event_class=event_class, selected_offsets=selected_offsets, action=action, origin=origin, max_steps=max_steps)
    try:
        initial = passive.PassiveNoForwardCapture(inner, worker.model_runner, metadata_types=metadata_types,
            original_collector_ref=source_refs["bounded_native_full_step_collector.py"], metadata_source_ref=metadata_ref)
        return HeterogeneousCapture(initial, source_refs)
    except BaseException:
        inner.detach()
        raise


def validate_collection_capture(root, refs, capture, frontend, *, run_id, expected_ordinals, source_ref, original_join):
    """New U consumer: original CUDA/output proof with complete per-row geometry.

    It cannot be passed to an original finite-cell issuer or I prerequisite.
    Failed historical capture bytes are never rewritten or reclassified.
    """
    frame, frame_ref = load_frame_adapter(root, refs)
    metadata = capture.get("heterogeneous_observation")
    require(type(metadata) is dict and metadata.get("schema") == "source_bound_heterogeneous_original_U_observation_v1" and
            metadata.get("mixed_frames_preserved") is True and metadata.get("exact_finite_cost_cells_qualified") is False and
            metadata.get("table_issued") is False and metadata.get("ordinary_I_authorized") is False and
            metadata.get("pending_event_bound") == 128 and metadata.get("full_frame_bound") == 4096 and
            metadata.get("no_added_synchronization") is True and
            metadata.get("progress_retirement") == "original_query_only_resolve_ready_after_original_step",
            "source-bound mixed stream observation, never exact cost qualification")
    declared = metadata.get("source_refs")
    require(type(declared) is dict and declared.get("heterogeneous_full_step_frame_adapter.py") == frame_ref,
            "actual heterogeneous frame source closure")
    for name, row in declared.items():
        require(name in PINS and type(row) is dict, "known observation source role")
        _source(root, Path(root) / row["path"], refs, PINS[name])
    require(set(declared) == {"bounded_native_full_step_collector.py", "bounded_native_full_step_collector_v2.py",
        "heterogeneous_full_step_frame_adapter.py", "worker", "scalar", "frame"}, "complete source-derived observation dependency set")
    require(refs.get(source_ref.get("path")) == source_ref and
            Path(source_ref["path"]).name == "bounded_native_full_step_collector_v3.py", "actual new U collector source")
    raw_self = (Path(root) / source_ref["path"]).read_bytes()
    require(len(raw_self) == source_ref["bytes"] and hashlib.sha256(raw_self).hexdigest() == source_ref["sha256"],
            "actual collector source bytes")
    raw, _, join_path = _source(root, original_join.__file__, refs, PINS["reserve_join"])
    require(inspect.getsourcefile(original_join._capture) == str(join_path), "exact original capture consumer source")
    frames = capture.get("frames")
    require(type(frames) is list and 128 <= len(frames) <= 4096, "complete bounded full stream")
    last, heterogeneous = {}, 0
    for item in frames:
        require(type(item) is dict, "actual original scalar frame")
        rows = frame.validate_prepared_document(item.get("prepared"), item.get("native_step_ordinal"))
        heterogeneous += item["prepared"]["exact_cell_eligible"] is False
        outputs = item.get("outputs")
        require(type(outputs) is list and len(outputs) == len(rows) and
                len({out[0] for out in outputs if type(out) is list and len(out) == 2}) == len(rows) and
                {out[0] for out in outputs} == {row.request_id for row in rows}, "complete original per-frame sampled cohort")
        for row in rows:
            previous = last.get(row.request_id)
            require(previous is None or (row.prompt_tokens == previous.prompt_tokens and
                row.pre_context == previous.pre_context + previous.scheduled_tokens), "complete actual per-request context progression")
            last[row.request_id] = row
        require(len(last) <= 128, "bounded complete original request cohort")
    require(metadata.get("heterogeneous_frame_count") == heterogeneous and
            type(metadata.get("heterogeneous_frame_count")) is int, "actual mixed frames counted without dropping")
    # A private pure consumer is derived from the pinned original function. All
    # event timing, causal enclosures, ordinals, full128 outputs/timelines and
    # complete cohort checks remain byte-source-derived unchanged.
    namespace = dict(vars(original_join))
    namespace["_collection_geometry"] = frame.validate_prepared_document
    namespace["_collection_output"] = frame.validate_output_document
    tree = ast.parse(raw)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_capture")
    matched = [0, 0]
    class Consumer(ast.NodeTransformer):
        def visit_Expr(self, value):
            if isinstance(value.value, ast.Call) and isinstance(value.value.func, ast.Name) and value.value.func.id == "require":
                args = value.value.args
                if len(args) == 2 and isinstance(args[1], ast.Constant):
                    if args[1].value == "actual prepared metadata scalars":
                        matched[0] += 1
                        return ast.copy_location(ast.parse("_collection_geometry(prepared, ordinal)").body[0], value)
                    if args[1].value == "each actual original sampled increment exactly one token":
                        matched[1] += 1
                        return ast.copy_location(ast.parse("_collection_output(item, prepared)").body[0], value)
            return self.generic_visit(value)
    node = Consumer().visit(node)
    require(matched == [1, 1], "only two exact geometry/output pure consumer clauses differ")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(join_path), "exec", dont_inherit=True), namespace)
    result = namespace["_capture"](capture, frontend, run_id=run_id, expected_ordinals=expected_ordinals, source_ref=source_ref)
    return dict(result, schema="complete_heterogeneous_original_U_stream_observation_v1",
        heterogeneous_frame_count=heterogeneous, mixed_frames_preserved=True,
        complete_stream_observation_valid=True, exact_finite_cost_cells_qualified=False,
        full_step_evidence_qualified=False, table_issued=False, ordinary_I_authorized=False)
