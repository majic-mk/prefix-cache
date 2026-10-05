"""CPU scalar contracts for a future observer of the original vLLM methods.

This module does not install hooks, execute models, record CUDA events, submit
I/O, or release resources. It records copied CPU values only. A closed frame
covers original execute_model returning None followed by sample_tokens returning
actual CPU token IDs. It is not a GPU measurement or a production qualification.
"""
from dataclasses import dataclass

MAX_STEPS = 4096
MAX_REQUESTS = 128
MAX_FULL_OUTPUT_TOKENS = 262144


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(value, name, minimum=0):
    if type(value) is not int and type(value).__module__.split(".")[0] == "numpy":
        value = value.item()
    require(type(value) is int and value >= minimum, name + " must be an explicit integer")
    return value


def text(value, name):
    require(type(value) is str and bool(value.strip()) and len(value) <= 128,
            name + " must be bounded text")
    return value


def digest(value, name):
    require(type(value) is str and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value), name + " must be an exact SHA-256")
    return value


@dataclass(frozen=True)
class PreparedRow:
    request_id: str
    pre_context: int
    prompt_tokens: int
    scheduled_tokens: int


@dataclass(frozen=True)
class PreparedFrame:
    native_step_ordinal: int
    rows: tuple
    batch: int
    active_decode: int
    prefill_tokens: int
    context_length: int
    step_kind: str
    input_seq_lens_from_cpu_inputs: tuple
    context_basis: str = "pre_computed_tokens"


@dataclass(frozen=True)
class ClosedFrame:
    native_step_ordinal: int
    start_ns: int
    end_ns: int
    prepared: PreparedFrame
    outputs: tuple
    intended_timing_scope: str = "full_decode_step"
    gpu_elapsed_ns: object = None
    existing_io: object = None
    new_io: object = None

    @property
    def production_qualified(self):
        return False

    @property
    def gpu_verified(self):
        return False


@dataclass(frozen=True)
class StageTotal:
    ops: int
    nbytes: int


def totals(values):
    require(type(values) is tuple and len(values) == 4, "four immutable physical stage totals")
    for value in values:
        require(type(value) is StageTotal, "explicit physical stage total")
        integer(value.ops, "physical operations")
        integer(value.nbytes, "physical bytes")
        require((value.ops == 0) == (value.nbytes == 0), "stage total geometry")
    return values


@dataclass(frozen=True)
class DrainEvidence:
    run_id: str
    native_source_sha256: str
    accepted: tuple
    completed: tuple
    transferred_bytes: tuple
    failed_ops: tuple
    accounting_valid: bool
    outstanding_records: int
    worker_joined: bool
    ring_closed: bool
    ring_drained: bool
    ring_outstanding: int
    pending_copies: int
    active_native: int
    inflight_parents: int
    observation_failures: int
    boundary: str = "after_original_handler_shutdown"

    def validate(self, run_id, native_source_sha256):
        require(self.run_id == run_id and self.native_source_sha256 == native_source_sha256,
                "drain run/native source differs")
        require(self.boundary == "after_original_handler_shutdown", "original shutdown boundary required")
        for name in ("accounting_valid", "worker_joined", "ring_closed", "ring_drained"):
            require(type(getattr(self, name)) is bool and getattr(self, name),
                    "actual native closure field missing: " + name)
        for name in ("outstanding_records", "ring_outstanding", "pending_copies",
                     "active_native", "inflight_parents", "observation_failures"):
            require(integer(getattr(self, name), name) == 0, "native closure is not empty: " + name)
        require(totals(self.accepted) == totals(self.completed),
                "all actual accepted physical I/O must complete")
        require(type(self.transferred_bytes) is tuple and len(self.transferred_bytes) == 4 and
                type(self.failed_ops) is tuple and len(self.failed_ops) == 4,
                "all four actual native result counters required")
        for i, accepted in enumerate(self.accepted):
            require(integer(self.failed_ops[i], "actual failed operations") == 0 and
                    integer(self.transferred_bytes[i], "actual transferred bytes") == accepted.nbytes,
                    "partial or failed native I/O cannot qualify full completion")


@dataclass(frozen=True)
class ClosedFrameRun:
    run_id: str
    source_sha256: str
    frames: tuple
    outputs: tuple
    drain: DrainEvidence
    status: str = "CPU_FRAME_RECONCILIATION_ONLY"
    runtime_hook_status: str = "not_installed"

    @property
    def production_qualified(self):
        return False

    @property
    def gpu_verified(self):
        return False


def capture_prepared_frame(runner, scheduler_output, native_step_ordinal):
    """Read actual post-_prepare_inputs CPU geometry; retain no runner or request."""
    require(getattr(runner, "speculative_config", object()) is None, "speculative work unsupported")
    require(getattr(runner, "use_async_scheduling", None) is False,
            "async_scheduling=False must be explicit in the resolved original runner")
    require(getattr(runner, "is_pooling_model", None) is False, "generation runner required")
    parallel = runner.parallel_config
    for name in ("pipeline_parallel_size", "data_parallel_size", "tensor_parallel_size"):
        value = getattr(parallel, name, None)
        require(type(value) is int and value == 1, "single original TP/PP/DP domain required")
    require(integer(runner._profile_step, "post-execute profile step", 1) == native_step_ordinal + 1,
            "original execute ordinal differs from prepared frame")
    batch = runner.input_batch
    count = integer(batch.num_reqs, "actual prepared requests", 1)
    require(count <= MAX_REQUESTS, "actual request bound exceeded")
    ids = tuple(batch.req_ids[:count])
    require(len(ids) == count and len(set(ids)) == count, "actual prepared identities")
    for rid in ids:
        text(rid, "actual prepared request id")
    scheduled = scheduler_output.num_scheduled_tokens
    require(type(scheduled) is dict and set(scheduled) == set(ids),
            "scheduled/prepared cohort differs")
    require(not getattr(scheduler_output, "scheduled_spec_decode_tokens", None),
            "speculative scheduled work unsupported")
    rows = tuple(PreparedRow(rid,
        integer(batch.num_computed_tokens_cpu[i], "actual pre-context"),
        integer(batch.num_prompt_tokens[i], "actual prompt length", 1),
        integer(scheduled[rid], "actual scheduled tokens", 1)) for i, rid in enumerate(ids))
    require(integer(scheduler_output.total_num_scheduled_tokens, "actual scheduled token total", 1) ==
            sum(row.scheduled_tokens for row in rows), "scheduled token total differs")
    require(len({row.pre_context for row in rows}) == 1, "mixed context has no exact cell")
    decode = all(row.pre_context >= row.prompt_tokens and row.scheduled_tokens == 1 for row in rows)
    prefill = all(row.pre_context < row.prompt_tokens and
                  row.pre_context + row.scheduled_tokens <= row.prompt_tokens for row in rows)
    require(decode or prefill, "mixed, crossing-prompt, or multi-token decode unsupported")
    return PreparedFrame(native_step_ordinal, rows, count, count if decode else 0,
                         0 if decode else sum(row.scheduled_tokens for row in rows),
                         rows[0].pre_context, "decode" if decode else "prefill",
                         tuple(row.pre_context + row.scheduled_tokens for row in rows))


def capture_sampled_outputs(output, frame):
    """Copy original synchronous ModelRunnerOutput CPU IDs, never use tensors."""
    ids = getattr(output, "req_ids", None)
    tokens = getattr(output, "sampled_token_ids", None)
    indices = getattr(output, "req_id_to_index", None)
    require(type(ids) is list and type(tokens) is list and type(indices) is dict,
            "actual synchronous CPU ModelRunnerOutput required")
    require(len(ids) == frame.batch and len(tokens) == frame.batch and len(set(ids)) == frame.batch,
            "actual sample cohort size differs")
    require(set(ids) == {row.request_id for row in frame.rows} and
            indices == {rid: i for i, rid in enumerate(ids)}, "actual sample identity/index differs")
    copied = []
    for rid, values in zip(ids, tokens):
        require(type(values) is list and len(values) <= 1, "ordinary sampling permits at most one token")
        copied.append((rid, tuple(integer(token, "actual sampled token") for token in values)))
    require(frame.step_kind != "decode" or all(len(values) == 1 for _, values in copied),
            "every actual decode request must produce its original sampled token")
    if frame.step_kind == "prefill":
        by_id = dict(copied)
        require(all(row.pre_context + row.scheduled_tokens == row.prompt_tokens or
                    len(by_id[row.request_id]) == 0 for row in frame.rows),
                "partial original prefill cannot emit a sampled output token")
    return tuple(copied)


class FullStepFrameAdapter:
    """Bounded contract state, not an executor, hook installer, timer or IO owner."""
    runtime_hook_status = "not_installed"
    production_qualified = False
    gpu_verified = False

    def __init__(self, run_id, source_sha256, native_source_sha256, *, enabled=False,
                 max_steps=MAX_STEPS):
        self.run_id = text(run_id, "run identity")
        self.source_sha256 = digest(source_sha256, "original runner source")
        self.native_source_sha256 = digest(native_source_sha256, "original native source")
        require(type(enabled) is bool, "explicit optional observer switch")
        require(type(max_steps) is int and 1 <= max_steps <= MAX_STEPS, "bounded frame journal")
        self.enabled = enabled
        self.max_steps = max_steps
        self.valid = True
        self.last_reason = "not_installed"
        self._pending = None
        self._steps = []
        self._last_rows = {}
        self._output_tokens = 0
        self._first_ordinal = None
        self._closed = False

    @property
    def frames(self):
        return tuple(self._steps)

    def invalidate(self, reason):
        self.valid = False
        self.last_reason = str(reason)[:160]
        self._pending = None

    def _guard(self, function):
        if not self.enabled:
            return None
        try:
            require(self.valid and not self._closed, "frame journal invalid or closed")
            return function()
        except Exception as exc:
            self.invalidate(exc)
            raise

    def begin(self, native_step_ordinal, host_start_ns):
        def action():
            require(self._pending is None, "previous execute/sample step is still open")
            require(len(self._steps) < self.max_steps, "complete frame bound exceeded")
            ordinal = integer(native_step_ordinal, "native step ordinal")
            started = integer(host_start_ns, "actual host start", 1)
            if self._first_ordinal is None:
                self._first_ordinal = ordinal
            require(ordinal == self._first_ordinal + len(self._steps), "missing or duplicate native ordinal")
            require(not self._steps or started >= self._steps[-1].end_ns, "host step order overlaps")
            self._pending = dict(ordinal=ordinal, start_ns=started, phase="begun", frame=None)
        return self._guard(action)

    def prepared(self, runner, scheduler_output):
        def action():
            require(self._pending is not None and self._pending["phase"] == "begun",
                    "post-prepare observation has no matching original execute")
            frame = capture_prepared_frame(runner, scheduler_output, self._pending["ordinal"])
            require(len(set(self._last_rows) | {row.request_id for row in frame.rows}) <= MAX_REQUESTS,
                    "whole-run original request bound exceeded")
            for row in frame.rows:
                previous = self._last_rows.get(row.request_id)
                if previous is not None:
                    require(row.prompt_tokens == previous.prompt_tokens and
                            row.pre_context == previous.pre_context + previous.scheduled_tokens,
                            "actual original context/prompt progression differs")
            self._pending["frame"] = frame
            self._pending["phase"] = "prepared"
            return frame
        return self._guard(action)

    def executed(self, *, returned_none):
        def action():
            require(self._pending is not None and self._pending["phase"] == "prepared",
                    "original execute completed before valid prepared geometry")
            require(type(returned_none) is bool and returned_none,
                    "supported original generation path must await sample_tokens")
            self._pending["phase"] = "awaiting_sample"
        return self._guard(action)

    def sampled(self, output, host_end_ns):
        def action():
            require(self._pending is not None and self._pending["phase"] == "awaiting_sample",
                    "sample output cannot close an unexecuted or duplicate step")
            pending = self._pending
            ended = integer(host_end_ns, "actual host end", 1)
            require(ended > pending["start_ns"], "positive original execute/sample host interval")
            outputs = capture_sampled_outputs(output, pending["frame"])
            token_count = sum(len(tokens) for _, tokens in outputs)
            require(self._output_tokens + token_count <= MAX_FULL_OUTPUT_TOKENS,
                    "complete output token bound exceeded")
            frame = ClosedFrame(pending["ordinal"], pending["start_ns"], ended,
                                pending["frame"], outputs)
            self._steps.append(frame)
            self._output_tokens += token_count
            self._last_rows.update({row.request_id: row for row in frame.prepared.rows})
            self._pending = None
            return frame
        return self._guard(action)

    def close_run(self, expected_outputs, drain):
        def action():
            require(self._pending is None and bool(self._steps), "full original step/sample work is incomplete")
            require(type(expected_outputs) is dict and 1 <= len(expected_outputs) <= MAX_REQUESTS,
                    "complete independent original output cohort required")
            expected = {}
            for rid, tokens in expected_outputs.items():
                text(rid, "independent native request identity")
                require(type(tokens) is list and len(tokens) <= MAX_STEPS, "bounded independent full output")
                expected[rid] = tuple(integer(token, "independent actual token") for token in tokens)
            require(sum(len(tokens) for tokens in expected.values()) <= MAX_FULL_OUTPUT_TOKENS,
                    "independent output token bound exceeded")
            actual = {}
            for frame in self._steps:
                for row in frame.prepared.rows:
                    actual.setdefault(row.request_id, [])
                for rid, tokens in frame.outputs:
                    actual[rid].extend(tokens)
            require({rid: tuple(tokens) for rid, tokens in actual.items()} == expected,
                    "complete original token IDs do not reconstruct independent outputs")
            require(type(drain) is DrainEvidence, "original native closure evidence required")
            drain.validate(self.run_id, self.native_source_sha256)
            result = ClosedFrameRun(self.run_id, self.source_sha256, self.frames,
                                    tuple(sorted(expected.items())), drain)
            self._closed = True
            return result
        return self._guard(action)

