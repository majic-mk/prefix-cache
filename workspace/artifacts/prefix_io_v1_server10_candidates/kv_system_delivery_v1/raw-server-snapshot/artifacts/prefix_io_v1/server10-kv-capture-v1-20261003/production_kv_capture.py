"""Bounded native KV-byte diagnosis. No qualification or timing authority.

Only install in the existing guarded size=128/domain=1024/reps=3 acquisition.
Callbacks are inserted into copies of four source-pinned original methods. They
do not replace an executor, consume completions, or alter original ownership.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
import threading
import types

RECEIPT_FILE = "native-kv-byte-receipt.json"
SCHEMA = "server10-native-kv-byte-diagnostic-v1"
MAX_FILES, FILE_BYTES, MAX_RECORDS = 24, 917504, 512
REACTOR = "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py"
RUNNER = "third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py"
PINS = {
    REACTOR: (169906, "2b0ac1e8e68f82adb42ade03b74b49843eabe55361a00f0ac1f7a738ac423e18"),
    RUNNER: (329690, "61828af13a5f523c9683df560374d75bc1df93254a9ac4a449c95447100e9075"),
    "third_party/work/py-kvcache-p4-02-cpu/py_kvcache/transfer.py":
        (12061, "1dcc5f4370e3db694d34924038a8fdd5a2eb12cf7f04ba81072dc969f275c1ba"),
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_exact(path, size, digest=None):
    p = Path(path)
    require(p.is_file() and not p.is_symlink() and p.stat().st_size == size,
            "regular bounded bytes required: " + str(p))
    raw = p.read_bytes()
    require(len(raw) == size and (digest is None or sha(raw) == digest), "payload/source drift")
    return raw


def code_shape(code):
    return (code.co_code, code.co_names, code.co_varnames, code.co_flags,
            code.co_argcount, code.co_kwonlyargcount,
            tuple(code_shape(x) if isinstance(x, types.CodeType) else x for x in code.co_consts))


def code_at(code, names):
    for name in names:
        found = [x for x in code.co_consts if isinstance(x, types.CodeType) and x.co_name == name]
        require(len(found) == 1, "unique source code path")
        code = found[0]
    return code


def transformed_method(original, raw, filename, class_name, name, callback):
    """Keep every original statement; insert exactly one guarded observation."""
    target = inspect.unwrap(original.__func__ if hasattr(original, "__func__") else original)
    expected = code_at(compile(raw, filename, "exec", dont_inherit=True), (class_name, name))
    require(code_shape(target.__code__) == code_shape(expected), "live original callable drift: " + name)
    module = ast.parse(raw, filename)
    klass = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    fn = copy.deepcopy(next(x for x in klass.body if isinstance(x, ast.FunctionDef) and x.name == name))
    # Preserve decorators, in particular torch.inference_mode on execute_model.
    n = 0
    if name == "_drain_cuda_copies":
        for node in ast.walk(fn):
            if isinstance(node, ast.For):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "not copy.end_event.query()":
                        require(len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Continue), "native event branch drift")
                        node.body.insert(i + 1, ast.parse("__kv_capture(self, copy)").body[0]); n += 1
                        break
    elif name == "_on_write_complete":
        for node in ast.walk(fn):
            if isinstance(node, ast.Try):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == "self.file_store.finish_write":
                        node.body.insert(i + 1, ast.parse("__kv_capture(self, op, result_nbytes)").body[0]); n += 1
                        break
    elif name == "_on_read_complete":
        fn.body.insert(0, ast.parse("__kv_capture(self, op, result_nbytes)").body[0]); n = 1
    elif name == "execute_model":
        for node in ast.walk(fn):
            if isinstance(node, ast.With):
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call) and ast.unparse(stmt.value.func) == "self._model_forward":
                        node.body.insert(i, ast.parse("__kv_capture(self, scheduler_output)").body[0]); n += 1
                        break
    require(n == 1, "exactly one native observation site: " + name)
    # Retain the module's future flags, but do not execute its imports/body.
    futures = [x for x in module.body if isinstance(x, ast.ImportFrom) and x.module == "__future__"]
    emitted = ast.fix_missing_locations(ast.Module(body=futures + [fn], type_ignores=[]))
    namespace = dict(target.__globals__)
    require("__kv_capture" not in namespace, "private callback name collision")
    namespace["__kv_capture"] = callback
    exec(compile(emitted, filename, "exec", dont_inherit=True), namespace)
    return namespace[name]


class CaptureHandle:
    def __init__(self, project, refs, output_dir, mode, reactor, runner, *, producer_manifest=None):
        require(mode in ("populate", "paired"), "fixed acquisition mode")
        self.project = Path(project).resolve(); self.output = Path(output_dir).resolve()
        require(self.output.is_relative_to(self.project) and not self.output.exists(), "new project-contained output")
        self.mode, self.reactor, self.runner = mode, reactor, runner
        self.records, self.failures, self.files, self.jobs, self.computes = [], [], {}, {}, {}
        self.producer_prompts = {}
        self.patches, self.lock, self.sources = [], threading.RLock(), {}
        self.detached = False; self.finished = False
        self.producer_ref = None; self.producer_files = {}
        for relative, (size, digest) in PINS.items():
            row = refs.get(relative)
            require(type(row) is dict and row.get("bytes") == size and row.get("sha256") == digest,
                    "locked original source required: " + relative)
            self.sources[relative] = read_exact(self.project / relative, size, digest)
        self._geometry()
        if mode == "paired":
            p = Path(producer_manifest).resolve()
            require(p.is_relative_to(self.project) and p.stat().st_size < 1000000, "bounded producer receipt")
            raw = p.read_bytes(); previous = json.loads(raw)
            validate_receipt(previous, mode="populate")
            require(previous.get("schema") == SCHEMA and previous.get("status") == "PASS_NATIVE_KV_BYTE_DIAGNOSTIC"
                    and previous.get("mode") == "populate" and previous.get("diagnostic_real_bytes_verified") is True,
                    "successful actual producer required")
            require(previous["geometry"]["logical_file_bytes"] == FILE_BYTES and len(previous["files"]) == 24,
                    "exact prior producer domain")
            self.producer_ref = dict(path=str(p), bytes=len(raw), sha256=sha(raw))
            self.producer_files = previous["files"]
            self.producer_prompts = previous["producer_prompts"]
            self.producer_dir = p.parent
        else:
            require(producer_manifest is None, "populate has no inherited producer")
        self.output.mkdir(parents=True, exist_ok=False)
        try:
            self._patch(reactor, "_drain_cuda_copies", REACTOR, "IoReactor", self._copies)
            self._patch(reactor, "_on_write_complete", REACTOR, "IoReactor", self._written)
            self._patch(reactor, "_on_read_complete", REACTOR, "IoReactor", self._read)
            self._patch(runner, "execute_model", RUNNER, "GPUModelRunner", self._before_compute)
        except BaseException:
            self.detach()
            raise

    def _geometry(self):
        layout = self.reactor.layout
        require(len(self.runner.kv_cache_config.kv_cache_groups) == 1, "single canonical KV group")
        require(self.runner.speculative_config is None and self.runner.use_async_scheduling is False,
                "ordinary synchronous runner only")
        require(all(getattr(self.runner.parallel_config, n) == 1 for n in
                    ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")), "single rank domain")
        require(self.runner.model_config.max_model_len == 1040, "fixed domain1024 plus16")
        require(layout.storage_block_size_factor == 1 and layout.storage_block_bytes == FILE_BYTES,
                "fixed Qwen 16-token complete storage pages")
        require(self.reactor.file_store.io_size == FILE_BYTES, "no padded/partial storage")
        require(1 <= len(layout.gpu_tensors) <= 64, "bounded layers")
        tensors = []
        for tensor, page in zip(layout.gpu_tensors, layout.bytes_per_kernel_block):
            require(str(tensor.dtype) == "torch.int8" and tensor.is_cuda and tensor.ndim == 2
                    and tensor.element_size() == 1 and tensor.stride(0) == page
                    and tensor.stride(1) == 1 and tensor.shape[1] == page, "canonical contiguous GPU bytes")
            tensors.append(dict(data_ptr=int(tensor.data_ptr()), shape=list(tensor.shape), stride=list(tensor.stride()),
                                dtype=str(tensor.dtype), device=str(tensor.device), page_bytes=page))
        require(len(tensors) == len(layout.gpu_tensors) == len(layout.bytes_per_kernel_block), "layout zip complete")
        self.geometry = dict(tensors=tensors, logical_file_bytes=FILE_BYTES, storage_factor=1, group_count=1,
                             prefix_tokens=128, acquisition_domain=1024, repetitions=3)

    def _patch(self, owner, name, relative, klass, callback):
        old = getattr(owner, name)
        require(Path(inspect.unwrap(old).__code__.co_filename).resolve() == (self.project / relative).resolve(),
                "live callable file differs")
        def safe(*args):
            with self.lock:
                if self.failures: return
                try: callback(*args)
                except Exception as exc: self._fail(name + ": " + type(exc).__name__ + ": " + str(exc))
                except BaseException as exc:
                    self._fail(name + ": termination: " + type(exc).__name__)
                    raise
        function = transformed_method(old, self.sources[relative], str(self.project / relative), klass, name, safe)
        present = name in owner.__dict__; prior = owner.__dict__.get(name)
        installed = types.MethodType(function, owner)
        self.patches.append((owner, name, present, prior, installed))
        setattr(owner, name, installed)

    def _fail(self, reason):
        if len(self.failures) < 16: self.failures.append(str(reason)[:500])

    def _record(self, stage, **fields):
        require(len(self.records) < MAX_RECORDS, "capture record bound")
        row = dict(sequence=len(self.records) + 1, stage=stage, **fields)
        self.records.append(row); return row

    def _identity(self, job, index):
        require(type(job.job_id) is int and type(job.profile.req_id) is str and len(job.profile.req_id) <= 128,
                "actual job/request scalar identity")
        require(job.total_files == 8 and len(job.block_hashes) == len(job.block_chunks) == 8,
                "one full size128 prefix per native job")
        require(type(index) is int and 0 <= index < 8, "file index")
        key = job.block_hashes[index]
        require(type(key) is bytes and 1 <= len(key) <= 64, "original hash bytes")
        blocks = [int(x) for x in job.block_chunks[index]]
        require(len(blocks) == 1 and 0 <= blocks[0] < self.geometry["tensors"][0]["shape"][0], "complete GPU block mapping")
        jid = ("store:" if job.is_store else "load:") + str(job.job_id)
        if jid not in self.jobs:
            count = sum(x["is_store"] == job.is_store for x in self.jobs.values())
            limit = 3 if job.is_store or self.mode == "populate" else 6
            require(count < limit, "bounded native jobs")
            self.jobs[jid] = dict(is_store=job.is_store, request_id=job.profile.req_id, files={},
                                  native_job_object_id=id(job), native_reactor_id=id(self.reactor))
        require(self.jobs[jid]["request_id"] == job.profile.req_id, "job reuse/identity drift")
        require(self.jobs[jid]["native_job_object_id"] == id(job), "native job object replaced")
        return key.hex(), blocks, jid

    def _slot(self, reactor, slot):
        view = reactor._slot_view(slot)
        require(view.nbytes == FILE_BYTES and str(view.dtype) == "uint8" and view.flags.c_contiguous,
                "actual bounded staging bytes")
        raw = view.tobytes(order="C"); require(len(raw) == FILE_BYTES, "full staging")
        return raw

    def _gpu(self, blocks):
        # One canonical page at a time; no persistent tensor/stream/event copies.
        chunks = []
        for tensor, geometry in zip(self.reactor.layout.gpu_tensors, self.geometry["tensors"]):
            require(int(tensor.data_ptr()) == geometry["data_ptr"] and list(tensor.shape) == geometry["shape"],
                    "registered GPU tensor identity changed")
            page = tensor[blocks[0]].detach().cpu().contiguous().numpy().tobytes(order="C")
            require(len(page) == geometry["page_bytes"], "complete live CUDA page capture")
            chunks.append(page)
        raw = b"".join(chunks); require(len(raw) == FILE_BYTES, "full logical GPU bytes")
        return raw

    def _source(self, key):
        files, directory = (self.files, self.output) if self.mode == "populate" else (self.producer_files, self.producer_dir)
        require(key in files, "actual producer bytes absent for key")
        row = files[key]; name = row["payload_file"]
        require(name == "producer-" + key + ".bin", "fixed producer payload name")
        return read_exact(directory / name, FILE_BYTES, row["sha256"])

    def _copies(self, reactor, completed):
        require(reactor is self.reactor and completed.accounting_copy_accepted is True, "actual accepted nonempty copy")
        # This callback is reachable only after the pinned original event query returned true.
        members = [(completed.slot_index, completed.job, completed.file_index, None, None)] if completed.is_store else completed.members
        require(type(members) is list and 1 <= len(members) <= 24, "bounded original fused members")
        for slot, job, index, shared, cache in members:
            require(not job.future_set and not job.future.done() and job.failed is None, "capture before parent publication")
            key, blocks, jid = self._identity(job, index)
            require(bool(job.is_store) == bool(completed.is_store), "native direction")
            staging = self._slot(reactor, slot); gpu = self._gpu(blocks)
            require(gpu == staging, "GPU and original staging bytes differ")
            if completed.is_store:
                require(self.mode == "populate" and key not in self.files and len(self.files) < MAX_FILES,
                        "one new producer capture per bounded key")
                filename = "producer-" + key + ".bin"
                with (self.output / filename).open("xb") as stream: stream.write(gpu)
                self.files[key] = dict(payload_file=filename, bytes=FILE_BYTES, sha256=sha(gpu),
                                       job_id=jid, file_index=index, request_id=job.profile.req_id, block_ids=blocks)
                stage = "d2h_complete_source_and_staging"
            else:
                require(gpu == self._source(key), "restored GPU differs from actual producer")
                stage = "h2d_complete_dest_and_staging"
            require(str(index) not in self.jobs[jid]["files"], "duplicate native file completion")
            row = self._record(stage, key=key, job_id=jid, request_id=job.profile.req_id, file_index=index,
                               block_ids=blocks, slot_index=slot, bytes=FILE_BYTES, sha256=sha(gpu),
                               original_end_event_id=id(completed.end_event), original_event_query_true=True,
                               parent_not_published=True, complete_byte_equality=True,
                               source_kind="store" if completed.is_store else ("cache" if cache is not None else "shared" if shared is not None else "file"))
            self.jobs[jid]["files"][str(index)] = row

    def _written(self, reactor, op, result_nbytes):
        require(reactor is self.reactor and op.op_kind == "write" and op.is_write is True
                and result_nbytes == FILE_BYTES, "actual successful write CQE")
        key, blocks, jid = self._identity(op.job, op.file_index)
        require(self.mode == "populate" and key in self.files, "observed producer before published write")
        expected_path = Path(reactor.file_mapper.get_file_name(op.job.block_hashes[op.file_index])).resolve()
        require(Path(op.final_path).resolve() == expected_path and expected_path.is_relative_to(self.project), "actual mapper publication")
        payload = read_exact(expected_path, FILE_BYTES)
        require(payload == self._source(key), "actual published SSD differs from producer")
        self._record("ssd_write_published", key=key, job_id=jid, file_index=op.file_index,
                     path=str(expected_path), bytes=FILE_BYTES, sha256=sha(payload), cqe_nbytes=result_nbytes,
                     finish_write_returned=True, complete_byte_equality=True)

    def _read(self, reactor, op, result_nbytes):
        require(reactor is self.reactor and op.op_kind == "read" and op.is_write is False, "actual read CQE op")
        require(result_nbytes == FILE_BYTES, "failed/short read CQE")
        key = op.preload_hash if op.preload_hash is not None else op.job.block_hashes[op.file_index]
        require(type(key) is bytes and 1 <= len(key) <= 64, "read key")
        key = key.hex(); payload = self._slot(reactor, op.slot_index)
        require(payload == self._source(key), "actual SSD read staging differs from producer")
        self._record("ssd_read_complete", key=key, slot_index=op.slot_index, fd=op.fd,
                     bytes=FILE_BYTES, sha256=sha(payload), cqe_nbytes=result_nbytes,
                     preload=op.preload_hash is not None, complete_byte_equality=True)

    def _before_compute(self, runner, scheduler_output):
        require(runner is self.runner, "registered runner")
        for rid in scheduler_output.num_scheduled_tokens:
            state = runner.requests[rid]
            prompt = state.prompt_token_ids
            if len(prompt) == 1: continue  # Original populate's real flush request.
            require(len(prompt) == 129, "fixed original prefix128 plus query")
            jobs = [(jid, data) for jid, data in self.jobs.items() if not data["is_store"] and data["request_id"] == rid]
            if not jobs:
                require(self.mode == "populate", "consumer compute before captured native load")
                require(rid not in self.producer_prompts and len(self.producer_prompts) < 3,
                        "one prefill for each of three original producers")
                digest = sha(json.dumps(list(prompt), separators=(",", ":")).encode())
                self.producer_prompts[rid] = digest
                self._record("producer_model_forward", request_id=rid, prompt_sha256=digest, prompt_tokens=129)
                continue  # Producer computation before its later store.
            require(len(jobs) == 1 and rid not in self.computes, "one original ordinary consumer forward")
            jid, data = jobs[0]
            require(set(data["files"]) == {str(x) for x in range(8)}, "compute before all actual H2D pages captured")
            expected = [block for i in range(8) for block in data["files"][str(i)]["block_ids"]]
            require(len(state.block_ids) == 1 and list(state.block_ids[0][:8]) == expected,
                    "actual consumer request prefix block IDs differ")
            require(len(set(expected)) == 8, "duplicate destination pages")
            source_files = self.files if self.mode == "populate" else self.producer_files
            source_requests = {source_files[data["files"][str(i)]["key"]]["request_id"] for i in range(8)}
            require(len(source_requests) == 1, "consumer combines different producer prefixes")
            producer_request = next(iter(source_requests))
            digest = sha(json.dumps(list(prompt), separators=(",", ":")).encode())
            require(self.producer_prompts.get(producer_request) == digest, "actual producer/consumer prompt identity")
            self.computes[rid] = self._record("before_model_forward", request_id=rid, job_id=jid,
                                             block_ids=expected, prompt_tokens=129,
                                             prompt_sha256=digest, producer_request_id=producer_request,
                                             captured_file_sequences=[data["files"][str(i)]["sequence"] for i in range(8)])

    def detach(self):
        with self.lock:
            if self.detached: return not self.failures
            for owner, name, present, old, installed in reversed(self.patches):
                if owner.__dict__.get(name) is not installed:
                    self._fail("installed method drift at detach: " + name); continue
                if present: setattr(owner, name, old)
                else: delattr(owner, name)
            self.detached = True
            return not self.failures

    def finish(self):
        with self.lock:
            require(not self.finished, "write receipt once")
            self.finished = True
            try:
                require(self.detached, "restore original methods first")
                require(self.reactor._closed is True and not self.reactor._pending_copies, "original reactor shutdown/drain required")
                loads = [x for x in self.jobs.values() if not x["is_store"]]
                require(len(loads) == (3 if self.mode == "populate" else 6) and len(self.computes) == len(loads),
                        "complete original consumer request cohort")
                require(all(len(x["files"]) == 8 for x in self.jobs.values()), "all observed native files complete")
                writes = [x for x in self.records if x["stage"] == "ssd_write_published"]
                reads = [x for x in self.records if x["stage"] == "ssd_read_complete"]
                if self.mode == "populate":
                    require(len(self.files) == 24 and len(writes) == 24 and {x["key"] for x in writes} == set(self.files),
                            "all24 producer files actually published")
                    require(not reads, "populate g_mem must retain original staging route")
                else:
                    require(len(reads) == 24 and len({x["key"] for x in reads}) == 24 and not writes,
                            "paired exact24 real SSD reads and no stores")
                    cache = [x for x in self.records if x["stage"] == "h2d_complete_dest_and_staging" and x["source_kind"] == "cache"]
                    require(len(cache) == 24, "paired exact24 staging-hit restores")
                if self.producer_ref:
                    read_exact(self.producer_ref["path"], self.producer_ref["bytes"], self.producer_ref["sha256"])
            except BaseException as exc: self._fail(type(exc).__name__ + ": " + str(exc))
            result = dict(schema=SCHEMA, mode=self.mode,
                          status="FAIL_NATIVE_KV_BYTE_DIAGNOSTIC" if self.failures else "PASS_NATIVE_KV_BYTE_DIAGNOSTIC",
                          diagnostic_real_bytes_verified=not self.failures,
                          production_qualified=False, cost_qualified=False, effect_verified=False,
                          load_planner="off", timing_usable=False, failures=self.failures,
                          methods_restored=self.detached, geometry=self.geometry, files=self.files,
                          native_jobs=self.jobs, records=self.records, producer_manifest=self.producer_ref,
                          producer_files=self.producer_files, producer_prompts=self.producer_prompts,
                          native_reactor_id=id(self.reactor), native_runner_id=id(self.runner),
                          source_refs={k: dict(bytes=v[0], sha256=v[1]) for k, v in PINS.items()},
                          capture_order_note="Producer source is read after native D2H completion while whole-parent protection is still held.")
            if not self.failures:
                try: validate_receipt(result, mode=self.mode)
                except BaseException as exc:
                    self._fail("receipt closure: " + str(exc))
                    result["status"] = "FAIL_NATIVE_KV_BYTE_DIAGNOSTIC"
                    result["diagnostic_real_bytes_verified"] = False
            with (self.output / RECEIPT_FILE).open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
            return result


def validate_receipt(receipt, *, mode):
    """Pure-CPU closure check of a live receipt, not independent GPU authority.

The guarded caller must additionally bind this exact JSON file/ref, source lock,
original acquisition rows, GPU identity, producer publication and shutdown.
"""
    require(type(receipt) is dict and mode in ("populate", "paired") and receipt.get("schema") == SCHEMA
            and receipt.get("mode") == mode, "native diagnostic receipt identity")
    require(receipt.get("status") == "PASS_NATIVE_KV_BYTE_DIAGNOSTIC"
            and receipt.get("diagnostic_real_bytes_verified") is True and receipt.get("failures") == []
            and receipt.get("methods_restored") is True, "successful restored native observations required")
    require(all(receipt.get(k) is False for k in ("production_qualified", "cost_qualified", "effect_verified", "timing_usable"))
            and receipt.get("load_planner") == "off", "diagnostic-only boundary")
    geometry = receipt["geometry"]
    require(geometry["logical_file_bytes"] == FILE_BYTES and geometry["storage_factor"] == 1
            and geometry["group_count"] == 1 and geometry["prefix_tokens"] == 128
            and geometry["acquisition_domain"] == 1024 and geometry["repetitions"] == 3, "fixed geometry")
    records, jobs = receipt["records"], receipt["native_jobs"]
    require(type(records) is list and 1 <= len(records) <= MAX_RECORDS and type(jobs) is dict, "bounded live record shape")
    require([x["sequence"] for x in records] == list(range(1, len(records) + 1)), "strict live observation order")
    stages = {}
    for row in records: stages.setdefault(row["stage"], []).append(row)
    counts = ({"producer_model_forward": 3, "d2h_complete_source_and_staging": 24, "ssd_write_published": 24,
               "h2d_complete_dest_and_staging": 24, "before_model_forward": 3} if mode == "populate" else
              {"ssd_read_complete": 24, "h2d_complete_dest_and_staging": 48, "before_model_forward": 6})
    require({k: len(v) for k, v in stages.items()} == counts, "exact native stage cohort")
    files = receipt["files"] if mode == "populate" else receipt["producer_files"]
    require(type(files) is dict and len(files) == MAX_FILES and len(receipt["producer_prompts"]) == 3, "three actual producer prefixes")
    if mode == "paired":
        require(receipt["files"] == {} and type(receipt["producer_manifest"]) is dict, "paired immutable producer ancestry")
        require(len(receipt["producer_manifest"]["sha256"]) == 64, "producer receipt ref")
    elif receipt["producer_manifest"] is not None:
        raise ValueError("populate cannot inherit ancestry")
    for row in records:
        if "key" not in row: continue
        source = files[row["key"]]
        require(row["bytes"] == source["bytes"] == FILE_BYTES and row["sha256"] == source["sha256"]
                and row["complete_byte_equality"] is True, "full-byte equality against actual producer")
        if row["stage"] in ("ssd_write_published", "ssd_read_complete"):
            require(row["cqe_nbytes"] == FILE_BYTES, "actual full native CQE")
        else:
            require(row["original_event_query_true"] is True and row["parent_not_published"] is True,
                    "successful original CUDA completion before publication")
    writes = {x["key"]: x for x in stages.get("ssd_write_published", [])}
    reads = {x["key"]: x for x in stages.get("ssd_read_complete", [])}
    if mode == "populate":
        require(set(writes) == set(files), "all producer writes")
        store_jobs = {key: value for key, value in jobs.items() if value["is_store"] is True}
        require(len(store_jobs) == 3, "exact three producer native jobs")
        for jid, job in store_jobs.items():
            require(set(job["files"]) == {str(i) for i in range(8)}, "all8 producer files")
            pages = [job["files"][str(i)] for i in range(8)]
            require(len({x["key"] for x in pages}) == 8 and len({x["block_ids"][0] for x in pages}) == 8,
                    "unique source keys and GPU pages")
            require(all(x in records and x["stage"] == "d2h_complete_source_and_staging"
                        and x["job_id"] == jid and x["request_id"] == job["request_id"] for x in pages), "actual producer job records")
        for row in stages["d2h_complete_source_and_staging"]:
            require(row["sequence"] < writes[row["key"]]["sequence"], "write cannot precede source capture")
    else:
        require(set(reads) == set(files), "all persisted bytes actually read")
        require(all(x["is_store"] is False for x in jobs.values()), "paired no producer jobs")
    consumers = stages["before_model_forward"]
    require(len({x["request_id"] for x in consumers}) == len(consumers), "unique consumer requests")
    load_jobs = {key: value for key, value in jobs.items() if value["is_store"] is False}
    require(len(load_jobs) == len(consumers) and set(load_jobs) == {x["job_id"] for x in consumers}, "all live load jobs reach compute gate")
    consumed_cohorts = {}
    for row in consumers:
        job = load_jobs[row["job_id"]]
        require(job["request_id"] == row["request_id"] and job["native_reactor_id"] == receipt["native_reactor_id"],
                "same live reactor job/request")
        require(set(job["files"]) == {str(i) for i in range(8)}, "all8 files before consumer compute")
        pages = [job["files"][str(i)] for i in range(8)]
        require(all(x in records and x["stage"] == "h2d_complete_dest_and_staging" and x["job_id"] == row["job_id"]
                    and x["request_id"] == row["request_id"] and x["sequence"] < row["sequence"] for x in pages), "native pages before this compute")
        require(row["captured_file_sequences"] == [x["sequence"] for x in pages]
                and row["block_ids"] == [x["block_ids"][0] for x in pages]
                and len(set(row["block_ids"])) == 8, "actual consumer block-table binding")
        require(row["prompt_sha256"] == receipt["producer_prompts"][row["producer_request_id"]]
                and all(files[x["key"]]["request_id"] == row["producer_request_id"] for x in pages), "same logical producer prompt")
        cohort = {key for key, source in files.items() if source["request_id"] == row["producer_request_id"]}
        require(len(cohort) == 8 and {x["key"] for x in pages} == cohort, "each consumer covers exactly its complete producer prefix")
        routes = {x["source_kind"] for x in pages}
        require(routes <= {"file", "shared", "cache"} and ("cache" not in routes or routes == {"cache"}), "uniform cache versus real-read cohort")
        consumed_cohorts.setdefault(row["producer_request_id"], []).append("cache" if routes == {"cache"} else "read")
        for page in pages:
            previous = writes if mode == "populate" else reads
            require(previous[page["key"]]["sequence"] < page["sequence"], "published/read bytes precede target capture")
    require(set(consumed_cohorts) == set(receipt["producer_prompts"]), "all three producer cohorts consumed")
    expected_routes = ["cache"] if mode == "populate" else ["read", "cache"]
    require(all(routes == expected_routes for routes in consumed_cohorts.values()), "each prefix follows its original medium order")
    return True


def install_capture(worker, project, source_refs, output_dir, mode, *, producer_manifest=None):
    state = sys.modules.get("vllm.distributed.kv_transfer.kv_transfer_state")
    require(state is not None, "actual registered connector module")
    connector = state.__dict__.get("_KV_CONNECTOR_AGENT")
    require(connector is not None, "actual registered connector")
    registry = connector.connector_worker.worker
    require(type(registry.handlers) is set and len(registry.handlers) == 1, "sole actual handler")
    handler = next(iter(registry.handlers))
    require(set(registry.transfer_type_to_handler) == {("GPU", "SHARED_STORAGE"), ("SHARED_STORAGE", "GPU")}
            and all(x is handler for x in registry.transfer_type_to_handler.values()), "actual bidirectional registry")
    return CaptureHandle(project, source_refs, output_dir, mode, handler.coordinator.reactor, worker.model_runner,
                         producer_manifest=producer_manifest)
