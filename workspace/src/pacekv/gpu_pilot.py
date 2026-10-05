"""Real model + controlled KV DMA microprobe; NOT integrated offload serving.

All torch/transformers imports are lazy. Copy destinations are disjoint from the
active model cache. This can demonstrate interference, not causal serving gain.
"""
import hashlib
import json
from pathlib import Path
import time
import subprocess

from .audit import file_sha, host_available_bytes, write_new_json
from .evidence import quantile
from .ownership import OwnershipLedger
from .plan import arm_order, verify_plan


def legacy_cache(cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    return tuple(tuple(t.detach().contiguous() for t in layer) for layer in cache)


def cache_digest(cache):
    """Qualification only; full D2H/hash is never inside timed decode."""
    h = hashlib.sha256()
    for layer in cache:
        for t in layer:
            h.update(str((tuple(t.shape), str(t.dtype))).encode())
            h.update(t.detach().contiguous().cpu().view(__import__("torch").uint8).numpy().tobytes())
    return h.hexdigest()


def kv_bytes(cache):
    return sum(t.numel() * t.element_size() for layer in cache for t in layer)


def ledger_direction(direction):
    """Map probe labels to the ownership ledger's canonical transfer direction."""
    if direction not in ("h2d", "d2h"):
        raise ValueError("unknown DMA probe direction")
    return direction.upper()


def transfer_cache(torch, cache, device, *, pinned=False):
    copied = []
    stream = torch.cuda.Stream()
    start = time.perf_counter_ns()
    with torch.cuda.stream(stream):
        for layer in cache:
            dest = []
            for t in layer:
                v = torch.empty(t.shape, dtype=t.dtype, device=device,
                                pin_memory=pinned if device == "cpu" else False)
                v.copy_(t, non_blocking=True)
                dest.append(v)
            copied.append(tuple(dest))
        event = torch.cuda.Event()
        event.record(stream)
    event.synchronize()
    return tuple(copied), (time.perf_counter_ns() - start) / 1e6


class DmaProbe:
    """One outstanding tile per direction; uses captured KV, not random payload.

    H2D/D2H here represent CONTROLLED pressure, not a real second request. Device
    source copies, pinned buffers, and destination storage remain owned until
    completion. No per-step CUDA global synchronize is inserted.
    """
    def __init__(self, torch, cache, plan):
        self.torch = torch
        self.plan = plan
        flattened = [t.reshape(-1).view(torch.uint8) for layer in cache for t in layer]
        # Outside measurement: packed immutable mirror of actual captured KV.
        self.source = torch.cat(flattened)
        self.tile = min(plan["tile_bytes"], self.source.numel())
        self.host_read = torch.empty(self.tile, dtype=torch.uint8, pin_memory=True)
        self.host_read.copy_(self.source[:self.tile].cpu())
        self.host_write = torch.empty_like(self.host_read, pin_memory=True)
        self.destination = torch.empty_like(self.source[:self.tile])
        self.streams = {d: torch.cuda.Stream() for d in ("h2d", "d2h")}
        self.pending = {}
        self.events = []
        self.total = {"h2d": 0, "d2h": 0}
        self.ledger = OwnershipLedger(self.tile * 2, self.tile * 2)
        self.sequence = 0

    def poll_and_issue(self, arm, origin_ns):
        now = time.perf_counter_ns()
        for direction, row in list(self.pending.items()):
            if row["end"].query():
                row["completion_observed_ns"] = now - origin_ns
                self.ledger.dma_complete(row["ticket_id"], 0, now)
                self.events.append(row)
                del self.pending[direction]
        for direction in ("h2d", "d2h"):
            if arm not in (direction, "bidirectional") or direction in self.pending:
                continue
            left = self.plan["transfer_bytes_per_direction_per_round"] - self.total[direction]
            if left <= 0:
                continue
            count = min(self.tile, left)
            stream = self.streams[direction]
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            issue = time.perf_counter_ns() - origin_ns
            ticket_id = "%s-%s" % (direction, self.sequence)
            self.sequence += 1
            stamp = time.perf_counter_ns()
            if not self.ledger.reserve(ticket_id, ledger_direction(direction), 0, count, count, True,
                                       stamp, self.ledger.epoch):
                raise RuntimeError("physical probe capacity and ownership ledger disagree")
            self.ledger.issue(ticket_id, 0, time.perf_counter_ns())
            with self.torch.cuda.stream(stream):
                start.record(stream)
                if direction == "h2d":
                    self.destination[:count].copy_(self.host_read[:count], non_blocking=True)
                else:
                    self.host_write[:count].copy_(self.source[:count], non_blocking=True)
                end.record(stream)
            self.total[direction] += count
            self.pending[direction] = dict(direction=direction, bytes=count,
                                           ticket_id=ticket_id, issued_ns=issue, start=start, end=end)

    def drain(self, origin_ns):
        for row in self.pending.values():
            row["end"].synchronize()
            row["completion_observed_ns"] = time.perf_counter_ns() - origin_ns
            self.ledger.dma_complete(row["ticket_id"], 0, time.perf_counter_ns())
            self.events.append(row)
        self.pending.clear()
        rows = []
        for event in self.events:
            row = {k: v for k, v in event.items() if k not in ("start", "end")}
            row["cuda_copy_ms"] = event["start"].elapsed_time(event["end"])
            # Upper bound observed at polling/drain, NOT exact allocator lifetime.
            row["observed_retention_byte_ns_upper"] = row["bytes"] * (
                row["completion_observed_ns"] - row["issued_ns"])
            rows.append(row)
        return rows


def decode(torch, model, prompt, prefix_cache, first_logits, steps, *, probe=None,
           arm="decode_only", collect_logits=False):
    past = prefix_cache
    logits = first_logits
    ids, arrivals, vectors = [], [], []
    torch.cuda.synchronize()
    origin = time.perf_counter_ns()
    try:
        for step in range(steps):
            if probe:
                probe.poll_and_issue(arm, origin)
            token = logits.argmax(-1)
            ids.append(int(token.item()))  # Host-visible token, same endpoint in every arm.
            arrivals.append(time.perf_counter_ns() - origin)
            if collect_logits:
                vectors.append(logits.detach().float().cpu())
            if step + 1 < steps:
                output = model(input_ids=token.reshape(1, 1), past_key_values=past,
                               use_cache=True, return_dict=True)
                past = legacy_cache(output.past_key_values)
                logits = output.logits[:, -1]
        end = time.perf_counter_ns() - origin
    finally:
        transfers = probe.drain(origin) if probe else []
    drained = time.perf_counter_ns() - origin
    return dict(token_ids=ids, token_visible_ns=arrivals, decode_ns=end,
                total_with_drain_ns=drained, transfer_events=transfers,
                timing_scope="prefix_already_ready_to_last_token_plus_separate_drain",
                full_request_ttft_measured=False), vectors


def compare_logits(torch, reference, candidate):
    if len(reference) != len(candidate) or not reference:
        raise ValueError("missing teacher positions")
    values = []
    for a, b in zip(reference, candidate):
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite logits")
        values.append(float(torch.linalg.vector_norm(a - b) /
                            torch.linalg.vector_norm(a).clamp_min(1e-12)))
    return values


def run(plan, audit, output):
    verify_plan(plan)
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device; no GPU work performed")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported")
    gpu_name = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if (gpu_name != plan["gpu_model"] or capability != tuple(plan["compute_capability"])
            or torch.version.cuda != plan["torch_cuda_version"]
            or torch.__version__ != plan["torch_version"] + "+cu128"):
        raise RuntimeError("RTX 5090 pilot requires SM120 and Torch %s+cu128; do not reuse A800 environment" % plan["torch_version"])
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    if any(s.strip().isdigit() and int(s.strip()) != os.getpid() for s in processes.splitlines()):
        raise RuntimeError("another GPU process is active; do not disturb its experiment")
    gpu_identity = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,uuid,driver_version,pci.bus_id,memory.total",
        "--format=csv,noheader"], text=True).strip()
    if torch.cuda.device_count() != 1:
        raise RuntimeError("pilot requires one explicitly isolated GPU")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < plan["minimum_free_gpu_bytes"]:
        raise RuntimeError("insufficient free GPU memory; do not silently shrink task")
    host_free = host_available_bytes()
    if host_free is None or host_free < plan["minimum_host_headroom_bytes"]:
        raise RuntimeError("insufficient host/cgroup headroom for CPU-first model load")
    out = Path(output)
    torch.manual_seed(plan["seed"])
    runtime = dict(gpu=gpu_name, compute_capability=list(capability), torch=torch.__version__,
                   cuda=torch.version.cuda, code_digest=audit["code_digest"],
                   plan_sha256=plan["plan_sha256"], evidence_scope=plan["evidence_scope"],
                   gpu_properties=str(torch.cuda.get_device_properties(0)),
                   nvidia_smi_identity=gpu_identity,
                   imports=dict(torch=torch.__file__))
    write_new_json(out / "runtime.json", runtime)
    tokenizer = AutoTokenizer.from_pretrained(audit["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        audit["model_path"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=False,
        attn_implementation=plan["attention"], local_files_only=True).eval().to("cuda:0")
    # Controlled shape workload, explicitly NOT natural QA or a serving trace.
    text = "A document describes a river, a library, and a bicycle. Read it carefully. "
    unit = tokenizer.encode(text, add_special_tokens=False)
    if not unit:
        raise RuntimeError("empty diagnostic prompt")
    correctness, all_rows = [], []
    with torch.inference_mode():
        for n in plan["prompt_tokens"]:
            token_ids = (unit * (n // len(unit) + 1))[:n]
            write_new_json(out / ("input-%s.json" % n), dict(token_ids=token_ids,
                           input_origin="constructed_shape_microprobe_not_QA",
                           full_prefix_identity=True, natural_workload=False))
            prompt = torch.tensor([token_ids], device="cuda")
            begin = time.perf_counter_ns()
            capture = model(input_ids=prompt, use_cache=True, return_dict=True)
            cache = legacy_cache(capture.past_key_values)
            first_logits = capture.logits[:, -1].detach().clone()
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter_ns() - begin) / 1e6
            del capture
            qualification_start = time.perf_counter_ns()
            original = cache_digest(cache)
            reference, ref_logits = decode(torch, model, prompt, cache, first_logits,
                                           plan["teacher_positions"], collect_logits=True)
            host, d2h = transfer_cache(torch, cache, "cpu", pinned=True)
            variants = [("cpu_pinned", host)]
            disk_path = out / ("roundtrip-%s.pt" % n)
            write_start = time.perf_counter_ns()
            with disk_path.open("xb") as f:
                torch.save(host, f)
                f.flush()
                os.fsync(f.fileno())
            write_ms = (time.perf_counter_ns() - write_start) / 1e6
            read_start = time.perf_counter_ns()
            # Local self-created tensor-only file, not an untrusted checkpoint.
            loaded = torch.load(disk_path, map_location="cpu", weights_only=True)
            disk_host = tuple(tuple(t.pin_memory() for t in layer) for layer in loaded)
            read_staging_ms = (time.perf_counter_ns() - read_start) / 1e6
            variants.append(("ssd_staged_page_cache_uncontrolled", disk_host))
            for name, payload in variants:
                restored, h2d = transfer_cache(torch, payload, "cuda")
                output_row, candidate_logits = decode(torch, model, prompt, restored,
                    first_logits, plan["teacher_positions"], collect_logits=True)
                l2 = compare_logits(torch, ref_logits, candidate_logits)
                row = dict(tokens=n, path=name, kv_bytes=kv_bytes(cache),
                           source_before=original, host_digest=cache_digest(payload),
                           destination_digest=cache_digest(restored), source_after=cache_digest(cache),
                           token_ids_reference=reference["token_ids"], token_ids=output_row["token_ids"],
                           logits_relative_l2_by_position=l2, capture_ms=capture_ms,
                           d2h_ms=d2h, h2d_ms=h2d, disk_write_fsync_ms=write_ms,
                           disk_read_and_pin_ms=read_staging_ms, disk_sha256=file_sha(disk_path))
                row["passed"] = (len(set(row[k] for k in ("source_before", "host_digest", "destination_digest", "source_after"))) == 1
                                 and row["token_ids"] == row["token_ids_reference"]
                                 and max(l2) <= plan["logits_relative_l2_max"])
                torch.save(dict(reference=ref_logits, candidate=candidate_logits),
                           out / ("logits-%s-%s.pt" % (n, name)))
                correctness.append(row)
                write_new_json(out / ("correctness-%s-%s.json" % (n, name)), row)
                del restored
                if not row["passed"]:
                    raise RuntimeError("exact KV roundtrip correctness failed")
            write_new_json(out / ("qualification-cost-%s.json" % n), dict(
                qualification_total_ms=(time.perf_counter_ns() - qualification_start) / 1e6,
                hash_and_evidence_included=True, excluded_from_decode_measurements=True))
            del variants, host, disk_host, loaded, payload
            expected_ids = None
            for round_index in range(plan["warmup_rounds"] + plan["repetitions"]):
                for arm in arm_order(round_index, plan["arms"]):
                    allocation_start = time.perf_counter_ns()
                    probe = DmaProbe(torch, cache, plan)  # Same allocation in ALL arms.
                    torch.cuda.synchronize()
                    setup_ms = (time.perf_counter_ns() - allocation_start) / 1e6
                    row, _ = decode(torch, model, prompt, cache, first_logits,
                                    plan["output_tokens"], probe=probe, arm=arm)
                    if expected_ids is None:
                        expected_ids = row["token_ids"]
                    row.update(prompt_tokens=n, arm=arm, round_index=round_index,
                               warmup=round_index < plan["warmup_rounds"], setup_ms=setup_ms,
                               ids_equal_reference=row["token_ids"] == expected_ids,
                               controlled_pressure=True, native_serving=False)
                    itl = [(b - a) / 1e6 for a, b in zip(row["token_visible_ns"], row["token_visible_ns"][1:])]
                    row["p95_itl_ms"] = quantile(itl, .95)
                    row["transfer_bytes"] = dict(probe.total)
                    row["dma_ownership"] = probe.ledger.snapshot(time.perf_counter_ns())
                    row["persistent_probe_allocation_bytes"] = dict(
                        gpu=probe.source.numel() + probe.destination.numel(),
                        pinned=probe.host_read.numel() + probe.host_write.numel())
                    all_rows.append(row)
                    write_new_json(out / ("decode-%s-%02d-%s.json" % (n, round_index, arm)), row)
                    del probe
                    if not row["ids_equal_reference"]:
                        raise RuntimeError("controlled copy changed generated token IDs")
            if cache_digest(cache) != original:
                raise RuntimeError("active cache mutated")
            del cache, first_logits
    write_new_json(out / "summary.json", dict(
        gpu_transport_correctness_passed=all(r["passed"] for r in correctness),
        controlled_decode_interference_measured=True, measurements=len(all_rows),
        native_serving_integrated=False, exact_cache_hit_validated_in_serving=False,
        production_writeback_debt_measured=False, positive_system_gain_validated=False,
        formal_profile_frozen=False, paper_evidence=False,
        note="Analyze paired raw rows; this test cannot establish production goodput or novelty."))
    files = {p.name: file_sha(p) for p in sorted(out.iterdir())
             if p.is_file() and p.suffix in (".json", ".pt")}
    write_new_json(out / "evidence_manifest.json", dict(files=files,
                   code_digest=audit["code_digest"], plan_sha256=plan["plan_sha256"]))
