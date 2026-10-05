"""Paired raw-file analysis for the controlled pilot, never a serving GO gate."""
import json
from pathlib import Path
import random
import statistics
import math

from .audit import file_sha
from .evidence import quantile
from .plan import arm_order, plan_for_sha


def paired_interval(deltas, seed=20260726, draws=2000):
    if len(deltas) < 2:
        raise ValueError("insufficient paired observations")
    rng = random.Random(seed)
    means = [statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(draws)]
    return dict(mean=statistics.mean(deltas), lower=quantile(means, .025),
                upper=quantile(means, .975), resampling_unit="complete_round",
                interpretation="within-process interval; independent-process replication pending")


def analyze(output):
    root = Path(output)
    if (root / "failed.json").exists():
        raise ValueError("failed or partial pilot cannot be aggregated as complete")
    manifest = json.loads((root / "evidence_manifest.json").read_text())
    plan = plan_for_sha(manifest.get("plan_sha256"))
    files = manifest["files"]
    expected = {"runtime.json", "summary.json", "launch.json"}
    for n in plan["prompt_tokens"]:
        for path in ("cpu_pinned", "ssd_staged_page_cache_uncontrolled"):
            expected.add("correctness-%s-%s.json" % (n, path))
            expected.add("logits-%s-%s.pt" % (n, path))
        expected.update({"input-%s.json" % n, "roundtrip-%s.pt" % n,
                         "qualification-cost-%s.json" % n})
        for r in range(plan["warmup_rounds"] + plan["repetitions"]):
            for arm in arm_order(r, plan["arms"]):
                expected.add("decode-%s-%02d-%s.json" % (n, r, arm))
    if not expected <= set(files):
        raise ValueError("missing required raw evidence")
    for name, sha in files.items():
        if Path(name).name != name or file_sha(root / name) != sha:
            raise ValueError("invalid path or evidence digest: " + name)
    runtime = json.loads((root / "runtime.json").read_text())
    if runtime.get("code_digest") != manifest.get("code_digest"):
        raise ValueError("runtime and manifest code mismatch")
    comparisons = []
    for n in plan["prompt_tokens"]:
        for path in ("cpu_pinned", "ssd_staged_page_cache_uncontrolled"):
            row = json.loads((root / ("correctness-%s-%s.json" % (n, path))).read_text())
            checksums = [row[k] for k in ("source_before", "host_digest", "destination_digest", "source_after")]
            l2 = row["logits_relative_l2_by_position"]
            if (len(set(checksums)) != 1 or row["token_ids"] != row["token_ids_reference"]
                or len(row["token_ids"]) != plan["teacher_positions"] or len(l2) != plan["teacher_positions"]):
                raise ValueError("incomplete/failed correctness")
            if any(not math.isfinite(v) or v < 0 or v > plan["logits_relative_l2_max"] for v in l2):
                raise ValueError("failed logits")
            import torch
            from .gpu_pilot import compare_logits
            raw = torch.load(root/("logits-%s-%s.pt" % (n, path)),
                             map_location="cpu", weights_only=True)
            recomputed = compare_logits(torch, raw["reference"], raw["candidate"])
            if len(recomputed) != plan["teacher_positions"] or any(
                    abs(a-b) > 1e-10 for a, b in zip(recomputed, l2)):
                raise ValueError("logit report does not match raw tensor evidence")
        measurements = {}
        expected_ids = None
        for r in range(plan["warmup_rounds"] + plan["repetitions"]):
            for arm in plan["arms"]:
                row = json.loads((root / ("decode-%s-%02d-%s.json" % (n, r, arm))).read_text())
                stamps = row["token_visible_ns"]
                if (len(stamps) != plan["output_tokens"] or any(b < a for a, b in zip(stamps, stamps[1:]))
                    or row.get("round_index") != r or row.get("arm") != arm or row.get("prompt_tokens") != n
                    or not row["ids_equal_reference"] or row["dma_ownership"]["outstanding"] != 0):
                    raise ValueError("invalid raw decode data")
                if expected_ids is None:
                    expected_ids = row["token_ids"]
                if len(expected_ids) != plan["output_tokens"] or row["token_ids"] != expected_ids:
                    raise ValueError("actual generated tokens disagree")
                for direction in ("h2d", "d2h"):
                    expected_bytes = (plan["transfer_bytes_per_direction_per_round"]
                                      if arm in (direction, "bidirectional") else 0)
                    actual_bytes = sum(e["bytes"] for e in row["transfer_events"] if e["direction"] == direction)
                    if actual_bytes != expected_bytes or row["transfer_bytes"][direction] != actual_bytes:
                        raise ValueError("incomplete or inconsistent controlled I/O exposure")
                for e in row["transfer_events"]:
                    if (e["completion_observed_ns"] < e["issued_ns"] or
                        not math.isfinite(e["cuda_copy_ms"]) or e["cuda_copy_ms"] < 0):
                        raise ValueError("invalid transfer event")
                if row["total_with_drain_ns"] < row["decode_ns"] or stamps[-1] > row["decode_ns"]:
                    raise ValueError("nonclosing time scope")
                itls = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
                measurements[r, arm] = quantile(itls, .95)
        for arm in plan["arms"][1:]:
            deltas = [measurements[r, arm] - measurements[r, "decode_only"]
                      for r in range(plan["warmup_rounds"], plan["warmup_rounds"] + plan["repetitions"])]
            comparisons.append(dict(prompt_tokens=n, arm=arm, p95_itl_difference_ms=paired_interval(deltas)))
    return dict(comparisons=comparisons, evidence_scope=plan["evidence_scope"],
                production_serving_go=None, production_writeback_debt_validated=False,
                dual_slo_goodput_measured=False, no_sci_publication_guarantee=True)
