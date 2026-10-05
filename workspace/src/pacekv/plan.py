"""Frozen bounded R1a diagnostic, explicitly not a native serving benchmark."""
from .exact_key import digest


def pilot_plan(stack="torch27"):
    if stack not in ("torch27", "torch28"):
        raise ValueError("unknown frozen 5090 runtime stack")
    plan = dict(
        protocol=("pacekv-r1a-5090-20260923" if stack == "torch27"
                  else "pacekv-r1a-5090-torch28-20260924"),
        evidence_scope="controlled_hf_kv_io_microprobe",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        revision="c170c708c41dac9275d15a8fff4eca08d52bab71",
        engine="transformers_4.40.2", dtype="bfloat16", attention="sdpa",
        model_loading="cpu_then_cuda_no_accelerate",
        gpu_model="NVIDIA GeForce RTX 5090", compute_capability=[12, 0],
        torch_version=("2.7.1" if stack == "torch27" else "2.8.0"),
        torch_cuda_version="12.8",
        minimum_free_gpu_bytes=24 * 1024**3,
        minimum_host_headroom_bytes=32 * 1024**3,
        prompt_tokens=[512, 2048], output_tokens=64, decode_batch=1,
        arms=["decode_only", "h2d", "d2h", "bidirectional"],
        repetitions=10, warmup_rounds=2, tile_bytes=16 * 1024**2,
        transfer_bytes_per_direction_per_round=256 * 1024**2,
        max_inflight_per_direction=1, seed=20260726,
        teacher_positions=32, logits_relative_l2_max=1e-4,
        kv_bytes_identical_required=True, generated_ids_identical_required=True,
        ssd_test="durable_roundtrip_only_not_cold_read_benchmark",
        native_serving_integration=False, formal_profile=False,
        research_gain_claim_allowed=False, gpu_execution_authorized=False)
    plan["plan_sha256"] = digest(plan)
    return plan


def verify_plan(plan):
    if plan != plan_for_protocol(plan.get("protocol")):
        raise ValueError("plan altered; register a new pilot rather than silently retune")


def plan_for_protocol(protocol):
    plans = (pilot_plan("torch27"), pilot_plan("torch28"))
    for plan in plans:
        if plan["protocol"] == protocol:
            return plan
    raise ValueError("unknown frozen 5090 protocol")


def plan_for_sha(sha):
    for stack in ("torch27", "torch28"):
        plan = pilot_plan(stack)
        if plan["plan_sha256"] == sha:
            return plan
    raise ValueError("unknown frozen 5090 plan SHA")


def arm_order(round_index, arms):
    """Balanced cyclic order, reversed after one full cycle; freeze before results."""
    x = list(arms)
    if round_index % (2 * len(x)) >= len(x):
        x.reverse()
    i = round_index % len(x)
    return x[i:] + x[:i]
