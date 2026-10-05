"""Honest token-level metrics; missing ITL or failed requests never become passes."""
import math


def quantile(values, p):
    if not values or not 0 <= p <= 1 or any(not math.isfinite(v) for v in values):
        raise ValueError("invalid quantile data")
    x = sorted(values)
    at = (len(x) - 1) * p
    lo = int(at)
    hi = min(lo + 1, len(x) - 1)
    return x[lo] + (at - lo) * (x[hi] - x[lo])


def token_metrics(arrival_ns, token_ns, *, completed, ttft_slo_ms, itl_slo_ms):
    if any(not math.isfinite(x) or x <= 0 for x in (ttft_slo_ms, itl_slo_ms)):
        raise ValueError("explicit positive SLO required")
    if type(arrival_ns) is not int or arrival_ns < 0 or type(completed) is not bool:
        raise ValueError("invalid request metadata")
    last = arrival_ns
    for t in token_ns:
        if type(t) is not int or t < last:
            raise ValueError("invalid token arrival order")
        last = t
    ttft = (token_ns[0] - arrival_ns) / 1e6 if token_ns else None
    itls = [(b - a) / 1e6 for a, b in zip(token_ns, token_ns[1:])]
    p95 = quantile(itls, .95) if itls else None
    return dict(ttft_ms=ttft, itl_ms=itls, p95_itl_ms=p95,
                completed=completed, dual_slo_met=bool(completed and ttft is not None
                and p95 is not None and ttft <= ttft_slo_ms and p95 <= itl_slo_ms))


def dual_slo_goodput(requests, trace_elapsed_ns):
    if type(trace_elapsed_ns) is not int or trace_elapsed_ns <= 0 or not requests:
        raise ValueError("complete trace extent required")
    if any(type(r.get("dual_slo_met")) is not bool for r in requests):
        raise ValueError("missing outcomes")
    return sum(r["dual_slo_met"] for r in requests) / (trace_elapsed_ns / 1e9)
