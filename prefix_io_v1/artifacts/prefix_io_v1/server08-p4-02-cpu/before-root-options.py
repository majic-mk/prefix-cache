"""Strict isolated P4 option. It does not change P3 source or acceptance rules."""
from dataclasses import dataclass
from .simple_stage_options import (
    PARENT_KEY, POLICY_KEY, parse_simple_options, build_simple_kwargs, identity,
)

P4_KEY = "prefix_io_p4_policy"

@dataclass(frozen=True)
class Options:
    original: object
    run_id: str | None = None
    config: object | None = None
    fixed: object | None = None

def parse_p4_options(extra):
    if type(extra) is not dict:
        raise TypeError("connector options must be an exact mapping")
    if P4_KEY not in extra:
        return Options(parse_simple_options(extra))
    if POLICY_KEY in extra:
        raise ValueError("P4 option and P3 policy option are mutually exclusive")
    base = {k: v for k, v in extra.items() if k != P4_KEY}
    original = parse_simple_options(base)
    raw = extra[P4_KEY]
    if raw == {"mode": "off"} and type(raw) is dict:
        return Options(original)
    keys = {"schema_version", "run_id", "mode", "sample_max_age_ns", "max_wait_ns",
            "internal_step_budget_ns", "candidate_batches", "fixed_stage_policy", "cost_table"}
    if type(raw) is not dict or set(raw) != keys:
        raise ValueError("strict active P4 fields required")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported P4 schema")
    if original.parent_cap is None or original.run_id != identity(raw["run_id"]):
        raise ValueError("P4 requires same-run common admission/accounting")
    if raw["cost_table"] is not None:
        raise ValueError("no P4 production cost table has a GPU qualification gate")
    from .p4_types import P4Config
    batches = raw["candidate_batches"]
    if type(batches) is not list:
        raise ValueError("explicit finite JSON batch list required")
    config = P4Config(raw["mode"], raw["sample_max_age_ns"], raw["max_wait_ns"],
                      raw["internal_step_budget_ns"], tuple(batches))
    if config.mode == "off":
        raise ValueError("off must not carry policy state")
    fixed = None
    if config.mode == "dependency_only":
        fixed_raw = raw["fixed_stage_policy"]
        if type(fixed_raw) is not dict or fixed_raw.get("mode") != "fixed":
            raise ValueError("dependency-only requires frozen original fixed stage budget")
        fixed = parse_simple_options(dict(base, **{POLICY_KEY:fixed_raw}))
        if fixed.policy is None:
            raise ValueError("fixed stage configuration missing")
    elif raw["fixed_stage_policy"] is not None:
        raise ValueError("shadow/unsupported I/J must retain baseline U allowances")
    return Options(original, raw["run_id"], config, fixed)

def build_p4_kwargs(options):
    if type(options) is not Options:
        raise TypeError("parsed immutable P4 options required")
    result = build_simple_kwargs(options.fixed or options.original)
    if options.config is not None:
        from .p4_bridge import make_native_bridge
        result["p4_bridge"] = make_native_bridge(options.run_id, options.config)
    return result
