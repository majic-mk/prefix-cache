"""Strict isolated P4 startup; prepared CPU evidence cannot enable production."""
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
    prepared_cost_request: object | None = None

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
    from .p4_types import P4Config
    batches = raw["candidate_batches"]
    if type(batches) is not list:
        raise ValueError("explicit finite JSON batch list required")
    config = P4Config(raw["mode"], raw["sample_max_age_ns"], raw["max_wait_ns"],
                      raw["internal_step_budget_ns"], tuple(batches))
    if config.mode == "off":
        raise ValueError("off must not carry policy state")
    fixed = None
    fixed_raw = raw["fixed_stage_policy"]
    if fixed_raw is not None:
        if type(fixed_raw) is not dict or fixed_raw.get("mode") != "fixed":
            raise ValueError("only frozen original fixed stage budget can be declared")
        fixed = parse_simple_options(dict(base, **{POLICY_KEY:fixed_raw}))
        if fixed.policy is None or fixed.run_id != raw["run_id"]:
            raise ValueError("same-run fixed stage configuration required")
    if config.mode == "dependency_only" and fixed is None:
        raise ValueError("dependency-only requires frozen original fixed stage budget")
    request = None
    if raw["cost_table"] is not None:
        if config.mode not in ("shadow","interference","joint"):
            raise ValueError("D cannot consume an interference-table input")
        from .p4_startup_evidence import parse_prepared_cost_request
        request=parse_prepared_cost_request(raw["cost_table"])
        if request.expected_context.run_id != raw["run_id"]:
            raise ValueError("paired measurement startup context run differs")
    return Options(original, raw["run_id"], config, fixed, request)

def build_p4_kwargs(options, *, evidence_root=None):
    if type(options) is not Options:
        raise TypeError("parsed immutable P4 options required")
    prepared = None
    if options.prepared_cost_request is not None:
        if evidence_root is None:
            raise ValueError("independently bound project evidence root required")
        prepared=options.prepared_cost_request.load(evidence_root)
        if prepared.production_qualified:
            raise ValueError("GPU-qualified activation is not implemented by the CPU prepared loader")
    # An explicit fixed common base is usable for D/shadow. I/J without real
    # qualification still use U; declaring caps is not permission to apply them.
    fixed_applies=(options.config is not None and
        options.config.mode in ("dependency_only","shadow") and options.fixed is not None)
    result = build_simple_kwargs(options.fixed if fixed_applies else options.original)
    if options.config is not None:
        from .p4_bridge import make_native_bridge
        bridge=make_native_bridge(options.run_id,options.config)
        bridge.prepared_cost_summary = ((("status","NO_PREPARED_TABLE"),) if prepared is None else
            (("status",prepared.status),("candidate_sha256",prepared.source_ref.sha256),
             ("cell_count",prepared.cell_count),("gpu_verified",False)))
        bridge.declared_common_fixed_base = options.fixed is not None
        result["p4_bridge"] = bridge
    return result
