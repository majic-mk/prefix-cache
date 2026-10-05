"""Strict P3 adapter; common admission and ordinary research policy are separate."""
from dataclasses import dataclass, fields
import json
from pathlib import Path
from .order_options import parse_order_options, build_order_kwargs
from .dispatch_budget import Amount, STAGES
from .stage_accounting import StageAccounting

PARENT_KEY = "prefix_io_parent_admission"
POLICY_KEY = "prefix_io_stage_policy"
CAPS = (1, 2, 4, 8, 16, 32, 64)

@dataclass(frozen=True)
class Options:
    original: object
    run_id: str | None = None
    parent_cap: int | None = None
    policy: object | None = None

def identity(value):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ValueError("bounded run identity required")
    return value

def parse_simple_options(extra):
    original = parse_order_options({k: v for k, v in extra.items()
                                    if k not in (PARENT_KEY, POLICY_KEY)})
    if PARENT_KEY not in extra:
        if POLICY_KEY in extra and extra[POLICY_KEY] != {"mode": "off"}:
            raise ValueError("policy needs the shared native admission bound")
        return Options(original)
    parent = extra[PARENT_KEY]
    if type(parent) is not dict or set(parent) != {"schema_version", "run_id", "max_accepted_parents"}:
        raise ValueError("strict common admission fields required")
    if type(parent["schema_version"]) is not int or parent["schema_version"] != 1:
        raise ValueError("unsupported admission schema")
    cap = parent["max_accepted_parents"]
    if type(cap) is not int or cap not in CAPS:
        raise ValueError("parent bound outside frozen candidates")
    run_id = identity(parent["run_id"])
    if original.start.observation[0] == "shadow" and original.start.observation[1] != run_id:
        raise ValueError("observer/common admission run differs")
    if original.start.mode != "off" or original.mode != "off":
        raise ValueError("do not combine old experimental policies")
    raw = extra.get(POLICY_KEY, {"mode": "off"})
    if raw == {"mode": "off"}:
        return Options(original, run_id, cap)
    if type(raw) is dict and raw.get("mode") == "off":
        raise ValueError("off must have no policy state")
    from .simple_stage_policy import SimpleStageConfig
    required = {f.name for f in fields(SimpleStageConfig)} - {"max_waiting_keys", "max_records"}
    if type(raw) is not dict or set(raw) != required | {"schema_version", "run_id"}:
        raise ValueError("missing or unknown stage policy fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported stage policy schema")
    if raw["run_id"] != run_id or raw["max_accepted_parents"] != cap:
        raise ValueError("stage policy/common admission identities differ")
    values = {k: v for k, v in raw.items() if k in required}
    for key in ("cumulative", "inflight"):
        mapping = values[key]
        if type(mapping) is not dict or set(mapping) != set(STAGES):
            raise ValueError("all four stage amounts required")
        amounts = []
        for stage in STAGES:
            amount = mapping[stage]
            if type(amount) is not dict or set(amount) != {"ops", "bytes"}:
                raise ValueError("strict op/byte amount required")
            amounts.append(Amount(amount["ops"], amount["bytes"]))
        values[key] = tuple(amounts)
    for key in ("shared_ssd_cumulative", "shared_ssd_inflight"):
        amount = values[key]
        if type(amount) is not dict or set(amount) != {"ops", "bytes"}:
            raise ValueError("strict shared SSD op/byte amount required")
        values[key] = Amount(amount["ops"], amount["bytes"])
    return Options(original, run_id, cap, SimpleStageConfig(**values))

def build_simple_kwargs(options):
    if type(options) is not Options:
        raise TypeError("parsed immutable options required")
    result = build_order_kwargs(options.original)
    if options.parent_cap is None:
        return result
    result.update(progress_run_id=options.run_id,
                  max_accepted_parents=options.parent_cap,
                  stage_accounting=StageAccounting())
    if options.policy is not None:
        from .simple_stage_policy import DispatchController
        result["dispatch_controller"] = DispatchController(options.run_id, options.policy)
    return result

def model_file(path, run_id):
    """Validate CPU-only, before storage/GPU work; inject only this run identity."""
    raw = json.loads(Path(path).read_text())
    if type(raw) is not dict or set(raw) != {"schema_version", "parent_admission", "stage_policy"}:
        raise ValueError("strict model control file required")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("unsupported model control schema")
    parent = raw["parent_admission"]
    policy = raw["stage_policy"]
    if type(parent) is not dict or set(parent) != {"schema_version", "max_accepted_parents"}:
        raise ValueError("model common admission must omit mutable runtime identity")
    if type(policy) is not dict or "run_id" in policy:
        raise ValueError("model stage policy must omit runtime identity")
    extra = {PARENT_KEY: dict(parent, run_id=identity(run_id)),
             POLICY_KEY: dict(policy) if policy == {"mode": "off"} else dict(policy, run_id=run_id)}
    parse_simple_options(extra)
    return extra
