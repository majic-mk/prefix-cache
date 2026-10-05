"""Strict opt-in P3 experimental start-budget adapter, separate from P2 options."""
from dataclasses import dataclass
from .native_options import parse_options,build_observer_kwargs
from .start_budget import StartBudgetConfig,StartBudget
from .stage_accounting import StageAccounting
KEY="prefix_io_start_budget"
EPOCHS=(1_000_000,10_000_000,100_000_000,1_000_000_000)

@dataclass(frozen=True)
class Options:
    observation:tuple
    mode:str
    run_id:str|None=None
    budget:StartBudgetConfig|None=None

def parse_start_options(extra):
    observation=parse_options({k:v for k,v in extra.items() if k!=KEY})
    if KEY not in extra:return Options(observation,"off")
    raw=extra[KEY]
    if type(raw) is not dict:raise ValueError("start-budget options must be a mapping")
    mode=raw.get("mode")
    if mode=="off":
        if raw!={"mode":"off"}:raise ValueError("off must have no start-budget state")
        return Options(observation,"off")
    if mode not in ("shadow","fixed","pressure"):
        raise ValueError("P3 entry only supports off/shadow/fixed/pressure starts")
    required={"schema_version","mode","run_id"}
    if mode in ("fixed","pressure"):required|={"epoch_ns","starts_per_epoch","max_wait_ns","reserve_free_slots"}
    if set(raw)!=required:raise ValueError("missing or unknown start-budget fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"]!=1:
        raise ValueError("unsupported start-budget schema")
    run_id=raw["run_id"]
    if type(run_id) is not str or not run_id.strip() or len(run_id)>128:
        raise ValueError("bounded nonempty run_id required")
    if observation[0]=="shadow" and observation[1]!=run_id:
        raise ValueError("observer and start-budget run IDs must match")
    config=None
    if mode in ("fixed","pressure"):
        if type(raw["epoch_ns"]) is not int or raw["epoch_ns"] not in EPOCHS:
            raise ValueError("epoch outside frozen finite candidates")
        if type(raw["max_wait_ns"]) is not int or raw["max_wait_ns"]!=10*raw["epoch_ns"]:
            raise ValueError("max wait must equal ten epochs")
        config=StartBudgetConfig(mode,raw["epoch_ns"],raw["starts_per_epoch"],
            raw["max_wait_ns"],raw["reserve_free_slots"])
        if config.reserve_free_slots not in (0,1,2,4):raise ValueError("reserve outside finite candidates")
    return Options(observation,mode,run_id,config)

def build_start_kwargs(options):
    if not isinstance(options,Options):raise TypeError("parsed immutable options required")
    kwargs=build_observer_kwargs(options.observation)
    if options.mode=="off":return kwargs
    kwargs.update(progress_run_id=options.run_id,stage_accounting=StageAccounting())
    if options.budget is not None:kwargs["start_budget"]=StartBudget(options.budget)
    return kwargs
