"""P3 store-order adapter; original start/observation parsers remain unchanged."""
from dataclasses import dataclass
from .start_options import parse_start_options,build_start_kwargs,KEY as START_KEY
from .store_order import MandatoryStoreOrder

KEY="prefix_io_store_order"

@dataclass(frozen=True)
class Options:
    start: object
    mode: str="off"
    run_id: str|None=None

def parse_order_options(extra):
    base={k:v for k,v in extra.items() if k!=KEY}
    if KEY not in extra:
        return Options(parse_start_options(base))
    raw=extra[KEY]
    if type(raw) is not dict:
        raise ValueError("store-order configuration must be a mapping")
    if raw=={"mode":"off"}:
        return Options(parse_start_options(base))
    if set(raw)!={"schema_version","mode","run_id","candidate_parents"}:
        raise ValueError("missing or unknown store-order fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"]!=1:
        raise ValueError("unsupported store-order schema")
    if raw["mode"]!="pressure":
        raise ValueError("P3 store order only supports off/pressure")
    if type(raw["candidate_parents"]) is not int or raw["candidate_parents"]!=32:
        raise ValueError("candidate window is frozen at 32")
    run_id=raw["run_id"]
    if type(run_id) is not str or not run_id.strip() or len(run_id)>128:
        raise ValueError("bounded nonempty run identity required")
    start=parse_start_options(base)
    if start.mode!="off":
        raise ValueError("do not combine ordering with experimental start allowances")
    if start.observation[0]=="shadow" and start.observation[1]!=run_id:
        raise ValueError("observer and order run identity must match")
    return Options(start,"pressure",run_id)

def build_order_kwargs(options):
    if type(options) is not Options:
        raise TypeError("parsed immutable order options required")
    kwargs=build_start_kwargs(options.start)
    if options.mode=="off":
        return kwargs
    kwargs.update(progress_run_id=options.run_id,store_order=MandatoryStoreOrder())
    return kwargs
