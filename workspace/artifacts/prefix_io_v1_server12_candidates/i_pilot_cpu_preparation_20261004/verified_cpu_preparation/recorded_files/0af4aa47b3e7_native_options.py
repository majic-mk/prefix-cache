"""Project-owned opt-in adapter; these are NOT upstream vLLM config keys."""
from .publication import SnapshotPublisher
KEYS={"prefix_io_observation_mode","prefix_io_observation_run_id","prefix_io_observation_interval_ns"}

def parse_options(extra):
    unknown={k for k in extra if str(k).startswith("prefix_io_")} - KEYS
    if unknown:raise ValueError("unknown project observer keys: "+str(sorted(unknown)))
    mode=extra.get("prefix_io_observation_mode","off")
    if mode not in ("off","shadow"):raise ValueError("P2 only supports off/shadow; ordinary policies remain disabled")
    run_id=extra.get("prefix_io_observation_run_id")
    interval=extra.get("prefix_io_observation_interval_ns",10_000_000)
    if type(interval) is not int or interval<=0:raise ValueError("observer interval must be a positive integer")
    if mode=="shadow" and (type(run_id) is not str or not run_id.strip()):
        raise ValueError("shadow requires a nonempty observation run id")
    if mode=="off" and run_id is not None:
        raise ValueError("off must not configure an observation run id")
    return mode,run_id,interval

def build_observer_kwargs(options):
    mode,run_id,interval=options
    if mode=="off":return {}
    return dict(progress_run_id=run_id,
        observation_sink=SnapshotPublisher(run_id=run_id,interval_ns=interval))
