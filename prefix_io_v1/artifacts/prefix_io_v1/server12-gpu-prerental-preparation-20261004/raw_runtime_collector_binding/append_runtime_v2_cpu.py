"""Append strict pre-model collector source selection; no runtime is executed."""
import hashlib
from pathlib import Path

PREP=Path(__file__).resolve().parent.parent
native=PREP/"runner/native_runtime.py"
trace=PREP/"runner/strong_trace_runner.py"
assert hashlib.sha256(native.read_bytes()).hexdigest()=="a293f98841e34bb9357e8dacb62cda43be7dfc7da66167f3df5577c4b5b11709"
assert hashlib.sha256(trace.read_bytes()).hexdigest()=="239771d85459b795d82624e23aa96510954ce301b2fc6554b2e575057edb9ce9"
helper='''def preflight_collector_binding(root, gates, *, driver):
    """Source binding only: cannot issue a table or replace the native executor.

    The existing activation verifier owns the full calibration ancestry/deadline
    gate. This repeats exact collector/delegate bytes before model imports.
    """
    config, refs = gates["config"], gates["refs"]
    directory = Path(config["runtime_ref"]["path"]).parent
    legacy = (directory / "bounded_native_full_step_collector.py").as_posix()
    if config["phase"] not in ("development", "effect"):
        driver.require(legacy in refs, "unchanged native off/shadow collector source frozen")
        driver.check_ref(root, refs[legacy])
        return refs[legacy]
    activation = gates.get("finite_activation")
    driver.require(type(activation) is dict and type(activation.get("descriptor")) is dict and
                   type(activation.get("calibration_plan")) is dict,
                   "closed finite calibration source gate before framework/model")
    descriptor, plan = activation["descriptor"], activation["calibration_plan"]
    selected = descriptor.get("collector_source_ref", descriptor.get("collector_ref"))
    driver.require(type(selected) is dict and descriptor.get("collector_ref") == selected and
                   ("collector_source_ref" not in descriptor or descriptor["collector_source_ref"] == selected),
                   "one exact closed startup collector reference")
    expected = (directory / "bounded_native_full_step_collector_v2.py").as_posix()
    driver.require(selected.get("path") == expected and
                   selected.get("sha256") == "a710f3f55ecbc6d8fbce8f3cf85ce4587daf0e1a863783df25a4ba8bc37e2aa0" and
                   selected.get("bytes") == 13016 and refs.get(expected) == selected and
                   activation.get("runtime_refs", {}).get(expected) == selected,
                   "exact actual V2 collector in current runtime closure")
    driver.check_ref(root, selected)
    driver.require(plan.get("collector_source_ref") == selected and
                   plan.get("source_lock_ref") == descriptor.get("calibration_source_lock_ref") and
                   plan.get("gpu_uuid") == config["gpu_uuid"] and
                   plan.get("common_runtime_domain_sha256") == gates["common_runtime_domain_sha256"],
                   "same calibration collector/device/common domain")
    declared_plan = driver.read(driver.check_ref(root, descriptor["plan_ref"]))
    driver.require(declared_plan == plan, "actual frozen calibration plan bytes before model")
    original = plan.get("original_collector_source_ref")
    driver.require(type(original) is dict and original.get("path") == legacy and
                   original.get("sha256") == "9915ca0c18147eb26f44e2292f541da95e33d0321c6236b224d19d4446e09e9d" and
                   original.get("bytes") == 25070 and refs.get(legacy) == original and
                   activation["runtime_refs"].get(legacy) == original,
                   "same immutable original collector delegate in runtime closure")
    driver.check_ref(root, original)
    calibration = driver.read(driver.check_ref(root, descriptor["calibration_source_lock_ref"]))
    driver.require(type(calibration.get("files")) is list and
                   all(refs.get(row["path"]) == row for row in calibration["files"]) and
                   selected in calibration["files"] and original in calibration["files"],
                   "exact inherited calibration source rows; appended runtime is allowed")
    return selected


'''
text=native.read_text(encoding="utf-8")
before="def execute(root, gates, reservation, *, driver):"
assert text.count(before)==1
text=text.replace(before,helper+before)
before='    driver.require(reservation["label"] == config["run_id"], "same original guarded runtime")'
after=before+'\n    collector_source_ref = preflight_collector_binding(root, gates, driver=driver)'
assert text.count(before)==1
text=text.replace(before,after)
before='        collector_relative = Path(config["runtime_ref"]["path"]).with_name("bounded_native_full_step_collector.py").as_posix()'
after='        collector_relative = collector_source_ref["path"]'
assert text.count(before)==1
text=text.replace(before,after)
target=PREP/"runner/native_runtime_v2.py"
with target.open("xb") as stream:
    stream.write(text.encode("utf-8"))
text=trace.read_text(encoding="utf-8")
before='    return dict(config=config, refs=refs, pair=pair, workload=workload, storage=storage, out=out, ledger=ledger,\n                standing_authorization_ref=grant_ref, activation_gap=gap, full_source_verified=full,\n                source_count=len(refs), common_runtime_domain_sha256=common_domain_sha(pair), finite_activation=finite_activation)'
after='''    gates = dict(config=config, refs=refs, pair=pair, workload=workload, storage=storage, out=out, ledger=ledger,
                standing_authorization_ref=grant_ref, activation_gap=gap, full_source_verified=full,
                source_count=len(refs), common_runtime_domain_sha256=common_domain_sha(pair), finite_activation=finite_activation)
    if config["phase"] in ("development", "effect"):
        require(Path(config["runtime_ref"]["path"]).name == "native_runtime_v2.py",
                "finite startup requires exact new runtime collector binding before GPU")
        selected_runtime = load(root, config["runtime_ref"], "_strong_finite_cpu_collector_gate_" + str(time.monotonic_ns()))
        collector_ref = selected_runtime.preflight_collector_binding(root,gates,driver=sys.modules[__name__])
        gates["collector_binding_preflight"] = dict(source_ref=collector_ref, source_only=True,
                                                   actual_table_issued=False, GPU_qualification_issued=False)
    return gates'''
assert text.count(before)==1
text=text.replace(before,after)
target2=PREP/"runner/strong_trace_runner_v2.py"
with target2.open("xb") as stream:
    stream.write(text.encode("utf-8"))
for path in (target,target2):
    print(path.name,len(path.read_bytes()),hashlib.sha256(path.read_bytes()).hexdigest())
