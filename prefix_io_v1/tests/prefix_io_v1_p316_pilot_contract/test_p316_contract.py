"""Pure P316 workload, provenance and diagnostic fault guards; no CUDA/model run."""
import copy, hashlib, json, sys, types
from pathlib import Path
import pytest
import concurrent_pilot_contract_p316 as contract
import simple_stage_worker_probe_p316 as probe

GPU = "GPU-cpu-fixture"
Q = 917504

def dump(path, value):
    path.write_text(json.dumps(value))
    return path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def registration(tmp_path):
    original=dict(max_num_seqs=1,kv_cache_memory_bytes=2147483648,
                  max_model_len=16400,max_num_batched_tokens=16400,enable_prefix_caching=True)
    candidate=dict(provenance=dict(gpu_uuid=GPU,engine=original))
    families=[dict(name="f"+str(i),tokens=[31000+i]+(list(range(1000,5096))*4)[:16255]+[13],
                   initial_ssd_present=i<3) for i in range(5)]
    parent=dict(partition="development",formal_goodput=False,slo=None,policy="shadow",
        load_planner="on",artificial_io_delay=False,cache_resets=0,gpu_uuid=GPU,
        engine=dict(original,max_num_seqs=2),staging_bytes=1073741824,output_tokens=128,
        families=families,io_depth=8,baseline_tuning=True,profile="all_hit",
        requests=[dict(request_id=i,family_index=f,scheduled_time=.5+i*.25)
                  for i,f in enumerate(contract.PATTERNS["all_hit"])],
        arrival_seed=1704,arrival_rate=1.0,scope="CPU fixture")
    hot=copy.deepcopy(parent);hot["profile"]="gpu_hot"
    for row in hot["requests"]:row["family_index"]=0
    parent_path=dump(tmp_path/"parent.json",parent);current=dump(tmp_path/"hot.json",hot)
    curves=dump(tmp_path/"candidate.json",candidate);qualification=dump(tmp_path/"permit.json",{"frozen":"fixture"})
    golden=list(range(128))
    reference=dump(tmp_path/"reference.json",dict(rows=[dict(family="f0",kind="gpu_hot",
        prompt_token_ids=families[0]["tokens"],output_tokens=golden)]))
    raw=dict(schema_version=1,scope=contract.SUPPLEMENT_SCOPE,gpu_uuid=GPU,
        manifest_sha256=sha(current),candidate_sha256=sha(curves),qualification_sha256=sha(qualification),
        parent_manifest=dict(path="parent.json",sha256=sha(parent_path)),
        reference_results=dict(path="reference.json",sha256=sha(reference)))
    reg=dump(tmp_path/"registration.json",raw)
    permit=dict(manifests={"all_hit":dict(sha256=sha(parent_path))})
    return dict(root=tmp_path,reg=reg,raw=raw,current=current,curves=curves,qualification=qualification,
                parent_path=parent_path,reference=reference,permit=permit,hot=hot,parent=parent,
                candidate=candidate,golden=golden)

def check(r):
    return contract.validate_supplement(r["root"],r["reg"],r["current"],r["curves"],
                                       r["qualification"],r["permit"],GPU)

def test_registered_gpu_hot_keeps_original_c2_d8_contract(registration):
    r=registration
    assert contract.validate(r["hot"],r["candidate"],GPU)==r["hot"]["engine"]
    assert check(r)["ssd_write_max_bytes"]==128*Q
    assert check(r)["golden_output_tokens"]==r["golden"]
    assert contract.validate(r["parent"],r["candidate"],GPU)==r["parent"]["engine"]

@pytest.mark.parametrize("field",["manifest_sha256","candidate_sha256","qualification_sha256"])
def test_supplement_input_hash_cannot_be_reused(registration,field):
    r=registration;r["raw"][field]="0"*64;dump(r["reg"],r["raw"])
    with pytest.raises(ValueError):check(r)

@pytest.mark.parametrize("field",["parent_manifest","reference_results"])
def test_supplement_bound_file_hash(registration,field):
    r=registration;r["raw"][field]["sha256"]="0"*64;dump(r["reg"],r["raw"])
    with pytest.raises(ValueError):check(r)

def test_supplement_rejects_unqualified_parent(registration):
    r=registration;r["permit"]["manifests"]["all_hit"]["sha256"]="0"*64
    with pytest.raises(ValueError):check(r)

def test_reference_must_remain_inside_registered_root(registration):
    r=registration;r["raw"]["reference_results"]["path"]="../reference.json";dump(r["reg"],r["raw"])
    with pytest.raises(ValueError):check(r)

@pytest.mark.parametrize("change",["arrival","resources","family","seed"])
def test_supplement_cannot_change_parent_resource_or_arrival(registration,change):
    r=registration;hot=r["hot"]
    if change=="arrival":hot["requests"][1]["scheduled_time"]+=.1
    if change=="resources":hot["engine"]["max_num_seqs"]=1
    if change=="family":hot["requests"][0]["family_index"]=1
    if change=="seed":hot["arrival_seed"]=1705
    dump(r["current"],hot);r["raw"]["manifest_sha256"]=sha(r["current"]);dump(r["reg"],r["raw"])
    with pytest.raises(ValueError):check(r)

@pytest.mark.parametrize("change",["length","prompt","kind"])
def test_reference_requires_full_matching_gpu_hot_golden(registration,change):
    r=registration;ref=json.loads(r["reference"].read_text())
    if change=="length":ref["rows"][0]["output_tokens"].pop()
    if change=="prompt":ref["rows"][0]["prompt_token_ids"][0]+=1
    if change=="kind":ref["rows"][0]["kind"]="cold"
    dump(r["reference"],ref);r["raw"]["reference_results"]["sha256"]=sha(r["reference"]);dump(r["reg"],r["raw"])
    with pytest.raises(ValueError):check(r)

def test_duplicate_registration_key_rejected(registration):
    r=registration;text=r["reg"].read_text()
    r["reg"].write_text(text[:-1]+',"schema_version":1}')
    with pytest.raises(ValueError,match="duplicate"):check(r)

def test_hot_profile_cannot_reuse_old_three_family_pattern(registration):
    r=registration;hot=copy.deepcopy(r["hot"]);hot["requests"]=r["parent"]["requests"]
    with pytest.raises(ValueError):contract.validate(hot,r["candidate"],GPU)

@pytest.fixture
def hot_rows():
    return [dict(num_cached_tokens=16256,output_tokens=list(range(128))) for _ in range(10)]

@pytest.mark.parametrize("paid_write",[0,Q,128*Q])
def test_gpu_hot_allows_only_bounded_paid_generation_writes(hot_rows,paid_write):
    contract.validate_gpu_hot_measurement(hot_rows,0,paid_write,Q,list(range(128)))

@pytest.mark.parametrize("change",["read","write","prefix","output","quantum","missing"])
def test_gpu_hot_actual_measurement_hard_gates(hot_rows,change):
    read=0;write=0;quantum=Q
    if change=="read":read=Q
    if change=="write":write=128*Q+1
    if change=="prefix":hot_rows[0]["num_cached_tokens"]=16240
    if change=="output":hot_rows[0]["output_tokens"][-1]+=1
    if change=="quantum":quantum=Q//2
    if change=="missing":hot_rows.pop()
    with pytest.raises(ValueError):
        contract.validate_gpu_hot_measurement(hot_rows,read,write,quantum,list(range(128)))

def test_optional_observer_default_and_off_control_domain():
    assert contract.validate_optional_sampling("on","mixed_readwrite",None)=="on"
    assert contract.validate_optional_sampling("off","gpu_hot",{"prefix_io_stage_policy":{"mode":"off"}},True)=="off"

@pytest.mark.parametrize("args",[
    ("off","mixed_readwrite",{"prefix_io_stage_policy":{"mode":"off"}},True),
    ("off","gpu_hot",{"prefix_io_stage_policy":{"mode":"pressure"}},True),
    ("off","gpu_hot",None,False),("on","gpu_hot",None,True),
    ("unknown","gpu_hot",{},False),("on","gpu_hot",{},1)])
def test_optional_observer_and_cpu_flags_fail_closed(args):
    with pytest.raises(ValueError):contract.validate_optional_sampling(*args)

def capacity(cache=65,preload=0,shared=0):
    return dict(staging_capacity=dict(valid=True,observation_valid=True,error=None,
        physical_release_credit=False,cache_slots=cache,preload_cached_slots=preload,
        shared_cached_slots=shared))

def test_pressure_live_capacity_uses_shared_slot_only_once():
    contract.validate_pressure_capacity("pressure","on",[capacity(60,10,10)])
    with pytest.raises(ValueError):
        contract.validate_pressure_capacity("pressure","on",[capacity(60,0,60)])

@pytest.mark.parametrize("change",["off","64","none","error","release","invalid","no_sample"])
def test_pressure_live_known_large_registry_is_required(change):
    live=[capacity()];sampling="on"
    if change=="off":sampling="off"
    if change=="64":live=[capacity(64)]
    if change=="none":live[0]["staging_capacity"]["cache_slots"]=None
    if change=="error":live[0]["staging_capacity"]["error"]="unknown"
    if change=="release":live[0]["staging_capacity"]["physical_release_credit"]=True
    if change=="invalid":live[0]["staging_capacity"]["observation_valid"]=False
    if change=="no_sample":live=[]
    with pytest.raises(ValueError):contract.validate_pressure_capacity("pressure",sampling,live)

class NativeError(RuntimeError):pass

def test_clock_start_failure_still_calls_native_once(monkeypatch):
    m=probe.MetadataCpu();calls=[]
    def bad_clock():raise RuntimeError("clock")
    monkeypatch.setattr(probe.time,"thread_time_ns",bad_clock)
    assert m.measured("_prefix_stage",lambda:calls.append(1) or 7)()==7
    assert calls==[1] and m.valid is False and m.local.depth==0
    assert m.export()["valid"] is False

def test_clock_end_failure_preserves_native_exception_and_finally(monkeypatch):
    m=probe.MetadataCpu();calls=[];clock_calls=[];original=NativeError("native")
    def clock():
        clock_calls.append(1)
        if len(clock_calls)==1:return 10
        raise RuntimeError("end clock")
    def native():calls.append(1);raise original
    monkeypatch.setattr(probe.time,"thread_time_ns",clock)
    with pytest.raises(NativeError) as e:m.measured("_prefix_stage",native)()
    assert e.value is original and calls==[1] and m.local.depth==0 and not m.valid

def test_counter_failure_cannot_override_native_return(monkeypatch):
    m=probe.MetadataCpu();calls=[];m.counts["warmup"]["_prefix_stage"]=None
    monkeypatch.setattr(probe.time,"thread_time_ns",lambda:1)
    assert m.measured("_prefix_stage",lambda:calls.append(1) or "native")()=="native"
    assert calls==[1] and m.local.depth==0 and not m.valid

def test_nested_hooks_are_not_double_counted(monkeypatch):
    m=probe.MetadataCpu();ticks=iter([10,20]);calls=[]
    monkeypatch.setattr(probe.time,"thread_time_ns",lambda:next(ticks))
    inner=m.measured("_prefix_stage",lambda:calls.append("inner") or 4)
    outer=m.measured("_prefix_stage_decide",lambda:calls.append("outer") or inner())
    assert outer()==4 and calls==["outer","inner"]
    assert m.counts["warmup"]["_prefix_stage"]["calls"]==0
    assert m.counts["warmup"]["_prefix_stage_decide"]==dict(calls=1,thread_cpu_ns=10)
    assert m.local.depth==0

def test_nested_native_error_is_never_recalled(monkeypatch):
    m=probe.MetadataCpu();calls=[];original=NativeError("inner")
    monkeypatch.setattr(probe.time,"thread_time_ns",lambda:10)
    def native():calls.append(1);raise original
    inner=m.measured("_prefix_stage",native)
    outer=m.measured("_prefix_stage_decide",inner)
    with pytest.raises(NativeError) as e:outer()
    assert e.value is original and calls==[1] and m.local.depth==0

def test_publication_failure_exports_invalid_without_good_metrics(monkeypatch):
    m=probe.MetadataCpu()
    def fail_copy(_):raise RuntimeError("publish")
    monkeypatch.setattr(probe.copy,"deepcopy",fail_copy)
    value=m.export()
    assert value["valid"] is False and value["phase_regions"] is None and value["phase_totals"] is None

class FakeReactor:
    def __init__(self):
        self._observation_sink=lambda r:None
        self.file_store=types.SimpleNamespace(io_size=Q)
        self.actual_staging_bytes=1073741824
        self._prefix_dispatch_controller=None
        self.protocol=object()
    def inspect_snapshot(self,timeout):return capacity()
    def _prefix_stage_decide(self,*a,**k):return "native"
    def _prefix_stage_settle(self,*a,**k):return "native"
    def _prefix_stage(self,*a,**k):return "native"
    def _prefix_stage_copy(self,*a,**k):return "native"
    def _prefix_capacity_snapshot(self,*a,**k):return "native"
    def _capture_owner_snapshot(self,*a,**k):return "native"

@pytest.fixture
def fake_worker(monkeypatch):
    reactors=[FakeReactor()]
    worker=types.SimpleNamespace()
    cw=types.SimpleNamespace(worker=types.SimpleNamespace(
        handlers=[types.SimpleNamespace(coordinator=types.SimpleNamespace(reactor=r)) for r in reactors]))
    module=types.ModuleType("vllm.distributed.kv_transfer.kv_transfer_state")
    module.get_kv_transfer_group=lambda:types.SimpleNamespace(connector_worker=cw)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    expected=Path(probe.__file__).resolve().parents[3]/"third_party/work/py-kvcache-p3-16-cpu/py_kvcache/reactor.py"
    monkeypatch.setattr(probe.inspect,"getfile",lambda _:str(expected))
    calls=[]
    def original(w,action,limits):calls.append((action,limits));return {"original":True}
    monkeypatch.setattr(probe,"original_probe",original)
    return worker,reactors,cw,calls

def test_default_probe_does_not_allocate_timer_wrap_or_read_clock(fake_worker,monkeypatch):
    worker,rs,cw,calls=fake_worker;r=rs[0];sink=r._observation_sink
    def forbidden(*a,**k):raise AssertionError("optional CPU timing called")
    monkeypatch.setattr(probe,"MetadataCpu",forbidden)
    monkeypatch.setattr(probe.time,"thread_time_ns",forbidden)
    limits=dict(kv_budget_bytes=2147483648,staging_budget_bytes=1073741824)
    result=probe.worker_probe(worker,"install",limits)
    assert calls==[("install",limits)] and result["original"] is True
    assert r._observation_sink is sink and not any(n in r.__dict__ for n in probe.CPU_METHODS)
    assert "native_metadata_cpu" not in result
    assert result["optional_native_sampling"]["mode"]=="on"

def test_optional_off_sink_restore_chain_preserves_protocol(fake_worker):
    worker,rs,cw,calls=fake_worker;r=rs[0];original_sink=r._observation_sink;protocol=r.protocol
    lease=probe.OptionalSampling();lease.suppress(rs)
    assert r._observation_sink is None and r.protocol is protocol
    lease.restore_sinks();assert r._observation_sink is original_sink
    combined=lambda r:original_sink(r)
    r._observation_sink=combined
    lease.suppress(rs);lease.restore_sinks()
    assert r._observation_sink is combined and r.protocol is protocol

def test_partial_cpu_install_restores_all_prior_methods(fake_worker):
    worker,rs,cw,calls=fake_worker;r=rs[0];old=r._prefix_stage
    other=FakeReactor();other._prefix_capacity_snapshot=None
    cw.worker.handlers.append(types.SimpleNamespace(coordinator=types.SimpleNamespace(reactor=other)))
    with pytest.raises(RuntimeError,match="hooks"):
        probe.worker_probe(worker,"install",dict(kv_budget_bytes=2147483648,
            staging_budget_bytes=1073741824,native_cpu_probe=True))
    assert r._prefix_stage==old and not worker._p316_metadata_cpu.restores and len(calls)==1

def test_start_failure_restores_timer_sink_and_preserves_error(fake_worker,monkeypatch):
    worker,rs,cw,calls=fake_worker;r=rs[0];old_sink=r._observation_sink;old_method=r._prefix_stage
    probe.worker_probe(worker,"install",dict(kv_budget_bytes=2147483648,
        staging_budget_bytes=1073741824,native_cpu_probe=True))
    original=NativeError("start");native_calls=[]
    def fail(*args):native_calls.append(1);raise original
    monkeypatch.setattr(probe,"original_probe",fail)
    with pytest.raises(NativeError) as e:probe.worker_probe(worker,"start")
    assert e.value is original and native_calls==[1]
    assert r._prefix_stage==old_method and r._observation_sink is old_sink

def test_finish_cleanup_failure_never_masks_native_error(fake_worker,monkeypatch):
    worker,rs,cw,calls=fake_worker;original=NativeError("native finish");native_calls=[]
    def cleanup():raise RuntimeError("cleanup")
    worker._p316_metadata_cpu=types.SimpleNamespace(phase="warmup",unwrap_sinks=lambda:None,restore=cleanup)
    def fail(*args):native_calls.append(1);raise original
    monkeypatch.setattr(probe,"original_probe",fail)
    with pytest.raises(NativeError) as e:probe.worker_probe(worker,"finish")
    assert e.value is original and native_calls==[1]
    assert any("cleanup" in n for n in original.__notes__)


def test_probe_off_suppresses_only_sampling_across_safe_boundaries(fake_worker,monkeypatch):
    worker,rs,cw,calls=fake_worker;r=rs[0];old=r._observation_sink;protocol=r.protocol
    result=probe.worker_probe(worker,"install",dict(kv_budget_bytes=2147483648,
        staging_budget_bytes=1073741824,native_observation="off"))
    assert r._observation_sink is None and r.protocol is protocol
    assert result["optional_native_sampling"]["optional_sink_suppressed"] is True
    combined=lambda r:old(r)
    def start_original(w,action,limits):
        assert action=="start" and r._observation_sink is old
        r._observation_sink=combined
        return {"original":True}
    monkeypatch.setattr(probe,"original_probe",start_original)
    probe.worker_probe(worker,"start")
    assert r._observation_sink is None and r.protocol is protocol
    original=NativeError("finish")
    def finish_original(w,action,limits):
        assert action=="finish" and r._observation_sink is combined
        raise original
    monkeypatch.setattr(probe,"original_probe",finish_original)
    with pytest.raises(NativeError) as e:probe.worker_probe(worker,"finish")
    assert e.value is original and r._observation_sink is combined and r.protocol is protocol
