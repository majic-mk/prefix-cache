"""P3 CPU-qualified start-budget primitive, not a full production controller.
A unit is the initiation of one native storage-file chain. Read/store/preload
share one allowance; existing continuations and mandatory work keep progressing.
This does not claim per-stage byte ceilings or change native resource ownership.
"""
from dataclasses import dataclass, asdict
from threading import get_ident

@dataclass(frozen=True)
class StartBudgetConfig:
    mode: str
    epoch_ns: int
    starts_per_epoch: int
    max_wait_ns: int
    reserve_free_slots: int = 0
    max_waiting_keys: int = 32

    def __post_init__(self):
        if self.mode not in ("fixed", "pressure"):
            raise ValueError("start budget only supports fixed/pressure")
        for name in ("epoch_ns", "starts_per_epoch", "max_wait_ns", "max_waiting_keys"):
            value=getattr(self,name)
            if type(value) is not int or value<=0:
                raise ValueError(name+" must be a positive integer")
        if self.starts_per_epoch not in (1,2,4,8):
            raise ValueError("start candidates must be 1,2,4,8")
        if self.max_wait_ns<self.epoch_ns:
            raise ValueError("max_wait_ns must cover an epoch")
        if self.max_waiting_keys>32:
            raise ValueError("waiting metadata is bounded to 32 keys")
        if type(self.reserve_free_slots) is not int or not 0<=self.reserve_free_slots<=4096:
            raise ValueError("invalid free-slot reserve")
        if self.mode=="fixed" and self.reserve_free_slots!=0:
            raise ValueError("fixed mode has no pressure reserve")

@dataclass(frozen=True)
class _Ticket:
    epoch: int
    kind: str
    key: object
    reason: str

class StartBudget:
    """Reactor-owner state only; no queue, callback, Future completion or I/O."""
    KINDS=frozenset(("load","store","preload"))
    REASONS=("ordinary","mandatory","mandatory_support","continuation","shutdown","age","fallback")
    def __init__(self,config):
        if not isinstance(config,StartBudgetConfig):
            raise TypeError("frozen StartBudgetConfig required")
        self.config=config
        self.bound=False
        self.owner=None
        self.epoch=None
        self.last_now=None
        self.used=0
        self.waiting={}
        self._ticket=None
        self.faulted=False
        self.fault_reason=None
        self.errors=0
        self.denied_quota=0
        self.denied_pressure=0
        self.epoch_refreshes=0
        self.ordinary_starts=0
        self.starts={k:0 for k in self.KINDS}
        self.reasons={k:0 for k in self.REASONS}

    def bind(self):
        if self.bound:
            raise ValueError("start budget already belongs to another reactor")
        self.bound=True

    def _owner(self):
        ident=get_ident()
        if self.owner is None:self.owner=ident
        if self.owner!=ident:raise RuntimeError("start budget accessed outside reactor owner")

    def fail(self,reason):
        self.errors+=1
        self.faulted=True
        self.fault_reason=str(reason)[:120]
        self.waiting.clear()
        self._ticket=None

    def allow(self,kind,key,*,now_ns,free_slots,mandatory=False,support=False,
              continuation=False,shutdown=False):
        self._owner()
        if kind not in self.KINDS:raise ValueError("unknown native chain kind")
        if type(now_ns) is not int or now_ns<0:raise ValueError("invalid monotonic timestamp")
        if type(free_slots) is not int or free_slots<0:raise ValueError("invalid free slots")
        if self.last_now is not None and now_ns<self.last_now:
            self.fail("monotonic clock regressed")
        self.last_now=now_ns
        epoch=now_ns//self.config.epoch_ns
        if self.epoch is None or epoch>self.epoch:
            self.epoch=epoch;self.used=0;self.epoch_refreshes+=1
        reason=None
        if self.faulted:reason="fallback"
        elif shutdown:reason="shutdown"
        elif mandatory:reason="mandatory"
        elif support:reason="mandatory_support"
        elif continuation:reason="continuation"
        wait_key=(kind,key)
        if reason is None:
            if wait_key not in self.waiting:
                if len(self.waiting)>=self.config.max_waiting_keys:
                    self.fail("bounded waiting metadata exhausted")
                    reason="fallback"
                else:self.waiting[wait_key]=now_ns
            if reason is None:
                age=now_ns-self.waiting[wait_key]
                if age>=self.config.max_wait_ns:reason="age"
                elif (self.config.mode=="pressure" and kind!="load" and
                      free_slots<=self.config.reserve_free_slots):
                    self.denied_pressure+=1;return False
                elif self.used>=self.config.starts_per_epoch:
                    self.denied_quota+=1;return False
                else:reason="ordinary"
        ticket=_Ticket(epoch,kind,key,reason)
        self._ticket=ticket
        return ticket

    def commit(self,ticket):
        self._owner()
        if ticket is not self._ticket:
            raise RuntimeError("stale or duplicate start ticket")
        self._ticket=None
        if ticket.reason=="ordinary":
            if ticket.epoch!=self.epoch or self.used>=self.config.starts_per_epoch:
                raise RuntimeError("ordinary allowance changed before commit")
            self.used+=1;self.ordinary_starts+=1
        self.starts[ticket.kind]+=1
        self.reasons[ticket.reason]+=1
        self.waiting.pop((ticket.kind,ticket.key),None)

    def retire(self,future):
        self._owner()
        for kind in ("load","store"):self.waiting.pop((kind,future),None)
        if self._ticket is not None and self._ticket.key is future:self._ticket=None

    def clear(self):
        self.waiting.clear();self._ticket=None

    def snapshot(self):
        # Read on the owner thread or after the native reactor has terminated.
        return dict(config=asdict(self.config),bound=self.bound,epoch=self.epoch,used=self.used,
                    waiting_keys=len(self.waiting),faulted=self.faulted,errors=self.errors,
                    fault_reason=self.fault_reason,denied_quota=self.denied_quota,
                    denied_pressure=self.denied_pressure,epoch_refreshes=self.epoch_refreshes,
                    ordinary_starts=self.ordinary_starts,starts=dict(self.starts),
                    reasons=dict(self.reasons),unit="native storage-file chain initiation",
                    full_per_stage_byte_caps=False,gpu_qualified=False)
