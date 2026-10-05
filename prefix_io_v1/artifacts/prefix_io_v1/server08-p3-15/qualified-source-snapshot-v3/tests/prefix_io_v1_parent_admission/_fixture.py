"""P315 native common/controller fixtures; CPU fake I/O, never GPU evidence."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
from prefix_io_control.dispatch_budget import Amount, ZERO
from prefix_io_control.simple_stage_policy import SimpleStageConfig, DispatchController
from tests.prefix_io_v1_start_budget.test_stage_accounting import attach as attach_accounting

def common(x, limit=2):
    x.r._init_parent_admission(limit)
    return x.r

def controlled(x, *, mode="fixed", limit=2, quota=0, reserve=0, age=100,
               epoch=1_000_000):
    r=common(x,limit)
    accounting=attach_accounting(x)
    stream=NS(wait_event=lambda event:None,synchronize=Mock())
    r._streams=[stream]*r.staging_pool.slot_count
    r._copy_stream=NS(synchronize=Mock())
    caps=tuple(Amount(quota,quota*4096) for _ in range(4))
    config=SimpleStageConfig(mode,epoch,4096,caps,caps,
        Amount(quota*2,quota*8192),Amount(quota*2,quota*8192),
        quota*8192,quota*8192,reserve,limit,1000,age)
    controller=DispatchController("cpu-run",config)
    controller.bind();r._prefix_dispatch_controller=controller
    return accounting,controller

def physical_streams(x):
    accounting=attach_accounting(x)
    stream=NS(wait_event=lambda event:None,synchronize=Mock())
    x.r._streams=[stream]*x.r.staging_pool.slot_count
    x.r._copy_stream=NS(synchronize=Mock())
    return accounting,stream,x.r._copy_stream
