"""CPU config/constructor checks; original reactor body remains intact."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from py_kvcache.fs_config import SharedFileConfig
from py_kvcache.linux_aio import LinuxAioRing
from py_kvcache import reactor

@pytest.mark.parametrize("mode", ["io_uring", "linux_aio"])
def test_backend_explicit_parse(mode):
    c = SharedFileConfig.from_extra_config({"shared_storage_path":"/tmp/x", "io_backend":mode,
                                           "aio_metadata_workers":"2"})
    assert c.io_backend == mode and c.aio_metadata_workers == 2

def test_default_and_original_positional_api():
    assert SharedFileConfig("/tmp/x", True).sync_on_store is True
    assert SharedFileConfig.from_extra_config({"shared_storage_path":"/tmp/x"}).io_backend == "io_uring"

@pytest.mark.parametrize("bad", ["auto", "", "Linux_AIO", 1, False])
def test_bad_backend_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        SharedFileConfig.from_extra_config({"shared_storage_path":"/tmp/x","io_backend":bad})

@pytest.mark.parametrize("bad", [0, 9, -1, True, 1.5])
def test_bad_metadata_worker_budget(bad):
    with pytest.raises((TypeError,ValueError)):
        SharedFileConfig.from_extra_config({"shared_storage_path":"/tmp/x",
                                           "io_backend":"linux_aio","aio_metadata_workers":bad})

def test_actual_constructor_selection_default_does_not_fallback():
    # Execute the actual selection node without constructing CUDA streams/staging.
    tree = ast.parse(Path(reactor.__file__).read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "config.io_backend == 'io_uring'"]
    assert len(nodes) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                   reactor.__file__, "exec")
    calls=[]
    def native(*args, **kwargs):
        calls.append((args,kwargs))
        raise PermissionError("native ring denied")
    scope={"__package__":"py_kvcache","self":SimpleNamespace(),"ring_depth":2,
           "config":SharedFileConfig("/tmp/x"),"LiburingRing":native}
    with pytest.raises(PermissionError):
        exec(code,scope)
    assert len(calls)==1
    scope["config"]=SharedFileConfig("/tmp/x",io_backend="linux_aio")
    exec(code,scope)
    assert isinstance(scope["self"].ring,LinuxAioRing)
    scope["self"].ring.close()
