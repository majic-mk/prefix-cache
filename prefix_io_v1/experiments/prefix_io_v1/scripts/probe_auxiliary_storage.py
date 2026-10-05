"""Real native Linux AIO / O_DIRECT probe in the explicitly authorized root."""
import ctypes as C,json,os,time,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from experiment_storage import permission,preflight
from py_kvcache.linux_aio import LinuxAioRing
root=Path(permission()["approved_auxiliary_storage"]["root"]);out=root/"probe"
pre=preflight(out,4*917504+1024**2);out.mkdir(exist_ok=False)
size=917504;fds=[];owners=[];ring=LinuxAioRing(4);evidence=[]
def buffer(fill):
    owner=C.create_string_buffer(size+4096);owners.append(owner)
    address=(C.addressof(owner)+4095)&~4095;C.memset(address,fill,size)
    return address
def reap(count):
    end=time.monotonic()+10;rows=[]
    while len(rows)<count and time.monotonic()<end:
        rows.extend(ring.poll_all())
        if len(rows)<count:time.sleep(.0001)
    assert len(rows)==count
    return dict(rows)
try:
    a=[buffer(17+i) for i in range(4)];b=[buffer(0) for i in range(4)]
    for i in range(4):
        fd=os.open(out/(str(i)+".bin"),os.O_CREAT|os.O_EXCL|os.O_RDWR|os.O_DIRECT,0o600);fds.append(fd)
        ring.queue_rw(user_data=i,fd=fd,ptr=C.c_void_p(a[i]),nbytes=size,write=True)
    ring.submit_pending();writes=reap(4);assert writes=={i:size for i in range(4)}
    for i,fd in enumerate(fds):
        ring.queue_rw(user_data=4+i,fd=fd,ptr=C.c_void_p(b[i]),nbytes=size,write=False)
    ring.submit_pending();reads=reap(4);assert reads=={4+i:size for i in range(4)}
    assert all(C.string_at(a[i],size)==C.string_at(b[i],size) for i in range(4))
    for fd in fds:os.fsync(fd)
finally:
    ring.close()
    for fd in fds:os.close(fd)
result=dict(status="PASSED_REAL_CPU_AIO_DIRECT_ROUNDTRIP",gpu_executed=False,
    backend="linux_aio",io_size=size,write_bytes=4*size,read_bytes=4*size,exact_bytes=True,
    aio=ring.snapshot(),storage_before=pre,storage_after=preflight(out,0),
    filesystem=subprocess.check_output(["findmnt","-no","FSTYPE,SOURCE","-T",str(out)],text=True).strip())
(out/"result.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
