import ast,hashlib,json,difflib
from pathlib import Path
base=Path(__file__).parent
candidate=base/"candidate"
candidate.mkdir(exist_ok=True)
def replace_once(text,old,new):
    if old.startswith(" "):
        old="\n"+old
        new="\n"+new
    if text.count(old)!=1: raise ValueError("replacement count "+repr(old)+": "+str(text.count(old)))
    return text.replace(old,new)
text=(base/"baseline/qualify_p4_native_gpu.py").read_bytes().decode()
text=replace_once(text,'BINARY_NAMES = ("_C.abi3.so","_C_stable_libtorch.abi3.so")',
'''BINARY_NAMES = ("_C.abi3.so","_C_stable_libtorch.abi3.so")
PERMISSION = "experiments/prefix_io_v1/configs/permissions.yaml"

def permission_relative(args):
    return getattr(args, "permissions_path", PERMISSION)

def permission_arguments(relative):
    # Omit the default argument to preserve the original guarded child argv.
    return [] if relative == PERMISSION else ["--permissions-path", relative]
''')
text=replace_once(text,'def preview(root, mode, name, source_lock=None,scope_record=GPU_SCOPE):',
'''def preview(root, mode, name, source_lock=None,scope_record=GPU_SCOPE,
            permissions_path=PERMISSION):
    safe(root, permissions_path)
    permission_args=permission_arguments(permissions_path)''')
text=replace_once(text,'           "--scope-record",scope_record,"--source-lock",source_lock or SOURCE_LOCK]',
                  '           "--scope-record",scope_record,"--source-lock",source_lock or SOURCE_LOCK,*permission_args]')
text=replace_once(text,'           "--label",name,"--seconds",str(TIME_LIMIT),"--",*child]',
                  '           *permission_args,"--label",name,"--seconds",str(TIME_LIMIT),"--",*child]')
text=replace_once(text,'            "--scope-record",scope_record,"--source-lock",source_lock or SOURCE_LOCK]',
                  '            "--scope-record",scope_record,"--source-lock",source_lock or SOURCE_LOCK,*permission_args]')
text=replace_once(text,'        source_lock=source_lock,output=str(out),ordinary_policy_controller=False,',
                  '        source_lock=source_lock,permissions_path=permissions_path,output=str(out),ordinary_policy_controller=False,')
text=replace_once(text,'    permission_path=safe(root,"experiments/prefix_io_v1/configs/permissions.yaml")',
                  '    relative=permission_relative(args)\n    permission_path=safe(root,relative)')
text=replace_once(text,'        path="experiments/prefix_io_v1/configs/permissions.yaml",bytes=len(raw),',
                  '        path=relative,bytes=len(raw),')
text=replace_once(text,'    refs=source_refs(root,args.source_lock)',
'''    refs=source_refs(root,args.source_lock)
    if relative != PERMISSION:
        require(relative in refs and refs[relative] == scope["base_permissions"],
                "effective permissions must be in the frozen source lock")''')
text=replace_once(text,'    return dict(scope=scope,source_refs=refs),"SCOPE_SOURCE_VALIDATED_WITHOUT_GPU"',
                  '    return dict(scope=scope,source_refs=refs,permit=permit),"SCOPE_SOURCE_VALIDATED_WITHOUT_GPU"')
old='    permit=permission_fields(safe(root,"experiments/prefix_io_v1/configs/permissions.yaml").read_text())'
if text.count(old)!=2: raise ValueError("expected two permission reads")
text=text.replace(old,'    permit=gates["permit"]')
text=replace_once(text,'    import subprocess\n    child=',
                  '    import subprocess\n    permission_args=permission_arguments(permission_relative(args))\n    child=')
text=replace_once(text,'           "--scope-record",args.scope_record,"--source-lock",args.source_lock]',
                  '           "--scope-record",args.scope_record,"--source-lock",args.source_lock,*permission_args]')
text=replace_once(text,'             "--label",args.name,"--seconds",str(TIME_LIMIT),"--",*child]',
                  '             *permission_args,"--label",args.name,"--seconds",str(TIME_LIMIT),"--",*child]')
text=replace_once(text,'              "--scope-record",args.scope_record,"--source-lock",args.source_lock]',
                  '              "--scope-record",args.scope_record,"--source-lock",args.source_lock,\n              *permission_arguments(permission_relative(args))]')
text=replace_once(text,'    require(active.get("command")==expected,"exact guarded child argv required")',
'''    require(active.get("command")==expected,"exact guarded child argv required")
    if permission_relative(args) != PERMISSION:
        require(active.get("permissions")==scope["base_permissions"],
                "actual guard effective-permissions binding required")''')
text=replace_once(text,'    return dict(scope=scope,source_refs=refs,out=out),"READY_FOR_AUTHORIZED_NATIVE_GPU"',
'''    return dict(scope=scope,source_refs=refs,out=out,permit=permit),"READY_FOR_AUTHORIZED_NATIVE_GPU"

def storage_contract(root,args,gates):
    if permission_relative(args) == PERMISSION:
        from experiment_storage import preflight
        return preflight(gates["out"],STORAGE_RESERVE,project=root)
    # G1 only writes PRIMARY. Do not borrow an old GPU's AUX authorization.
    import shutil
    primary=safe(root,"experiments/prefix_io_v1/runs")
    require(Path(gates["permit"]["approved_experiment_root"]).resolve()==primary,
            "effective permission PRIMARY root differs")
    out=safe(root,gates["out"].relative_to(root).as_posix())
    require(out==primary/args.name/"details" and primary.is_dir(),
            "effective G1 storage must remain under PRIMARY")
    free=shutil.disk_usage(primary).free
    require(type(free) is int and free-STORAGE_RESERVE>=FLOOR,
            "effective PRIMARY storage floor would be crossed")
    return dict(path=str(out),root=str(primary),free_bytes=free,
                reserved_new_bytes=STORAGE_RESERVE,
                permissions=gates["scope"]["base_permissions"],auxiliary_scope_used=False)
''')
text=replace_once(text,'    ap.add_argument("--source-lock")',
                  '    ap.add_argument("--source-lock")\n    ap.add_argument("--permissions-path",default=PERMISSION)')
text=replace_once(text,'    result=preview(root,args.mode,args.name,None if args.launch else args.source_lock,args.scope_record)',
                  '    result=preview(root,args.mode,args.name,None if args.launch else args.source_lock,args.scope_record,args.permissions_path)')
text=replace_once(text,'            result=preview(root,args.mode,args.name,args.source_lock,args.scope_record)',
                  '            result=preview(root,args.mode,args.name,args.source_lock,args.scope_record,args.permissions_path)')
text=replace_once(text,'            from experiment_storage import preflight\n            contract=preflight(gates["out"],STORAGE_RESERVE,project=root)',
                  '            contract=storage_contract(root,args,gates)')
ast.parse(text)
(candidate/"qualify_p4_native_gpu.py").write_bytes(text.encode())

text=(base/"baseline/run_gpu_stage.py").read_bytes().decode()
text=replace_once(text,'import fcntl','import fcntl\nimport hashlib')
text=replace_once(text,'RESERVE_SECONDS = 20',
'''RESERVE_SECONDS = 20
PERMISSION = "experiments/prefix_io_v1/configs/permissions.yaml"

def permission_source(root,relative=PERMISSION):
    if (not isinstance(relative,str) or not relative or "\\\\" in relative
            or Path(relative).is_absolute()
            or any(part in ("",".","..") for part in relative.split("/"))):
        raise RuntimeError("project-relative POSIX permission path required")
    path=root/relative
    cursor=path
    while cursor!=root:
        if cursor.is_symlink():
            raise RuntimeError("permission path symlink rejected")
        cursor=cursor.parent
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise RuntimeError("permission path outside project")
    raw=path.read_bytes()
    permission=yaml.safe_load(raw.decode())
    if relative!=PERMISSION:
        # A new device-specific G1 grant cannot enlarge the original contract.
        base=yaml.safe_load((root/PERMISSION).read_text())
        if positive_number(permission["max_gpu_hours"],"max_gpu_hours")>positive_number(base["max_gpu_hours"],"base max_gpu_hours"):
            raise RuntimeError("effective GPU budget exceeds base contract")
        for key in ("approved_experiment_root","approved_dependency_root"):
            if permission.get(key)!=base.get(key):
                raise RuntimeError("effective permission root differs from base: "+key)
        for key in ("allow_model_downloads","allow_driver_or_system_changes",
                    "allow_shared_data_deletion","allow_payment","allow_new_cloud_rental"):
            if permission.get(key) is not False:
                raise RuntimeError("effective G1 scope prohibits "+key)
        if permission.get("approved_auxiliary_storage") is not None:
            raise RuntimeError("effective G1 scope is PRIMARY-only")
    return permission,dict(path=relative,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
''')
text=replace_once(text,'    p.add_argument("--seconds", type=int, required=True)',
                  '    p.add_argument("--seconds", type=int, required=True)\n    p.add_argument("--permissions-path",default=PERMISSION)')
text=replace_once(text,'    permission = yaml.safe_load((root / "experiments/prefix_io_v1/configs/permissions.yaml").read_text())',
                  '    permission,permission_ref = permission_source(root,a.permissions_path)')
text=replace_once(text,'        budget["active_reservation"] = reservation',
                  '        if a.permissions_path != PERMISSION:\n            reservation["permissions"] = permission_ref\n        budget["active_reservation"] = reservation')
text=replace_once(text,'        budget["gpu_wall_seconds"] = used + elapsed',
                  '        if a.permissions_path != PERMISSION:\n            event["permissions"] = permission_ref\n        budget["gpu_wall_seconds"] = used + elapsed')
ast.parse(text)
(candidate/"run_gpu_stage.py").write_bytes(text.encode())
files=[]
for name in ("qualify_p4_native_gpu.py","run_gpu_stage.py"):
    before=(base/"baseline"/name).read_bytes();after=(candidate/name).read_bytes()
    diff="".join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),
                                   fromfile="a/experiments/prefix_io_v1/scripts/"+name,
                                   tofile="b/experiments/prefix_io_v1/scripts/"+name))
    (base/(name+".diff")).write_bytes(diff.encode())
    files.append(dict(path="experiments/prefix_io_v1/scripts/"+name,
                      old_bytes=len(before),old_sha256=hashlib.sha256(before).hexdigest(),
                      new_bytes=len(after),new_sha256=hashlib.sha256(after).hexdigest()))
(base/"CANDIDATE_HASHES.json").write_text(json.dumps(dict(status="CPU_CANDIDATE_NOT_APPLIED_OR_GPU_AUTHORIZED",files=files),indent=2)+"\n")
print(json.dumps(files,indent=2))

