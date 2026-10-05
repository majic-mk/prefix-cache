import hashlib,os
from pathlib import Path
import pytest
import consolidate_private_cache_copies as mod

def pair(tmp_path):
    a=tmp_path/"a.bin";b=tmp_path/"b.bin";a.write_bytes(b"archived exact KV");b.write_bytes(a.read_bytes())
    return a,b,hashlib.sha256(a.read_bytes()).hexdigest()

def test_verified_private_copy_preserves_bytes_and_paths(tmp_path):
    a,b,sha=pair(tmp_path);old=b.stat().st_ino
    mod.consolidate_one(a,b,sha)
    assert os.path.samefile(a,b) and b.stat().st_ino!=old
    assert b.read_bytes()==a.read_bytes()==b"archived exact KV"
    assert sorted(p.name for p in tmp_path.iterdir())==["a.bin","b.bin"]

def test_shared_target_is_never_replaced(tmp_path):
    a,b,sha=pair(tmp_path);os.link(b,tmp_path/"shared.bin")
    old=b.stat().st_ino
    with pytest.raises(ValueError,match="hardlink"):mod.consolidate_one(a,b,sha)
    assert b.stat().st_ino==old and not os.path.samefile(a,b)

def test_mismatch_rejected_without_mutation(tmp_path):
    a,b,sha=pair(tmp_path);b.write_bytes(b"different");old=b.stat().st_ino
    with pytest.raises(ValueError,match="content mismatch"):mod.consolidate_one(a,b,sha)
    assert b.stat().st_ino==old and b.read_bytes()==b"different"

def test_failure_before_replace_keeps_original_inode(tmp_path):
    a,b,sha=pair(tmp_path);old=b.stat().st_ino
    def fail():raise RuntimeError("injected before replace")
    with pytest.raises(RuntimeError):mod.consolidate_one(a,b,sha,before_replace=fail)
    assert b.stat().st_ino==old and b.read_bytes()==a.read_bytes()
    assert len(list(tmp_path.iterdir()))==2

def test_post_replace_verification_failure_restores_private_inode(tmp_path,monkeypatch):
    a,b,sha=pair(tmp_path);old=b.stat().st_ino
    monkeypatch.setattr(mod.os.path,"samefile",lambda *args:False)
    with pytest.raises(ValueError,match="post-replacement"):mod.consolidate_one(a,b,sha)
    assert b.stat().st_ino==old and b.read_bytes()==a.read_bytes()
    assert len(list(tmp_path.iterdir()))==2

def test_audit_metadata_staleness_rejected(tmp_path):
    root=tmp_path.resolve();p=root/"runs/x.bin";p.parent.mkdir();p.write_bytes(b"v")
    s=p.stat();r=dict(path=str(p),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,private_completed_run=True)
    assert mod.verified_path(r,root,private=True)==p
    p.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed since audit"):mod.verified_path(r,root,private=True)

def test_redirected_or_outside_target_rejected(tmp_path):
    root=tmp_path/"scope";root.mkdir();p=tmp_path/"external.bin";p.write_bytes(b"v")
    s=p.stat();r=dict(path=str(p),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,private_completed_run=True)
    with pytest.raises(ValueError):mod.verified_path(r,root,private=True)
    link=root/"link.bin";link.symlink_to(p);r["path"]=str(link)
    with pytest.raises(ValueError):mod.verified_path(r,root,private=False)
