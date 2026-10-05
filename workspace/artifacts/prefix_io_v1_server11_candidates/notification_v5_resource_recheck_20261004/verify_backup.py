import hashlib,json,pathlib
root=pathlib.Path(__file__).resolve().parent
receipt=json.loads((root/'SERVER_RECEIPT.json').read_text(encoding='utf-8'))
result={}
for name,expected in receipt['files'].items():
    assert pathlib.Path(name).name==name
    path=root/name
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected['sha256']:
        assert data.endswith(b'\n')
        repaired=data[:-1]
        assert len(repaired)==expected['bytes']
        assert hashlib.sha256(repaired).hexdigest()==expected['sha256']
        path.write_bytes(repaired)
        data=path.read_bytes()
    assert len(data)==expected['bytes']
    assert hashlib.sha256(data).hexdigest()==expected['sha256']
    result[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
proof={'status':'PASS','matched_files':len(result),'server_evidence_dir':receipt['remote_evidence_dir'],'files':result,'repair_note':'Local patch writer appended one newline. Removed only that newline after proving proposed bytes equal the server hash; server files unchanged.','gpu_runs':0}
(root/'LOCAL_BACKUP_VERIFICATION.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(proof,ensure_ascii=False))

