"""Generate exact byte-preserving C3 -> C4 diffs and strictly replay every hunk.

Reads immutable candidates only. Writes new patch/proof files beneath --output;
never modifies source candidates, contacts a server, imports model code or CUDA.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys


GROUPS = {
    'source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py': '01-common-idle',
    'test_retained_idle_wait.py': '01-common-idle',
    'test_empty_p4_window.py': '01-common-idle',
    'IDLE_CHANGE_NOTES.md': '01-common-idle',
    'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_bridge.py': '02-compact-bridge',
    'test_enriched_snapshot_reuse.py': '02-compact-bridge',
    'ENRICHED_CHANGE_NOTES.md': '02-compact-bridge',
    'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py': '03-v6-receipt-binding',
    'test_single_file_receipt.py': '03-v6-receipt-binding',
    'CANDIDATE_MANIFEST.json': '04-candidate-metadata',
}
POLICY = 'source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_policy.py'
HUNK = re.compile(rb'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n$')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ref(path, raw):
    return dict(path=str(path.resolve()).replace('\\','/'), bytes=len(raw), sha256=digest(raw))


def files(root):
    result={}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts:
            continue
        if path.is_symlink():
            raise ValueError('symlink inputs are not supported: '+str(path))
        if path.is_file():
            result[path.relative_to(root).as_posix()]=path.read_bytes()
    return result


def replay(base, patch):
    """Apply line-numbered unified hunks exactly: no offset, fuzz or EOL changes."""
    old=base.splitlines(keepends=True); lines=patch.splitlines(keepends=True)
    if len(lines)<3 or not lines[0].startswith(b'--- ') or not lines[1].startswith(b'+++ '):
        raise ValueError('one unified file patch required')
    output=[]; cursor=0; index=2; records=[]
    while index<len(lines):
        match=HUNK.fullmatch(lines[index])
        if match is None:
            raise ValueError('exact unified hunk header required')
        old_start=int(match[1]); old_count=int(match[2] or b'1')
        new_start=int(match[3]); new_count=int(match[4] or b'1')
        at=old_start if old_count==0 else old_start-1
        if not cursor<=at<=len(old):
            raise ValueError('overlapping, reordered or out-of-range hunk')
        output.extend(old[cursor:at]); cursor=at
        new_at=new_start if new_count==0 else new_start-1
        if new_at!=len(output):
            raise ValueError('new hunk location disagrees with exact replay')
        seen_old=seen_new=added=removed=context=0
        before=[]; after=[]; index+=1
        while index<len(lines) and not lines[index].startswith(b'@@ '):
            marker=lines[index][:1]; content=lines[index][1:]
            if marker not in (b' ',b'+',b'-'):
                raise ValueError('unsupported patch marker; no implicit EOF handling')
            if marker in (b' ',b'-'):
                if cursor>=len(old) or old[cursor]!=content:
                    raise ValueError('exact old context/deletion mismatch at '+str(cursor+1))
                before.append(content);cursor+=1;seen_old+=1
            if marker in (b' ',b'+'):
                output.append(content);after.append(content);seen_new+=1
            added+=marker==b'+';removed+=marker==b'-';context+=marker==b' '
            index+=1
        if (seen_old,seen_new)!=(old_count,new_count):
            raise ValueError('hunk line counts disagree with body')
        records.append(dict(index=len(records)+1, old_start=old_start,old_count=old_count,
            new_start=new_start,new_count=new_count,added_lines=added,removed_lines=removed,
            context_lines=context,old_hunk_sha256=digest(b''.join(before)),
            reconstructed_hunk_sha256=digest(b''.join(after)),exact_context=True))
    output.extend(old[cursor:])
    return b''.join(output), records


def put(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:
        stream.write(raw)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--target',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=args.base.resolve(strict=True);target=args.target.resolve(strict=True)
    output=args.output.resolve()
    if base==target or output.is_relative_to(base) or output.is_relative_to(target):
        raise ValueError('distinct immutable inputs and separate output required')
    old=files(base);new=files(target)
    if old.get(POLICY)!=new.get(POLICY) or POLICY not in old:
        raise ValueError('strategy policy bytes unexpectedly changed')
    rows=[];bundle=[];groups={};hunks=0
    for name in sorted(set(old)|set(new)):
        before=old.get(name);after=new.get(name)
        row=dict(relative_path=name,base_ref=None if before is None else ref(base/name,before),
                 target_ref=None if after is None else ref(target/name,after), changed=before!=after)
        if before==after:
            row.update(status='BYTE_IDENTICAL_NO_PATCH',reconstructed_sha256=digest(before))
            rows.append(row);continue
        if name not in GROUPS:
            raise ValueError('unclassified change: '+name)
        for raw in (before,after):
            if raw and not raw.endswith(b'\n'):
                raise ValueError('missing final newline requires separate explicit support')
        patch=b''.join(difflib.diff_bytes(difflib.unified_diff,
            (before or b'').splitlines(keepends=True),(after or b'').splitlines(keepends=True),
            fromfile=b'/dev/null' if before is None else ('a/'+name).encode(),
            tofile=b'/dev/null' if after is None else ('b/'+name).encode(),n=3,lineterm=b'\n'))
        reconstructed,proof=replay(before or b'',patch)
        if reconstructed!=(after or b''):
            raise ValueError('exact replay failed: '+name)
        group=GROUPS[name];patch_name=group+'/'+Path(name).name+'.patch'
        put(output/patch_name,patch)
        groups.setdefault(group,[]).append(patch)
        bundle.append(patch);hunks+=len(proof)
        row.update(category=group,patch_ref=ref(output/patch_name,patch),
            status='EXACT_HUNK_REPLAY_MATCHES_TARGET',hunks=proof,
            reconstructed_bytes=len(reconstructed),reconstructed_sha256=digest(reconstructed),
            byte_equal=True,base_missing=before is None,target_missing=after is None)
        rows.append(row)
    for name,patches in groups.items():
        put(output/(name+'.patch'),b''.join(patches))
    all_patch=b''.join(bundle);put(output/'C3_TO_C4.patch',all_patch)
    # Detect any concurrent source edits before writing a successful proof.
    if files(base)!=old or files(target)!=new:
        raise ValueError('input drift while constructing patches')
    changed=[row for row in rows if row['changed']]
    map_data=dict(scope='LOCAL_C3_TO_C4_EXACT_SOURCE_DIFF_ONLY',
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--base',str(base),
                 '--target',str(target),'--output',str(output)],
        generator_ref=ref(Path(__file__),Path(__file__).read_bytes()),
        base_directory=str(base),target_directory=str(target),
        category_descriptions={
            '01-common-idle':'Original incoming Queue wait for retained-only idle; shutdown uses original _has_work. Applies to all modes.',
            '02-compact-bridge':'Reuse derived snapshots only for identical inputs; does not cache policy decisions or change policy rules.',
            '03-v6-receipt-binding':'Require fresh v6 raw calibration and exact common source pins; no strategy algorithm change.',
            '04-candidate-metadata':'Manifest describing the candidate; not executable policy.'},
        old_file_count=len(old),new_file_count=len(new),changed_file_count=len(changed),
        unchanged_file_count=len(rows)-len(changed),hunks_replayed=hunks,
        complete_target_tree_reconstructed=True,strict_no_offset_no_fuzz=True,
        line_endings_preserved=True,policy_byte_identical=True,
        policy_ref=ref(target/POLICY,new[POLICY]),
        combined_patch_ref=ref(output/'C3_TO_C4.patch',all_patch),files=rows,
        GPU_operations=0,native_execution_verified=False,performance_claim=False,
        limitations=['Local candidate files only; complete server package-copy closure is verified separately.',
            'Exact source reconstruction is not a CPU/GPU behavioral test or a qualification result.',
            'Apply combined patch OR per-category/per-file patches to a clean C3 copy; never apply both.',
            'No strategy thresholds, model, workload, original cost formula or author executor are modified by these patches.'])
    put(output/'PATCH_MAP.json',(json.dumps(map_data,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    print(json.dumps(dict(changed_files=len(changed),unchanged_files=len(rows)-len(changed),
        hunks_replayed=hunks,policy_sha256=digest(new[POLICY]),exact_replay=True)))


if __name__=='__main__':main()
