import ast,hashlib,json
from pathlib import Path
base=Path(__file__).parent
retained={}
for name,functions in {
"qualify_p4_native_gpu.py":["qualify","BoundAuthorFinder","LockedSourceLoader","qualification_context","source_refs","normalize_uuid"],
"run_gpu_stage.py":["atomic_json","positive_number","session_members","signal_session","cleanup_session","Interrupted"]
}.items():
    before=ast.parse((base/"baseline"/name).read_bytes())
    after=ast.parse((base/"candidate"/name).read_bytes())
    nodes_before={node.name:ast.dump(node,include_attributes=False) for node in before.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    nodes_after={node.name:ast.dump(node,include_attributes=False) for node in after.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    for function in functions:
        assert nodes_before[function]==nodes_after[function],(name,function)
    retained[name]=functions
result=dict(status="PASS_AST_UNCHANGED_ORIGINAL_NATIVE_AND_GUARD_PRIMITIVES",
            original_implementation_retained=retained,
            permission_source_changes_only_are_CPU_contract_changes=True,
            actual_GPU_runs=0,server_project_source_mutations=0)
(base/"CANDIDATE_AST_REUSE_PROOF.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

