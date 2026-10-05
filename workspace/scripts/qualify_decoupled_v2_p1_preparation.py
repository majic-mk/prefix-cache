"""Read-only P0/data/Source qualification; never starts a model or GPU task."""
import argparse,json
from pathlib import Path
from probekv.p0_stage_readiness_v2 import _read_ref
from probekv.p0_evidence_v2 import write_new_json
from probekv.source_comparison_v2 import runtime_binding_digest
from probekv.p1_qualification_v2 import assess_p1_preparation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--contract-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    contract,_=_read_ref(dict(path=str(a.contract),sha256=a.contract_sha256),'qualification contract')
    report=assess_p1_preparation(contract,observed_runtime_digest=runtime_binding_digest())
    report['contract_file_sha256']=a.contract_sha256
    # Keep the canonical report hash valid after binding the external bytes.
    from probekv.v8_schema10_execution import digest_json
    report['report_sha256']=digest_json({k:v for k,v in report.items() if k!='report_sha256'})
    write_new_json(a.output,report)
    print(json.dumps({k:report[k] for k in ('status','checks','blockers','execution_blockers','P1_execution_allowed')}))
    return 0 if report['build_plan_preparation_ready'] else 2


if __name__=='__main__':raise SystemExit(main())
