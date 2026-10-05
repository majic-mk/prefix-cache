"""Re-read immutable raw P0 evidence, without model initialization or CUDA work."""
import argparse
import json
from pathlib import Path

from probekv.p0_acceptance_v2 import assess_numerical_correctness, file_sha
from probekv.p0_evidence_v2 import write_new_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--index-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if file_sha(args.index)!=args.index_sha256:
        raise ValueError('evidence index checksum differs')
    index=json.loads(args.index.read_text(encoding='utf-8'))
    report=assess_numerical_correctness(index['batches'])
    report['index_sha256']=args.index_sha256
    write_new_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('status','counts','blockers','P1_execution_allowed')}))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
