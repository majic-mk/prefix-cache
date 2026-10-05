"""CPU-only preflight. Hashes assets; does not import torch or load a model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pacekv.audit import audit_environment, write_new_json
from pacekv.plan import pilot_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stack", choices=("torch27", "torch28"), default="torch27")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    write_new_json(out / "plan.json", pilot_plan(args.stack))
    audit = audit_environment(ROOT, args.model, stack=args.stack)
    write_new_json(out / "cpu_preflight.json", audit)
    print("CPU preflight:", "PASS" if audit["cpu_asset_preflight_passed"] else "BLOCKED")
    for reason in audit["pending"]:
        print(reason)
    for reason in audit["runtime_capacity_pending"]:
        print("Before GPU execution:", reason)
    print("GPU execution NOT authorized or performed; native serving NOT qualified.")
    return 0 if audit["cpu_asset_preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
