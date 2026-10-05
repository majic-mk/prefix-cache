import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pacekv.analysis import analyze
from pacekv.audit import write_new_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    write_new_json(args.output, analyze(args.input))


if __name__ == "__main__":
    main()
