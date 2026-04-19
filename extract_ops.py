"""Extract all unique namespaced operations from the curated dataset.

Usage:
    python extract_ops.py
"""

import json
import re
import pathlib

DATA_PATH = pathlib.Path(__file__).parent / "curated_100.jsonl"

PATTERN = re.compile(r"\b(tl|tl_math|triton_helpers|libdevice)\.[a-zA-Z_][a-zA-Z0-9_]*")


def extract_ops() -> dict[str, set[str]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    by_ns: dict[str, set[str]] = {}
    for s in samples:
        for m in PATTERN.finditer(s["kernel_code"]):
            op = m.group()
            ns = op.split(".")[0]
            by_ns.setdefault(ns, set()).add(op)
    return by_ns


def main():
    by_ns = extract_ops()
    for ns in sorted(by_ns):
        print(f"{ns} ({len(by_ns[ns])}):")
        for op in sorted(by_ns[ns]):
            print(f"  {op}")
        print()

    total = sum(len(v) for v in by_ns.values())
    print(f"--- {total} unique ops total ---")


if __name__ == "__main__":
    main()
