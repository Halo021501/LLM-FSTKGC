from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def collect(run_root: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for name in sorted(os.listdir(run_root)):
        path = os.path.join(run_root, name)
        if not os.path.isdir(path) or name == "logs":
            continue
        metrics_path = os.path.join(path, "metrics.json")
        if not os.path.exists(metrics_path):
            continue
        metrics = load_json(metrics_path)
        meta_path = os.path.join(path, "run_meta.json")
        meta = load_json(meta_path) if os.path.exists(meta_path) else {}
        exit_path = os.path.join(path, "exit_code.txt")
        exit_code = ""
        if os.path.exists(exit_path):
            with open(exit_path, "r", encoding="utf-8") as handle:
                exit_code = handle.read().strip()
        row = {
            "run": name,
            "exit_code": exit_code,
            "disabled_modules": ",".join(meta.get("disabled_modules", [])),
        }
        row.update({key: f"{value:.6f}" for key, value in metrics.items()})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    rows = collect(args.run_root)
    output = args.output or os.path.join(args.run_root, "summary.csv")
    core_fields = [
        "run",
        "exit_code",
        "disabled_modules",
        "valid_mrr",
        "valid_hits1",
        "valid_hits3",
        "valid_hits10",
        "test_mrr",
        "test_hits1",
        "test_hits3",
        "test_hits10",
    ]
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in core_fields
        }
    )
    fieldnames = core_fields + extra_fields
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
