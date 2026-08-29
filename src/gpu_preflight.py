from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import torch


def query_rows(query: str) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    return [[part.strip() for part in row] for row in csv.reader(output.splitlines()) if row]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select idle GPUs without touching existing processes")
    parser.add_argument("--candidates", default="0,2,3,4,5")
    parser.add_argument("--excluded", default="1")
    parser.add_argument("--max-used-mb", type=int, default=512)
    parser.add_argument("--max-util", type=int, default=10)
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = {int(value) for value in args.candidates.split(",") if value.strip()}
    excluded = {int(value) for value in args.excluded.split(",") if value.strip()}
    inventory = []
    selected = []
    try:
        rows = query_rows("index,name,uuid,memory.total,memory.used,utilization.gpu")
        backend = "nvml"
        normalized = [
            (int(index), name, uuid, int(total), int(used), int(utilization))
            for index, name, uuid, total, used, utilization in rows
        ]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # This server has a known broken NVML device handle while CUDA runtime
        # allocation remains healthy. Validate each card with the runtime.
        backend = "cuda_runtime"
        normalized = []
        if not torch.cuda.is_available():
            raise SystemExit("neither NVML nor CUDA runtime is available")
        for index in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(index):
                    free, total = torch.cuda.mem_get_info()
                    probe = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{index}")
                    torch.cuda.synchronize(index)
                    del probe
                    torch.cuda.empty_cache()
                normalized.append(
                    (index, torch.cuda.get_device_name(index), "unavailable", total // (1024**2), (total - free) // (1024**2), -1)
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                normalized.append((index, "cuda_error", "unavailable", 0, 10**9, -1))

    for index, name, uuid, total, used, utilization in normalized:
        item = {
            "index": index,
            "name": name,
            "uuid": uuid,
            "memory_total_mb": total,
            "memory_used_mb": used,
            "utilization_percent": utilization,
        }
        eligible = (
            item["index"] in candidates
            and item["index"] not in excluded
            and item["memory_used_mb"] <= args.max_used_mb
            and (item["utilization_percent"] < 0 or item["utilization_percent"] <= args.max_util)
        )
        item["eligible"] = eligible
        inventory.append(item)
        if eligible:
            selected.append(item["index"])

    payload = {
        "candidates": sorted(candidates),
        "excluded": sorted(excluded),
        "selected": selected,
        "inventory": inventory,
        "backend": backend,
        "thresholds": {"max_used_mb": args.max_used_mb, "max_util": args.max_util},
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if len(selected) < args.min_gpus:
        raise SystemExit(f"need at least {args.min_gpus} idle GPUs, found {selected}")
    print(" ".join(str(value) for value in selected))


if __name__ == "__main__":
    main()
