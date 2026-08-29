from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import HistoryIndex, load_temporal_kg, tensorize
from .model import NineFuseTKG


LOW_MB = 4096.0
TARGET_HIGH_MB = 7680.0
HARD_HIGH_MB = 8192.0


def repeated_support(batch, shot: int, device: torch.device) -> torch.Tensor:
    rows = []
    for s, r, _, t in batch:
        rows.append([(s, r, s, max(0, t - 1))] * shot)
    return torch.as_tensor(rows, dtype=torch.long, device=device)


def worker(args: argparse.Namespace) -> dict[str, float | int | str]:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    kg = load_temporal_kg(args.data_dir)
    train_aug = kg.train_aug
    history = HistoryIndex(train_aug, kg.num_entities, kg.num_relations * 2, args.history_len)
    model = NineFuseTKG(
        kg.num_entities,
        kg.num_relations * 2,
        kg.num_times,
        dim=args.dim,
        history_len=args.history_len,
        channels=args.channels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    batch = [train_aug[index % len(train_aug)] for index in range(args.train_batch)]
    query = tensorize(batch, device)
    support = repeated_support(batch, 5, device)
    features = {key: value.to(device) for key, value in history.build(batch).items()}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    log_probs_a, aux_a = model(query, support, features)
    log_probs_b, aux_b = model(query, support, features)
    target = query[:, 2]
    loss = F.nll_loss(log_probs_a, target) + F.nll_loss(log_probs_b, target)
    loss = loss + 1e-3 * (aux_a["freq_reg"] + aux_b["freq_reg"])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    eval_batch = [train_aug[index % len(train_aug)] for index in range(args.eval_batch)]
    eval_query = tensorize(eval_batch, device)
    eval_support = repeated_support(eval_batch, 5, device)
    eval_features = {key: value.to(device) for key, value in history.build(eval_batch).items()}
    with torch.no_grad():
        model(eval_query, eval_support, eval_features)
    torch.cuda.synchronize(device)
    return {
        "dim": args.dim,
        "channels": args.channels,
        "train_batch": args.train_batch,
        "eval_batch": args.eval_batch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0**2),
        "gpu_name": torch.cuda.get_device_name(device),
    }


def run_candidate(args: argparse.Namespace, dim: int, channels: int, batch: int) -> dict:
    output = Path(args.output).with_suffix(f".candidate_{dim}_{channels}_{batch}.json")
    command = [
        sys.executable,
        "-m",
        "src.calibrate_memory",
        "--worker",
        "--data-dir",
        args.data_dir,
        "--output",
        str(output),
        "--dim",
        str(dim),
        "--channels",
        str(channels),
        "--train-batch",
        str(batch),
        "--eval-batch",
        str(max(256, min(512, batch * 2))),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    completed = subprocess.run(command, env=env, text=True, capture_output=True)
    if completed.returncode != 0:
        return {
            "dim": dim,
            "channels": channels,
            "train_batch": batch,
            "status": "oom" if "out of memory" in (completed.stderr + completed.stdout).lower() else "failed",
            "stderr": completed.stderr[-4000:],
        }
    return json.loads(output.read_text(encoding="utf-8"))


def auto(args: argparse.Namespace) -> None:
    trials = []
    for dim, channels in ((256, 64), (128, 32), (512, 128)):
        first = run_candidate(args, dim, channels, 256)
        trials.append(first)
        peak = first.get("peak_allocated_mb", float("inf"))
        if first.get("status") is None and LOW_MB <= peak <= HARD_HIGH_MB:
            selected = first
            break
        batches = (384, 512) if first.get("status") is None and peak < LOW_MB else (192, 128)
        selected = None
        for batch in batches:
            trial = run_candidate(args, dim, channels, batch)
            trials.append(trial)
            trial_peak = trial.get("peak_allocated_mb", float("inf"))
            if trial.get("status") is None and LOW_MB <= trial_peak <= HARD_HIGH_MB:
                selected = trial
                break
        if selected is not None:
            break
    else:
        selected = None

    if selected is None:
        successful = [trial for trial in trials if trial.get("status") is None and "peak_allocated_mb" in trial]
        if successful:
            selected = min(successful, key=lambda trial: abs(float(trial["peak_allocated_mb"]) - 6144.0))
    in_target = bool(selected and LOW_MB <= float(selected["peak_allocated_mb"]) <= HARD_HIGH_MB)
    payload = {
        "status": "selected_in_target" if in_target else ("selected_outside_target" if selected else "no_usable_profile"),
        "target_mb": [LOW_MB, TARGET_HIGH_MB],
        "hard_range_mb": [LOW_MB, HARD_HIGH_MB],
        "selected": selected,
        "trials": trials,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if selected is None:
        raise SystemExit("no calibration profile completed successfully")
    print(json.dumps(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a single global v1.5 memory profile")
    parser.add_argument("--data-dir", default="data/ICEWS14")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--train-batch", type=int, default=256)
    parser.add_argument("--eval-batch", type=int, default=512)
    parser.add_argument("--history-len", type=int, default=10)
    args = parser.parse_args()
    if args.worker:
        result = worker(args)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
    else:
        auto(args)


if __name__ == "__main__":
    main()
