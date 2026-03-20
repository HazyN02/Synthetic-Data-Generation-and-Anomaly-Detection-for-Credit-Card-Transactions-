# src/results.py
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RunRecord:
    timestamp: str
    fold: int
    method: str                      # "baseline" | "ctgan"
    target_pos_rate: Optional[float] # None for baseline
    train_rows: int
    val_rows: int
    train_pos: int
    train_neg: int
    synth_rows: int
    final_train_rows: int
    final_pos_rate: float
    pr_auc: float
    recall_at_1pct_fpr: float
    notes: str = ""


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_config(config: Dict[str, Any], out_dir: str = "results") -> str:
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def append_metrics(record: RunRecord, out_dir: str = "results") -> str:
    _ensure_dir(out_dir)
    path = os.path.join(out_dir, "metrics.csv")

    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(record).keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(asdict(record))
    return path


def now_ts() -> str:
    # ISO-ish, filesystem friendly
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
