#!/usr/bin/env python3
"""Normalize results/protocol/results.csv: handle mixed 14 vs 16 column rows."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "results" / "protocol" / "results.csv"
BACKUP = CSV_PATH.with_suffix(".csv.bak")

HEADER_16 = "timestamp,fold,delay_days,run_id,method,target_pos_rate,train_rows,val_rows,train_pos,train_neg,synth_rows,final_train_rows,final_pos_rate,pr_auc,recall_at_1pct_fpr,notes"


def main():
    if not CSV_PATH.exists():
        print(f"Not found: {CSV_PATH}")
        sys.exit(1)

    rows = []
    with open(CSV_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
        n_header = len(header)
        for i, row in enumerate(reader):
            n = len(row)
            if n == 16:
                rows.append(row)
            elif n == 14:
                # Insert delay_days, run_id (empty) after fold (index 2)
                rows.append(row[:2] + ["", ""] + row[2:])
            else:
                print(f"Skip line {i+2}: {n} fields (expected 14 or 16)")
    assert len(rows) > 0, "No rows to write"

    # Backup
    import shutil
    shutil.copy2(CSV_PATH, BACKUP)
    print(f"Backed up to {BACKUP}")

    with open(CSV_PATH, "w", newline="") as f:
        f.write(HEADER_16 + "\n")
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)
    print(f"Normalized {len(rows)} rows -> {CSV_PATH}")


if __name__ == "__main__":
    main()
