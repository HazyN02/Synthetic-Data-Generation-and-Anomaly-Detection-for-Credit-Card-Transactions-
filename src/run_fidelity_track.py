#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from src.fidelity.metrics import compute_fidelity_metrics
from src.fidelity.tabddpm_decode import decode_tabddpm_samples, sanitize_tabddpm_decoded
from src.folds import get_temporal_folds
from src.preprocess_synth import preprocess_fold, get_cat_cols_for_synth
import src.preprocess_synth as preprocess_mod
from src.synth_ctgan import make_synthetic_positives
from src.synth_tabddpm import fit_tabddpm, TARGET_COL


TIME_COL = "TransactionDT"


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_train_df(repo_root: str) -> pd.DataFrame:
    data_dir = os.path.join(repo_root, "data")
    for name in ("train_merged.parquet", "train_transaction.csv"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("No train data in data/ (need train_merged.parquet or train_transaction.csv)")


@contextmanager
def _preprocess_variant(hash_bins: int, max_cols: int):
    old_hash_bins = preprocess_mod.HASH_BINS
    old_max_cols = preprocess_mod.MAX_COLS
    preprocess_mod.HASH_BINS = int(hash_bins)
    preprocess_mod.MAX_COLS = int(max_cols)
    try:
        yield
    finally:
        preprocess_mod.HASH_BINS = old_hash_bins
        preprocess_mod.MAX_COLS = old_max_cols


def _run_ctgan(
    train_df: pd.DataFrame,
    used_cols: List[str],
    cat_cols: List[str],
    cfg: Dict[str, Any],
    target_pos_rate: float,
    max_synth: int,
    seed: int,
) -> pd.DataFrame:
    synth = make_synthetic_positives(
        train_df=train_df,
        cat_cols=cat_cols,
        used_cols=used_cols,
        target_pos_rate=float(target_pos_rate),
        max_synth=int(max_synth),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        pac=int(cfg["pac"]),
        seed=int(seed),
        verbose=False,
    )
    return synth[used_cols].copy()


def _run_tabddpm(
    train_df: pd.DataFrame,
    used_cols: List[str],
    cat_cols: List[str],
    cfg: Dict[str, Any],
    target_pos_rate: float,
    max_synth: int,
    seed: int,
) -> pd.DataFrame:
    pos_df = train_df[train_df[TARGET_COL] == 1].copy()
    neg_df = train_df[train_df[TARGET_COL] == 0].copy()
    n_pos_full = len(pos_df)
    n_neg = len(neg_df)
    total = n_pos_full + n_neg
    target_pos = int(np.ceil(float(target_pos_rate) * total / (1.0 - float(target_pos_rate))))
    synth_add = int(min(int(max_synth), max(0, target_pos - n_pos_full)))
    if synth_add <= 0:
        return pd.DataFrame(columns=used_cols)

    diffusion, artifacts, denoiser = fit_tabddpm(
        pos_df,
        cat_cols,
        used_cols,
        timesteps=int(cfg["timesteps"]),
        epochs=int(cfg["epochs"]),
        batch_size=1024,
        lr=float(cfg["lr"]),
        hidden_dims=list(cfg["hidden_dims"]),
        seed=int(seed),
        device="cpu",
        verbose=False,
    )
    denoiser.eval()
    device = next(denoiser.parameters()).device
    y_dist = torch.ones(1, device=device)
    with torch.no_grad():
        sample_batch = min(2048, synth_add)
        x_gen, _ = diffusion.sample_all(synth_add, sample_batch, y_dist, ddim=False)
        raw = x_gen.cpu().numpy()

    decoded = decode_tabddpm_samples(raw, artifacts)
    decoded = sanitize_tabddpm_decoded(decoded, artifacts)
    return decoded[used_cols].copy()


def _rank_score(row: pd.Series) -> float:
    # Lower is better
    return (
        abs(float(row["real_vs_synth_auc"]) - 0.5) * 2.0
        + float(row["cat_tv_distance"]) * 1.2
        + float(row["num_wasserstein_norm"]) * 0.8
        + float(row["corr_mad"]) * 0.5
        + (1.0 - float(row["schema_validity_rate"])) * 2.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separate fidelity optimization track.")
    parser.add_argument("--ctgan-config", default="configs/fidelity/ctgan_grid.json")
    parser.add_argument("--tabddpm-config", default="configs/fidelity/tabddpm_grid.json")
    parser.add_argument("--schema-config", default="configs/fidelity/base_schema_checks.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--folds", type=int, default=None, help="Override number of temporal folds for this run.")
    parser.add_argument("--max-configs-per-method", type=int, default=None, help="Optional cap on number of configs per method.")
    parser.add_argument("--max-target-rates", type=int, default=None, help="Optional cap on target rates per method.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(__file__))
    run_id = args.run_id or datetime.now().strftime("fidelity_%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, "results", "fidelity", "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    paper_out = os.path.join(root, "paper", "tables", "fidelity")
    os.makedirs(paper_out, exist_ok=True)

    ct_cfg = _load_json(os.path.join(root, args.ctgan_config))
    td_cfg = _load_json(os.path.join(root, args.tabddpm_config))
    sc_cfg = _load_json(os.path.join(root, args.schema_config))

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "ctgan_config": ct_cfg,
        "tabddpm_config": td_cfg,
        "schema_config": sc_cfg,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    df = _load_train_df(root)
    n_folds = int(args.folds) if args.folds is not None else int(ct_cfg.get("n_folds", 4))
    folds_raw = get_temporal_folds(df, n_folds=n_folds, time_col=TIME_COL)
    min_real_pos = int(sc_cfg.get("min_real_pos_per_fold", 80))
    forbidden_tokens = set(sc_cfg.get("gate_a_forbidden_tokens", ["__SYNTH__"]))

    rows: List[Dict[str, Any]] = []
    validity_rows: List[Dict[str, Any]] = []

    preprocess_variants = sc_cfg.get("preprocess_variants", [{"name": "default", "hash_bins": 20, "max_cols": 100}])
    for pv in preprocess_variants:
        pv_name = pv["name"]
        with _preprocess_variant(hash_bins=int(pv["hash_bins"]), max_cols=int(pv["max_cols"])):
            for fold_info in folds_raw:
                fold = int(fold_info["fold"])
                train_df = fold_info["train_df"]
                val_df = fold_info["val_df"]
                train_prep, val_prep, used_cols = preprocess_fold(train_df, val_df)
                cat_cols = get_cat_cols_for_synth(train_prep, used_cols)
                real_pos = train_prep[train_prep[TARGET_COL] == 1].copy()
                if len(real_pos) < min_real_pos:
                    continue

                # CTGAN sweep
                ct_configs = ct_cfg["configs"][: args.max_configs_per_method] if args.max_configs_per_method else ct_cfg["configs"]
                ct_rates = ct_cfg["target_pos_rates"][: args.max_target_rates] if args.max_target_rates else ct_cfg["target_pos_rates"]
                for cfg in ct_configs:
                    for target_pos_rate in ct_rates:
                        for max_synth in ct_cfg["max_synth_values"]:
                            synth = _run_ctgan(
                                train_df=train_prep,
                                used_cols=used_cols,
                                cat_cols=cat_cols,
                                cfg=cfg,
                                target_pos_rate=float(target_pos_rate),
                                max_synth=int(max_synth),
                                seed=int(ct_cfg["seed"]),
                            )
                            if len(synth) == 0:
                                continue
                            m = compute_fidelity_metrics(real_pos, synth, used_cols, seed=int(ct_cfg["seed"]))
                            row = {
                                "preprocess_variant": pv_name,
                                "fold": fold,
                                "method": "ctgan",
                                "config_name": cfg["name"],
                                "target_pos_rate": float(target_pos_rate),
                                "max_synth": int(max_synth),
                                **m,
                            }
                            rows.append(row)

                # TabDDPM sweep
                td_configs = td_cfg["configs"][: args.max_configs_per_method] if args.max_configs_per_method else td_cfg["configs"]
                td_rates = td_cfg["target_pos_rates"][: args.max_target_rates] if args.max_target_rates else td_cfg["target_pos_rates"]
                for cfg in td_configs:
                    for target_pos_rate in td_rates:
                        for max_synth in td_cfg["max_synth_values"]:
                            synth = _run_tabddpm(
                                train_df=train_prep,
                                used_cols=used_cols,
                                cat_cols=cat_cols,
                                cfg=cfg,
                                target_pos_rate=float(target_pos_rate),
                                max_synth=int(max_synth),
                                seed=int(td_cfg["seed"]),
                            )
                            if len(synth) == 0:
                                continue
                            bad_tokens = 0
                            for c in cat_cols:
                                if c in synth.columns:
                                    bad_tokens += int(synth[c].astype("string").isin(forbidden_tokens).sum())
                            m = compute_fidelity_metrics(real_pos, synth, used_cols, seed=int(td_cfg["seed"]))
                            row = {
                                "preprocess_variant": pv_name,
                                "fold": fold,
                                "method": "tabddpm",
                                "config_name": cfg["name"],
                                "target_pos_rate": float(target_pos_rate),
                                "max_synth": int(max_synth),
                                "forbidden_token_count": int(bad_tokens),
                                **m,
                            }
                            rows.append(row)
                            validity_rows.append(
                                {
                                    "preprocess_variant": pv_name,
                                    "fold": fold,
                                    "method": "tabddpm",
                                    "config_name": cfg["name"],
                                    "forbidden_token_count": int(bad_tokens),
                                    "schema_validity_rate": float(m["schema_validity_rate"]),
                                }
                            )

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("No fidelity rows produced in separate track.")
    all_df.to_csv(os.path.join(run_dir, "synthetic_fidelity_v2.csv"), index=False)

    agg_cols = [
        "num_quantile_l1",
        "num_wasserstein_norm",
        "cat_tv_distance",
        "cat_js_divergence",
        "corr_mad",
        "real_vs_synth_auc",
        "rare_cat_coverage",
        "schema_validity_rate",
        "n_real",
        "n_synth",
    ]
    summary = (
        all_df.groupby(["method", "preprocess_variant", "config_name", "target_pos_rate", "max_synth"], as_index=False)[agg_cols]
        .mean()
    )
    summary["rank_score"] = summary.apply(_rank_score, axis=1)
    summary = summary.sort_values(["method", "rank_score", "real_vs_synth_auc"])
    summary.to_csv(os.path.join(run_dir, "synthetic_fidelity_v2_summary.csv"), index=False)

    ct_leaderboard = summary[summary["method"] == "ctgan"].copy()
    td_leaderboard = summary[summary["method"] == "tabddpm"].copy()
    ct_leaderboard.to_csv(os.path.join(run_dir, "ctgan_fidelity_leaderboard.csv"), index=False)
    td_leaderboard.to_csv(os.path.join(run_dir, "tabddpm_fidelity_leaderboard.csv"), index=False)

    preprocess_sensitivity = (
        summary.groupby(["method", "preprocess_variant"], as_index=False)[
            ["rank_score", "real_vs_synth_auc", "cat_tv_distance", "num_wasserstein_norm", "schema_validity_rate"]
        ]
        .mean()
        .sort_values(["method", "rank_score"])
    )
    preprocess_sensitivity.to_csv(os.path.join(run_dir, "preprocess_sensitivity.csv"), index=False)

    validity_df = pd.DataFrame(validity_rows)
    if validity_df.empty:
        validity_df = all_df[["method", "preprocess_variant", "config_name", "schema_validity_rate"]].copy()
        if "forbidden_token_count" in all_df.columns:
            validity_df["forbidden_token_count"] = all_df["forbidden_token_count"].fillna(0).astype(int)
        else:
            validity_df["forbidden_token_count"] = 0
    validity_df.to_csv(os.path.join(run_dir, "synthetic_validity_checks.csv"), index=False)

    # Mirror key outputs to dedicated paper fidelity folder (separate namespace)
    summary.to_csv(os.path.join(paper_out, "synthetic_fidelity_v2_summary.csv"), index=False)
    all_df.to_csv(os.path.join(paper_out, "synthetic_fidelity_v2.csv"), index=False)
    validity_df.to_csv(os.path.join(paper_out, "synthetic_validity_checks.csv"), index=False)
    ct_leaderboard.to_csv(os.path.join(paper_out, "ctgan_fidelity_leaderboard.csv"), index=False)
    td_leaderboard.to_csv(os.path.join(paper_out, "tabddpm_fidelity_leaderboard.csv"), index=False)
    preprocess_sensitivity.to_csv(os.path.join(paper_out, "preprocess_sensitivity.csv"), index=False)

    print(f"[fidelity-track] run_dir={run_dir}")
    print(f"[fidelity-track] wrote {len(all_df)} per-fold rows")
    print("[fidelity-track] best configs:")
    for method in ("ctgan", "tabddpm"):
        top = summary[summary["method"] == method].head(1)
        if len(top):
            print(top[["method", "preprocess_variant", "config_name", "target_pos_rate", "rank_score", "real_vs_synth_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
