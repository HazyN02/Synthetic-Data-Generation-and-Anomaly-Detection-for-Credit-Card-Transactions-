# src/synth_tabddpm.py
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------
# 🔧 PERMANENT PATH FIX (DO NOT TOUCH)
# ------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
TABDDPM_PATH = os.path.join(ROOT, "external", "tab_ddpm")

if TABDDPM_PATH not in sys.path:
    sys.path.insert(0, TABDDPM_PATH)

from tab_ddpm.gaussian_multinomial_diffusion import GaussianMultinomialDiffusion

TARGET_COL = "isFraud"


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class SimpleDenoiser(nn.Module):
    """Lightweight denoiser compatible with tab_ddpm diffusion: forward(x, t, **kwargs) -> x_pred."""

    def __init__(self, dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.dim_t = 128
        self.time_embed = nn.Sequential(
            nn.Linear(self.dim_t, self.dim_t),
            nn.SiLU(),
            nn.Linear(self.dim_t, self.dim_t),
        )
        layers: List[nn.Module] = [nn.Linear(dim + self.dim_t, hidden_dims[0]), nn.SiLU(), nn.Dropout(dropout)]
        for i in range(len(hidden_dims) - 1):
            layers.extend([nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.SiLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dims[-1], dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        emb = _timestep_embedding(t, self.dim_t).to(x.dtype)
        emb = self.time_embed(emb)
        h = torch.cat([x, emb], dim=-1)
        return self.mlp(h)


# ------------------------------------------------------------------
# Artifacts (for clean sampling + reproducibility)
# ------------------------------------------------------------------
@dataclass
class TabDDPMArtifacts:
    used_cols: List[str]
    cat_cols: List[str]
    cont_cols: List[str]
    cat_maps: Dict[str, List[str]]
    cont_means: Dict[str, float]
    cont_stds: Dict[str, float]


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def _prepare_tabddpm_data(
    df: pd.DataFrame,
    used_cols: List[str],
    cat_cols: List[str],
) -> Tuple[np.ndarray, TabDDPMArtifacts]:
    """
    One-hot encode categoricals, z-score continuous features.
    """
    X = df[used_cols].copy()

    cat_cols = [c for c in cat_cols if c in X.columns]
    cont_cols = [c for c in X.columns if c not in cat_cols]

    cat_maps: Dict[str, List[str]] = {}
    one_hot_parts = []

    for c in cat_cols:
        X[c] = X[c].astype("string").fillna("__MISSING__")
        cats = sorted(X[c].unique().tolist())
        cat_maps[c] = cats
        oh = pd.get_dummies(X[c], prefix=c)
        one_hot_parts.append(oh)

    cont_means, cont_stds = {}, {}
    cont_arrs = []

    for c in cont_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        med = float(X[c].median())
        X[c] = X[c].fillna(med)
        mean = float(X[c].mean())
        std = float(X[c].std()) + 1e-6
        cont_means[c] = mean
        cont_stds[c] = std
        arr = ((X[c] - mean) / std).to_numpy().astype(np.float32)
        # Clip to avoid extreme values causing NaN in diffusion
        arr = np.clip(arr, -10.0, 10.0)
        cont_arrs.append(arr[:, None])

    X_num = np.hstack(cont_arrs) if cont_arrs else np.empty((len(X), 0))
    X_cat = pd.concat(one_hot_parts, axis=1).to_numpy() if one_hot_parts else np.empty((len(X), 0))

    X_all = np.hstack([X_num, X_cat]).astype(np.float32)
    # Final NaN/inf sanitization
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    artifacts = TabDDPMArtifacts(
        used_cols=used_cols,
        cat_cols=cat_cols,
        cont_cols=cont_cols,
        cat_maps=cat_maps,
        cont_means=cont_means,
        cont_stds=cont_stds,
    )

    return X_all, artifacts


# ------------------------------------------------------------------
# Fit TabDDPM
# ------------------------------------------------------------------
def fit_tabddpm(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    used_cols: List[str],
    timesteps: int = 100,
    epochs: int = 5,
    batch_size: int = 1024,
    lr: float = 1e-4,
    hidden_dims: List[int] | None = None,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[GaussianMultinomialDiffusion, TabDDPMArtifacts, nn.Module]:

    torch.manual_seed(seed)
    np.random.seed(seed)

    X, artifacts = _prepare_tabddpm_data(train_df, used_cols, cat_cols)
    X_tensor = torch.from_numpy(X).to(device)

    num_features = X_tensor.shape[1]

    if verbose:
        print(
            f"[TabDDPM] Training rows: {len(X_tensor)} | "
            f"features={num_features} | "
            f"cat={len(artifacts.cat_cols)} | cont={len(artifacts.cont_cols)}"
        )

    if hidden_dims is None:
        hidden_dims = [1024, 1024]
    denoiser = SimpleDenoiser(dim=num_features, hidden_dims=hidden_dims, dropout=0.1).to(device)
    diffusion = GaussianMultinomialDiffusion(
        denoise_fn=denoiser,
        num_classes=np.array([0]),  # pure Gaussian
        num_numerical_features=num_features,
        num_timesteps=timesteps,
        gaussian_loss_type="mse",
        device=torch.device(device),
    )
    diffusion._denoise_fn = denoiser

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    denoiser.train()
    for ep in range(epochs):
        losses = []
        for (x,) in loader:
            x = x.to(device)
            if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
                continue
            out_dict = {"y": torch.zeros(x.size(0), dtype=torch.long, device=device)}
            optimizer.zero_grad()
            loss_multi, loss_gauss = diffusion.mixed_loss(x, out_dict)
            loss = loss_multi + loss_gauss
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        if verbose:
            print(f"[TabDDPM] Epoch {ep+1}/{epochs} | loss={np.mean(losses):.4f}")

    return diffusion, artifacts, denoiser


def _apply_recency_tabddpm(
    pos_df: pd.DataFrame,
    recency_frac: float | None,
    time_col: str,
    min_pos: int,
    verbose: bool,
) -> pd.DataFrame:
    if recency_frac is None or recency_frac >= 1.0:
        return pos_df
    if time_col not in pos_df.columns:
        raise ValueError(f"[recency] time_col={time_col} not in dataframe")
    n_pos = len(pos_df)
    n_recent = max(min_pos, int(np.ceil(recency_frac * n_pos)))
    if n_recent >= n_pos:
        return pos_df
    pos_sorted = pos_df.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    pos_recent = pos_sorted.tail(n_recent)
    if verbose:
        print(f"[TabDDPM recency] n_pos={n_pos} -> n_recent={n_recent} (frac={recency_frac})")
    return pos_recent


# ------------------------------------------------------------------
# Sample positives
# ------------------------------------------------------------------
def make_synthetic_positives_tabddpm(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    used_cols: List[str],
    target_pos_rate: float,
    max_synth: int = 50000,
    recency_frac: float | None = None,
    time_col: str = "TransactionDT",
    min_pos_for_recency: int = 50,
    verbose: bool = True,
    fit_pos_df: pd.DataFrame | None = None,
    **fit_kwargs,
) -> pd.DataFrame:
    """
    If fit_pos_df is set, TabDDPM is fit on these rows (expected fraud-only) while
    synth_add and mixing still use the original train_df positive count (n_pos_full).
    """

    real_pos = train_df[train_df[TARGET_COL] == 1]
    n_pos_full = len(real_pos)
    neg_df = train_df[train_df[TARGET_COL] == 0]

    if fit_pos_df is not None:
        pos_df = fit_pos_df[used_cols + [TARGET_COL]].copy()
    else:
        pos_df = real_pos.copy()
        pos_df = _apply_recency_tabddpm(pos_df, recency_frac, time_col, min_pos_for_recency, verbose)

    n_pos, n_neg = len(pos_df), len(neg_df)
    total = n_pos_full + n_neg
    target_pos = int(np.ceil(target_pos_rate * total / (1.0 - target_pos_rate)))
    synth_add = min(max_synth, max(0, target_pos - n_pos_full))

    if verbose:
        print(f"[TabDDPM] n_pos={n_pos} (gen), n_pos_full={n_pos_full}, n_neg={n_neg}, synth_add={synth_add}")

    if synth_add == 0:
        return pd.DataFrame(columns=used_cols + [TARGET_COL])

    diffusion, artifacts, denoiser = fit_tabddpm(
        pos_df, cat_cols, used_cols, **fit_kwargs
    )

    denoiser.eval()
    device = next(denoiser.parameters()).device
    y_dist = torch.ones(1, device=device)
    with torch.no_grad():
        sample_batch = min(4096, synth_add)
        x_gen, _ = diffusion.sample_all(synth_add, sample_batch, y_dist, ddim=False)
        samples = x_gen[:, : len(artifacts.cont_cols)].cpu().numpy()

    # Only keep continuous part
    out = pd.DataFrame(samples, columns=artifacts.cont_cols)

    for c in artifacts.cont_cols:
        out[c] = out[c] * artifacts.cont_stds[c] + artifacts.cont_means[c]

    for c in artifacts.cat_cols:
        out[c] = "__SYNTH__"

    out[TARGET_COL] = 1
    return out[used_cols + [TARGET_COL]]
