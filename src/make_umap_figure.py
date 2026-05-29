# src/make_umap_figure.py
"""
UMAP visualization of fraud clusters with CTGAN synthetic overlay.

Self-contained: loads raw data, preprocesses, recomputes cluster labels from
cluster_per_fold_hdbscan.csv, regenerates CTGAN fold-4 synthetics, routes
them to clusters via nearest-neighbour, runs UMAP, plots.

Outputs:
  figures/fraud_cluster_umap.pdf
  figures/fraud_cluster_umap.png
"""
from __future__ import annotations
import os, sys, gc
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.folds import get_temporal_folds
from src.preprocess_synth import (
    TARGET_COL, TIME_COL, PRIORITY_COLS,
    get_cat_cols_for_synth, preprocess_for_synth,
)
from src.synth_ctgan import make_synthetic_positives

PARQUET_PATH = os.path.join(_ROOT, "data", "train_merged.parquet")
RUN_DIR      = os.path.join(_ROOT, "results", "protocol", "run_20260330_180216")
UMAP_CACHE   = os.path.join(_ROOT, "results", "umap_embedding_fraud.npy")
FIG_DIR      = os.path.join(_ROOT, "figures")
FOLD         = 4
N_FOLDS      = 8
COLS_TO_READ = 100

os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("[UMAP] Loading parquet...")
import pyarrow.parquet as pq
pf = pq.ParquetFile(PARQUET_PATH)
all_cols = pf.schema_arrow.names
selected = list(dict.fromkeys(
    [TIME_COL, TARGET_COL] +
    [c for c in PRIORITY_COLS if c in all_cols] +
    [c for c in all_cols]
))[:COLS_TO_READ + 2]
selected = list(dict.fromkeys(selected))
df_raw = pf.read(columns=selected).to_pandas()
print(f"[UMAP] df shape={df_raw.shape}")

# ── 2. Temporal folds ─────────────────────────────────────────────────────────
folds = get_temporal_folds(df_raw, n_folds=N_FOLDS, time_col=TIME_COL)
fold_info = folds[FOLD]
train_df  = fold_info["train_df"]

# ── 3. Preprocess for synthesis, extract fraud ────────────────────────────────
print("[UMAP] Preprocessing full dataset for fraud feature matrix...")
full_proc, used_cols = preprocess_for_synth(df_raw)
cat_cols = get_cat_cols_for_synth(full_proc, used_cols)
cont_cols = [c for c in used_cols if c not in cat_cols and c != TARGET_COL]

fraud_all = full_proc[full_proc[TARGET_COL] == 1].copy()
print(f"[UMAP] Total fraud rows: {len(fraud_all)}, cont_cols: {len(cont_cols)}")

# Scale continuous features for UMAP + distance computation
scaler = StandardScaler()
fraud_matrix = scaler.fit_transform(fraud_all[cont_cols].fillna(0).values)
print(f"[UMAP] fraud_matrix shape: {fraud_matrix.shape}")

# ── 4. Load cluster labels ────────────────────────────────────────────────────
print("[UMAP] Loading cluster assignments...")
cluster_csv = os.path.join(RUN_DIR, "cluster_per_fold_hdbscan.csv")
cdf = pd.read_csv(cluster_csv)
fraud_assign = cdf[cdf["record_type"] == "fraud_cluster_assignment"].copy()
fraud_assign = fraud_assign.dropna(subset=["global_row_idx", "cluster_id"])
fraud_assign = fraud_assign.sort_values("global_row_idx")

# global_row_idx are indices into the time-sorted full_proc fraud rows
# Align: fraud_all is already in time-sorted order (from full_proc which was
# built from df_raw sorted by TIME_COL in preprocess_for_synth)
global_idx = fraud_assign["global_row_idx"].astype(int).values
cluster_labels_raw = fraud_assign["cluster_id"].astype(int).values

# Reindex fraud_matrix to match global_row_idx ordering
# fraud_all has a reset_index — map global_row_idx to row positions
fraud_all_reset = fraud_all.reset_index(drop=True)
# global_row_idx maps into the original full_proc index
# Build mapping: full_proc index -> position in fraud_all
full_proc_fraud_idx = fraud_all.index.values  # original full_proc row indices

# Map global_row_idx to positions in fraud_matrix
idx_map = {orig_idx: pos for pos, orig_idx in enumerate(full_proc_fraud_idx)}
valid_mask = np.array([g in idx_map for g in global_idx])
valid_global = global_idx[valid_mask]
cluster_labels = cluster_labels_raw[valid_mask]
fraud_positions = np.array([idx_map[g] for g in valid_global])

fraud_matrix_aligned = fraud_matrix[fraud_positions]
print(f"[UMAP] Aligned fraud matrix: {fraud_matrix_aligned.shape}, "
      f"clusters: {np.unique(cluster_labels)}")

# ── 5. Compute / load UMAP embedding ─────────────────────────────────────────
if os.path.exists(UMAP_CACHE):
    print(f"[UMAP] Loading cached embedding from {UMAP_CACHE}")
    embedding = np.load(UMAP_CACHE)
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1,
                             n_components=2, random_state=42, verbose=True)
    reducer.fit(fraud_matrix_aligned)
else:
    print("[UMAP] Computing UMAP embedding (n=20663, may take ~5 min)...")
    import umap as umap_lib
    reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1,
                             n_components=2, random_state=42, verbose=True)
    embedding = reducer.fit_transform(fraud_matrix_aligned)
    np.save(UMAP_CACHE, embedding)
    print(f"[UMAP] Saved embedding to {UMAP_CACHE}")

# ── 6. Regenerate CTGAN synthetics for fold 4 ────────────────────────────────
print(f"\n[UMAP] Regenerating CTGAN synthetics for fold {FOLD}...")
train_proc, _ = preprocess_for_synth(train_df)
synth_df = make_synthetic_positives(
    train_df=train_proc,
    cat_cols=cat_cols,
    used_cols=used_cols,
    target_pos_rate=0.05,
    epochs=150,
    batch_size=500,
    discriminator_steps=5,
    pac=1,
    seed=0,
    verbose=True,
)
print(f"[UMAP] Generated {len(synth_df)} synthetic fraud rows")

# Scale synthetics with same scaler
synth_matrix = scaler.transform(
    synth_df[[c for c in cont_cols if c in synth_df.columns]].fillna(0).values
)
# Align columns (scaler expects exactly cont_cols)
synth_cont = pd.DataFrame(0.0, index=range(len(synth_df)), columns=cont_cols)
for c in cont_cols:
    if c in synth_df.columns:
        synth_cont[c] = synth_df[c].values
synth_matrix = scaler.transform(synth_cont.fillna(0).values)

# ── 7. Route synthetics to clusters via nearest-neighbour ────────────────────
print("[UMAP] Routing synthetics to clusters...")
nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=1)
nn.fit(fraud_matrix_aligned)
distances, nn_idx = nn.kneighbors(synth_matrix)
synth_routed_labels = cluster_labels[nn_idx.flatten()]
print(f"[UMAP] Routing distribution: "
      + str({int(c): int((synth_routed_labels == c).sum())
             for c in np.unique(synth_routed_labels)}))

# ── 8. Project synthetics into UMAP space ────────────────────────────────────
print("[UMAP] Projecting synthetics into UMAP space...")
synth_embedding = reducer.transform(synth_matrix)

# ── 9. Plot ───────────────────────────────────────────────────────────────────
print("[UMAP] Plotting...")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cluster_colors = {
    -1: "#999999",  # noise — grey
     0: "#7B2D8B",  # cluster 0 — purple
     1: "#2CA02C",  # cluster 1 — green
     2: "#FF7F0E",  # cluster 2 — orange
     3: "#1F77B4",  # cluster 3 — blue
}
cluster_names = {
    -1: "Noise (−1)",
     0: "Cluster 0",
     1: "Cluster 1",
     2: "Cluster 2",
     3: "Cluster 3",
}

fig, ax = plt.subplots(figsize=(8, 6))

# Real fraud (small dots, semi-transparent)
for c in sorted(set(cluster_labels)):
    mask = cluster_labels == c
    ax.scatter(
        embedding[mask, 0], embedding[mask, 1],
        c=cluster_colors[c], s=3, alpha=0.4,
        rasterized=True,
    )

# Synthetic samples (cross markers, larger, colored by routed cluster)
for c in sorted(set(synth_routed_labels)):
    mask = synth_routed_labels == c
    ax.scatter(
        synth_embedding[mask, 0], synth_embedding[mask, 1],
        c=cluster_colors.get(c, "black"),
        s=18, marker="x", linewidths=0.9, alpha=0.75,
        rasterized=True,
    )

# ── Legend ────────────────────────────────────────────────────────────────────
real_handles = [
    mpatches.Patch(color=cluster_colors[c],
                   label=f"Real fraud — {cluster_names[c]} "
                         f"(n={int((cluster_labels==c).sum()):,})")
    for c in sorted(set(cluster_labels))
]

routed_unique = sorted(set(synth_routed_labels))
synth_handles = [
    plt.Line2D([0], [0], marker="x", color=cluster_colors.get(c, "black"),
               markersize=6, linewidth=0,
               label=f"CTGAN synth → {cluster_names.get(c, str(c))} "
                     f"(n={int((synth_routed_labels==c).sum()):,})")
    for c in routed_unique
]

# Note for clusters with zero synthetics
zero_clusters = [c for c in sorted(set(cluster_labels))
                 if c not in routed_unique and c != -1]
zero_handles = [
    mpatches.Patch(facecolor=cluster_colors[c], edgecolor="black",
                   linewidth=0.5, alpha=0.4,
                   label=f"CTGAN synth → {cluster_names[c]} (n=0, routing gap)")
    for c in zero_clusters
]

ax.legend(
    handles=real_handles + synth_handles + zero_handles,
    loc="upper right", fontsize=7.5, framealpha=0.85,
    handlelength=1.5,
)

n_synth = len(synth_df)
ax.set_xlabel("UMAP dimension 1", fontsize=11)
ax.set_ylabel("UMAP dimension 2", fontsize=11)
ax.set_title(
    f"Fraud cluster structure and CTGAN synthetic allocation (fold {FOLD})\n"
    f"Real fraud n=20,663 · CTGAN synthetic n={n_synth:,} · "
    f"Clusters 0 and 2 receive zero synthetic samples",
    fontsize=10,
)
ax.tick_params(labelsize=9)

plt.tight_layout()
pdf_path = os.path.join(FIG_DIR, "fraud_cluster_umap.pdf")
png_path = os.path.join(FIG_DIR, "fraud_cluster_umap.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[UMAP] Saved:\n  {pdf_path}\n  {png_path}")
print("Send fraud_cluster_umap.png to Claude to review the figure.")
