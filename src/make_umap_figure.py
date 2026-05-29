# src/make_umap_figure.py
"""
Two-panel UMAP figure: CTGAN vs TVAE synthetic allocation over fraud clusters.

Left panel  — CTGAN (150 ep): broad routing across clusters 1, 3, noise
Right panel — TVAE  (300 ep): mode-collapse onto cluster 1

All synthetics generated at r=0.05, fold 4 (representative mid-run fold).
Synthetic samples and UMAP embedding are cached for fast re-runs.

Outputs:
  figures/fraud_cluster_umap.pdf   (300 dpi, vector)
  figures/fraud_cluster_umap.png   (150 dpi, raster)
"""
from __future__ import annotations
import os, sys, gc, pickle

# Force single-threaded joblib/loky — prevents OOM from parallel worker spawning
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"]     = "1"
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
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
from src.synth_tvae import fit_tvae, sample_tvae

PARQUET_PATH    = os.path.join(_ROOT, "data", "train_merged.parquet")
RUN_DIR         = os.path.join(_ROOT, "results", "protocol", "run_20260330_180216")
UMAP_CACHE      = os.path.join(_ROOT, "results", "umap_embedding_fraud.npy")
CTGAN_CACHE     = os.path.join(_ROOT, "results", "umap_ctgan_synth_fold4.pkl")
TVAE_CACHE      = os.path.join(_ROOT, "results", "umap_tvae_synth_fold4.pkl")
FIG_DIR         = os.path.join(_ROOT, "figures")

FOLD            = 4
N_FOLDS         = 8
CTGAN_EPOCHS    = 150
TVAE_EPOCHS     = 300
TARGET_RATE     = 0.05
COLS_TO_READ    = 100

os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour / label scheme (shared by both panels) ────────────────────────────
CLUSTER_COLORS = {-1: "#999999", 0: "#7B2D8B", 1: "#2CA02C", 2: "#FF7F0E", 3: "#1F77B4"}
CLUSTER_NAMES  = {-1: "Noise (-1)", 0: "Cluster 0", 1: "Cluster 1",
                   2: "Cluster 2",  3: "Cluster 3"}

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load & preprocess
# ═══════════════════════════════════════════════════════════════════════════════
print("[UMAP] Loading parquet...")
import pyarrow.parquet as pq
pf = pq.ParquetFile(PARQUET_PATH)
all_cols = pf.schema_arrow.names
selected = list(dict.fromkeys(
    [TIME_COL, TARGET_COL] + [c for c in PRIORITY_COLS if c in all_cols] + all_cols
))[:COLS_TO_READ + 2]
df_raw = pf.read(columns=selected).to_pandas()
print(f"[UMAP] df shape={df_raw.shape}")

folds     = get_temporal_folds(df_raw, n_folds=N_FOLDS, time_col=TIME_COL)
train_df  = folds[FOLD]["train_df"]

print("[UMAP] Preprocessing...")
full_proc, used_cols = preprocess_for_synth(df_raw)
cat_cols  = get_cat_cols_for_synth(full_proc, used_cols)
cont_cols = [c for c in used_cols if c not in cat_cols and c != TARGET_COL]

fraud_all    = full_proc[full_proc[TARGET_COL] == 1].copy()
scaler       = StandardScaler()
fraud_matrix = scaler.fit_transform(fraud_all[cont_cols].fillna(0).values)
print(f"[UMAP] fraud_matrix={fraud_matrix.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cluster labels
# ═══════════════════════════════════════════════════════════════════════════════
cdf          = pd.read_csv(os.path.join(RUN_DIR, "cluster_per_fold_hdbscan.csv"))
fraud_assign = (cdf[cdf["record_type"] == "fraud_cluster_assignment"]
                .dropna(subset=["global_row_idx", "cluster_id"])
                .sort_values("global_row_idx"))
global_idx        = fraud_assign["global_row_idx"].astype(int).values
cluster_labels_raw = fraud_assign["cluster_id"].astype(int).values

idx_map      = {orig: pos for pos, orig in enumerate(fraud_all.index.values)}
valid        = np.array([g in idx_map for g in global_idx])
cluster_labels    = cluster_labels_raw[valid]
fraud_positions   = np.array([idx_map[g] for g in global_idx[valid]])
fraud_matrix_al   = fraud_matrix[fraud_positions]
print(f"[UMAP] Aligned: {fraud_matrix_al.shape}, clusters={np.unique(cluster_labels)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. UMAP — load cached embedding, refit reducer for transform()
# ═══════════════════════════════════════════════════════════════════════════════
import umap as umap_lib

print("[UMAP] Fitting UMAP reducer (needed for transform)...")
reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1, n_components=2,
                         random_state=42, verbose=False)

if os.path.exists(UMAP_CACHE):
    embedding = np.load(UMAP_CACHE)
    reducer.fit(fraud_matrix_al)          # refit to enable .transform()
    print(f"[UMAP] Loaded cached embedding {embedding.shape}")
else:
    embedding = reducer.fit_transform(fraud_matrix_al)
    np.save(UMAP_CACHE, embedding)
    print(f"[UMAP] Computed & cached embedding {embedding.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. NN index on real fraud (shared by CTGAN & TVAE routing)
# ═══════════════════════════════════════════════════════════════════════════════
nn = NearestNeighbors(n_neighbors=1, metric="euclidean", n_jobs=1)
nn.fit(fraud_matrix_al)

def _scale_and_route(synth_df):
    """Scale synth cont cols with shared scaler, route via NN, project into UMAP."""
    sc = pd.DataFrame(0.0, index=range(len(synth_df)), columns=cont_cols)
    for c in cont_cols:
        if c in synth_df.columns:
            sc[c] = synth_df[c].values
    mat = scaler.transform(sc.fillna(0).values)
    _, idx = nn.kneighbors(mat)
    labels = cluster_labels[idx.flatten()]
    proj   = reducer.transform(mat)
    dist   = nn.kneighbors(mat)[0].flatten()
    return mat, labels, proj, dist

# ═══════════════════════════════════════════════════════════════════════════════
# 5. CTGAN synthetics (cached)
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(CTGAN_CACHE):
    print("[UMAP] Loading cached CTGAN synthetics...")
    with open(CTGAN_CACHE, "rb") as f:
        ctgan_data = pickle.load(f)
    ctgan_df, ctgan_labels, ctgan_proj = (
        ctgan_data["df"], ctgan_data["labels"], ctgan_data["proj"])
else:
    print(f"[UMAP] Training CTGAN ({CTGAN_EPOCHS} ep, fold {FOLD})...")
    train_proc, _ = preprocess_for_synth(train_df)
    ctgan_df = make_synthetic_positives(
        train_df=train_proc, cat_cols=cat_cols, used_cols=used_cols,
        target_pos_rate=TARGET_RATE, epochs=CTGAN_EPOCHS,
        batch_size=500, discriminator_steps=5, pac=1, seed=0, verbose=True,
    )
    _, ctgan_labels, ctgan_proj, _ = _scale_and_route(ctgan_df)
    with open(CTGAN_CACHE, "wb") as f:
        pickle.dump({"df": ctgan_df, "labels": ctgan_labels, "proj": ctgan_proj}, f)
    print(f"[UMAP] CTGAN: {len(ctgan_df)} rows, routing={dict(zip(*np.unique(ctgan_labels, return_counts=True)))}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TVAE synthetics (cached)
# ═══════════════════════════════════════════════════════════════════════════════
if os.path.exists(TVAE_CACHE):
    print("[UMAP] Loading cached TVAE synthetics...")
    with open(TVAE_CACHE, "rb") as f:
        tvae_data = pickle.load(f)
    tvae_df, tvae_labels, tvae_proj = (
        tvae_data["df"], tvae_data["labels"], tvae_data["proj"])
else:
    print(f"[UMAP] Training TVAE ({TVAE_EPOCHS} ep, fold {FOLD})...")
    train_proc, _ = preprocess_for_synth(train_df)
    fraud_train = train_proc[train_proc[TARGET_COL] == 1]
    synth_add   = max(0, int(
        (len(train_proc[train_proc[TARGET_COL]==0]) * TARGET_RATE / (1 - TARGET_RATE))
        - len(fraud_train)
    ))
    synthesizer, artifacts = fit_tvae(
        fraud_train[used_cols], cat_cols=cat_cols, used_cols=used_cols,
        epochs=TVAE_EPOCHS, batch_size=500, seed=0, verbose=True,
    )
    tvae_df = sample_tvae(synthesizer, n=synth_add, artifacts=artifacts, verbose=True)
    del synthesizer; gc.collect()
    _, tvae_labels, tvae_proj, _ = _scale_and_route(tvae_df)
    with open(TVAE_CACHE, "wb") as f:
        pickle.dump({"df": tvae_df, "labels": tvae_labels, "proj": tvae_proj}, f)
    print(f"[UMAP] TVAE: {len(tvae_df)} rows, routing={dict(zip(*np.unique(tvae_labels, return_counts=True)))}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Print routing comparison
# ═══════════════════════════════════════════════════════════════════════════════
cluster_ids = sorted(set(cluster_labels))
total_fraud = len(cluster_labels)
print("\n=== ROUTING COMPARISON (fold 4) ===")
print(f"{'Cluster':>10}  {'Real%':>6}  {'CTGAN%':>8}  {'TVAE%':>8}")
print("-" * 40)
for c in cluster_ids:
    pct_real  = (cluster_labels == c).sum() / total_fraud * 100
    pct_ctgan = (ctgan_labels == c).sum() / len(ctgan_labels) * 100 if len(ctgan_labels) else 0
    pct_tvae  = (tvae_labels  == c).sum() / len(tvae_labels)  * 100 if len(tvae_labels)  else 0
    print(f"  {CLUSTER_NAMES[c]:>10}  {pct_real:>5.1f}%  {pct_ctgan:>7.1f}%  {pct_tvae:>7.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Two-panel figure
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _draw_panel(ax, synth_proj, synth_labels, method_label, n_synth_total):
    # Real fraud (dots)
    for c in cluster_ids:
        mask = cluster_labels == c
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=CLUSTER_COLORS[c], s=3, alpha=0.35, rasterized=True,
                   linewidths=0)

    # Synthetics (crosses)
    routed_cs = sorted(set(synth_labels))
    for c in routed_cs:
        mask = synth_labels == c
        ax.scatter(synth_proj[mask, 0], synth_proj[mask, 1],
                   c=CLUSTER_COLORS.get(c, "black"),
                   s=20, marker="x", linewidths=0.9, alpha=0.8,
                   rasterized=True)

    # Per-panel routing annotation (bottom-left box)
    lines = []
    for c in cluster_ids:
        pct_synth = (synth_labels == c).sum() / n_synth_total * 100
        marker = "×" if c in routed_cs else "–"
        lines.append(f"{CLUSTER_NAMES[c]:>12}: {pct_synth:>5.1f}%  {marker}")
    annotation = "Routing:\n" + "\n".join(lines)
    ax.text(0.02, 0.02, annotation, transform=ax.transAxes,
            fontsize=7, family="monospace",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85))

    ax.set_xlabel("UMAP dimension 1", fontsize=10)
    ax.set_title(f"{method_label}\nn={n_synth_total:,} synthetic", fontsize=10)
    ax.tick_params(labelsize=8)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True, sharex=True)

_draw_panel(axes[0], ctgan_proj, ctgan_labels,
            f"CTGAN ({CTGAN_EPOCHS} ep)  —  fold {FOLD}", len(ctgan_labels))
_draw_panel(axes[1], tvae_proj,  tvae_labels,
            f"TVAE  ({TVAE_EPOCHS} ep)  —  fold {FOLD}", len(tvae_labels))

axes[0].set_ylabel("UMAP dimension 2", fontsize=10)

# Shared legend (real fraud clusters only — synthetics explained by routing box)
real_handles = [
    mpatches.Patch(color=CLUSTER_COLORS[c],
                   label=f"{CLUSTER_NAMES[c]}  (n={int((cluster_labels==c).sum()):,}, "
                         f"{int((cluster_labels==c).sum())/total_fraud*100:.1f}%)")
    for c in cluster_ids
]
synth_handle = plt.Line2D([0], [0], marker="x", color="black",
                           markersize=6, linewidth=0, label="Synthetic sample (×)")
fig.legend(handles=real_handles + [synth_handle],
           loc="upper center", ncol=3, fontsize=8,
           bbox_to_anchor=(0.5, 1.01), framealpha=0.9)

fig.suptitle(
    "Fraud cluster structure (UMAP) and synthetic allocation: CTGAN vs TVAE\n"
    "Real fraud n=20,663 · colours = HDBSCAN cluster · crosses = synthetic samples",
    fontsize=10, y=1.10,
)

plt.tight_layout()
pdf_path = os.path.join(FIG_DIR, "fraud_cluster_umap.pdf")
png_path = os.path.join(FIG_DIR, "fraud_cluster_umap.png")
plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\n[UMAP] Saved:\n  {pdf_path}\n  {png_path}")
