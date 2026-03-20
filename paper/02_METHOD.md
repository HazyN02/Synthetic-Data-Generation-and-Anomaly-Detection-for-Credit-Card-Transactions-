# 2. Method

## 2.1 Dataset and Temporal Evaluation

- **Dataset**: IEEE‑CIS Fraud Detection (Kaggle; real e‑commerce transactions from Vesta), ≈590k transactions with ≈3.5% labeled as fraudulent (`isFraud = 1`).
- **Time variable**: `TransactionDT`, a monotonically increasing time delta in seconds (not a calendar timestamp) that orders transactions.
- **Target**: Binary fraud label `isFraud`.

We follow the Fraud Detection Handbook’s recommendation to **respect time in both training and validation**. Let transactions be ordered by `TransactionDT`. **Preprocessing is fit per fold on the training block only** and then applied to the validation block, avoiding leakage from global preprocessing before folding. We construct **four temporal folds**; in fold \(i\), we train on an initial prefix and evaluate on the immediately following block:

- Fold \(i\) trains on transactions up to time \(t_i\) and validates on transactions in \((t_i, t_{i+1}]\).
- There is **no random shuffling**: validation always lies strictly in the future of training.
- All methods—baseline and every oversampling variant—use the *same* fold boundaries.

To mimic real‑world label latency, we also implement a **label‑delay variant** of this protocol. For a given delay \(\delta_{delay}\) (in days), we:

- Identify the start time of the validation block, \(t_{val\_start}\).
- Remove from the training set any transaction with `TransactionDT > t_{val_start} − δ_delay` (i.e., drop the most recent \(\delta_{delay}\) days of training data).

This mirrors the Handbook’s \(\delta_{train}\) / \(\delta_{delay}\) / \(\delta_{valid}\) setup: training uses only sufficiently old, labeled transactions; the validation block simulates future test data. We run a **7‑day delay** on all folds in “medium” mode, and a **14‑day delay** as a smaller “quick” check (fewer folds), and report whenever folds become too sparse after applying the delay.

## 2.2 Models and Oversampling Protocols

All classifiers are **LightGBM** models trained with the same hyperparameters across methods. Before modeling, we apply a **single shared preprocessing pipeline**:

- Drop high‑cardinality and sparsely populated columns.
- Apply hashing or grouping to email domains and selected categorical features.
- Limit the feature set to ≈100 mixed numerical/categorical columns suitable for both tree models and tabular generators.

This pipeline is **fit only on the training data in each fold** and then applied to validation data, preventing temporal leakage from feature engineering.

Within each fold, after preprocessing:

| Method | Description |
|--------|-------------|
| **Baseline** | LightGBM trained on real data only (no explicit oversampling), using class weights as in standard IEEE‑CIS practice. |
| **SMOTE** | Interpolation‑based Synthetic Minority Over‑sampling (on the 100‑feature space). We target several post‑oversampling fraud rates (e.g., 5%, 10%, 20%); **5%** is our canonical setting and the one emphasized in the main tables. |
| **CTGAN** | Conditional Tabular GAN trained **only on positive (fraud) examples** in the preprocessed space. We sample synthetic positives to reach the same target fraud rates as SMOTE and then concatenate them with real data before fitting LightGBM. |
| **TabDDPM** | Tabular diffusion model (TabDDPM) trained on positive examples only, analogously to CTGAN, and used to generate additional synthetic fraud cases up to the same target rates. |

### Recency‑Aware Synthesis Ablation

To test the intuition that **recent frauds are more relevant than older ones**, we implement a **recency‑aware oversampling ablation**:

- For CTGAN/TabDDPM, we restrict the generator’s training set to the most recent fraction \(\rho\) of positive examples in the fold (sorted by `TransactionDT`), keeping all negatives unchanged.
- For SMOTE, we down‑weight history by keeping all negatives but **only** the most recent fraction \(\rho\) of positives before running SMOTE in the feature space.

In practice we use \(\rho = 0.3\) (the latest 30% of positives) with a safeguard that falls back to all positives if this would leave too few fraud cases. These “recency” variants share everything else—features, folds, classifier, target fraud rate—with their non‑recency counterparts. They are treated as **ablations**, not stand‑alone methods.

## 2.3 Drift Quantification

We quantify **distribution shift between train and validation blocks** via a **domain classifier AUC**:

- For each fold, we label training transactions as domain 0 and validation transactions as domain 1, excluding purely temporal/ID columns (such as `TransactionDT`, transaction IDs, and obvious row counters).
- We train a LightGBM classifier to distinguish domain 0 vs. 1 and report its ROC‑AUC (“domain AUC”).
- A domain AUC close to 0.5 indicates little shift; higher values indicate stronger covariate shift.

We then examine, fold by fold, how oversampling impacts PR‑AUC and Recall@1% FPR as a function of this domain AUC. Given that we have **only four temporal folds**, all such correlations are interpreted as **exploratory patterns**, not precise causal statements.

## 2.4 Evaluation Metrics

We focus on metrics standard in fraud detection with extreme imbalance:

- **PR‑AUC (Average Precision)** on transactions: summarizes the full precision–recall curve and is more informative than ROC‑AUC under strong imbalance.
- **Recall@1% FPR**: the fraction of actual frauds caught when we constrain the transaction‑level false positive rate to 1%; this approximates the operating region relevant for production fraud systems.

All metrics are reported **per‑fold** and averaged across folds; confidence intervals are obtained by simple fold‑to‑fold variation (we do not bootstrap individual transactions). Whenever a configuration (e.g., large label delay plus strong recency restriction) leaves too few frauds to train a meaningful model, we explicitly mark that fold as **skipped** rather than silently aggregating over it.

