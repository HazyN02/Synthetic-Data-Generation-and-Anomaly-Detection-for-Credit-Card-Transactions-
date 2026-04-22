# 3. Method

> **Locked (canonical).** This section is the source of truth for §3 in the camera-ready paper.

## 3.1 Dataset and temporal evaluation

**Data.** We use the **IEEE-CIS Fraud Detection** training split (Kaggle): real e‑commerce transactions from Vesta, **≈590 k** rows, **≈3.5%** fraudulent (`isFraud = 1`)—the same **minority‑class** regime as in §1. We do **not** use the Kaggle **test** file (labels withheld); all scores are on **held‑out blocks** carved from the training data. The time column **`TransactionDT`** is a **monotonically increasing** offset in **seconds** (competition‑relative, not a calendar clock); it **orders** rows. The target is **binary fraud**.

**Temporal folds (expanding window).** We sort by `TransactionDT` and divide the timeline into **\(K{+}1\)** **equal‑sized** row blocks (\(K=4\) folds). **Fold** \(i\) **trains** on the **concatenation** of blocks **0 through \(i\)** and **validates** on block **\(i{+}1\)**. So the training set **grows** with later folds; validation is always the **next** unseen segment—**strictly in the future** of that fold’s training data. There is **no random shuffling**. **Every** method shares the **same** fold definitions.

**Preprocessing and leakage.** Feature construction uses one **shared** pipeline (§3.2 below). The pipeline is **fit only on the training rows of each fold** and **applied** to validation—**never** fit on validation or on the full table before splitting. That blocks **leakage** from **global** statistics that would **peek** at future transactions.

**Where synthetic data appears.** SMOTE and generative models run **only** on **training** rows **after** preprocessing. **Synthetic fraud rows are never added to validation**; reported metrics are always **real** transactions at **true** labels.

**Label delay.** To mimic **label latency**, we remove training rows that are **too recent** relative to validation. Let \(t_{\mathrm{val\_start}} = \min(\texttt{TransactionDT})\) on the validation block. For delay \(\delta\) **days**, we keep only training rows with \(\texttt{TransactionDT} \le t_{\mathrm{val\_start}} - \delta \times 86400\) **seconds**. (Equivalently: a **gap** of \(\delta\) days between the **end** of usable training time and the **start** of validation.) We report \(\delta \in \{0, 7, 14\}\). If too few rows or **fewer than 50** training frauds remain, we **skip** that fold.

## 3.2 Preprocessing and downstream classifier

**Features.** We **drop** high‑cardinality / sparse columns, **hash or group** selected categoricals (e.g. email‑domain fields), and keep **≈100** numerical/categorical columns suitable for **LightGBM** and the **generators**. Categorical handling is aligned between **tree** training and **synthetic** pipelines where required.

**Classifier (fixed across methods).** All oversampling conditions use the **same** **LightGBM** hyperparameters so that differences reflect **data augmentation**, not a **different learner**. Our implementation uses **`n_estimators=300`**, **`learning_rate=0.05`**, **`num_leaves=64`**, **`min_child_samples=200`**, **`subsample=0.8`**, **`colsample_bytree=0.8`**, **`objective=binary`**, **`random_state=42`** (see appendix for a one‑line table). The **baseline** is also **class‑weighted** via **`scale_pos_weight = n_neg / n_pos`** while keeping **real** labels only and **no** synthetic oversampling.

## 3.3 Oversampling protocols

**Target fraud rates.** For SMOTE, CTGAN, and TabDDPM we vary the **post‑augmentation** minority proportion via a **target positive rate** \(\in \{5\%, 10\%, 20\%\}\). **Main** tables in this paper **aggregate** results by taking, **per fold and method**, the **best** validation **PR‑AUC** across that grid (same validation fold used for scoring). **5%** is the **canonical** setting we emphasize for **label‑delay** runs and for **interpretability**; sensitivity to the rate is part of the **experimental** design.

**SMOTE [1].** **Synthetic Minority Over‑sampling** runs in the **preprocessed** feature space. We use **`k_neighbors=5`** (capped by minority count − 1), **`random_state=42`**, and optionally cap synthetic count (**`max_synth`**) for speed. SMOTE **only** touches **training** rows.

**CTGAN [2].** We fit the **ctgan** implementation on **fraud (positive) training** rows only in the **generator** feature space, then **sample** synthetic fraud until the **target** positive rate is reached, and **concatenate** with **real** training data before fitting LightGBM. In our **canonical** (medium) protocol runs used for the main tables, CTGAN uses **`epochs=7`**, **`batch_size=512`**, **`pac=1`**, **`seed=0`**. **GPU** is disabled for reproducibility across environments.

**TabDDPM [3].** We train a **Gaussian** tabular diffusion model on **positive** training rows, **sample** synthetic positives to the **same** target rates, then train LightGBM. In our **canonical** (medium) protocol runs used for the main tables, TabDDPM uses **`timesteps=75`**, **`epochs=4`**, **`hidden_dims=[768,768]`**, **Adam** **lr** \(10^{-4}\) and **batch_size=1024** (gradient clip as in the implementation).

To address concerns about undertraining, we ran a **max-convergence sanity check** with much larger budgets (**CTGAN epochs=50**, **TabDDPM epochs=50**). Mean PR-AUC shifts remained small and inconsistent, suggesting that the small effect sizes in the main protocol are not driven solely by insufficient generator training time.

**Recency ablation (\(\rho=0.3\)).** For each generator and for SMOTE, we restrict **positive** training examples to the **latest 30%** by `TransactionDT` (with a **minimum** count fallback to **all** positives). **Negatives** stay **unchanged** for SMOTE; generators see only **recent** fraud for training.

## 3.4 Drift quantification

We train a **domain classifier** (**LightGBM**) to separate **training** (domain 0) from **validation** (domain 1), **excluding** pure time/ID features. We report **ROC‑AUC** (**domain AUC**): **~0.5** suggests **little** shift; **higher** values indicate **stronger** covariate shift. We relate this to **per‑fold** changes in **PR‑AUC** vs. the baseline (**exploratory**, **four** folds).

## 3.5 Evaluation metrics and statistics

**Primary:** **PR‑AUC** (Average Precision)—**threshold‑free** and standard under **extreme** imbalance.

**Operating point:** **Recall at 1% FPR**—fraction of frauds caught when the **false‑positive rate** on transactions is **1%**.

We report **mean ± std** **across** folds. **Method** comparisons use **paired permutation tests** over folds (**§4 Experiments**).
