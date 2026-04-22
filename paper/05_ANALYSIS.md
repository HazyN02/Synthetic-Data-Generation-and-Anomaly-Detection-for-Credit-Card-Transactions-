# 5. Analysis

> **Role.** §4 reports **numbers**; here we interpret **patterns** without repeating tables. Broader implications and limitations are in §6.

## 5.1 Why low-power results matter

The paired tests in §4.4 do not reach α = 0.05, despite moderate mean PR-AUC deltas versus the baseline. Under temporal evaluation, that is still informative: it characterizes what generic SMOTE, CTGAN, and TabDDPM pipelines achieve on a standard benchmark when leakage is controlled. Importantly, fold-level **n = 4** gives these tests limited power, so we treat p-values as weak evidence and instead emphasize effect sizes, fold variability, and protocol realism.

## 5.2 Fold heterogeneity

Rankings move by fold (§4.3) because validation regimes differ and training grows in the expanding window. Oversampling perturbs training only; it cannot stabilize rankings across shifts. Random train/test splits suppress this variance—one reason offline leaderboards can mislead.

## 5.3 Generators, drift, and drift-dependent effects

Domain AUC spans roughly 0.82–0.92 (§4.3). Because our drift metric is correlated with fold difficulty, the correlation sign depends on how “effect” is defined; in our tables we report deltas relative to the baseline, so negative r values (n=4) are exploratory and best read as: **higher drift is associated with larger absolute augmentation effects** (often benefits in this run), not as a stable causal relationship. A plausible hypothesis remains that generators fit the training fraud distribution; under covariate shift, synthetic points can either improve coverage or amplify mismatch, depending on how fraud modes evolve. SMOTE interpolates between real points; GANs and diffusion sample new points—different failure modes.

## 5.4 Recency

Recency restricts positives to the latest **30%**. Under temporal evaluation, this creates an asymmetry: **CTGAN-recency** and **TabDDPM-recency** remain close to their non-recency variants (mean PR‑AUC ≈ 0.569 and 0.564), whereas **SMOTE-recency** drops noticeably (mean PR‑AUC ≈ 0.552). In §4.6, SMOTE-recency is therefore **neutral overall** (though it hurts in some folds). A plausible reading is that thinning positives reduces minority coverage for interpolation-based SMOTE more severely than it damages the generators’ learned mapping.

## 5.5 Label delay

Delay removes labeled training rows before validation (§3.1). Oversampling cannot recover features from transactions that were never in training; the parallel decline across methods in §4.7 matches that story.

## 5.6 Selection on validation

Choosing the best target fraud rate per fold and method on validation PR-AUC (§3.3, §4.1) can be optimistic relative to a fixed rate chosen a priori. Mitigations: an inner temporal split for rate selection, or reporting a fixed 5% configuration—see §3.3 for the honest statement.

## 5.7 Why fidelity checks matter

The supplementary fidelity audit (§4.9) gives a useful mechanistic signal: synthetic samples are easy to distinguish from real frauds (real-vs-synth AUC \(\approx 1.0\) for both CTGAN and TabDDPM in that run), and TabDDPM shows especially weak categorical fidelity there. This does not invalidate the downstream benchmark results, but it clarifies why augmentation effects can be unstable: utility can improve in some folds even when synthetic realism is imperfect, and weak fidelity can cap generalization under shift.
