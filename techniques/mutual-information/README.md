# Mutual Information (Reference §4.14)

Information-theoretic association measure. For discrete `X` and `Y` with joint distribution `p(x, y)` and marginals `p(x)`, `p(y)`:

`I(X; Y) = Σ_{x,y} p(x, y) · log( p(x, y) / (p(x) · p(y)) )`

Units depend on the log base: **bits** (`log₂`), **nats** (`ln`), **Hartleys** (`log₁₀`).

## Properties

- `I(X; Y) ≥ 0`, with equality **iff** `X` and `Y` are independent.
- Symmetric: `I(X; Y) = I(Y; X)`.
- Captures **any** dependence — linear, nonlinear, non-monotonic. Pearson / Spearman / Kendall can miss everything except their specific assumption; MI can't.
- Unbounded above. To get a `[0, 1]`-scaled quantity, divide by an entropy term:

  | Variant | Definition |
  |---------|------------|
  | NMI (arithmetic) | `I / ((H(X) + H(Y))/2)` |
  | NMI (geometric)  | `I / √(H(X) · H(Y))` |
  | NMI (max)        | `I / max(H(X), H(Y))` |

  `H(X) = −Σ p(x) log p(x)` is the Shannon entropy.

## Continuous variables

There's no single canonical estimator. Options:
- **Binning** (this file's default): bin each variable (Freedman–Diaconis by default) and apply the discrete formula. Biased and bin-choice dependent; fine for a first pass.
- **KSG / k-NN** (Kraskov, Stögbauer, Grassberger): the modern default. `sklearn.feature_selection.mutual_info_regression` / `mutual_info_classif` ship this.
- **Kernel density**: smoother but parameter-sensitive.

## Maximal Information Coefficient (MIC)

Reshef et al. (2011). Searches over many `(bx, by)` bin grids subject to a budget `bx · by < B(n) = n^α` (default α = 0.6), and returns

`MIC = max_{bx·by < B(n)} I(X; Y; bx, by) / log₂(min(bx, by))`

rescaled to `[0, 1]`. Designed to be "equitable" — a noisy line, parabola, and sinusoid with the same noise level should give similar MIC values.

A **simple grid-search implementation** is included here (`maximal_information_coefficient`). It's `O(B²)` and roughly matches the paper's MIC on smooth functions but is slower / slightly less optimized than the **ApproxMaxMI** heuristic in the paper. For production use, install **`minepy`** (Python) or **`minerva`** (R) — both wrap the original ApproxMaxMI C library.

## Files
- `python/mutual_information.py` — from-scratch discrete MI + entropy + all three NMI variants; binning helper for continuous data with Freedman–Diaconis defaults; compares against `sklearn.metrics.mutual_info_score` and `normalized_mutual_info_score`. Demo shows the killer use case: `y = x² + noise` has Pearson `r ≈ 0` but `NMI ≈ 0.48`.
- `r/mutual_information.R` — from-scratch + `infotheo::mutinformation`; notes on `entropy::mi.empirical` and `mpmi::mmi`.
- PySpark: N/A — for huge categorical-by-categorical data, aggregate the joint counts with `groupBy(x, y).count()` (as in `techniques/chi-square-tests/pyspark/`), then apply the discrete formula on the driver. Continuous MI at Spark scale typically uses sub-sampling.

## Run
```
python techniques/mutual-information/python/mutual_information.py
Rscript techniques/mutual-information/r/mutual_information.R
```

**Refs:** Shannon, "A Mathematical Theory of Communication," *Bell System Technical Journal* 27, 379–423 & 623–656, 1948; Kraskov, Stögbauer & Grassberger, "Estimating Mutual Information," *Phys. Rev. E* 69(6), 066138, 2004; Reshef et al., "Detecting Novel Associations in Large Data Sets," *Science* 334(6062), 1518–1524, 2011.
