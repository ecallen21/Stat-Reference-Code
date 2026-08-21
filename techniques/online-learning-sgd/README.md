# Online Learning via SGD (Reference §21.x extra)

Fit a model **one example at a time** — never touching more than the current
`(x_t, y_t)` in memory.

```
w ← w − η_t · ∇_w ℓ(y_t, x_t; w)
```

Common losses and their derivatives:

| Loss | Task | ∇_w ℓ |
|---|---|---|
| **Squared** | regression | `(w·x − y) · x` |
| **Log** | binary classification | `(σ(w·x) − y) · x` |
| **Hinge** | SVM classifier | `−y · x` if `y · w·x < 1` else 0 |
| **Passive-Aggressive** | max-margin | `w += τ · y · x` with `τ = max(0, 1 − y·w·x) / ‖x‖²` — no learning rate |

Learning rate schedules:

- **Constant** `η_0` — tracks a drifting distribution.
- **Inverse-time** `η_0 / (1 + t/t₀)` — Robbins-Monro guarantees, collapses on the first regime.
- **Inverse-sqrt** `η_0 / √t` — common default.
- **AdaGrad / RMSProp / Adam** — per-feature adaptive rates.

## When to use

- **Streaming data** — logs, sensor readings, per-request scoring.
- **Data too big to hold** — one pass through disk or memory-mapped storage.
- **Concept drift** — a constant / cyclic learning rate lets the model chase moving distributions.
- **Warm start** for a batch trainer.

## Files

- `python/online_learning_sgd.py` — from-scratch online SGD with squared / log / hinge / passive-aggressive losses and three learning-rate schedules; sklearn `SGDRegressor` cross-check. Demo (n=5000, p=5): squared-loss ‖ŵ − w‖ = 0.022 (sklearn 0.021); hinge and PA both hit 0.8% classification error; constant-η tracks a mid-stream sign flip and recovers `−w_true` to within 0.17.
- `r/online_learning_sgd.R` — `biglm::bigglm`, `RSGD::rsgd`, `stream::DSC*`; Python `sklearn.linear_model.SGD*` and `river` for production streaming ML.

## Assumptions & caveats

- **Feature scaling matters more, not less** than in batch — no chance to standardize globally; use running-standardization or a data-plane preprocessor.
- **Loss surface curvature** — constant `η` can diverge on ill-conditioned features; consider AdaGrad / Adam or L2 regularisation.
- **Sample order matters** — SGD on an ordered stream (e.g. time-sorted) may overfit early patterns. Shuffle when the stream permits.
- **Passive-Aggressive has no hyperparameter for step size**, but PA-II / PA-III variants add slack budgets (`C`) to trade robustness against noisy labels.
- **Convergence** is to a *neighbourhood*, not a point — average over the last window (Polyak-Ruppert averaging) if you need a point estimate.

## Related

- **Mini-batch SGD** — process `B` examples per step; smoother gradients, better GPU utilisation.
- **Follow-The-Regularised-Leader (FTRL-Proximal)** — Google's ads-CTR workhorse; L1-sparse online logistic.
- **Hoeffding trees** — streaming decision trees with `river::HoeffdingTreeClassifier`.
- **Concept-drift detectors** — Page-Hinkley, ADWIN.

## Run

```
python techniques/online-learning-sgd/python/online_learning_sgd.py
Rscript techniques/online-learning-sgd/r/online_learning_sgd.R
```

**Refs:** Robbins, H. & Monro, S. "A stochastic approximation method." *Ann. Math. Statist.* 22(3), 400–407, 1951; Bottou, L. "Large-scale machine learning with stochastic gradient descent." *COMPSTAT*, 2010; Crammer, K. et al. "Online passive-aggressive algorithms." *JMLR* 7, 551–585, 2006.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
