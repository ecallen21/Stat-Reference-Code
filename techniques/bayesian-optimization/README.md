# Bayesian Optimization (Reference §14.28)

Sequential black-box optimization when each function evaluation is **expensive** — hyperparameter tuning, physical experiments, drug screens, A/B tests. Model the unknown `f(x)` with a Gaussian process, pick the next `x` by maximizing an **acquisition function** that balances exploration and exploitation.

## Loop

```
1. Fit GP posterior to observed (x_i, y_i).
2. Compute acquisition α(x) over a candidate set.
3. Query f at argmax α(x); append; repeat.
```

## Gaussian process surrogate

```
K = kernel matrix + σ_n² I         (squared-exponential kernel is standard)
μ(x*) = K_{x*,X} K⁻¹ y
Σ(x*) = K_{x*,x*} − K_{x*,X} K⁻¹ K_{X,x*}
```

Hyperparameters (length-scale, noise) either fixed or learned by marginal-likelihood optimization.

## Acquisition functions

- **Expected Improvement (Mockus 1978)**:

```
EI(x) = (μ(x) − f_best − ξ) Φ(z) + σ(x) φ(z)
z = (μ(x) − f_best − ξ) / σ(x)
```

- **Upper Confidence Bound (Srinivas 2010)**: `μ(x) + κ σ(x)`.
- **Probability of Improvement**: `Φ((μ(x) − f_best − ξ) / σ(x))`. Simpler, but tends to over-exploit.
- **Thompson Sampling**: sample `f̂ ~ GP posterior`, take `argmax f̂`.

## Files

- `python/bayesian_optimization.py` — from-scratch GP with squared-exponential kernel + expected-improvement acquisition on a 1-D grid. Demo on a multimodal target: BayesOpt reaches `f = 1.4190` in 18 evaluations (grid truth 1.4195), slightly better than random search (1.4154) at the same budget.
- `r/bayesian_optimization.R` — `ParBayesianOptimization::bayesOpt` or `DiceOptim::EGO.nsteps`.

## When to use

- **Expensive** `f`: minutes-to-days per evaluation.
- **Low-dimensional** parameter spaces: `d ≤ 20`. Above `d ≈ 20`, GP scales poorly; use tree-structured Parzen estimators (Hyperopt, Optuna) or trust-region BO (TuRBO).
- **Continuous / mixed** search spaces with smooth-ish response surface.

## When NOT to use

- **Cheap** evaluations: random search / grid search / evolutionary algorithms often win.
- **Very high `d`**: standard GP degrades; use low-dimensional embeddings, additive GPs, or hyperparameter-search libraries.
- **Discrete / categorical** search space: use TPE or Bayesian bandits.

## Production libraries

- Python: **scikit-optimize** (skopt), **GPyOpt**, **Ax + BoTorch** (Meta), **Optuna** (TPE, not GP-based).
- R: **DiceOptim**, **ParBayesianOptimization**, **rBayesianOptimization**.

## Assumptions & caveats

- **Smoothness assumption**: GP kernel encodes it. Wrong kernel → BayesOpt wastes evaluations. Use Matern-3/2 or Matern-5/2 for less-smooth functions.
- **Warm-start**: with 3–5 initial random / Latin-hypercube points, EI can find the optimum in tens of evaluations for smooth `f`.
- **Exploration parameter** `ξ` (or `κ` for UCB) controls exploration vs exploitation; try `ξ ∈ {0, 0.01, 0.1}`.

## Run

```
python techniques/bayesian-optimization/python/bayesian_optimization.py
Rscript techniques/bayesian-optimization/r/bayesian_optimization.R
```

**Refs:** Mockus, J. "On Bayesian methods for seeking the extremum." *Optim. Tech.* 400–404, 1978; Jones, D.R., Schonlau, M. & Welch, W.J. "Efficient global optimization of expensive black-box functions." *J. Global Optim.* 13(4), 455–492, 1998; Shahriari, B. et al. "Taking the human out of the loop: a review of Bayesian optimization." *Proc. IEEE* 104(1), 148–175, 2016.

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
