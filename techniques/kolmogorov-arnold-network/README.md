# Kolmogorov-Arnold Network (KAN) (Reference §47.70)

Liu, Wang, Vaidya, Ruehle, Halverson, Šoljačić, Hou & Tegmark
(2024). Motivated by the Kolmogorov-Arnold representation
theorem: any continuous `f: [0,1]ⁿ → ℝ` can be written as
compositions of learned UNIVARIATE functions. A KAN replaces
standard MLP linear-then-nonlinear layers with learnable
univariate ACTIVATIONS on the edges (typically splines):

    z_l = Σ_i φ_{l, i}(x_i).

Parameter-efficient for additive / composed functions and each
φ is interpretable.

## Files

- `python/kolmogorov_arnold_network.py` — one-layer additive
  KAN with B-spline edges + alternating least-squares fit. Demo
  (n=500, 3 inputs, additive nonlinear truth `sin(2 x₁) + 0.5 x₂² −
  exp(−x₃²)`):
  - Linear regression R² = 0.033
  - 1-layer KAN (10 basis) R² = 0.997
  - 32-unit ReLU MLP R² = 0.970.
- `r/kolmogorov_arnold_network.R` — no established R
  implementation; `pykan`, `efficient-kan` (Python).

## When to use

- **Additive / low-order interaction structure** — KANs shine.
- **Interpretability required** — plot each learned univariate φ.
- **Scientific discovery** — Liu et al showcase symbolic-regression
  applications.
- **Small / structured tabular data** — beats naive MLPs.

## When NOT to use

- **Large-scale vision / language** — MLPs and Transformers still
  dominate; KANs slower per parameter.
- **Highly nonseparable interactions** — need multi-layer KANs,
  which lose some interpretability.
- **Real-time inference** — spline evaluation slower than linear
  + ReLU.

## Assumptions & caveats

- **Basis size** and knot placement matter; too few → underfit,
  too many → overfit.
- **Grid extension** trick (progressively finer knots) needed for
  training stability.
- **L1 regularisation** on edge functions promotes sparsity, key
  for interpretability.
- **Fit stability** in ALS is sensitive to initialisation; multi-
  restart or gradient descent alternatives.

## Related in this repo

- `splines-regression`, `functional-basis-smoothing`,
  `functional-linear-model`, `additive-quantile-regression`,
  `single-index-model` — basis / additive-model cousins.
- `neural-network-mlp`, `residual-connections`,
  `neural-ode`, `neural-tangent-kernel` — deep-learning
  neighbours.
- `explainable-boosting-machine`, `shape-constrained-regression`,
  `varying-coefficient-model` — interpretable-model peers.
- `bart-bayesian-additive-regression-trees`, `gradient-boosting` —
  alternative additive nonparametric regressors.

## Run

```
python techniques/kolmogorov-arnold-network/python/kolmogorov_arnold_network.py
Rscript techniques/kolmogorov-arnold-network/r/kolmogorov_arnold_network.R
```

**Refs:** Liu, Z. et al. "KAN: Kolmogorov-Arnold Networks." *arXiv:2404.19756*, 2024; Kolmogorov, A.N. "On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition." *Dokl. Akad. Nauk SSSR* 114: 953-956, 1957.

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
