# Neural Tangent Kernel (Reference §46.19)

Jacot, Gabriel & Hongler (2018, NeurIPS). In the **infinite-width
limit**, gradient-descent training of a neural network is equivalent
to **kernel regression** with the NTK:

    Θ(x, x') = ⟨∇_θ f(x),  ∇_θ f(x')⟩

which for standard initialisations is deterministic and does not
change during training. The full training trajectory becomes a
linear-in-parameters problem with closed-form predictions

    f_∞(x) = K(x, X) · K(X, X)⁻¹ · y

## Files

- `python/neural_tangent_kernel.py` — analytical NTK for a 2-layer
  ReLU network (arc-cosine kernel) + finite-width MLP trained by
  GD, both from scratch. Demo (d=5, n_train=200, H=400): NTK
  kernel-regression MSE 0.028, wide-MLP MSE 0.027; correlation
  between the two prediction vectors = 0.996 — matches Jacot 2018.
- `r/neural_tangent_kernel.R` — no R implementations;
  `neural-tangents` (JAX), `ntk-pytorch` / `functorch` (Python).

## When to use

- **Understanding wide-NN generalisation** — theoretical linearisation.
- **Kernel-methods as NN baseline** — NTK regression is competitive on
  small datasets.
- **Feature-learning diagnostics** — comparing NTK vs finite-width
  behaviour reveals whether features are moving.
- **Uncertainty quantification** — NNGP / NTK-Bayes gives Gaussian-
  process predictive intervals.

## When NOT to use

- **Deep / narrow networks** — feature learning dominates; NTK does
  not describe the behaviour.
- **Large datasets** — kernel regression is `O(n³)` in n;
  approximate methods (KRR + subsampling) required.
- **Trained-model interpretation** — the NTK is about the training
  trajectory, not the final learned features.

## Assumptions & caveats

- **Infinite width** — approximate at finite width; use `neural-tangents`
  for the analytic limit and inspect finite-width deviations.
- **NTK parameterisation** — Jacot's original scaling `1/√fan_in`
  matters; PyTorch defaults differ.
- **Depth scaling** — NTK for depth-L nets stacks; `neural-tangents`
  computes it.
- **Empirical NTK** — Novak et al. 2022 shows finite-width empirical
  NTK converges to infinite-width NTK as width grows.

## Related in this repo

- `deep-mlp-backprop`, `bayesian-neural-network`, `gaussian-process-regression`
  — deep-learning / kernel siblings.
- `kernel-density-estimation`, `nadaraya-watson-kernel-regression`
  — classical kernel methods.
- `rademacher-complexity`, `vc-dimension`, `pac-bayes-bounds` —
  generalisation-theory cousins.

## Run

```
python techniques/neural-tangent-kernel/python/neural_tangent_kernel.py
Rscript techniques/neural-tangent-kernel/r/neural_tangent_kernel.R
```

**Refs:** Jacot, A., Gabriel, F. & Hongler, C. "Neural tangent kernel: convergence and generalization in neural networks." *NeurIPS*, 2018; Lee, J. et al. "Wide neural networks of any depth evolve as linear models under gradient descent." *NeurIPS*, 2019.

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
