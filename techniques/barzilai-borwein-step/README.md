# Barzilai-Borwein Step (Reference §47.379)

Barzilai & Borwein (1988). Quasi-Newton flavour of gradient
descent using SPECTRAL step sizes derived from the secant
condition on consecutive iterates:

```
s_k = x_k − x_{k−1}
y_k = ∇f(x_k) − ∇f(x_{k−1})

α_BB1 = (sᵀs) / (sᵀy)     (long BB step)
α_BB2 = (sᵀy) / (yᵀy)     (short BB step)
```

The BB step makes plain gradient descent super-linearly
convergent on quadratic problems and highly competitive with
CG on general smooth problems. Cornerstone of many modern
proximal-gradient solvers.

## Files

- `python/barzilai_borwein_step.py` — Ill-conditioned SPD
  quadratic in d=50, `κ = 1 000`. 200 iterations: fixed-step
  `1/L` GD leaves `f − f* ≈ 14`; BB1 (long) reaches 4e−4;
  BB2 (short) 5e−4 — ~ 30 000× tighter than fixed-step GD.
- `r/barzilai_borwein_step.R` — `BB::BBoptim`,
  `optimx(method='BB')` (R); `pyproximal`, from-scratch
  (Python).

## When to use

- **Ill-conditioned quadratic / smooth-convex problems** —
  BB dramatically speeds up plain GD.
- **Proximal-gradient methods on composite `f + g`** —
  BB-accelerated ISTA / FISTA variants.
- **Non-monotone GD variants** — BB combined with
  Grippo-Lampariello-Lucidi safeguarding.

## When NOT to use

- **Very small problems** — L-BFGS / Newton is trivial.
- **Highly non-convex neural nets** — BB does not have the
  same track record as Adam / AdamW / SGD+momentum.
- **When each gradient is very expensive** — the two-step
  memory is fine, but L-BFGS uses more history efficiently.

## Assumptions & caveats

- **Non-monotone descent** — BB can INCREASE `f` at some
  steps; use Grippo-Lampariello-Lucidi's non-monotone line
  search for safety.
- **Alternate BB1 / BB2** — cyclic BB and adaptive
  Dai-Yuan variants improve robustness.
- **Negative curvature** — `s·y ≤ 0` ⇒ default to a small
  safeguard step.
- **First step** — no prior `s_k`; use one gradient step of
  `α = 1 / ‖g‖` or a line search.
- **Comparison to L-BFGS** — BB is a memory-1 quasi-Newton;
  L-BFGS uses memory-m for the same order of work per step.

## Related in this repo

- `lbfgs-quasi-newton`, `nesterov-accelerated-gradient`,
  `adam-optimizer`, `adamw-decoupled-weight-decay`,
  `conjugate-gradient-cg` — first- and quasi-Newton
  neighbours.
- `proximal-gradient-method`, `fista-accelerated-proximal`,
  `douglas-rachford-splitting`, `chambolle-pock-primal-dual`
  — splitting / proximal cousins.
- `spg-spectral-projected-gradient` (not present) — the
  constrained-BB generalisation.

## Run

```
python techniques/barzilai-borwein-step/python/barzilai_borwein_step.py
Rscript techniques/barzilai-borwein-step/r/barzilai_borwein_step.R
```

**Refs:** Barzilai, J. and Borwein, J.M. "Two-point step size gradient methods." *IMA J. Numer. Anal.*, 8(1): 141-148, 1988.

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
