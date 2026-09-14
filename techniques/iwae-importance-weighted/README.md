# IWAE — Importance Weighted Autoencoder (Reference §47.289)

Burda, Grosse & Salakhutdinov (2016). Tighter ELBO for VAEs
via K importance samples:

```
log p(x) ≥ L_K = E[log (1/K) Σ_k w_k]
w_k = p(x, z_k) / q(z_k | x),   z_k ~ q(z | x)
```

`L_K` increases monotonically with `K`, and `L_1` recovers
the plain ELBO. Trades compute for a tighter bound;
important for posteriors that a factorised `q(z | x)`
misfits.

## Files

- `python/iwae_importance_weighted.py` — 1-D toy: `x = 1.5`,
  mis-specified `q = N(0.5, 1)`. Bound tightens monotonically
  as K grows: at K=1 gap to truth is ~0.2 nats; at K=500 the
  gap is < 0.02 nats. Illustrates Burda's Theorem 1.
- `r/iwae_importance_weighted.R` — reticulate + Pyro (R);
  `pyro.infer.RenyiELBO`, TFP `monte_carlo_variational_loss`,
  from-scratch (Python).

## When to use

- **Complex posteriors** that a mean-field q(z | x) can't
  capture — IWAE tightens the bound without changing q.
- **Model comparison** — likelihood estimates from IWAE are
  more accurate at moderate K.
- **Training as a bound** — IWAE is a drop-in ELBO
  replacement.

## When NOT to use

- **Well-specified q** — plain ELBO is already tight; K > 1
  is wasted compute.
- **Very large K** — variance reduces slowly; K=5-50 is
  usually enough.
- **When posterior collapse is the problem** — IWAE can
  worsen it; use β-VAE or free-bits.

## Assumptions & caveats

- **Rao-Blackwellisation** — Tucker et al 2019 showed IWAE
  gradient wrt φ has issues; DReG estimator fixes.
- **Compute cost** — K× the forward passes per gradient step.
- **Numerical stability** — implement `log-mean-exp(w_k)`
  with `logsumexp`, not naive exp+mean+log.
- **Bound tightness monotonic in K**, but variance of the
  bound estimate does not shrink monotonically.

## Related in this repo

- `variational-autoencoder` — the K=1 baseline.
- `variational-inference` — the surrounding framework.
- `beta-vae-disentangle` — β-scaled ELBO cousin.
- `importance-sampling` — the estimator IWAE builds on.
- `rejection-sampling` — related MC technique.

## Run

```
python techniques/iwae-importance-weighted/python/iwae_importance_weighted.py
Rscript techniques/iwae-importance-weighted/r/iwae_importance_weighted.R
```

**Refs:** Burda, Y., Grosse, R.B. and Salakhutdinov, R.R. "Importance weighted autoencoders." In *ICLR*, 2016.

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
