# Bridge Sampling for Marginal Likelihood (Reference §25.7)

Meng & Wong (1996); Gronau et al. (2017). Given samples
`θᵢ ~ p(θ | y)` (target) with unnormalised density
`p_u(θ) = p(y|θ) · p(θ)` (evidence `Z = p(y)` unknown), estimate `Z`
by comparing to a **proposal** `g(θ)` with known normalising constant.

## Meng-Wong iteration (Gronau 2017 eq. 5-6)

    l1[i] = p_u(θ_target^i) / g(θ_target^i)
    l2[j] = p_u(θ_proposal^j) / g(θ_proposal^j)
    s1, s2 = N1/(N1+N2),  N2/(N1+N2)

    r ← mean_j( l2 / (s1·l2 + s2·r) )  /  mean_i( 1 / (s1·l1 + s2·r) )

Iterate to convergence; `Z = r`.

## Files

- `python/bridge_sampling_evidence.py` — Meng-Wong iterative
  bridge estimator from scratch. Demo (target N(3, 1) with
  synthetic scale 5, proposal N(3.5, 1.5)): bridge recovers
  Z = 5.04 vs truth 5.00; naive importance sampling 5.06;
  harmonic-mean estimator 0.46 (classic instability).
- `r/bridge_sampling_evidence.R` — `bridgesampling::bridge_sampler`
  (one-liner over stan/JAGS/nimble output), `BayesFactor` (R);
  from-scratch (Python; no first-class library).

## When to use

- **Marginal likelihood / model evidence** for Bayes factors.
- **Nested-model comparison** where evidence differences matter.
- **Downstream of stan / JAGS / pymc** — you already have posterior
  samples; bridge only needs proposal draws + log-density.

## When NOT to use

- **Predictive comparison** — WAIC / PSIS-LOO (`loo`) directly target
  out-of-sample fit and are usually preferred for prediction.
- **Very high dimensions** — proposal must overlap target well;
  moment-matched Gaussian proposals struggle for d > 100.
- **Improper priors** — evidence is undefined.

## Assumptions & caveats

- **Proposal overlap** with target is critical; use a moment-matched
  Gaussian on the transformed (unconstrained) parameter space, per
  `bridgesampling` default.
- **Warp bridge / warp-III** — transform samples to remove skew /
  multimodality before applying MW.
- **Sample independence** — MCMC autocorrelation shrinks effective
  sample size and inflates variance.
- **Numerical stability** — do computations in log space
  (log-sum-exp) for high-dim / peaky targets.

## Related in this repo

- `bayesian-model-comparison`, `bayesian-model-averaging`,
  `information-criteria` — companion tools.
- `reversible-jump-mcmc` — alternative model-choice sampler.
- `variational-inference`, `laplace-approximation` — evidence lower
  bound (ELBO) / Laplace evidence approximations.
- `importance-sampling` — related but usually less accurate.

## Run

```
python techniques/bridge-sampling-evidence/python/bridge_sampling_evidence.py
Rscript techniques/bridge-sampling-evidence/r/bridge_sampling_evidence.R
```

**Refs:** Meng, X.-L. & Wong, W.H. "Simulating ratios of normalising constants via a simple identity: a theoretical exploration." *Statistica Sinica*, 6(4): 831-860, 1996; Gronau, Q.F. et al. "A tutorial on bridge sampling." *Journal of Mathematical Psychology*, 81: 80-97, 2017.

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
