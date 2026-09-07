# Stochastic-Gradient MCMC (SGLD / SGHMC) (Reference §25.9)

Welling & Teh (2011, SGLD); Chen, Fox & Guestrin (2014, SGHMC).
Scales MCMC to large datasets by using **mini-batch** gradient
estimates in place of the full-data likelihood gradient.

## SGLD update

    θ_{t+1} = θ_t
              + (ε_t / 2) · (∇ log prior + (n/b) · Σ_{i ∈ batch_t} ∇ log lik_i)
              + N(0, ε_t · I)

The injected noise turns the descent into approximate Langevin
dynamics; annealing `ε_t → 0` gives asymptotic exactness (Teh, Thiery
& Vollmer 2016).

## Files

- `python/stochastic_gradient_mcmc.py` — from-scratch SGLD for
  Bayesian linear regression with step-size schedule
  `ε_t ∝ t⁻(α/2)`. Demo (n=5000, p=4): SGLD post mean =
  (0.806, −0.305, 0.488, −0.004) matches analytic
  (0.803, −0.305, 0.491, −0.005); post SDs are larger than analytic
  (0.038 vs 0.014) as expected without step-size vanishing to 0.
- `r/stochastic_gradient_mcmc.R` — `SGmcmc` (SGLD / SGHMC / SGRLD /
  SGNHT) (R); `tensorflow_probability.mcmc.LangevinDynamics`,
  `numpyro`, pymc custom stepper, torch-based SGMCMC (Python).

## When to use

- **Large-n / large-p Bayesian inference** — deep-net posteriors
  (Bayesian CNN, Bayesian MLP).
- **Streaming data** — mini-batch fits online.
- **Warm-start MCMC** — use SGLD trajectory as init for HMC.

## When NOT to use

- **Small n** — full-batch NUTS is exact and free of the tuning
  headache.
- **Peaked posteriors with strong curvature** — preconditioning is
  essential (pSGLD, SGRLD).
- **Absolute-tolerance inference** — SGLD has bias unless step size
  is annealed to zero (which trades variance).

## Assumptions & caveats

- **Independent minibatches** — random with-replacement or shuffled
  passes.
- **Step-size schedule** — decay too fast → chain freezes; too slow
  → residual bias. Rule of thumb `ε_t ∝ t⁻α` with 0.5 ≤ α ≤ 1.
- **Preconditioning** — RMSprop-style adaptive step per parameter
  (pSGLD, Li-Chen-Carlson-Carin 2016) helps ill-conditioned targets.
- **Variance underestimation** — samples are dependent; ESS-based
  reporting is essential.

## Related in this repo

- `mcmc-metropolis-hastings`, `hmc-nuts`, `adaptive-metropolis-haario`
  — MCMC family.
- `variational-inference`, `bayesian-neural-network`, `mc-dropout`
  — scalable-Bayesian alternatives.
- `online-learning-sgd`, `adam-optimizer` — optimisation cousins.

## Run

```
python techniques/stochastic-gradient-mcmc/python/stochastic_gradient_mcmc.py
Rscript techniques/stochastic-gradient-mcmc/r/stochastic_gradient_mcmc.R
```

**Refs:** Welling, M. & Teh, Y.W. "Bayesian learning via stochastic gradient Langevin dynamics." *ICML*, 2011; Chen, T., Fox, E. & Guestrin, C. "Stochastic gradient Hamiltonian Monte Carlo." *ICML*, 2014; Teh, Y.W., Thiery, A.H. & Vollmer, S.J. "Consistency and fluctuations for stochastic gradient Langevin dynamics." *JMLR*, 17: 1-33, 2016.

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
