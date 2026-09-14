# Population-Based Training (Reference §47.277)

Jaderberg, Dalibard, Osindero et al (2017). Train a POPULATION
of N models in parallel; periodically:

- **exploit**: bottom P% copy top-Q%'s weights + hyperparams
- **explore**: perturb hyperparameters by × 0.8 or × 1.25

Result: HYPERPARAMETER SCHEDULES emerge (learning-rate warmup /
decay) rather than a single fixed value. Standard for
DeepMind-style RL and large-scale supervised training.

## Files

- `python/population_based_training.py` — Toy 1-D optimisation
  demo. N=12 population, T=20 PBT rounds. Best-loss trajectory
  converges to 0.0 (target weight w=3.0), and the top model's
  learning-rate schedule evolves over training as exploit /
  explore pairs shuffle winning HPs into losing seats.
- `r/population_based_training.R` — reticulate + Ray Tune
  (R); Ray Tune PopulationBasedTraining, Ax Service,
  from-scratch (Python).

## When to use

- **Long training runs** — reinforcement learning, LLM
  fine-tuning, style-transfer training where HPs matter over
  training.
- **Parallel compute** — a population must be trained
  concurrently.
- **Learning-rate scheduling under uncertainty** — PBT
  discovers schedules rather than requiring one to be
  pre-specified.

## When NOT to use

- **Small compute budget** — a population of size 8-32 is
  the minimum for signal.
- **Deterministic short training** — single-run HP search is
  simpler.
- **Non-continuous HPs** — perturbation (× 0.8 or × 1.25) is
  designed for positive real HPs; adapt for categorical.

## Assumptions & caveats

- **Exploit fraction** — 25%/25% (bottom copies top) is
  standard; adjust for very small populations.
- **Sync vs async** — sync PBT waits for all models; async
  PBT (Ray Tune) is more practical.
- **Weight copying overhead** — copying full model weights
  across workers can dominate for large models; async and
  bucketed copies mitigate.
- **Diverging populations** — over-aggressive perturbation
  can collapse the population; anneal perturbation range
  late in training.

## Related in this repo

- `hyperband-multi-fidelity`, `successive-halving-asha` —
  scheduler alternatives.
- `bohb-bayesian-hyperband` — Bayesian + Hyperband cousin.
- `bayesian-optimization`, `tpe-tree-parzen-estimator` — pure
  BO alternatives.

## Run

```
python techniques/population-based-training/python/population_based_training.py
Rscript techniques/population-based-training/r/population_based_training.R
```

**Refs:** Jaderberg, M., Dalibard, V., Osindero, S., et al. "Population Based Training of Neural Networks." *arXiv:1711.09846*, 2017.

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
