# Reptile Meta-Learning (Reference §47.134)

Nichol, Achiam & Schulman (2018). First-order approximation of
MAML:

    for each task T ~ p(T):
        theta' = SGD_K(theta, task_T, lr_inner)
        theta = theta + eps * (theta' - theta).

No second-order gradients (unlike MAML), yet performs comparably
on few-shot benchmarks. Interprets meta-training as pushing the
initialisation toward a manifold from which local SGD adapts to
any task efficiently.

## Files

- `python/reptile_meta_learning.py` — from-scratch Reptile on
  1-D sinusoid regression with a 5-basis sinusoidal linear model.
  Demo (1200 meta-tasks, 3-shot adaptation, 80 test tasks):
  - K=0 (no adaptation): meta 1.95 vs random 3.36
  - K=1 step:  meta 1.91 vs random 3.29
  - K=5 steps: meta 1.77 vs random 3.03
  - **K=20 steps: meta 1.43 vs random 2.44** (39 % better MSE).
- `r/reptile_meta_learning.R` — no first-class R port;
  `learn2learn`, `higher`, `torchmeta` (Python).

## When to use

- **Few-shot supervised learning** — image classification (miniImageNet),
  regression, sequence tasks.
- **Warm-start pretraining** when second-order MAML is too expensive.
- **Continual / lifelong learning** primer.

## When NOT to use

- **When abundant task-specific labels exist** — plain supervised
  learning wins.
- **When precise MAML second-order gradients help** — MAML edges
  Reptile on some benchmarks.
- **When task distribution has no shared structure**.

## Assumptions & caveats

- **Task distribution p(T)** must yield transferable initialisations.
- **Inner K** controls adaptation depth; too high → forgets
  meta-init.
- **Outer step ε** typically 0.1–1.0.
- **Full-batch inner steps** simplify analysis; SGD-style inner
  loop works too.

## Related in this repo

- `meta-learning-maml` — the second-order relative Reptile
  approximates.
- `transfer-learning`, `lora-peft`,
  `prefix-prompt-tuning` — parameter-efficient / adaptation
  cousins.
- `contrastive-learning`, `byol-simsiam`,
  `contrastive-predictive-coding`, `jepa-self-supervised` — SSL
  neighbours (also produce transferable inits).
- `multi-armed-bandits`, `thompson-sampling` — task-level
  exploration ideas.

## Run

```
python techniques/reptile-meta-learning/python/reptile_meta_learning.py
Rscript techniques/reptile-meta-learning/r/reptile_meta_learning.R
```

**Refs:** Nichol, A., Achiam, J. & Schulman, J. "On first-order meta-learning algorithms." *arXiv:1803.02999*, 2018; Finn, C., Abbeel, P. & Levine, S. "Model-agnostic meta-learning for fast adaptation of deep networks." *ICML*, 2017.

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
