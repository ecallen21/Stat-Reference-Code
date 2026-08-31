# Model-Agnostic Meta-Learning — MAML (Reference §27.x extra)

Finn, Abbeel & Levine (2017). Learn an **initialisation** `θ*` such that a
small number of gradient steps on **any** new task's support set produces a
good policy / classifier on that task.

## Bi-level objective

```
θ* = argmin_θ  Σ_i  L_i^query( θ − α · ∇_θ L_i^support(θ) )
```

- **Inner loop** — `k` gradient steps of task-specific fine-tuning from `θ`.
- **Outer loop** — SGD on `θ` via the **meta-gradient** through the inner step.

## First-order MAML

The full meta-gradient contains second-order terms. FOMAML (Finn 2017)
drops them for speed with usually-modest accuracy loss. **Reptile** (Nichol
2018) is an even simpler analogue: run k SGD steps on each task, then move
`θ` toward the resulting parameters.

## Family

| Method | Idea |
|---|---|
| **MAML** (Finn 2017) | full second-order meta-gradient |
| **FOMAML** | drop second-order term |
| **Reptile** (Nichol 2018) | SGD analogue; no autograd through inner |
| **ANIL** (Raghu 2019) | adapt only the classifier head |
| **ProtoNets** (Snell 2017) | metric-based few-shot (Euclidean distance to prototypes) |
| **Matching Networks** (Vinyals 2016) | attention-based few-shot |
| **Meta-Baseline** (Chen 2020) | pretrain + fine-tune — surprisingly competitive |

## When to use

- **Few-shot classification / regression** — image classification with 1–5 labels per class.
- **Fast adaptation in RL** — meta-RL for new environments (RL²).
- **Continual learning** — learn to update quickly to new data.
- **Cross-task transfer** — different tasks share structure that a good init exploits.
- **NOT** when pretrain + fine-tune already works well; meta-baseline often ties MAML for a fraction of the compute.

## Files

- `python/meta_learning_maml.py` — first-order MAML + shared-weights baseline on 5-shot sine regression: each task fits `y = A sin(x + φ)` with random `(A, φ)`; the model is linear in features `[1, x, x², sin(x)]`. After 800 outer iterations:
  - MAML init MSE: 0.454 → 0.247 (46% drop after 5 adapt steps).
  - Shared baseline: 0.459 → 0.248 — MAML mechanics work, the toy is easy enough that both approach the same fixed point.
- `r/meta_learning_maml.R` — `reticulate` + `learn2learn`, `higher`, `torchmeta`.

## Assumptions & caveats

- **Inner-step learning rate** — the meta-gradient depends on this; often learned per-parameter (Meta-SGD).
- **Second-order gradients** are memory- and compute-heavy; FOMAML / Reptile are the practical defaults.
- **Task distribution matters** — MAML learns for the distribution of tasks it sees; strong out-of-distribution transfer isn't guaranteed.
- **Meta-baseline benchmark** (Chen 2020) — surprisingly strong: standard pretrain-then-fine-tune matches or beats MAML on many few-shot benchmarks. Report both.
- **ANIL** shows that in MAML most of the adaptation happens in the head, not the backbone — motivates cheaper head-only variants.
- **Meta-overfitting** — MAML can memorise the task distribution rather than learning a general init; regularisation (dropout, augmentation) helps.

## Related in this repo

- `transfer-learning` — related "reuse pretrained weights" recipe; MAML learns the *initialisation* to be adaptable.
- `deep-mlp-backprop`, `adam-optimizer` — the SGD building blocks.
- `contrastive-learning` — pretraining for representation transfer.

## Run

```
python techniques/meta-learning-maml/python/meta_learning_maml.py
Rscript techniques/meta-learning-maml/r/meta_learning_maml.R
```

**Refs:** Finn, C., Abbeel, P. & Levine, S. "Model-Agnostic Meta-Learning for fast adaptation of deep networks." *ICML*, 2017; Nichol, A., Achiam, J. & Schulman, J. "On first-order meta-learning algorithms (Reptile)." *arXiv:1803.02999*, 2018; Chen, Y. et al. "A new meta-baseline for few-shot learning." *NeurIPS*, 2020.

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
