# Model Soups -- Fine-Tune Weight Averaging (Reference §47.28)

Wortsman et al. (2022, ICML). Instead of picking the single-best
fine-tuning run, **average** the weights of many runs (same
architecture, same init, different hyperparameters / seeds):

    θ_soup = (1 / M) · Σ_m θ_m           (uniform soup)
    θ_greedy = greedy add-if-improves-val (greedy soup)

Zero inference-time overhead vs a single model; often matches or
beats the best single run and even prediction ensembles on some
benchmarks.

## Files

- `python/model_soups_fine_tune_averaging.py` — M=12 linear models
  fine-tuned from a common init on bootstrap subsamples, then
  uniform / greedy soup + prediction ensemble baselines from
  scratch. Demo (n=400, d=10): best single 0.2566, uniform soup
  0.2561, ensemble 0.2561, greedy soup 0.2532 with 5 / 12 models
  included — soup matches ensemble quality with a single forward
  pass.
- `r/model_soups_fine_tune_averaging.R` — no R implementations;
  Wortsman code, torch state_dict averaging, mergekit / timm
  ensemble scripts (Python).

## When to use

- **Fine-tuning large models with many runs** — hyperparameter
  sweep runs would otherwise be discarded.
- **Latency-tight inference** — cannot afford ensemble forward
  passes.
- **Robustness / OOD** — soups often generalise better than the
  best single model (Wortsman 2022 Fig 6).

## When NOT to use

- **Different initialisations** — soup only works when models start
  from the same pre-trained weights (loss-basin alignment).
- **Very different architectures / tokenizers** — cannot average
  weights that are not shape-aligned.
- **You need per-model diversity / calibration** — ensembles preserve
  the disagreement information that soups collapse.

## Assumptions & caveats

- **Same starting point** — Wortsman requires shared pre-trained
  weights so all fine-tunes stay in one loss basin.
- **Batch-norm / running stats** — average them separately or
  re-estimate on a small calibration set.
- **Greedy selection** may overfit its held-out val split;
  cross-validate.
- **Task arithmetic** (Ilharco 2023) generalises: `θ_new = θ_pre +
  Σ_i α_i · (θ_i_ft − θ_pre)`.

## Related in this repo

- `deep-ensembles`, `swag`, `mc-dropout` — ensemble / posterior
  cousins.
- `lora-peft`, `transfer-learning`, `knowledge-distillation` —
  fine-tuning family.
- `bayesian-model-averaging` — statistical cousin of weight
  averaging.

## Run

```
python techniques/model-soups-fine-tune-averaging/python/model_soups_fine_tune_averaging.py
Rscript techniques/model-soups-fine-tune-averaging/r/model_soups_fine_tune_averaging.R
```

**Refs:** Wortsman, M. et al. "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time." *ICML*, 2022; Ilharco, G. et al. "Editing models with task arithmetic." *ICLR*, 2023.

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
