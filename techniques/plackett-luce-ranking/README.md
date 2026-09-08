# Plackett-Luce Ranking Model (Reference §47.53)

Plackett (1975); Luce (1959). For a full ranking π_1, …, π_K over
K items with `worth` parameters w_i > 0:

    P(π_1, …, π_K) = ∏_{k=1}^K  w_{π_k} / Σ_{j ≥ k} w_{π_j}.

Reduces to Bradley-Terry for K=2. Fit by Hunter's (2004) MM
iteration.

## Files

- `python/plackett_luce_ranking.py` — MM algorithm from
  scratch. Demo (5 items, 800 simulated rankings, true worths
  (4, 3, 2, 1, 0.5)): 15 iterations to converge; recovered worths
  (1.91, 1.42, 0.97, 0.47, 0.23) vs truth (1.90, 1.43, 0.95, 0.48,
  0.24); rank order exactly recovered.
- `r/plackett_luce_ranking.R` — `PlackettLuce`, `pmr` (R);
  `choix`, from-scratch (Python).

## When to use

- **Full or partial rankings** over K items (movies, athletes,
  political candidates).
- **Multi-competitor sports** modelling.
- **Learning-to-rank (LTR)** — Plackett-Luce loss for list
  ranking.
- **RLHF preference data** with K > 2 candidates per prompt.

## When NOT to use

- **Only pairwise comparisons** — Bradley-Terry is simpler.
- **Ordinal ratings without ranking** — use ordinal regression.
- **When ties are frequent** — extend to Plackett-Luce with ties or
  Rao-Kupper.

## Assumptions & caveats

- **IIA (Luce choice axiom)** — adding a new alternative doesn't
  change relative choice probabilities; violated in some choice
  data (nested/mixed logit alternative).
- **Complete rankings** — partial or top-k data handled via
  standard Plackett-Luce factorisation.
- **Worth identifiability** — only ratios matter; normalise sum or
  fix a reference.
- **MM guarantees** monotone convergence; MLE unique if the win
  graph is strongly connected.

## Related in this repo

- `bradley-terry` — pairwise special case.
- `conjoint-choice`, `multinomial-logistic`,
  `mixed-logit-mnl` — discrete-choice cousins.
- `rlhf-preferences`, `dpo-direct-preference-optimization` — LLM
  preference optimisation.
- `graded-response-model`, `nominal-response-model`,
  `partial-credit-model` — IRT ordinal cousins.

## Run

```
python techniques/plackett-luce-ranking/python/plackett_luce_ranking.py
Rscript techniques/plackett-luce-ranking/r/plackett_luce_ranking.R
```

**Refs:** Plackett, R.L. "The analysis of permutations." *Appl Stat* 24(2): 193-202, 1975; Luce, R.D. *Individual Choice Behavior.* Wiley, 1959; Hunter, D.R. "MM algorithms for generalized Bradley-Terry models." *Ann Stat* 32(1): 384-406, 2004.

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
