# Elo / Glicko Rating Systems (Reference §47.66)

Elo (1978); Glickman (1999). Online updates for competitive
ratings.

**Elo**: `E_ij = 1 / (1 + 10^((R_j − R_i)/400))`,
`R_i ← R_i + K (s − E_ij)`.

**Glicko** adds a rating deviation `RD` (uncertainty) and a
volatility parameter; batch update over multiple games gives
Bayesian shrinkage.

## Files

- `python/elo_glicko_rating.py` — from-scratch Elo and Glicko
  updates. Demo (8 players, log-odds skill on a linear grid,
  K=16, 4 000 games):
  - Corr(final Elo, true skill) = 0.987
  - Glicko single-round: R 1500 (RD 350) → 1592.6 (RD 181.8)
    after 2-1 vs three opponents.
- `r/elo_glicko_rating.R` — `PlayerRatings::runGL / runGlicko` (R);
  `skelo`, `trueskill`, `openskill` (Python).

## When to use

- **Competitive games** — chess, esports, MOBAs.
- **A/B/C testing of many arms with pairwise comparisons** — LLM
  eval arenas (Chatbot Arena).
- **Online-learning of latent skill / difficulty** — IRT-lite.
- **Any pairwise-comparison stream** — worker performance,
  proposal review.

## When NOT to use

- **Batch estimation from full history** — Bradley-Terry MLE or
  Plackett-Luce is more efficient.
- **Multi-outcome games** — extend to Glicko-2, TrueSkill (Xbox).
- **Small population** — Glicko / Bayes preferable to reduce
  variance early in the season.
- **Team compositions changing** — TrueSkill / OpenSkill needed.

## Assumptions & caveats

- **K choice** — trade responsiveness for stability; FIDE uses
  40/20/10 by rating band.
- **Rating deviation** in Glicko decays without games — inactive
  players drift.
- **Order matters** for Elo (online) but not for MLE fits.
- **Draws** count as s=0.5; extend to draw-aware models for
  frequent-draw games.

## Related in this repo

- `bradley-terry` — offline MLE for pairwise wins.
- `plackett-luce-ranking` — multi-competitor extension.
- `bayesian-hierarchical-models`, `bayesian-ab-testing` —
  full-Bayes rating alternatives.
- `multi-armed-bandits`, `thompson-sampling`,
  `mixed-logit-mnl` — related choice-model / decision toolkit.

## Run

```
python techniques/elo-glicko-rating/python/elo_glicko_rating.py
Rscript techniques/elo-glicko-rating/r/elo_glicko_rating.R
```

**Refs:** Elo, A.E. *The Rating of Chessplayers, Past and Present.* Arco Pub, 1978; Glickman, M.E. "Parameter estimation in large dynamic paired comparison experiments." *Appl Stat* 48(3): 377-394, 1999.

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
