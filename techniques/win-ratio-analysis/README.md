# Win Ratio Analysis (Reference §47.268)

Pocock, Ariti, Collier & Wang (2012). HIERARCHICAL COMPOSITE
endpoint that ranks outcomes by clinical importance (death >
hospitalisation > quality of life). For every treated-control
pair, decide who "wins" using the highest-order outcome that
discriminates them:

```
Win Ratio = (# treated wins) / (# control wins)
```

Popular in cardiovascular trials to combine mortality with
softer endpoints without the composite-event-first flaw
(where a hospitalisation "beats" a death by coming first in
time).

## Files

- `python/win_ratio_analysis.py` — Pairwise hierarchical
  comparison with Bebu-Lachin CI. Demo: N=300/arm,
  three-tier outcomes (time-to-death, time-to-hospitalisation,
  symptom score) with modest treatment benefit on all tiers.
  Treated wins 47 115 pairs vs control's 42 885 → WR = 1.099
  (95% CI 1.09-1.11). Naive time-to-first-event Wilcoxon
  gives z=2.68 (p=0.007) but misses the hierarchy: a death
  loss cannot be masked by a hospitalisation "win".
- `r/win_ratio_analysis.R` — `WWR`, `WINrat`, `hierBinom`,
  `survival + custom loop` (R); from-scratch pairwise
  comparison (Python).

## When to use

- **Cardiovascular / heart-failure trials** — Pocock's home
  ground; MACE endpoints combined with QoL / functional
  status.
- **Time-to-event composites** where higher-severity events
  should dominate.
- **Ordinal-heavy endpoints** — win ratio handles ties
  gracefully across tiers.

## When NOT to use

- **Single-endpoint trials** — plain t-test / logrank is
  simpler.
- **When lower-tier outcomes dominate power** — the WR
  ignores lower-tier information if the higher tier
  discriminates.
- **Very small N** — the pairwise structure has non-standard
  variance; small-N CIs can be optimistic.

## Assumptions & caveats

- **Hierarchy pre-specification** — the tier order MUST be
  fixed in the SAP; post-hoc reordering inflates Type-I.
- **Ties treatment** — how time-to-event ties are handled
  affects the answer; grid-based tie-breakers vs 0/1 credit
  matter.
- **Censoring** — the Pocock estimator handles censoring by
  restricting to pairs where the higher-severity event can
  be compared; sensitivity to censoring rules should be
  reported.
- **Variance approximation** — Bebu-Lachin, jackknife, and
  bootstrap variants can differ; bootstrap is safest.

## Related in this repo

- `fleming-harrington-weighted-logrank` — weighted
  time-to-event alternative.
- `log-rank-test`, `competing-risks` — standard survival
  analyses.
- `composite-likelihood` — related composite framework
  (though not the same).

## Run

```
python techniques/win-ratio-analysis/python/win_ratio_analysis.py
Rscript techniques/win-ratio-analysis/r/win_ratio_analysis.R
```

**Refs:** Pocock, S.J., Ariti, C.A., Collier, T.J. and Wang, D. "The win ratio: a new approach to the analysis of composite endpoints in clinical trials based on clinical priorities." *Eur. Heart J.*, 33(2): 176-182, 2012.

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
