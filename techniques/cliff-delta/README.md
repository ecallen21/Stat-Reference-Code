# Cliff's Delta (Reference §7.16)

Nonparametric effect size for two independent ordinal (or continuous) samples `X` and `Y`:

```
δ = Pr(X > Y) − Pr(X < Y)          in [−1, 1]
```

Interpretation:

- `δ = +1` — every `X > every Y` (perfect stochastic dominance).
- `δ = 0` — no stochastic dominance.
- `δ = −1` — every `X < every Y`.

## Relation to other measures

- **Mann-Whitney U**: `δ = 2 U / (n_X · n_Y) − 1`.
- **Vargha-Delaney A**: `A = (δ + 1) / 2`.

## Interpretation rule of thumb (Romano et al. 2006)

| |δ| range        | Effect size |
|-------------------|-------------|
| < 0.147          | negligible  |
| 0.147 – 0.33     | small       |
| 0.33 – 0.474     | medium      |
| > 0.474          | large       |

## Files

- `python/cliff_delta.py` — from-scratch δ + Cliff (1993) asymptotic Normal CI + magnitude classification. Demos: 0.5σ shift → δ = 0.05 (negligible on this n); 2σ shift → δ = 0.86 (large); null → δ ≈ 0.07 (negligible).
- `r/cliff_delta.R` — `effsize::cliff.delta`.

## When to use

- **Nonparametric alternative to Cohen's d** — no distributional assumption.
- **Reporting effect size alongside Mann-Whitney U p-value** — significance ≠ importance.
- **Ordinal outcomes** (Likert-style) where means are meaningless.

## Assumptions & caveats

- **Independent samples** — for paired data use the analog of Wilcoxon signed-rank (matched-pairs rank-biserial correlation).
- **Interpretation thresholds** are context-dependent; report the numeric value alongside the qualitative label.
- **Ties** — the formula handles ties by treating them as neither greater nor less (they contribute 0).

## Run

```
python techniques/cliff-delta/python/cliff_delta.py
Rscript techniques/cliff-delta/r/cliff_delta.R
```

**Refs:** Cliff, N. "Dominance statistics: ordinal analyses to answer ordinal questions." *Psychol. Bull.* 114(3), 494–509, 1993; Romano, J. et al. "Appropriate statistics for ordinal level data: should we really be using t-test and Cohen's d for evaluating group differences on the NSSE and other surveys?" *Annual meeting of the Florida Association of Institutional Research*, 2006.

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
