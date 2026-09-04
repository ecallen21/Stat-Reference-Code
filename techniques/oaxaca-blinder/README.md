# Oaxaca-Blinder Decomposition (Reference §35.21)

Blinder (1973), Oaxaca (1973). Decompose the **mean outcome gap**
between two groups (A, B) into components attributable to differences
in observable characteristics vs differences in returns to those
characteristics.

## Threefold decomposition (from A's perspective)

```
ȳ_A − ȳ_B  =  (X̄_A − X̄_B) β_B                       endowments
            + X̄_B (β_A − β_B)                          coefficients
            + (X̄_A − X̄_B) (β_A − β_B)                 interaction
```

## Twofold decomposition (reference β\*)

```
gap  =  (X̄_A − X̄_B) β*                                explained
       + X̄_A (β_A − β*) + X̄_B (β* − β_B)             unexplained
```

Common `β* = pooled OLS`; also Neumark's overall-model pooled with
group dummy.

## When to use

- **Wage-gap analysis** (gender, race, region) — the classic
  application.
- **Health / education inequality** decompositions.
- **Attribution** of an outcome gap to observables vs "residual" /
  discrimination.

## When NOT to use

- **Endogenous / selection-biased outcomes** — use Heckman
  decomposition variants.
- **Non-linear models** — extensions exist (Fairlie 2005 for logit /
  probit).
- **Very rich unobservables** — the "unexplained" component conflates
  discrimination and omitted variables.

## Files

- `python/oaxaca_blinder.py` — from-scratch threefold + twofold
  decompositions. Demo synthetic wage-gap data (10 % return in group
  B vs 6 % in group A). **Gap = −0.591**; threefold =
  endowments 0.052 + coefficients −0.599 + interaction −0.044;
  twofold = explained −0.075 + unexplained −0.517.
- `r/oaxaca_blinder.R` — `oaxaca` reference (R); `oaxaca-blinder`
  Python port.

## Assumptions & caveats

- **Reference-group choice matters** — swap A ↔ B changes the
  threefold split.
- **Detailed contribution** — each coefficient's contribution can be
  decomposed further; sensitive to the reference category of dummy
  variables (Oaxaca-Ransom 1994).
- **Non-linear extensions** — Fairlie 2005 (binary), Bourguignon 2007
  (RIF regression) for quantile decompositions.
- **Standard errors** — bootstrap or delta method.

## Related in this repo

- `fixed-effects-panel`, `heckman-selection` — sibling wage-equation
  tools.
- `quantile-regression`, `additive-quantile-regression`,
  `bayesian-quantile-regression` — distributional decompositions.
- `demographic-parity`, `disparate-impact` — related fairness-metric
  siblings.

## Run

```
python techniques/oaxaca-blinder/python/oaxaca_blinder.py
Rscript techniques/oaxaca-blinder/r/oaxaca_blinder.R
```

**Refs:** Blinder, A.S. "Wage discrimination: reduced form and structural estimates." *Journal of Human Resources*, 1973; Oaxaca, R. "Male-female wage differentials in urban labor markets." *International Economic Review*, 1973; Fairlie, R.W. "An extension of the Blinder-Oaxaca decomposition technique to logit and probit models." *Journal of Economic and Social Measurement*, 2005.

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
