# 2×2 Crossover Design (Reference §18.x extra)

Each subject receives **both** treatments A and B in one of two randomised
orders — Sequence 1 (AB) or Sequence 2 (BA) — usually with a **washout**
between periods to erase carryover.

## Grizzle (1965) analysis

Per-subject differences `d_i = y_{period 1} − y_{period 2}`:

- **Sequence 1** (AB): `E[d] = (A − B) + (P₁ − P₂)` = treatment + period.
- **Sequence 2** (BA): `E[d] = (B − A) + (P₁ − P₂)` = −treatment + period.

Two useful contrasts of the two per-sequence means:

```
treatment_est = ½ · (mean(d_seq1) − mean(d_seq2))    (isolates treatment)
period_est    = ½ · (mean(d_seq1) + mean(d_seq2))    (isolates period)
```

Both use the same standard error (a scaled Welch two-sample SE).

**Carryover** cannot be isolated cleanly from within a single 2×2 crossover;
a two-sample test on within-subject **sums** `y₁ + y₂` (compared across
sequences) is the traditional Grizzle carryover test — but it's confounded
with subject-by-treatment interaction and has notoriously low power. Use a
sufficient washout instead, or a design with more periods (Williams, higher-order
crossovers) if carryover is a real concern.

## Why crossover pays off

Because both treatments are given to the same subject, the between-subject
variance cancels in the difference. Sample size per arm can be dramatically
smaller than a parallel-arm RCT for the same power. The demo below shows the
correct within-subject t = 11.6 vs a naive between-subject t = 4.3 on the
same data.

## When to use

- **Chronic, stable conditions** (asthma, hypertension) with reversible short-term outcomes.
- **PK / PD** bioequivalence studies (regulator-required for generics).
- **Sensory / preference** trials.
- **Behavioural interventions** with quick washout.
- **Never** for cure-oriented outcomes (survival, one-shot events).

## Files

- `python/crossover_design.py` — from-scratch Grizzle two-sample t analysis with treatment, period, and carryover tests. Demo (n=30/sequence, true A−B = 3.0, period = 0.5, subject SD 4.0, residual SD 1.5): treatment estimate = +2.988 (t = 11.6, p < 0.0001, matches truth), carryover t = −0.03 (correctly null), period t = 2.57 (p = 0.013, matches the small period effect). Naive between-subject t = 4.3 loses power because it ignores within-subject correlation.
- `r/crossover_design.R` — `Crossover::analyze2x2`, `lme4::lmer(y ~ period + treatment + (1 | subject))`, `geepack::geeglm`.

## Assumptions & caveats

- **No carryover** — the Grizzle treatment test is only valid without carryover. Include an explicit washout.
- **Carryover test is under-powered** and biased when the treatment effect varies across sequences — the "pre-test" of carryover before running the treatment test is now widely discouraged; the modern recommendation is to design washout in and analyse via a mixed model.
- **Missing period-2 data** — subjects who drop out contribute only period 1 (a parallel-arm subset).
- **Order effects distinct from carryover** — practice / fatigue / expectation; adjust for period effect (always include it in the model).
- **Higher-order crossovers** (Williams squares, ABBA / BAAB, four-period designs) allow direct carryover estimation.
- **Bioequivalence uses log-transformed outcomes** and 90% confidence intervals for the ratio A/B — a special case of this analysis with regulatory conventions.

## Related in this repo

- `latin-square-design` — cross-classified designs including carryover-controlled schemes.
- `repeated-measures-anova`, `linear-mixed-models` — general within-subject analysis; crossover is a special case.
- `non-inferiority-test` — the endpoint framework for many bioequivalence trials.
- `kenward-roger` — small-sample df correction for the mixed-model analysis.

## Run

```
python techniques/crossover-design/python/crossover_design.py
Rscript techniques/crossover-design/r/crossover_design.R
```

**Refs:** Grizzle, J.E. "The two-period change-over design and its use in clinical trials." *Biometrics* 21(2), 467–480, 1965; Jones, B. & Kenward, M.G. *Design and Analysis of Cross-Over Trials*, 3rd ed., Chapman & Hall/CRC, 2014; Senn, S. *Cross-over Trials in Clinical Research*, 2nd ed., Wiley, 2002.

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
