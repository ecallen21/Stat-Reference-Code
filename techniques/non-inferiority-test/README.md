# Non-Inferiority Tests (Reference §17.7)

Show a new treatment is **at most a pre-specified margin worse** than a standard. Distinct from superiority (`H_0: δ = 0`) and equivalence (`|δ| < margin`).

```
H_0 : δ ≤ −margin      (new is inferior by more than margin)
H_a : δ > −margin      (new is at most margin worse)
```

One-sided test at α; equivalent to checking whether the **lower bound** of the two-sided (1 − 2α) CI (or one-sided (1 − α) CI) for δ exceeds `−margin`.

## Two common effect measures

### Mean difference (Normal outcomes)

Pooled-variance t on `d̂ = ȳ_new − ȳ_std`. Compare `(d̂ + margin) / SE(d̂)` against `t_{α, df}`.

### Proportion difference (binary outcomes)

**Farrington-Manning** (1990) restricted-MLE score test: solve for `(p̃_1, p̃_2)` satisfying `p_1 − p_2 = −margin` and maximizing the constrained likelihood; use the resulting SE under `H_0`. Widely used in FDA / EMA submissions.

## Contrast with equivalence (TOST)

- **Non-inferiority**: one-sided, only bounds one direction of harm.
- **TOST equivalence**: two-sided, bounds `|δ| < margin` in both directions — required when neither direction is preferred.

## Files

- `python/non_inferiority_test.py` — from-scratch NI t-test on mean difference and Farrington-Manning NI test on proportion difference. Demos: mean-diff test declares NI (p = 0.014, CI lower = −0.80 > −1.0); proportions 82/100 vs 80/100 with margin 0.10 gives z = 2.13, p_NI = 0.017.
- `r/non_inferiority_test.R` — base `t.test(alternative = "greater", mu = -margin)` for means; `DescTools::BinomDiffCI(sides = "left")` for proportions.

## When to use

- **Regulatory drug approval** — new therapy shown to be "not much worse" than the standard.
- **Non-inferior device / assay** — replace a well-studied gold-standard with something cheaper, faster, or safer.
- **Bioequivalence-style testing** on log-scale ratios (multiplicative margin).

## Margin choice

- **Not a statistical decision** — it comes from clinical / regulatory judgment, prior effect sizes, or minimum clinically important difference (MCID).
- **Half the effect of the standard vs placebo** is a common heuristic ("preserve at least 50% of the active effect").
- **Report the margin explicitly** so readers can judge substantive plausibility.

## Assumptions & caveats

- **Assay sensitivity** — the trial must be capable of detecting a difference if one existed; otherwise NI conclusions from an under-powered trial are meaningless.
- **Constancy assumption** — the standard's effect vs placebo is the same in this trial as in the historical trials that defined the margin.
- **Missing data and analysis population** — per-protocol and intention-to-treat analyses can diverge; regulators typically want both.

## Run

```
python techniques/non-inferiority-test/python/non_inferiority_test.py
Rscript techniques/non-inferiority-test/r/non_inferiority_test.R
```

**Refs:** FDA. *Guidance for Industry: Non-Inferiority Clinical Trials*, 2016; Farrington, C.P. & Manning, G. "Test statistics and sample size formulae for comparative binomial trials with null hypothesis of non-zero risk difference or non-unity relative risk." *Stat. Med.* 9(12), 1447–1454, 1990; D'Agostino, R.B. et al. "Non-inferiority trials: design concepts and issues." *Stat. Med.* 22(2), 169–186, 2003.

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
