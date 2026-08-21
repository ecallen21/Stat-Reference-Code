# Differential Item Functioning (Reference §22.11)

**DIF**: an item favours or disadvantages a subgroup (gender, ethnicity, language) at the **same ability level**. Distinct from item **impact** (group mean differences that are substantively legitimate).

## Mantel-Haenszel DIF (Holland & Thayer 1988)

Stratify examinees by total-score levels. Within each stratum, form a 2×2 table of {correct/incorrect × reference/focal}. Combine via MH odds ratio:

```
α_MH = (Σ_k n_{RC,k} n_{FI,k} / n_k) / (Σ_k n_{RI,k} n_{FC,k} / n_k)
Δ_MH = −2.35 · log(α_MH)              ETS delta scale
```

ETS categories:

- `|Δ| < 1.0` — negligible (Type A)
- `1.0 ≤ |Δ| < 1.5` — moderate (Type B)
- `|Δ| ≥ 1.5` — large (Type C)

## Logistic-regression DIF (Swaminathan & Rogers 1990)

```
logit P(y_j = 1) = β_0 + β_1 · total + β_2 · group + β_3 · total × group
```

- `β_2` significant → **uniform DIF**.
- `β_3` significant → **non-uniform DIF** (DIF that varies with ability).

Nested LR tests give χ² statistics with 1 df each.

## Files

- `python/dif_mantel_haenszel.py` — from-scratch MH-DIF with rest-score stratification + logistic-DIF via nested-model LR tests. Demo (K = 10, item 4 planted with 0.8-logit difficulty shift for focal group): MH Δ = −1.97 (Type C large); logistic uniform χ² = 31.9, p < 10⁻⁷ — both correctly flag item 4.
- `r/dif_mantel_haenszel.R` — `difR::difMH`, `difR::difLogistic`, `difR::difLord` (IRT-based).

## When to use

- **Test fairness reviews** — mandatory in large-scale operational testing (SAT, GRE, PISA).
- **Cross-cultural / cross-language** scale adaptation.
- **Any group comparison** where a scale is used to compare people.

## Interpret DIF flags carefully

- **Statistical DIF ≠ bias** — investigate substantively before dropping items.
- **Reference vs focal** assignment matters — flip labels flips sign.
- **Sample size** — MH is very sensitive with large n; report effect size (Δ) alongside χ².
- **Manifest vs matching variable** — MH conditions on total score; IRT-based methods condition on estimated latent ability.

## Related methods

- **IRT-based DIF** — Lord's χ² on item parameters across groups; SIBTEST for non-parametric multi-item DIF.
- **DIF vs bias vs impact** — three distinct concepts; DIF is statistical, bias needs substantive review.

## Run

```
python techniques/dif-mantel-haenszel/python/dif_mantel_haenszel.py
Rscript techniques/dif-mantel-haenszel/r/dif_mantel_haenszel.R
```

**Refs:** Holland, P.W. & Thayer, D.T. "Differential item performance and the Mantel-Haenszel procedure." In H. Wainer & H.I. Braun, *Test Validity*, Lawrence Erlbaum, 1988; Swaminathan, H. & Rogers, H.J. "Detecting differential item functioning using logistic regression procedures." *J. Educ. Meas.* 27(4), 361–370, 1990.

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
