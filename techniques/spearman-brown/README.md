# Split-half Reliability + Spearman-Brown (Reference §22.4)

## Split-half

Split a test into two halves; correlate the two half-scores → `r_hh`. This is the reliability of a **half-length** test.

## Spearman-Brown prophecy formula

Predicted reliability of a test lengthened by factor `k`:

```
ρ_k = k · ρ_1 / (1 + (k − 1) ρ_1)
```

Split-half specific case (`k = 2`, full-length from halves):

```
ρ_full = 2 r_hh / (1 + r_hh)
```

## Split methods

- **Odd-even** — interleave items (usually the best if item order carries no meaning).
- **First-vs-second half** — sensitive to fatigue / order effects.
- **Random split** — average over many random splits for stability.
- **Guttman lower bound** — minimum across all possible splits (most conservative).

## Files

- `python/spearman_brown.py` — split-half with three split methods + Spearman-Brown prophecy calculator. Demo (K = 10, unequal loadings not needed): odd-even split gives `r_hh = 0.90`, SB-corrected reliability = 0.947; matches other split methods to 3 decimals.
- `r/spearman_brown.R` — `psych::splitHalf` (Revelle; enumerates many splits + reports min/max/mean).

## When to use

- **Test-length planning**: "how many items do I need to reach reliability 0.9?" — invert Spearman-Brown.
- **Alternative to Cronbach's α** when items are heterogeneous (α underestimates then).
- **Historical context** — early psychometrics relied heavily on split-half.

## Assumptions & caveats

- **Random split assumption** — splits should be equivalent in content and difficulty.
- **Same underlying trait** — no multidimensionality.
- **Spearman-Brown assumes** the added items are of the same quality as the original ones; predicting from small samples of items to a much longer test is optimistic.

## Run

```
python techniques/spearman-brown/python/spearman_brown.py
Rscript techniques/spearman-brown/r/spearman_brown.R
```

**Refs:** Spearman, C. "Correlation calculated from faulty data." *Br. J. Psychol.* 3(3), 271–295, 1910; Brown, W. "Some experimental results in the correlation of mental abilities." *Br. J. Psychol.* 3(3), 296–322, 1910.

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
