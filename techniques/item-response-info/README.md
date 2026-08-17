# Item + Test Information Functions (Reference §22.14)

The **Fisher information** for item `j` at ability `θ` measures how precisely the item pins down `θ`:

```
I_j(θ) = a_j² · P_j(θ) · (1 − P_j(θ))         2PL model
```

Test information is the sum:

```
I(θ)  = Σ_j I_j(θ)
SE(θ) = 1 / √I(θ)
```

## Interpretation

- Peak of `I_j(θ)` occurs at `θ = b_j` (item's difficulty).
- Higher `a_j` → taller, narrower information peak.
- Test information curve is the sum → shows where the test measures precisely and where it doesn't.

## Applications

- **Test design** — pick items to spread information across the target `θ` range.
- **Computer Adaptive Testing (CAT)** — at each step, ask the item with maximum information at the current `θ̂`. Shortens tests dramatically.
- **Reliability at each θ** — SE(θ) is the local analog of the marginal reliability.

## Files

- `python/item_response_info.py` — from-scratch 2PL item information + test information curve + adaptive next-item selection. Demo (15 items spanning `b ∈ [−2.5, 2.5]`): information peaks near `θ = 0` (I = 2.83, SE = 0.60); adaptive selection picks item 8 (b = 0.36) at `θ = 0.5` and item 3 (b = −1.43) at `θ = −1.5`.
- `r/item_response_info.R` — `mirt::testinfo`, `mirt::iteminfo`, `catR` package for CAT simulations.

## When to use

- **Test blueprint validation** — verify the test measures well across the ability range you care about.
- **CAT simulation** — plan and validate an adaptive testing framework.
- **Reporting SEM(θ)** alongside estimated θ — modern operational testing.
- **Item-bank development** — identify gaps in information coverage.

## Extensions

- **3PL information** — `I_j(θ) = a² (P − c)² (1 − P) / (P (1 − c)²)`.
- **Polytomous** — GRM / GPCM have category-specific information contributions.
- **Multidimensional** — information is a matrix in multi-dimensional IRT.

## Assumptions & caveats

- **Item parameters treated as known** — information under-estimates SE when parameters are noisily estimated.
- **CAT bias** — early items chosen for one θ create attenuation in later θ estimates; exposure control and Sympson-Hetter methods needed.

## Run

```
python techniques/item-response-info/python/item_response_info.py
Rscript techniques/item-response-info/r/item_response_info.R
```

**Refs:** Lord, F.M. *Applications of Item Response Theory to Practical Testing Problems*, Lawrence Erlbaum, 1980; van der Linden, W.J. (ed.) *Elements of Adaptive Testing*, Springer, 2010.

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
