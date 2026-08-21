# Fractional Factorial Designs (Reference §16.4)

Full `2^k` factorial requires `2^k` runs — `2^7 = 128`, `2^10 = 1024`. **Fractional factorial** `2^(k − p)` runs only `1 / 2^p` of the grid by **aliasing** higher-order interactions with main effects and lower-order interactions.

## Generation

1. Start with a full `2^(k − p)` base design in the first `k − p` factors.
2. Assign each additional factor by a **generator** — a product of existing base columns.

Example: `k = 5, p = 1`, generator `E = ABCD` → 16-run design (vs 32 full).

## Resolution

| Resolution | Alias structure                                | Use                              |
|------------|-------------------------------------------------|----------------------------------|
| III        | main aliased with 2-factor interactions         | screening (many factors, cheap)  |
| IV         | main clear; 2fi aliased with 2fi               | most common workhorse            |
| V+         | main and 2fi both clear                        | when 2fi matter                  |

## Alias structure

For a defining relation like `I = ABCDE`, multiply each effect by the word (mod-2 XOR of factor sets) to find its alias. E.g. under `E = ABCD`:

```
A = BCDE     B = ACDE     C = ABDE     D = ABCE     E = ABCD
AB = CDE     AC = BDE     ...           (all 2fi aliased with 3fi)
```

Main effects and 2fi's stay clear of each other → **Resolution V**.

## Files

- `python/fractional_factorial.py` — from-scratch full and fractional-factorial generators + alias structure enumerator. Demos: `2^(5-1)` with `E = ABCD` (16 runs, Res V); `2^(7-2)` (32 runs, Res IV).
- `r/fractional_factorial.R` — `FrF2::FrF2(nruns, nfactors, generators)` (Groemping's canonical R package).

## When to use

- **Screening** many potential factors (5–15) cheaply.
- **Second-stage** experiments after RSM has narrowed the region.
- **Robustness studies** — Taguchi designs (fractional factorials for control × noise factors).

## Choosing a design

- **Screening**: Res III (Plackett-Burman is a special case) — many factors, main effects only.
- **Effect estimation**: Res IV (main effects clear) or Res V (main + 2fi clear).
- **Optimization**: augment with center points and consider CCD / Box-Behnken for the second-order fit (see `response-surface`).

## Assumptions & caveats

- **Effect sparsity** — most higher-order interactions are negligible. Report the fraction of variance explained by main effects.
- **Alias resolution** — only meaningful when the assumed aliases (usually higher-order) can plausibly be ignored.
- **Randomize run order** — protects against nuisance time trends.
- **Center points** — replicated center runs quantify pure error and detect curvature.

## Run

```
python techniques/fractional-factorial/python/fractional_factorial.py
Rscript techniques/fractional-factorial/r/fractional_factorial.R
```

**Refs:** Box, G.E.P., Hunter, W.G. & Hunter, J.S. *Statistics for Experimenters*, 2nd ed., Wiley, 2005; Montgomery, D.C. *Design and Analysis of Experiments*, 9th ed., Wiley, 2017.

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
