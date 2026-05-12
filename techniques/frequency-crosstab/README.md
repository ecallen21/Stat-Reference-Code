# Frequency Distributions & Cross-Tabulations (Reference §1.7)

Organizing categorical (or binned numeric) data into counts — the first move in exploratory analysis.

## One variable
- **Frequency table** — count per category.
- **Relative frequency** — `count / n` (proportion).
- **Cumulative frequency** — running total / running proportion (meaningful for *ordinal* categories).

## Two variables — cross-tabulation (contingency table)
Joint counts of two categorical variables, plus:
- **Marginal totals** — row sums, column sums, grand total.
- **Row %** — within each row (conditional distribution of the column variable given the row).
- **Column %** — within each column.
- **Total %** — share of the grand total.

Choosing which percentage to show depends on the question: "of the North region, what fraction said Yes?" → row %; "of the Yes responders, what fraction were North?" → column %.

## Binning numeric data
A frequency table of a continuous variable needs bins. Common bin-width rules (same ones a histogram uses, §1.9): **Sturges** `k = ⌈log₂ n + 1⌉`; **Scott** `h = 3.49·s·n^(−1/3)`; **Freedman–Diaconis** `h = 2·IQR·n^(−1/3)` (robust to outliers — usually the best default).

## Files
- `python/frequency_crosstab.py` — from-scratch frequency table, crosstab, margins, row/col/total %, FD/Scott/Sturges binning; compares against `pandas.value_counts` / `pandas.crosstab` / `scipy.stats.contingency.margins`
- `r/frequency_crosstab.R` — from-scratch + base `table`/`prop.table`/`xtabs`/`addmargins`, `hist(breaks=)`; notes on `janitor::tabyl`, `gmodels::CrossTable`
- `pyspark/frequency_crosstab.py` — `groupBy().count()` + window for cumulative totals; built-in `DataFrame.crosstab`; row % via column arithmetic

## Run
```
python techniques/frequency-crosstab/python/frequency_crosstab.py
Rscript techniques/frequency-crosstab/r/frequency_crosstab.R
python techniques/frequency-crosstab/pyspark/frequency_crosstab.py
```

**Ref:** Agresti & Finlay, *Statistical Methods for the Social Sciences*, 4th ed., Pearson, 2009. (Formal tests for contingency tables — chi-square, Fisher's exact — come in a later batch with Chapter 3.)
