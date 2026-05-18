"""Frequency distributions and cross-tabulations (Reference §1.7).

Building blocks of exploratory data analysis for categorical (and binned
numeric) data:

  - frequency table     : count per category
  - relative frequency  : proportion (count / n)
  - cumulative frequency: running total / running proportion (ordinal data)
  - cross-tabulation    : joint counts of two categorical variables
                          + row %, column %, total %, and marginal totals

The numeric-binning helper (`bin_counts`) uses the Sturges / Scott /
Freedman-Diaconis rules to choose the number of bins -- the same rules a
histogram uses (see Reference §1.9).
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

import math    # stdlib: scalar math (sqrt, log, exp, comb, lgamma, pi, ...)
from collections import Counter    # stdlib: dict subclass that counts occurrences (Counter([1,1,2]) -> {1:2, 2:1})
from typing import Hashable, Sequence    # stdlib: type hints (Hashable = dict-key; Sequence = indexable iterable)

import numpy as np    # numerical arrays + linear algebra (np.mean, np.linalg.lstsq, ...)


def frequency_table(x: Sequence[Hashable], sort_by: str = "value"):
    """Build a one-variable frequency distribution.

    Parameters
    ----------
    x : a sample of categorical (or already-binned) values.
    sort_by : ``"value"`` to sort by the category, ``"count"`` to sort by frequency descending.

    Returns
    -------
    list of dicts, one per category, with keys:
        ``"category"``  -- the category label
        ``"count"``     -- number of observations in this category
        ``"rel_freq"``  -- relative frequency (count / n), i.e. the proportion
        ``"cum_count"`` -- cumulative count up to and including this category
        ``"cum_rel"``   -- cumulative relative frequency (cum_count / n)

    Because each row is a dict, the output is self-describing: ``print(table)``
    shows the field names, and ``pandas.DataFrame(table)`` auto-names the columns.
    """
    counts = Counter(x)
    n = len(x)
    items = sorted(counts.items()) if sort_by == "value" else counts.most_common()
    out, run = [], 0
    for cat, c in items:
        run += c
        out.append({"category": cat, "count": c, "rel_freq": c / n,
                    "cum_count": run, "cum_rel": run / n})
    return out


def crosstab(row: Sequence[Hashable], col: Sequence[Hashable]):
    """Two-way contingency table of joint counts.

    Parameters
    ----------
    row, col : two parallel sequences of categorical values (same length).
        Each (row[i], col[i]) is one observation.

    Returns
    -------
    (row_labels, col_labels, counts) -- the labels in sorted order, and the
    contingency matrix as a list of lists.
    """
    if len(row) != len(col):
        raise ValueError("row and col must have the same length")
    rlabels = sorted(set(row))
    clabels = sorted(set(col))
    ridx = {v: i for i, v in enumerate(rlabels)}
    cidx = {v: i for i, v in enumerate(clabels)}
    table = [[0] * len(clabels) for _ in rlabels]
    for r, c in zip(row, col):
        table[ridx[r]][cidx[c]] += 1
    return rlabels, clabels, table


def margins(table):
    """Marginal totals from a 2D count table.

    Returns ``(row_totals, col_totals, grand_total)``.
    """
    row_tot = [sum(r) for r in table]
    col_tot = [sum(col) for col in zip(*table)]
    grand = sum(row_tot)
    return row_tot, col_tot, grand


def percentages(table, kind: str = "total"):
    """Convert a count table to a percent table.

    Parameters
    ----------
    table : 2D list of counts (rows x cols).
    kind : ``"row"`` -- each row sums to 100% (conditional on the row);
        ``"col"`` -- each column sums to 100% (conditional on the column);
        ``"total"`` -- the whole table sums to 100%.
    """
    row_tot, col_tot, grand = margins(table)
    out = []
    for i, r in enumerate(table):
        out_row = []
        for j, v in enumerate(r):
            if kind == "row":
                denom = row_tot[i]
            elif kind == "col":
                denom = col_tot[j]
            else:
                denom = grand
            out_row.append(100.0 * v / denom if denom else 0.0)
        out.append(out_row)
    return out


def bin_counts(x: Sequence[float], rule: str = "fd"):
    """Bin a numeric vector and count observations per bin.

    Parameters
    ----------
    x : numeric sample.
    rule : bin-width selector --
        ``"sturges"`` (k = ceil(log2 n + 1); fine for near-normal data),
        ``"scott"`` (3.49 * sd * n**(-1/3); assumes near-normal),
        ``"fd"`` (Freedman-Diaconis: 2 * IQR * n**(-1/3); robust to outliers, the usual default).

    Returns
    -------
    ``(edges, counts)`` -- bin edges (length k+1) and counts (length k).
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if rule == "sturges":
        k = math.ceil(math.log2(n) + 1)
        edges = np.linspace(arr.min(), arr.max(), k + 1)
    elif rule == "scott":
        h = 3.49 * arr.std(ddof=1) * n ** (-1 / 3)
        edges = np.arange(arr.min(), arr.max() + h, h)
    elif rule == "fd":  # Freedman-Diaconis
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        h = 2.0 * (q3 - q1) * n ** (-1 / 3)
        h = h or (arr.max() - arr.min()) / 10 or 1.0
        edges = np.arange(arr.min(), arr.max() + h, h)
    else:
        raise ValueError("rule must be 'sturges', 'scott', or 'fd'")
    counts, edges = np.histogram(arr, bins=edges)
    return edges, counts


def _print_crosstab(rlabels, clabels, table, title):
    print(title)
    row_tot, col_tot, grand = margins(table)
    width = max(8, *(len(str(c)) + 2 for c in clabels))
    header = "".ljust(10) + "".join(str(c).rjust(width) for c in clabels) + "Total".rjust(width)
    print(header)
    for lbl, r, rt in zip(rlabels, table, row_tot):
        print(str(lbl).ljust(10) + "".join(str(v).rjust(width) for v in r) + str(rt).rjust(width))
    print("Total".ljust(10) + "".join(str(v).rjust(width) for v in col_tot) + str(grand).rjust(width))


def library_versions(row, col):
    import pandas as pd
    from scipy.stats import contingency

    s = pd.Series(row)
    print("\npandas value_counts (relative):")
    print(s.value_counts(normalize=True).sort_index())
    ct = pd.crosstab(pd.Series(row, name="row"), pd.Series(col, name="col"), margins=True)
    print("\npandas crosstab with margins:")
    print(ct)
    print("\npandas crosstab row %:")
    print(pd.crosstab(pd.Series(row), pd.Series(col), normalize="index").round(3))
    obs = pd.crosstab(pd.Series(row), pd.Series(col)).values
    print("\nscipy.stats.contingency.margins:", contingency.margins(obs))


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    region = rng.choice(["North", "South", "East"], size=60, p=[0.5, 0.3, 0.2]).tolist()
    outcome = rng.choice(["Yes", "No"], size=60, p=[0.4, 0.6]).tolist()

    print("=== frequency table: region ===")
    for row in frequency_table(region):
        print(f"{row['category']:8s} count={row['count']:3d}  "
              f"rel={row['rel_freq']:.3f}  "
              f"cum={row['cum_count']:3d}  cumrel={row['cum_rel']:.3f}")

    rl, cl, tbl = crosstab(region, outcome)
    print()
    _print_crosstab(rl, cl, tbl, "=== cross-tab: region x outcome (counts) ===")
    print("\n=== row percentages ===")
    for lbl, r in zip(rl, percentages(tbl, "row")):
        print(f"{lbl:8s}: " + "  ".join(f"{cl[j]}={v:5.1f}%" for j, v in enumerate(r)))

    print("\n=== binned numeric (Freedman-Diaconis) ===")
    vals = rng.normal(50, 10, 200)
    edges, counts = bin_counts(vals, "fd")
    for lo, hi, c in zip(edges[:-1], edges[1:], counts):
        print(f"[{lo:6.2f}, {hi:6.2f}): {c}")

    library_versions(region, outcome)
