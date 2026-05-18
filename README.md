# Stat-Reference-Code

Working implementations of the techniques catalogued in
`stat_techniques_reference_v124.docx`, written in **R**, **Python**, and
**PySpark** (the latter only where distributed/large-data execution is
meaningful).

## Layout

Organized by technique, then language:

```
techniques/
  <technique-name>/
    README.md                # what it is, when to use it, formulas, assumptions, ref
    r/<technique>.R          # base-R "from scratch" + idiomatic-package version
    python/<technique>.py    # numpy/scipy "from scratch" + library version; example under __main__
    pyspark/<technique>.py   # Spark DataFrame version (when applicable)
```

Each implementation provides a *from-scratch* version (so the math is visible)
**and** an idiomatic library version (what you'd actually use), plus a small
runnable example.

## Running the examples

- Python: `pip install -r requirements.txt` then `python techniques/<name>/python/<name>.py`
- R: `Rscript techniques/<name>/r/<name>.R`
- PySpark: `python techniques/<name>/pyspark/<name>.py` (needs a local Spark install)

## Loading data

The examples in each technique file use small inline sample data so the script
is self-contained. To run the same technique on a real dataset, load the data
with your language's standard tools and pass the relevant **column** (a 1-D
sample) to the from-scratch / library function. Below is the boilerplate for
each common format in each language.

### Python (pandas / numpy)

The from-scratch functions accept any sequence (Python list, numpy array,
pandas `Series`, etc.). The library helpers use numpy/scipy directly.

```python
import pandas as pd
import numpy as np
from techniques.central_tendency.python.central_tendency import arithmetic_mean, geometric_mean

# --- CSV / TSV ------------------------------------------------------------
df = pd.read_csv("data.csv")                       # default comma-separated
df = pd.read_csv("data.tsv", sep="\t")
df = pd.read_csv("data.csv", parse_dates=["date_col"])

# --- Excel ----------------------------------------------------------------
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")     # needs `openpyxl`

# --- Parquet (preferred for large, typed datasets) ------------------------
df = pd.read_parquet("data.parquet")                     # needs `pyarrow` or `fastparquet`

# --- SQL database ---------------------------------------------------------
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pwd@host/db")  # or sqlite:///file.db
df = pd.read_sql("SELECT income FROM households WHERE year = 2024", engine)

# --- Plain text / numpy ---------------------------------------------------
arr = np.loadtxt("values.txt")                            # one number per line
arr = np.genfromtxt("data.csv", delimiter=",", skip_header=1)  # missing values -> NaN

# --- Pass a column into a technique function ------------------------------
x = df["income"].dropna().to_numpy()                     # drop NA before stats!
arithmetic_mean(x)
geometric_mean(x[x > 0])                                  # geometric mean needs x > 0
```

### R

Base R covers CSV/TSV; `readr`, `readxl`, `arrow`, and `DBI` cover the rest.
Pass a column with `df$col_name` or `df[["col_name"]]`.

```r
# --- CSV / TSV (base R; comma-separated assumed) --------------------------
df <- read.csv("data.csv", stringsAsFactors = FALSE)
df <- read.delim("data.tsv")                              # tab-separated

# --- Tidyverse / readr (faster, better defaults) --------------------------
library(readr)
df <- read_csv("data.csv")                                # auto-detects types
df <- read_tsv("data.tsv")

# --- Excel ----------------------------------------------------------------
library(readxl)
df <- read_excel("data.xlsx", sheet = "Sheet1")

# --- Parquet --------------------------------------------------------------
library(arrow)
df <- read_parquet("data.parquet")

# --- SQL database ---------------------------------------------------------
library(DBI); library(RPostgres)                          # or RSQLite, odbc, etc.
con <- dbConnect(Postgres(), dbname = "db", host = "host", user = "u", password = "p")
df  <- dbGetQuery(con, "SELECT income FROM households WHERE year = 2024")
dbDisconnect(con)

# --- Built-in datasets (handy for testing) --------------------------------
data(iris); data(mtcars)

# --- Pass a column into a technique function ------------------------------
source("techniques/central-tendency/r/central_tendency.R")
x <- na.omit(df$income)                                   # drop NAs first
arithmetic_mean(x)
geometric_mean_scratch(x[x > 0])
```

### PySpark

The PySpark functions in this repo take a `DataFrame` and the name of the
column(s) to summarize -- no need to collect the data to the driver.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.master("local[*]").appName("loader").getOrCreate()

# --- CSV (single file or a directory of files) ----------------------------
df = (spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv("data.csv"))                                  # or "s3a://bucket/path/*.csv"

# --- Parquet (preferred -- columnar, typed, splittable) -------------------
df = spark.read.parquet("data.parquet")                    # or a directory

# --- JSON, ORC, Avro ------------------------------------------------------
df = spark.read.json("data.json")
df = spark.read.orc("data.orc")

# --- JDBC (SQL databases) -------------------------------------------------
df = (spark.read.format("jdbc")
        .option("url", "jdbc:postgresql://host/db")
        .option("dbtable", "households")
        .option("user", "u").option("password", "p")
        .load())

# --- Hive / catalog tables (in a Spark cluster with a metastore) ----------
df = spark.table("default.households")
df = spark.sql("SELECT income FROM default.households WHERE year = 2024")

# --- From a local pandas DataFrame (great for unit tests) -----------------
import pandas as pd
df = spark.createDataFrame(pd.read_csv("data.csv"))

# --- Drop nulls before stats and pass into a technique function -----------
from techniques.central_tendency.pyspark.central_tendency import central_tendency
df = df.filter(F.col("income").isNotNull())
result = central_tendency(df, col="income")
```

**A few practical notes**

- **Missing values**: handle them *before* you summarize. Python: `df["col"].dropna()`; R: `na.omit(df$col)` or `complete.cases()`; PySpark: `df.filter(F.col("c").isNotNull())` or `df.na.drop(subset=["c"])`.
- **Numeric coercion**: if a column was read as text (common with CSV), cast it first. Python: `pd.to_numeric(s, errors="coerce")`; R: `as.numeric()`; PySpark: `F.col("c").cast("double")`.
- **Large data**: read Parquet, not CSV — typed, columnar, splittable across a Spark cluster. CSV is fine for small files (< ~1 GB) and easy interop.
- **Bridging Spark ↔ pandas**: small results — `df.toPandas()`; medium — `df.limit(n).toPandas()`; large — keep it in Spark and use the `pyspark/` variant of the technique.

## Language idiosyncrasies & gotchas

Real, repeated time-sinks when working across these three stacks. Skim this once
before you debug a "weird" result.

### Python / numpy / scipy / pandas

- **`\u` in string literals triggers a Unicode escape.** `"C:\users\file.csv"`
  is a `SyntaxError` because `\u` starts a `\uXXXX` escape. Fixes:
  - escape each backslash: `"C:\\users\\file.csv"`
  - use a raw string: `r"C:\users\file.csv"`
  - use forward slashes: `"C:/users/file.csv"` (works on Windows too)
  - or `pathlib.Path(r"C:\users\file.csv")`
  Same trap with `\n` (newline), `\t` (tab), `\r`, `\b`, `\x`, `\N`, `\0`.
  Python 3.12 turns most unrecognized `\x` into a `SyntaxWarning` that will
  become a `SyntaxError` later — fix them now.

- **`np.var` / `np.std` default to `ddof=0`** (population). Most other software
  (R `var`/`sd`, SAS, SPSS, Stata) defaults to `ddof=1` (sample). We pass
  `ddof=1` explicitly everywhere in this repo so the from-scratch and library
  numbers agree.

- **0-based indexing; half-open slices.** `x[0]` is the first element, `x[a:b]`
  excludes `b`. Coming from R, off-by-one bugs are very easy. Negative indices
  count from the end (`x[-1]` is the last element — different from R!).

- **`scipy.stats.mode` returns one value** (the smallest among ties); our
  from-scratch `mode()` returns a *list* of all tied values. Both are
  defensible; just know which you're calling.

- **`scipy.stats.kurtosis` returns *excess* kurtosis by default**
  (`fisher=True` → normal = 0). R's `moments::kurtosis` returns *non-excess*
  (normal = 3). `e1071::kurtosis(x, type = ...)` lets you choose.

- **`np.quantile` API churn.** Default is Hyndman–Fan type 7. In numpy < 1.22
  the option was `interpolation=`; from 1.22+ it's `method=`. In numpy 2.0
  `np.trapz` was renamed to `np.trapezoid` (the from-scratch `gini_trapezoid`
  in this repo `getattr`s its way around that).

- **`scipy.stats.mannwhitneyu` U1 sign convention.** It returns the U for
  the *first* argument, equal to `#(x1 > x2) + 0.5·#(x1 == x2)`. The
  rank-biserial conversion is therefore `r = 2·U1/(n1·n2) − 1`, not
  `1 − 2·U1/(n1·n2)` — the formula has flipped between textbooks depending on
  which U they used.

- **Floating-point traps.** `0.1 + 0.2 == 0.3` is `False`. Summing a million
  small floats with `sum()` accumulates rounding error; prefer
  `math.fsum(x)` or `np.add.reduce(x)`. Test floats with `math.isclose` /
  `np.isclose`, not `==`.

- **Mutable default arguments are shared across calls.** Never
  `def f(x=[]): ...` — the same list is reused every call. Default to `None`
  and create inside the function.

- **`is` vs `==`.** `is` checks object identity, `==` checks value. `[] is []`
  is `False` even though `[] == []` is `True`. Use `==` for value comparisons.

- **pandas `dropna()` behaviour.** Drops *any* row with a NaN by default; for
  one column do `df["c"].dropna()` (Series) or `df.dropna(subset=["c"])`
  (DataFrame). Don't forget — most stats functions silently propagate `nan` to
  the result, so a single missing value can poison a mean.

### R

- **1-based indexing; ranges `1:n` are inclusive on both ends.** `x[1]` is the
  first element. Negative indices *exclude* (`x[-1]` is everything but the
  first — different from Python!).

- **`var()` / `sd()` use the `n − 1` divisor by default**, no `ddof` knob. If
  you genuinely need the population version, multiply by `(n − 1) / n`.

- **`mad()` is scaled by default.** `mad(x)` uses `constant = 1.4826` so it's
  a consistent estimator of σ at the normal. Pass `constant = 1` for the
  raw MAD.

- **`quantile(type = ...)`.** Default is type 7 (numpy/pandas match this).
  `fivenum()` uses Tukey hinges, which can differ slightly from the
  type-7 Q1/Q3 — both are "correct," they're just different conventions.

- **Vector recycling silently zips short vectors.** `c(1, 2, 3, 4) + c(10, 20)`
  produces `c(11, 22, 13, 24)` with only a warning (and only if the long length
  isn't a multiple of the short). One of R's biggest footguns.

- **`NA` propagates.** `mean(c(1, 2, NA))` is `NA`. Most stat functions take
  `na.rm = TRUE`; some don't (e.g. `cor()` uses `use = ...` instead). Decide
  up front whether you want listwise deletion or per-pair.

- **`T` and `F` are *variables* aliased to `TRUE`/`FALSE`** — they can be
  overwritten (`T <- 0` is valid). Always use `TRUE`/`FALSE` in code that
  matters.

- **`df[, "col"]` may or may not return a data frame**, depending on
  `drop`. `df[, "col"]` on a base R `data.frame` returns a vector by default;
  on a `tibble` it stays a tibble. `df[["col"]]` and `df$col` are
  unambiguous — prefer them.

- **`<-` vs `=`.** Use `<-` for assignment. Inside a function call, `=` is
  argument binding, not assignment (`f(x = 5)` passes `5` as `x`;
  `f(x <- 5)` *assigns* `5` to a global `x` and then passes the value, which
  is almost never what you want).

- **`stringsAsFactors` history.** Before R 4.0, `data.frame()` and
  `read.csv()` converted character columns to factors by default. Many old
  tutorials and packages still assume this. R ≥ 4.0 defaults to `FALSE`.

- **`==` on factors compares levels, not labels** — and silently returns `NA`
  if the levels differ. Cast with `as.character()` first when in doubt.

### PySpark

- **Lazy evaluation.** `df.filter(...)`, `select(...)`, `withColumn(...)`,
  `groupBy(...)` build a plan but execute nothing. Computation triggers only
  on an **action**: `count()`, `collect()`, `show()`, `first()`, `write...`,
  `toPandas()`, etc. So timing a `filter` says nothing; time the action.

- **`approxQuantile` is approximate.** With `relativeError = 0.01` you might
  see a quantile off by up to 1% — fine for the median of a billion rows,
  not fine if you need an exact answer. Set `relativeError = 0` for exact
  (much more expensive — a full sort).

- **DataFrames are immutable; every operation returns a new one.**
  `df.withColumn("x", ...)` does *not* mutate `df` in place. Reassign:
  `df = df.withColumn("x", ...)`.

- **Schema inference on CSV is expensive** (it scans the file twice). For
  production, pass an explicit `schema=StructType([...])`. Parquet/ORC carry
  the schema natively — no scan needed.

- **`count()` is a full scan.** Don't sprinkle `df.count()` everywhere — each
  call re-executes the whole plan unless the DataFrame is cached. For
  diagnostics during dev, `.cache()` or `.persist()` before repeated actions.

- **Column references: `F.col("c")` vs string `"c"`.** Most functions accept
  both, but inside arithmetic / boolean expressions you need a `Column`:
  `df.filter(F.col("c") > 0)`, not `df.filter("c" > 0)` (that compares the
  *string* `"c"` to `0`).

- **Python UDFs are slow** — they ship rows over a JVM↔Python boundary and
  serialize each value. Prefer the built-in `pyspark.sql.functions` (`F.mean`,
  `F.when`, `F.regexp_replace`, ...). If you must write Python, use a
  **pandas UDF** (vectorized).

- **`null` is not `NaN`.** SQL nulls and floating-point NaN are different
  things in Spark. Filter nulls with `F.col("c").isNull()` /
  `.isNotNull()` or `df.na.drop(subset=["c"])`. `isnan()` is a separate
  function for the float NaN.

- **Window functions need an explicit frame** for `sum`/`avg`/etc. over an
  ordered window. `Window.orderBy("x")` alone defaults to a *range* frame
  from `unboundedPreceding` to `currentRow`, which causes subtle bugs with
  ties. Be explicit: `.rowsBetween(Window.unboundedPreceding, Window.currentRow)`.

- **`groupBy` doesn't preserve order;** add an `orderBy(...)` after.

- **PySpark string indexing.** `substring` is **1-based**, not 0-based —
  `substring("abc", 1, 2) == "ab"`. Inherited from SQL, surprising in Python.

### Cross-language traps (when porting code between R / Python / PySpark)

- **Indexing**: R is 1-based and inclusive; Python is 0-based and half-open;
  PySpark `substring` is 1-based.
- **Sample vs population default**: R `var`/`sd` use `n − 1`; numpy uses `n`;
  Spark's `var_samp`/`stddev_samp` use `n − 1`, `var_pop`/`stddev_pop` use `n`.
  When numbers disagree across languages, this is almost always why.
- **Excess vs non-excess kurtosis**: scipy `kurtosis(fisher=True)` (default)
  is excess; R `moments::kurtosis` is non-excess; Spark's `kurtosis`
  aggregation is excess. Subtract/add 3 as needed.
- **Quantile definitions**: R `quantile(type = 7)`, numpy default, pandas
  default, and Spark `percentile_approx` all match (linear interp /
  Hyndman–Fan type 7). SAS, Stata, and SPSS use *different* defaults — if
  you're reconciling against a SAS report, ask which type they used.
- **Missing values**: in R, `NA` is its own thing and propagates; in Python,
  pandas uses `NaN` for numerics (and `pd.NA` for nullable types); in
  Spark, SQL `NULL` and float `NaN` are different. Each language has its own
  null-handling idioms — don't mix them in your head.

## Notation & Conventions

These names mean the same thing in every file unless a function's own docstring overrides them.

### Data inputs
| Name | Meaning |
|------|---------|
| `x`, `x1`, `x2` | A 1-D sample (a Python sequence / numpy array / R numeric vector). `x1` and `x2` are two independent samples (e.g. group 1 and group 2). |
| `y` | A second variable paired with `x` (same length), for bivariate techniques. Not used in Batch 1. |
| `w` | Weights, one per observation in `x` (`len(w) == len(x)`). Used by `weighted_mean`. Survey weights, meta-analysis weights, etc. |
| `df` *(PySpark)* | A Spark `DataFrame`. |
| `col`, `row_col`, `col_col`, `value_col`, `weight_col`, `group_col` *(PySpark)* | Column **names** (strings) inside `df`. |
| `groups` | A list/list-of-lists, one sub-sample per group (used by one-way ANOVA effect sizes). |
| `subjects` | A list of repeated-measurements vectors, one per subject (used by within-subject CV). |
| `events`, `person_time_total` | Event count and total person-time at risk (for incidence rates). |
| `x` / `n` *(proportions)* | Number of successes / sample size, in functions that take a count rather than a vector. |

### Parameters & options
| Name | Meaning |
|------|---------|
| `n` | Sample size (`len(x)`). |
| `proportion` | Fraction trimmed/Winsorized from **each** tail. `0.2` means drop the bottom 20% and the top 20%. Must be in `[0, 0.5)`. |
| `ddof` | "Delta degrees of freedom" — divisor is `n − ddof`. `ddof=1` is the sample (Bessel-corrected) version; `ddof=0` is the population version. |
| `conf` | Confidence level for a CI (e.g. `0.95`). |
| `alpha` | Significance level / `1 − conf`. |
| `bias` *(skew/kurtosis)* | `True` → method-of-moments estimator; `False` → bias-corrected `G1` / `G2` that most software reports. |
| `excess` *(kurtosis)* | `True` → normal distribution has kurtosis 0; `False` → normal has kurtosis 3. |
| `kind` *(quantile)* | Hyndman–Fan quantile definition (1, 6, or 7 — see `techniques/quantiles`). |
| `as_percent` *(CV)* | If `True`, return `100 × SD/mean` instead of `SD/mean`. |
| `relative_error` *(PySpark)* | Tolerance for `approxQuantile` (`0` = exact, expensive; `0.001` is a good default). |
| `bias_corrected` *(Gini)* | Multiply by `n/(n−1)` for the small-sample correction. |
| `max_iter`, `tol` *(Huber)* | IRLS stopping criteria. |
| `k` *(Huber)* | Tuning constant for the Huber loss; `1.345` gives ~95% efficiency at the normal. |

### Conventions in the code
- Python's from-scratch functions default to **`ddof = 1`** (sample variance / SD). numpy defaults to `ddof = 0`, so we pass `ddof=1` explicitly when comparing.
- R's `var()` / `sd()` use `n − 1` by default; both languages therefore agree on the from-scratch defaults.
- Functions that need strictly positive inputs (geometric/harmonic mean, geometric CV, log-based things) raise `ValueError` (Python) / `stop()` (R) if a non-positive value is passed.
- "From-scratch" implementations exist for **transparency** (you can see the formula); for production code prefer the library version shown alongside.
- Library cross-checks use optional packages (`pingouin`, `lmoments3`, R's `effsize` / `lmom` / `ineq` / `MASS` / `psych` / `DescTools`); each file degrades to a note rather than erroring if the package is missing.

## Progress

Building in batches; we walk through each batch together before moving on.

### Batch 1 — Chapter 1: Descriptive Statistics

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [central-tendency](techniques/central-tendency) | 1.1 | ✅ | ✅ | ✅ |
| 2 | [dispersion](techniques/dispersion) | 1.2 | ✅ | ✅ | ✅ |
| 3 | [robust-location-scale](techniques/robust-location-scale) | 1.3, 1.26 | ✅ | ✅ | N/A |
| 4 | [shape-skewness-kurtosis](techniques/shape-skewness-kurtosis) | 1.4 | ✅ | ✅ | N/A |
| 5 | [quantiles](techniques/quantiles) | 1.5 | ✅ | ✅ | ✅ |
| 6 | [ecdf](techniques/ecdf) | 1.13 | ✅ | ✅ | ✅ |
| 7 | [effect-sizes](techniques/effect-sizes) | 1.6, 1.25 | ✅ | ✅ | N/A |
| 8 | [frequency-crosstab](techniques/frequency-crosstab) | 1.7 | ✅ | ✅ | ✅ |
| 9 | [rates-proportions](techniques/rates-proportions) | 1.8 | ✅ | ✅ | ✅ |
| 10 | [coefficient-of-variation](techniques/coefficient-of-variation) | 1.22, 1.33 | ✅ | ✅ | ✅ |
| 11 | [gini-lorenz](techniques/gini-lorenz) | 1.23 | ✅ | ✅ | ✅ |
| 12 | [l-moments](techniques/l-moments) | 1.24 | ✅ | ✅ | N/A |

### Batch 2 — Chapter 3: Basic Inferential Statistics

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [t-tests](techniques/t-tests) (one/two-sample, Student/Welch, paired) | 3.4 | ✅ | ✅ | ✅ |
| 2 | [z-tests](techniques/z-tests) (means + proportions) | 3.7, 3.22 | ✅ | ✅ | ✅ |
| 3 | [chi-square-tests](techniques/chi-square-tests) (GOF + independence) | 3.5 | ✅ | ✅ | ✅ |
| 4 | [fisher-exact](techniques/fisher-exact) (2×2 + OR) | 3.6 | ✅ | ✅ | N/A |
| 5 | [binomial-test](techniques/binomial-test) (exact + mid-p + normal) | 3.22 | ✅ | ✅ | N/A |
| 6 | [one-way-anova](techniques/one-way-anova) (classic + Welch + Brown-Forsythe) | 3.8, 3.9 | ✅ | ✅ | ✅ |
| 7 | [post-hoc-tests](techniques/post-hoc-tests) (Tukey HSD, Dunnett, Games-Howell) | 3.10, 3.11, 3.16 | ✅ | ✅ | N/A |
| 8 | [multiple-comparisons](techniques/multiple-comparisons) (Bonferroni, Holm, Hochberg, BH, BY) | 3.13, 3.14 | ✅ | ✅ | N/A |
| 9 | [normality-tests](techniques/normality-tests) (Shapiro-Wilk, D'Agostino, AD, Lilliefors, JB) | 3.19, 3.40 | ✅ | ✅ | N/A |
| 10 | [homogeneity-of-variance](techniques/homogeneity-of-variance) (Levene, Brown-Forsythe, Bartlett) | 3.20, 3.55 | ✅ | ✅ | N/A |
| 11 | [equivalence-testing-tost](techniques/equivalence-testing-tost) | 3.21 | ✅ | ✅ | N/A |
| 12 | [outlier-tests](techniques/outlier-tests) (Grubbs, Dixon's Q, Generalized ESD, IQR rule) | 3.25 | ✅ | ✅ | N/A |

### Batch 3 — Chapter 4: Correlation

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [pearson-correlation](techniques/pearson-correlation) (r + Fisher z CI) | 4.1 | ✅ | ✅ | ✅ |
| 2 | [spearman-rank-correlation](techniques/spearman-rank-correlation) | 4.2 | ✅ | ✅ | ✅ |
| 3 | [kendalls-tau](techniques/kendalls-tau) (τ-a, τ-b) | 4.3 | ✅ | ✅ | ✅ |
| 4 | [point-biserial-correlation](techniques/point-biserial-correlation) | 4.4 | ✅ | ✅ | N/A |
| 5 | [partial-correlation](techniques/partial-correlation) (partial + semi-partial) | 4.5 | ✅ | ✅ | N/A |
| 6 | [intraclass-correlation](techniques/intraclass-correlation) (ICC(1/2/3, 1/k)) | 4.6 | ✅ | ✅ | N/A |
| 7 | [polychoric-correlation](techniques/polychoric-correlation) (tetrachoric + polychoric) | 4.7 | ✅ | ✅ | N/A |
| 8 | [distance-correlation](techniques/distance-correlation) | 4.8 | ✅ | ✅ | N/A |
| 9 | [concordance-correlation](techniques/concordance-correlation) (Lin's CCC) | 4.9 | ✅ | ✅ | N/A |
| 10 | [cramers-v-phi](techniques/cramers-v-phi) (+ Bergsma bias correction) | 4.10 | ✅ | ✅ | N/A |
| 11 | [goodman-kruskal-somers](techniques/goodman-kruskal-somers) (γ, Somers' D, τ-b on tables) | 4.11, 4.12 | ✅ | ✅ | N/A |
| 12 | [mutual-information](techniques/mutual-information) (discrete + binned + MIC) | 4.14, 4.15 | ✅ | ✅ | N/A |

### Cleanup pass (Batch 3.5) — backfills for Batches 2 & 3

| Addition | Where | Ref §|
|----------|-------|------|
| Scheffé's method (arbitrary contrasts) + Tamhane T2 + Dunnett T3 | extends `post-hoc-tests` | 3.12, 3.37 |
| [wald-lrt-score](techniques/wald-lrt-score) — Wald, LRT, Score (Rao) tests | new | 3.18, 3.30, 3.31, 3.33 |
| [delta-method](techniques/delta-method) — SE/CI for a function of estimates | new | 3.29 |
| Goodman–Kruskal λ and G-K τ (nominal PRE measures) | extends `goodman-kruskal-somers` | 4.11 |
| [eta-correlation-ratio](techniques/eta-correlation-ratio) — η between categorical X and continuous Y | new | 4.13 |
| MIC (Maximal Information Coefficient) | extends `mutual-information` | 4.15 |

### Batch 4 — Chapter 5: Linear Regression

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [simple-linear-regression](techniques/simple-linear-regression) (OLS β, SE, t-test, CI/PI) | 5.1 | ✅ | ✅ | ✅ |
| 2 | [multiple-linear-regression](techniques/multiple-linear-regression) (OLS, ANOVA, R², adj R²) | 5.2 | ✅ | ✅ | ✅ |
| 3 | [regression-diagnostics](techniques/regression-diagnostics) (leverage, Cook's D, DFFITS, DFBETAS) | 5.6, 5.30, 5.39 | ✅ | ✅ | N/A |
| 4 | [specification-tests](techniques/specification-tests) (Breusch-Pagan, White, Durbin-Watson, RESET) | 5.7, 5.21 | ✅ | ✅ | N/A |
| 5 | [collinearity-diagnostics](techniques/collinearity-diagnostics) (VIF, condition index) | 5.23 | ✅ | ✅ | N/A |
| 6 | [polynomial-regression](techniques/polynomial-regression) (raw + orthogonal) | 5.3 | ✅ | ✅ | N/A |
| 7 | [interaction-terms](techniques/interaction-terms) (continuous × continuous + categorical × continuous + centering) | 5.16, 5.24, 5.25, 5.37 | ✅ | ✅ | N/A |
| 8 | [regularization](techniques/regularization) (ridge, lasso, elastic net + CV) | 5.9, 5.17 | ✅ | ✅ | ✅ |
| 9 | [variable-selection](techniques/variable-selection) (stepwise, best subsets, AIC/BIC) | 5.8, 5.18, 5.19, 5.36 | ✅ | ✅ | N/A |
| 10 | [weighted-least-squares](techniques/weighted-least-squares) (known + IRWLS) | 5.10 | ✅ | ✅ | N/A |
| 11 | [robust-regression](techniques/robust-regression) (Huber M via IRLS) | 5.11 | ✅ | ✅ | N/A |
| 12 | [splines-segmented](techniques/splines-segmented) (piecewise linear, natural cubic, breakpoint search) | 5.4, 5.22, 5.26, 5.34 | ✅ | ✅ | N/A |

### Cleanup pass (Batch 4.5) — backfills for Batch 4

| Addition | Where | Ref §|
|----------|-------|------|
| [categorical-variable-coding](techniques/categorical-variable-coding) (Dummy, Effect, Helmert, Deviation) | new | 5.32 |
| [standardized-coefficients](techniques/standardized-coefficients) (beta weights + dominance analysis) | new | 5.38 |

### Batch 5 — Chapter 6: Nonparametric Methods

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [sign-test](techniques/sign-test) | 6.1 | ✅ | ✅ | N/A |
| 2 | [wilcoxon-signed-rank](techniques/wilcoxon-signed-rank) | 6.2 | ✅ | ✅ | N/A |
| 3 | [mann-whitney](techniques/mann-whitney) | 6.3 | ✅ | ✅ | ✅ |
| 4 | [kruskal-wallis](techniques/kruskal-wallis) | 6.4 | ✅ | ✅ | N/A |
| 5 | [friedman-test](techniques/friedman-test) (+ Kendall's W) | 6.5 | ✅ | ✅ | N/A |
| 6 | [kolmogorov-smirnov](techniques/kolmogorov-smirnov) (1-/2-sample) | 6.7 | ✅ | ✅ | ✅ |
| 7 | [moods-median](techniques/moods-median) | 6.8 | ✅ | ✅ | N/A |
| 8 | [jonckheere-terpstra](techniques/jonckheere-terpstra) (ordered alternatives) | 6.10 | ✅ | ✅ | N/A |
| 9 | [kernel-density-estimation](techniques/kernel-density-estimation) (5 kernels + bandwidth rules) | 6.21 | ✅ | ✅ | ✅ |
| 10 | [local-regression-loess](techniques/local-regression-loess) | 6.22 | ✅ | ✅ | N/A |
| 11 | [hodges-lehmann](techniques/hodges-lehmann) (robust location + CI) | 6.29 | ✅ | ✅ | N/A |
| 12 | [theil-sen-slope](techniques/theil-sen-slope) (robust slope + Sen CI) | 6.32 | ✅ | ✅ | N/A |

### Batch 6 — Chapter 7: Generalized Linear Models

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [logistic-regression](techniques/logistic-regression) (binary; IRLS, ORs, deviance, McFadden R²) | 7.1 | ✅ | ✅ | ✅ |
| 2 | [ordinal-logistic](techniques/ordinal-logistic) (proportional odds) | 7.2 | ✅ | ✅ | N/A |
| 3 | [multinomial-logistic](techniques/multinomial-logistic) (softmax MLE) | 7.4 | ✅ | ✅ | ✅ |
| 4 | [probit-regression](techniques/probit-regression) (+ AME) | 7.10 | ✅ | ✅ | N/A |
| 5 | [poisson-regression](techniques/poisson-regression) (log link, offset for rates, IRR) | 7.12, 7.43 | ✅ | ✅ | ✅ |
| 6 | [negative-binomial-regression](techniques/negative-binomial-regression) (joint MLE on β, θ) | 7.13 | ✅ | ✅ | N/A |
| 7 | [modified-poisson](techniques/modified-poisson) (sandwich SEs for risk ratios) | 7.9, 7.53 | ✅ | ✅ | N/A |
| 8 | [firth-logistic](techniques/firth-logistic) (penalized MLE for separation) | 7.7, 7.51 | ✅ | ✅ | N/A |
| 9 | [gamma-regression](techniques/gamma-regression) (log link, dispersion) | 7.25 | ✅ | ✅ | ✅ |
| 10 | [glm-diagnostics](techniques/glm-diagnostics) (Pearson/deviance, Hosmer-Lemeshow, RQR) | 7.40, 7.41, 7.55 | ✅ | ✅ | N/A |
| 11 | [marginal-effects](techniques/marginal-effects) (AME, MEM, MER, discrete-change) | 7.37, 7.47 | ✅ | ✅ | N/A |
| 12 | [overdispersion-tests](techniques/overdispersion-tests) (Pearson φ, score, LRT Poisson vs NB) | 7.35, 7.42, 7.54 | ✅ | ✅ | N/A |

Later batches will cover the remaining chapters (Survival, Time Series,
Bayesian, Causal Inference, ML, ...).

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

For methodological background see related publications: [link to ORCID or selected DOIs].

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data — every example in this repository uses small inline or programmatically generated synthetic samples.

**References.** Each technique's README cites the original methodological references for its specific algorithm. General references underpinning the reference guide (`stat_techniques_reference_v124.docx`): Casella & Berger, *Statistical Inference*, 2nd ed.; Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, 2nd ed.; Wackerly, Mendenhall & Scheaffer, *Mathematical Statistics with Applications*, 7th ed.; Harrell, *Regression Modeling Strategies*, 2nd ed.

## License

[MIT](LICENSE)
