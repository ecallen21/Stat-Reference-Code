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

## Using a technique on your own data — a walkthrough

If you've never used Python before, you'll bump into three patterns the first time you try to apply a technique from this repo to your own data. This section spells them all out with one worked example.

### The end-to-end recipe

```python
# --- 1. Imports the technique needs (copy these from the top of the file) ----
from __future__ import annotations
import pandas as pd
import math
from collections import Counter
from typing import Hashable, Sequence

# --- 2. Load YOUR data ------------------------------------------------------
df = pd.read_excel(
    "path/to/your_file.xlsx",      # ← swap in your file path
    sheet_name="Sheet1",            # ← swap in your sheet name
)

# --- 3. Paste the function definition exactly as it is in the repo ----------
#        (you can also `from frequency_crosstab import frequency_table`
#         if the file is on your Python path)
def frequency_table(x: Sequence[Hashable], sort_by: str = "value"):
    counts = Counter(x)
    n = len(x)
    items = sorted(counts.items()) if sort_by == "value" else counts.most_common()
    out, run = [], 0
    for cat, c in items:
        run += c
        out.append({"category": cat, "count": c, "rel_freq": c / n,
                    "cum_count": run, "cum_rel": run / n})
    return out

# --- 4. Create a REAL variable from the column of df you want to analyze ----
#        Run print(df.columns.tolist()) once if you don't remember the names.
my_column = df["YOUR_COLUMN_NAME"]   # ← swap in the column you want to count

# --- 5. Call the function and PRINT the result ------------------------------
#        Each row is a dict, so the field names are right there in the output:
#        {"category": ..., "count": ..., "rel_freq": ..., "cum_count": ..., "cum_rel": ...}
result = frequency_table(my_column)
print(result)

# --- 6. (Optional) Pretty-print row by row using the dict keys --------------
print("\n  category                count   rel_freq   cum_count   cum_rel")
for row in result:
    print(f"  {str(row['category']):22s} {row['count']:5d}    "
          f"{row['rel_freq']:5.3f}      {row['cum_count']:5d}     {row['cum_rel']:6.3f}")

# --- 7. (Optional) Drop straight into a pandas DataFrame --------------------
#        Because the rows are dicts, the column headers come along for free.
print("\n", pd.DataFrame(result))
```

Change two things to make it work on **your** data: the file path on line 9 and the column name on line 29. Nothing else.

### The three traps to watch out for

The recipe above is built specifically to avoid these. They're the most common reasons "I copied the code and nothing happened":

**1. Parameter name ≠ variable name.** Inside a `def` line like

```python
def frequency_table(x, sort_by="value"):
```

the `x` is a **local placeholder** — the function's internal label for "whatever you hand in." It does not create a global variable called `x`. So this errors out:

```python
print(frequency_table(x))           # NameError: name 'x' is not defined
```

You need to pass *your* variable in. Whatever you named it. The function will internally call it `x` while it works with it; outside the function, `x` doesn't exist.

**2. Functions return values; they don't print them.** When you do:

```python
frequency_table(my_column)
```

…the function does compute the table, but the result vanishes into the void unless you capture or print it:

```python
result = frequency_table(my_column)   # capture it
print(frequency_table(my_column))      # or print it inline
```

This is a deliberate Python convention: a function that *computes* should return its result (so you can do more with it — print, plot, save to CSV); a function that *displays* should print. The repo's functions all *compute*, so you always have to print or capture.

**3. A DataFrame is not a column.** When you load Excel / CSV / Parquet, you get a **whole table** in a variable like `df`. Most functions in this repo expect a single 1-D column, not the full table. Pick the column you want with `df["column_name"]`:

```python
print(frequency_table(df))             # wrong: tries to count column names
print(frequency_table(df["Status"]))   # right: counts values in the 'Status' column
```

To see what columns your file has, run once:

```python
print(df.columns.tolist())
print(df.head())                       # first 5 rows, gives a sense of the values
```

### A 30-second template you can adapt to any technique

For any function in `techniques/<name>/python/<name>.py`:

```python
# 1. Imports (copy from the top of the file)
# 2. Load your data into a DataFrame `df`
# 3. Either copy the function in, or `from <name> import <function>`
# 4. column = df["YOUR_COLUMN_NAME"]      ← extract what you want
# 5. result = the_function(column)        ← compute
# 6. print(result)                        ← display
```

That's the whole pattern. The only file-to-file variation is which imports you need (look at the top of the file), what the function's parameters are (look at its `def` line and docstring), and what shape of input it expects (a 1-D column? two columns? a 2-D table?).

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

## Python imports glossary

Every Python file in this repo starts with a block of `import` statements that pull in tools from Python itself plus a few external packages. This glossary explains what each one is and where you'll see it.

### Python standard library (built in — no install needed)

| Import line | What it gives you | Where you see it |
|-------------|-------------------|------------------|
| `from __future__ import annotations` | A Python *language flag* that postpones the evaluation of type hints. Lets us write modern hints like `int \| None` and `list[str]` even on slightly older Python. Cosmetic; doesn't change runtime behavior. | Every file (the safe default). |
| `import math` | Python's basic **math module**. Scalar functions: `math.sqrt`, `math.log`, `math.exp`, `math.pi`, `math.comb(n, k)` (binomial coefficient), `math.floor`, `math.lgamma`. | Most files — anywhere we need a non-vector math operation. |
| `from collections import Counter` | A dictionary subclass that **counts occurrences**: `Counter([1, 1, 2, 3]) == {1: 2, 2: 1, 3: 1}`. Methods: `.most_common(k)`, `.values()`. | `frequency-crosstab`, `central-tendency`, `wilcoxon-signed-rank` (tie counts), `goodman-kruskal-somers`, `mutual-information`. |
| `from itertools import combinations` | Iterator over all `C(n, k)` subsets of size `k`. `combinations([1,2,3], 2)` → `(1,2), (1,3), (2,3)`. | `variable-selection` (best-subsets), `theil-sen-slope` (pairwise slopes), `effect-sizes` (Cliff's δ), `kendalls-tau`. |
| `import bisect` | Binary search on a *sorted* list. `bisect.bisect_right(sorted_x, t)` returns the insertion point — handy for counting `#{xᵢ ≤ t}` in `O(log n)`. | `ecdf`, `kolmogorov-smirnov`. |
| `from typing import Sequence, Hashable, Callable` | **Type hints** — purely documentation; not enforced at runtime. `Sequence` = "anything you can index and iterate over" (list, tuple, numpy array, pandas Series); `Hashable` = "can be a dict key or set element" (numbers, strings, tuples); `Callable` = "a function". | Most files, in function signatures. |

### Scientific Python (`pip install numpy scipy pandas statsmodels scikit-learn`)

| Import line | What it gives you | Where you see it |
|-------------|-------------------|------------------|
| `import numpy as np` | The numerical-array library. `np.array`, `np.mean`, `np.var`, `np.linalg.lstsq` (least squares), `np.linalg.pinv` (pseudo-inverse), `np.random.default_rng(seed)` (modern RNG), array broadcasting. | Anywhere we work with vectors / matrices. |
| `from scipy import stats` | Probability distributions, statistical tests, related utilities. `stats.norm`, `stats.t`, `stats.chi2`, `stats.f`, `stats.poisson`, `stats.ttest_ind`, `stats.kruskal`, `stats.kendalltau`, `stats.binomtest`, `stats.mannwhitneyu`. | Every file that needs a p-value, CDF, or PPF. |
| `from scipy import optimize` | Numerical optimizers. `minimize_scalar` (1-D minimization on a bracket), `minimize` (multi-dim BFGS / Nelder-Mead). | `polychoric-correlation`, `negative-binomial-regression`, `ordinal-logistic`, `multinomial-logistic`. |
| `from scipy import special` | Special functions. `special.gammaln` (log-gamma; numerically stable factorial logs), `special.beta`, `special.digamma`. | `negative-binomial-regression`, `overdispersion-tests`. |
| `import pandas as pd` | DataFrame library. `pd.read_csv`, `pd.DataFrame`, `df["col"].dropna()`, `pd.crosstab`. | Mainly the library-cross-check sections (e.g. `pd.Series` + `value_counts`). |
| `import statsmodels.api as sm` | Statistical models with full statistical output (SEs, CIs, p-values, F-tests). `sm.OLS`, `sm.Logit`, `sm.Probit`, `sm.MNLogit`, `sm.GLM`, `sm.NegativeBinomial`, `sm.WLS`, `sm.RLM` (robust). | The library-cross-check block of most regression files. |
| `import statsmodels.formula.api as smf` | The R-style formula interface to statsmodels (`y ~ x1 + C(group)`). | `categorical-variable-coding`. |
| `from sklearn... import ...` | scikit-learn for ML utilities. `Ridge`, `Lasso`, `ElasticNet`, `mutual_info_score`, `normalized_mutual_info_score`. | `regularization`, `mutual-information`. |

### Optional packages (only used in some files; install on demand)

| Package | What it adds | Where |
|---------|-------------|-------|
| `pingouin` | Pre-baked stats functions for psychologists: `compute_effsize`, `partial_corr`, `intraclass_corr`, `anova`. | `effect-sizes`, `partial-correlation`, `intraclass-correlation`, `eta-correlation-ratio`. |
| `lmoments3` | L-moment estimators and parameter fitting. | `l-moments`. |
| `firthlogist` | Firth's penalized logistic regression. | `firth-logistic`. |

### How an import statement reads

- `import X` → load module `X`, refer to its things as `X.foo`.
- `import X as Y` → same but give it a short alias `Y` (`numpy as np`).
- `from X import a, b` → only pull names `a` and `b` into this file's namespace (`from math import sqrt` lets you write `sqrt(2)` directly instead of `math.sqrt(2)`).
- `from X.Y import Z` → reach into a sub-module: `from scipy import stats` then `stats.norm.cdf(0)`.

If you ever wonder where something came from, search the file's top imports — there's no implicit "stdlib magic" in Python; everything is named explicitly.


These names are the **parameter names** used inside the function definitions throughout the repo. They are *labels* — placeholders the function uses for whatever you hand in. They are **not** variables that exist in your script, and you do **not** rename them to match your data.

### How to read these function signatures

Take this function from `techniques/frequency-crosstab`:

```python
def frequency_table(x, sort_by="value"):
    counts = Counter(x)
    ...
```

The `x` inside `(x, sort_by="value")` is the function's **internal label** — it's saying "whatever you pass in, I'll call it `x` while I'm working with it." It does **not** mean "go create a variable called `x`."

When you **call** the function, you pass your own variable (with whatever name you actually gave it):

```python
my_documents = ["a.pdf", "b.docx", "a.pdf"]
print(frequency_table(my_documents))         # ← your variable becomes 'x' inside
```

Inside the function, `x` refers to `my_documents`. Outside the function, `x` doesn't exist at all.

**The single most common mistake** is to read the parameter name as if it were the data itself and try to call the function with that exact name:

```python
print(frequency_table(x))                # NameError: x is not defined
print(frequency_table(documents))        # NameError if you never created 'documents'
```

Always pass *your* variable, with *its* name.

### Parameter names used in the repo

This is the dictionary the function signatures speak. When you see `x` in a `def` line, the function is saying "give me a 1-D sample." When you see `proportion`, the function expects a number between 0 and 0.5. Etc.

| Parameter name in `def` line | What the function expects you to pass in |
|------------------------------|------------------------------------------|
| `x`, `x1`, `x2` | A 1-D sample (Python list / numpy array / pandas Series / R numeric vector). `x1` and `x2` are two **independent** samples (e.g. group 1, group 2). |
| `y` | A second variable paired with `x` (same length), for bivariate techniques (regression, correlation, paired tests). |
| `w` | Weights, one per observation in `x` (`len(w) == len(x)`). Used by `weighted_mean`. Survey weights, meta-analysis weights, etc. |
| `df` *(PySpark only)* | A Spark `DataFrame`. (Not the same `df` as in pandas — the PySpark version.) |
| `col`, `row_col`, `col_col`, `value_col`, `weight_col`, `group_col` *(PySpark)* | Column **names** (strings) inside the Spark `DataFrame`. |
| `groups` | A list of per-group samples — i.e. a list of lists / list of arrays, one entry per group (used by ANOVA, Kruskal-Wallis, ...). |
| `subjects` | A list of per-subject repeated-measurement vectors, one entry per subject (used by within-subject CV, Friedman, ...). |
| `events`, `person_time_total` | Event count and total person-time at risk (for incidence-rate functions). |
| `x` / `n` *(proportions)* | When a function expects a **count + sample size** rather than a vector (e.g. `binomial_test(x=42, n=100)`), `x` is the number of successes and `n` is the total trials. |

### Option / tuning parameters

These are the *control knobs* of the functions — they tell the function *how* to compute, not what data to compute on.

| Parameter | Meaning |
|-----------|---------|
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

### Return values: what each function gives you back

Every function in this repo returns something **self-describing** — when you `print()` the result you can see what each value means without consulting the docstring. Three shapes show up:

**1. A dict** — used when the return is "one row of a table" with several columns, *or* when the return is a bag of named scalars (estimate, SE, p-value, CI, …).
```python
result = pearson_correlation(x, y)
# {'r': 0.78, 't': 6.4, 'df': 28, 'p_value': 1.3e-06, ...}
result["r"]            # access by key
```
List-of-dicts is used when the return is *several rows of a table* — e.g. `frequency_table(region)` returns one dict per category. You can drop that straight into pandas: `pd.DataFrame(result)` and the column headers come along for free.

**2. A `NamedTuple`** — used when the return is "a small fixed set of distinct things" (e.g. a CI's lower and upper, or a Lorenz curve's two parallel arrays).
```python
ci = ci_wilson(8, 100)
# CI(lower=0.041, upper=0.150)        ← prints with field names
ci.lower, ci.upper                     # access by name
lo, hi = ci_wilson(8, 100)             # still unpacks like a regular tuple
```
NamedTuples are the right tool here because they preserve the tuple-unpacking idiom (`lo, hi = func(...)`) *and* add field names — no caller code has to change.

**3. R named vector / named list** — R's equivalents. A function that would return a NamedTuple in Python returns `c(lower = ..., upper = ...)` in R; a function that would return a dict in Python returns `list(...)` in R. Both print with their labels and are indexed by name:
```r
ci <- ci_wilson(8, 100)            # c(lower = 0.041, upper = 0.150)
ci[["lower"]]                       # 0.041
```

Quick reference:

| Return shape | Python | R | When |
|---|---|---|---|
| One row of a table | `dict` | `list(...)` | Single record with several labeled fields |
| Several rows of a table | `list[dict]` | `data.frame(...)` | One dict per row; pandas-ready |
| Small fixed set of distinct things | `NamedTuple` | `c(name = value, ...)` | CI's `(lower, upper)`, fit's `(beta, mu)`, etc. |

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

### Batch 7 — Chapter 8: Categorical Data Analysis (Beyond GLMs)

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [mcnemar-test](techniques/mcnemar-test) (asymptotic, continuity, exact, mid-p, Newcombe CI) | 8.2, 8.18 | ✅ | ✅ | ✅ |
| 2 | [cochran-mantel-haenszel](techniques/cochran-mantel-haenszel) (CMH stat + MH common OR w/ RBG SE + Woolf) | 8.3, 8.16 | ✅ | ✅ | ✅ |
| 3 | [cohens-kappa](techniques/cohens-kappa) (two-rater κ + Fleiss ASE + PABAK) | 8.4 | ✅ | ✅ | ✅ |
| 4 | [fleiss-kappa](techniques/fleiss-kappa) (≥3 raters + per-category κ) | 8.4 | ✅ | ✅ | N/A |
| 5 | [weighted-kappa](techniques/weighted-kappa) (linear + quadratic; bootstrap SE) | 8.4 | ✅ | ✅ | N/A |
| 6 | [breslow-day](techniques/breslow-day) (OR homogeneity + Tarone correction) | 8.6 | ✅ | ✅ | N/A |
| 7 | [bowker-stuart-maxwell](techniques/bowker-stuart-maxwell) (symmetry + marginal homogeneity on k×k) | 8.7, 8.15 | ✅ | ✅ | N/A |
| 8 | [log-linear-models](techniques/log-linear-models) (Poisson GLM on multi-way tables + agreement models) | 8.1, 8.14 | ✅ | ✅ | N/A |
| 9 | [correspondence-analysis](techniques/correspondence-analysis) (CA via SVD + MCA via Burt) | 8.5 | ✅ | ✅ | N/A |
| 10 | [bradley-terry](techniques/bradley-terry) (MLE via MM algorithm; Wald SEs) | 8.8 | ✅ | ✅ | N/A |
| 11 | [continuation-ratio](techniques/continuation-ratio) (K-1 binomial GLMs + proportional variant) | 8.9 | ✅ | ✅ | N/A |
| 12 | [adjacent-category-logit](techniques/adjacent-category-logit) (common β via BFGS + pairwise) | 8.10 | ✅ | ✅ | N/A |

### Batch 8 — Chapter 9: Multivariate Methods

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [hotellings-t2](techniques/hotellings-t2) (one- and two-sample; F-approx) | 9.1, 9.28 | ✅ | ✅ | N/A |
| 2 | [manova](techniques/manova) (Wilks / Pillai / Hotelling-Lawley / Roy) | 9.2 | ✅ | ✅ | N/A |
| 3 | [pca](techniques/pca) (SVD-based; loadings + scores + explained var) | 9.3 | ✅ | ✅ | ✅ |
| 4 | [exploratory-factor-analysis](techniques/exploratory-factor-analysis) (PAF + varimax + promax) | 9.4 | ✅ | ✅ | N/A |
| 5 | [hierarchical-clustering](techniques/hierarchical-clustering) (single/complete/average/Ward + cophenetic) | 9.8 | ✅ | ✅ | N/A |
| 6 | [k-means](techniques/k-means) (Lloyd + k-means++ + multi-restart) | 9.9 | ✅ | ✅ | ✅ |
| 7 | [dbscan](techniques/dbscan) (density clustering + k-distance heuristic) | 9.11 | ✅ | ✅ | N/A |
| 8 | [gaussian-mixture-models](techniques/gaussian-mixture-models) (EM + BIC/AIC selection) | 9.12 | ✅ | ✅ | ✅ |
| 9 | [cluster-validation](techniques/cluster-validation) (silhouette, CH, DB, elbow, gap statistic) | 9.14 | ✅ | ✅ | N/A |
| 10 | [lda-qda](techniques/lda-qda) (pooled and per-class Σ; Bayes-optimal classifier) | 9.30 | ✅ | ✅ | N/A |
| 11 | [canonical-correlation](techniques/canonical-correlation) (generalized eigen + Bartlett) | 9.29 | ✅ | ✅ | N/A |
| 12 | [multidimensional-scaling](techniques/multidimensional-scaling) (classical + non-metric with PAV isotonic) | 9.32, 9.25 | ✅ | ✅ | N/A |

**Chapter 9 subsections also covered by earlier batches** (no separate implementation needed):
- **§9.28** Hotelling's T² — same technique as §9.1; both listed in [`hotellings-t2`](techniques/hotellings-t2).
- **§9.31** Correspondence Analysis / MCA — built in Batch 7 as [`correspondence-analysis`](techniques/correspondence-analysis) (§8.5).

### Batch 9 — Chapter 10: Resampling and Computationally Intensive Methods

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [nonparametric-bootstrap](techniques/nonparametric-bootstrap) (case resampling; percentile/basic/normal CIs) | 10.1 | ✅ | ✅ | ✅ |
| 2 | [parametric-bootstrap](techniques/parametric-bootstrap) (fit model then simulate) | 10.2 | ✅ | ✅ | N/A |
| 3 | [bca-bootstrap](techniques/bca-bootstrap) (BCa + comparison of all four CI methods) | 10.3, 10.14 | ✅ | ✅ | N/A |
| 4 | [block-bootstrap](techniques/block-bootstrap) (moving + circular blocks for dependent data) | 10.4 | ✅ | ✅ | N/A |
| 5 | [wild-bootstrap](techniques/wild-bootstrap) (Rademacher / Mammen weights for heteroscedasticity) | 10.5 | ✅ | ✅ | N/A |
| 6 | [jackknife](techniques/jackknife) (LOO SEs + bias correction + jackknife-after-bootstrap) | 10.6, 10.17 | ✅ | ✅ | N/A |
| 7 | [permutation-tests](techniques/permutation-tests) (two-sample + correlation + regression) | 10.7, 10.16 | ✅ | ✅ | ✅ |
| 8 | [cross-validation](techniques/cross-validation) (K-fold + stratified + LOOCV) | 10.8, 10.12 | ✅ | ✅ | ✅ |
| 9 | [monte-carlo-simulation](techniques/monte-carlo-simulation) (power + CI coverage) | 10.9 | ✅ | ✅ | N/A |
| 10 | [subsampling](techniques/subsampling) (Politis-Romano-Wolf + m-out-of-n bootstrap) | 10.10, 10.15 | ✅ | ✅ | N/A |
| 11 | [double-bootstrap](techniques/double-bootstrap) (one-step Beran calibration) | 10.11 | ✅ | ✅ | N/A |
| 12 | [nested-cv](techniques/nested-cv) (K_outer × K_inner + stratified repeated CV) | 10.13 | ✅ | ✅ | N/A |

### Batch 10 — Chapter 11: Survival Analysis

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [kaplan-meier](techniques/kaplan-meier) (KM + Greenwood + log-log CI + median + risk table) | 11.2, 11.1, 11.45, 11.46, 11.61, 11.68 | ✅ | ✅ | N/A |
| 2 | [nelson-aalen](techniques/nelson-aalen) (cumulative hazard + kernel-smoothed rate) | 11.3, 11.65 | ✅ | ✅ | N/A |
| 3 | [log-rank-test](techniques/log-rank-test) (weighted family + stratified) | 11.4, 11.5, 11.6, 11.7, 11.47, 11.62 | ✅ | ✅ | N/A |
| 4 | [cox-ph](techniques/cox-ph) (partial-likelihood; Efron+Breslow ties; counting-process input) | 11.8, 11.16, 11.42, 11.54, 11.59, 11.63, 11.64, 11.66 | ✅ | ✅ | N/A |
| 5 | [cox-diagnostics](techniques/cox-diagnostics) (Grambsch-Therneau + 4 residual types) | 11.33, 11.53 | ✅ | ✅ | N/A |
| 6 | [parametric-survival](techniques/parametric-survival) (exp / Weibull / lognormal / loglogistic AFT + piecewise-exp) | 11.10–11.15, 11.44, 11.58 | ✅ | ✅ | N/A |
| 7 | [competing-risks](techniques/competing-risks) (Aalen-Johansen + cause-specific Cox + Fine-Gray + Gray) | 11.22, 11.23, 11.24, 11.25 | ✅ | ✅ | N/A |
| 8 | [recurrent-events](techniques/recurrent-events) (Andersen-Gill + PWP + WLW + gap-time) | 11.17, 11.18, 11.19, 11.41, 11.51 | ✅ | ✅ | N/A |
| 9 | [frailty-models](techniques/frailty-models) (shared gamma frailty via moment estimator) | 11.26 | ✅ | ✅ | N/A |
| 10 | [multi-state-models](techniques/multi-state-models) (illness-death + state occupation) | 11.27, 11.52 | ✅ | ✅ | N/A |
| 11 | [rmst](techniques/rmst) (restricted mean survival time + difference test) | 11.29, 11.67 | ✅ | ✅ | N/A |
| 12 | [penalized-cox](techniques/penalized-cox) (elastic-net Cox via coordinate descent) | 11.21 | ✅ | ✅ | N/A |

**Chapter 11 subsections deferred** (specialized; will be picked up in later batches):
§11.9 Aalen additive · §11.20 Royston-Parmar splines · §11.28/40/56 cure models · §11.30 landmark analysis · §11.31 random survival forests (→ ML) · §11.32/57 joint longitudinal-survival · §11.34/60/69 interval-censored · §11.35/50 pseudo-observations · §11.36 IPCW · §11.37 win ratio · §11.38 time-dependent ROC · §11.39 dynamic prediction · §11.43 relative survival · §11.48 doubly-truncated · §11.49 conditional survival · §11.55 linear transformation · §11.70–72 (methodological discussions, not techniques).

**PySpark N/A across Batch 10** — MLlib does not ship survival models, and distributed survival is a research topic. For very large data, aggregate risk sets on Spark and run the fitter on the driver.

### Batch 11 — Chapter 12: Longitudinal and Repeated Measures

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [repeated-measures-anova](techniques/repeated-measures-anova) (SS decomposition + GG / HF corrections) | 12.1 | ✅ | ✅ | N/A |
| 2 | [linear-mixed-models](techniques/linear-mixed-models) (REML profile likelihood + BLUPs + ICC) | 12.2, 12.13, 12.16, 12.20, 12.25, 12.26, 12.27, 12.29, 12.30, 12.32, 12.33 | ✅ | ✅ | N/A |
| 3 | [generalized-linear-mixed-models](techniques/generalized-linear-mixed-models) (Gauss-Hermite MLE) | 12.3, 12.23 | ✅ | ✅ | N/A |
| 4 | [gee](techniques/gee) (IRWLS + sandwich SE; independence / exch / AR(1)) | 12.8, 12.24, 12.31 | ✅ | ✅ | N/A |
| 5 | [growth-curve-models](techniques/growth-curve-models) (random int + slope; quadratic) | 12.4 | ✅ | ✅ | N/A |
| 6 | [group-based-trajectory](techniques/group-based-trajectory) (EM K-class polynomial mixture + BIC) | 12.5, 12.6, 12.7 | ✅ | ✅ | N/A |
| 7 | [markov-transition-models](techniques/markov-transition-models) (MLE transition matrix + stationary + order test) | 12.9 | ✅ | ✅ | N/A |
| 8 | [cross-lagged-panel](techniques/cross-lagged-panel) (2-wave CLPM + person-centered RI-CLPM) | 12.10, 12.18 | ✅ | ✅ | N/A |
| 9 | [nonlinear-mixed-effects](techniques/nonlinear-mixed-effects) (two-stage NLME) | 12.12 | ✅ | ✅ | N/A |
| 10 | [kenward-roger](techniques/kenward-roger) (Satterthwaite-style contrast test) | 12.17 | ✅ | ✅ | N/A |
| 11 | [multilevel-mediation](techniques/multilevel-mediation) (within/between decomposition + MC CI) | 12.22 | ✅ | ✅ | N/A |
| 12 | [mixed-effects-location-scale](techniques/mixed-effects-location-scale) (two-stage MELS) | 12.21 | ✅ | ✅ | N/A |

**Chapter 12 subsections deferred** to later batches: §12.11 intensive longitudinal / EMA · §12.14 multivariate longitudinal · §12.15 measurement invariance (→ SEM Ch 19) · §12.19 doubly-robust (→ causal Ch 15) · §12.28 spaghetti plots (viz) · §12.34 pitfalls (discussion).

**PySpark N/A across Batch 11** — MLlib has no mixed-model support.

### Batch 12 — Chapter 13: Time Series Analysis

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [acf-pacf](techniques/acf-pacf) (ACF + PACF + Ljung-Box + CCF + Mann-Kendall) | 13.1, 13.9, 13.42, 13.48 | ✅ | ✅ | N/A |
| 2 | [stationarity-tests](techniques/stationarity-tests) (ADF + KPSS + Phillips-Perron + reconciliation) | 13.2, 13.8, 13.53 | ✅ | ✅ | N/A |
| 3 | [arima](techniques/arima) (from-scratch ARMA MLE + AIC order + LB residuals) | 13.4, 13.5, 13.52 | ✅ | ✅ | N/A |
| 4 | [sarima-arimax](techniques/sarima-arimax) (SARIMA + ARIMAX / regression w/ ARIMA errors) | 13.6, 13.25 | ✅ | ✅ | N/A |
| 5 | [exponential-smoothing](techniques/exponential-smoothing) (SES + Holt + Holt-Winters + ETS/TBATS notes) | 13.3, 13.43, 13.56 | ✅ | ✅ | N/A |
| 6 | [seasonal-decomposition](techniques/seasonal-decomposition) (classical + STL + X-13 note) | 13.24, 13.47, 13.54 | ✅ | ✅ | N/A |
| 7 | [var-cointegration](techniques/var-cointegration) (VAR + Engle-Granger + ECM + Johansen note) | 13.12, 13.13, 13.44 | ✅ | ✅ | N/A |
| 8 | [granger-causality](techniques/granger-causality) (F-test on nested regressions + caveats) | 13.50 | ✅ | ✅ | N/A |
| 9 | [garch](techniques/garch) (GARCH(1,1) MLE + notes on DCC/BEKK/CCC) | 13.11, 13.33 | ✅ | ✅ | N/A |
| 10 | [state-space-kalman](techniques/state-space-kalman) (Kalman filter + local level + local trend + forecast intervals) | 13.17, 13.20, 13.55 | ✅ | ✅ | N/A |
| 11 | [structural-breaks-its](techniques/structural-breaks-its) (Chow + Bai-Perron scan + ITS regression) | 13.7, 13.10 | ✅ | ✅ | N/A |
| 12 | [forecast-evaluation-cv](techniques/forecast-evaluation-cv) (expanding-window CV + MAE/MAPE/MASE + bottom-up reconciliation) | 13.23, 13.31, 13.35, 13.36, 13.45, 13.51 | ✅ | ✅ | N/A |

**Chapter 13 subsections deferred** (specialized / ML / frequency-domain, will be picked up in later batches):
§13.14 HMM · §13.15/26/46 regime switching / TAR / SETAR / TVAR · §13.16 ARFIMA · §13.18/19/58/59 spectral / wavelet / EMD / locally-stationary · §13.21/27/28 Prophet / BSTS / NN forecasting · §13.22/29/39/41/57 DTW / anomaly / features / classification / similarity · §13.30 forecast combination · §13.32 count TS · §13.34 functional-coefficient · §13.37 hierarchical TS · §13.40 stochastic volatility · §13.49 functional TS.

**PySpark N/A across Batch 12** — MLlib has no classical time-series support.

### Batch 13 — Catch-up: subsections deferred from earlier chapters

Twelve techniques deferred from Batches 7–12 (Chapters 8, 9, 11, 13), grouped for coverage across topics.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [k-medoids](techniques/k-medoids) (PAM: BUILD + SWAP) | 9.11 | ✅ | ✅ | N/A |
| 2 | [procrustes-analysis](techniques/procrustes-analysis) (SVD-based orthogonal alignment + optional scale) | 9.16 | ✅ | ✅ | N/A |
| 3 | [permanova](techniques/permanova) (pseudo-F on a distance matrix + label permutation) | 9.17 | ✅ | ✅ | N/A |
| 4 | [mantel-test](techniques/mantel-test) (correlation of two distance matrices + partial Mantel) | 9.18 | ✅ | ✅ | N/A |
| 5 | [box-m-mauchly](techniques/box-m-mauchly) (equality of covariances + sphericity + GG/HF ε) | 9.3, 12.2 | ✅ | ✅ | N/A |
| 6 | [multivariate-outlier-detection](techniques/multivariate-outlier-detection) (classical Mahalanobis + Fast-MCD) | 9.6, 9.7 | ✅ | ✅ | N/A |
| 7 | [anosim](techniques/anosim) (rank-based analog of PERMANOVA) | 9.19 | ✅ | ✅ | N/A |
| 8 | [generalized-ordered-logit](techniques/generalized-ordered-logit) (partial PO + Brant test) | 8.35 | ✅ | ✅ | N/A |
| 9 | [landmark-analysis](techniques/landmark-analysis) (immortal-time-bias-safe survival with time-varying exposure + super-landmark) | 11.24 | ✅ | ✅ | N/A |
| 10 | [interval-censored-survival](techniques/interval-censored-survival) (Turnbull NPMLE via EM + parametric Weibull MLE) | 11.20 | ✅ | ✅ | N/A |
| 11 | [regime-switching-markov](techniques/regime-switching-markov) (Hamilton 1989: forward-backward EM on Gaussian HMM) | 13.14, 13.15 | ✅ | ✅ | N/A |
| 12 | [spectral-analysis](techniques/spectral-analysis) (raw + Daniell + Welch periodograms) | 13.18 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 13** — all techniques are single-node inferential procedures on covariance / distance / hidden-state structures that need the full sample on one driver; aggregate in Spark then fit on the driver.

### Batch 14 — Chapter 14: Bayesian Inference

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [conjugate-priors](techniques/conjugate-priors) (Beta-Binomial + Gamma-Poisson + Normal-Normal) | 14.1, 14.2, 14.3 | ✅ | ✅ | N/A |
| 2 | [mcmc-metropolis-hastings](techniques/mcmc-metropolis-hastings) (random-walk MH + Haario adaptation + ESS + R-hat) | 14.6 | ✅ | ✅ | N/A |
| 3 | [gibbs-sampler](techniques/gibbs-sampler) (Normal-InvGamma + 8-schools hierarchical) | 14.7 | ✅ | ✅ | N/A |
| 4 | [hamiltonian-mc](techniques/hamiltonian-mc) (leapfrog HMC + NUTS notes) | 14.8 | ✅ | ✅ | N/A |
| 5 | [bayesian-linear-regression](techniques/bayesian-linear-regression) (Normal-InvGamma conjugate + Zellner g-prior + posterior predictive) | 14.10, 14.11 | ✅ | ✅ | N/A |
| 6 | [bayesian-hierarchical-models](techniques/bayesian-hierarchical-models) (partial pooling + 8-schools Gibbs) | 14.15, 14.16 | ✅ | ✅ | N/A |
| 7 | [bayesian-glms](techniques/bayesian-glms) (logistic + Poisson via MH + Laplace-approx proposal) | 14.12, 14.13 | ✅ | ✅ | N/A |
| 8 | [bayesian-model-comparison](techniques/bayesian-model-comparison) (WAIC + PSIS-LOO + DIC + Bayes-factor caveats) | 14.20, 14.21, 14.22 | ✅ | ✅ | N/A |
| 9 | [posterior-predictive-checks](techniques/posterior-predictive-checks) (Bayesian p-values + test-statistic overlay) | 14.19 | ✅ | ✅ | N/A |
| 10 | [variational-inference](techniques/variational-inference) (mean-field Gaussian VI with reparameterization gradient + CAVI) | 14.24, 14.25 | ✅ | ✅ | N/A |
| 11 | [empirical-bayes](techniques/empirical-bayes) (Beta-Binomial EB + James-Stein estimator) | 14.17, 14.18 | ✅ | ✅ | N/A |
| 12 | [credible-intervals-hpd](techniques/credible-intervals-hpd) (ETI + HPD + Kruschke ROPE decision) | 14.9, 14.23 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 14** — Bayesian inference is inherently sequential (sampling chains, ELBO ascent); Spark is used for pre-aggregation, not the fitter itself.

### Batch 15 — Catch-up: missed subsections from earlier batches

Twelve techniques deferred from Batches 11-12 (Chapters 12-13). Time-series-heavy: HMM, long-memory, wavelets, DTW, TS anomaly detection, forecast combination, hierarchical forecasting, stochastic volatility, decomposable / Prophet-like forecasting, count TS, TS features + classification, plus multivariate longitudinal.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [hmm](techniques/hmm) (categorical HMM: forward-backward + Viterbi + Baum-Welch EM) | 13.14 | ✅ | ✅ | N/A |
| 2 | [arfima](techniques/arfima) (fractional differencing + GPH log-periodogram estimator) | 13.16 | ✅ | ✅ | N/A |
| 3 | [wavelet-analysis](techniques/wavelet-analysis) (Haar + Daubechies-4 DWT + universal-threshold denoising) | 13.19 | ✅ | ✅ | N/A |
| 4 | [dynamic-time-warping](techniques/dynamic-time-warping) (DTW distance + Sakoe-Chiba band + alignment path) | 13.22 | ✅ | ✅ | N/A |
| 5 | [ts-anomaly-detection](techniques/ts-anomaly-detection) (Hampel + STL-residual + predictive-residual approaches) | 13.29 | ✅ | ✅ | N/A |
| 6 | [forecast-combination](techniques/forecast-combination) (simple + trimmed mean + Bates-Granger + Granger-Ramanathan) | 13.30 | ✅ | ✅ | N/A |
| 7 | [hierarchical-forecasting](techniques/hierarchical-forecasting) (bottom-up + top-down + MinT reconciliation) | 13.37 | ✅ | ✅ | N/A |
| 8 | [stochastic-volatility](techniques/stochastic-volatility) (SV model: particle filter + Kalman-QMLE) | 13.40 | ✅ | ✅ | N/A |
| 9 | [decomposable-forecasting](techniques/decomposable-forecasting) (Prophet-style trend + Fourier + holidays) | 13.21 | ✅ | ✅ | N/A |
| 10 | [count-time-series](techniques/count-time-series) (Poisson-INAR(1) MoM + Poisson-INGARCH conditional MLE) | 13.32 | ✅ | ✅ | N/A |
| 11 | [multivariate-longitudinal](techniques/multivariate-longitudinal) (two-stage bivariate random-intercept LMM) | 12.14 | ✅ | ✅ | N/A |
| 12 | [ts-features-classification](techniques/ts-features-classification) (14 features + 1-NN DTW + 1-NN feature classifiers) | 13.39, 13.41 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 15** — all techniques are single-series / single-node procedures; distribute over series (per-key groupby) then fit on each executor.

### Batch 16 — Catch-up: missed subsections across Chapters 5 / 8 / 10 / 11 / 12 / 14

Twelve techniques spread across earlier chapters. Fills common regression flavors (Tobit, quantile, beta, penalized), exact 2×2 tests, refined bootstrap, joint longitudinal-survival, additive-hazards and cure survival, and two more Bayesian tools (ABC, BMA).

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [fisher-exact-barnard](techniques/fisher-exact-barnard) (Fisher's conditional + Barnard's unconditional exact 2x2 tests) | 8.4 | ✅ | ✅ | N/A |
| 2 | [zero-inflated-regression](techniques/zero-inflated-regression) (ZIP + hurdle Poisson via mixture likelihood) | 5.24 | ✅ | ✅ | N/A |
| 3 | [tobit-regression](techniques/tobit-regression) (left / right / two-sided Tobit MLE for censored outcomes) | 5.19 | ✅ | ✅ | N/A |
| 4 | [quantile-regression](techniques/quantile-regression) (Koenker-Bassett pinball loss + smoothed BFGS + LP) | 5.15 | ✅ | ✅ | N/A |
| 5 | [beta-regression](techniques/beta-regression) (Ferrari-Cribari-Neto Beta(mu, phi) MLE + variable-precision) | 5.20 | ✅ | ✅ | N/A |
| 6 | [ridge-lasso-elasticnet](techniques/ridge-lasso-elasticnet) (closed-form ridge + coord-descent LASSO/EN + reg path) | 5.9, 5.10 | ✅ | ✅ | N/A |
| 7 | [studentized-bootstrap](techniques/studentized-bootstrap) (bootstrap-t + nested inner-boot variance) | 10.4 | ✅ | ✅ | N/A |
| 8 | [joint-longitudinal-survival](techniques/joint-longitudinal-survival) (two-stage LME BLUPs + time-varying Cox) | 12.10 | ✅ | ✅ | N/A |
| 9 | [additive-aalen](techniques/additive-aalen) (Aalen additive-hazards least-squares increments + sup test) | 11.14 | ✅ | ✅ | N/A |
| 10 | [cure-models](techniques/cure-models) (Berkson-Gage mixture cure with logistic pi + Weibull latency MLE) | 11.22 | ✅ | ✅ | N/A |
| 11 | [abc-approximate-bayesian](techniques/abc-approximate-bayesian) (rejection ABC + Beaumont local-regression adjust) | 14.27 | ✅ | ✅ | N/A |
| 12 | [bayesian-model-averaging](techniques/bayesian-model-averaging) (BIC-approx BMA + PIP over all 2^p subsets) | 14.26 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 16** — all techniques are single-node inferential procedures. For scale, split by group (`groupBy(...).applyInPandas(...)`) and fit on each executor.

### Batch 17 — Catch-up: another 12 across Chapters 5 / 9 / 11 / 12 / 14

Twelve more catch-ups. Focus on regression flavors that were missing (nonlinear, IV, Heckman, GAM, splines, sandwich SE), multivariate manifold and blind-source methods, competing-risks and trajectory-mixture models, plus two lighter Bayesian tools (Bayesian optimization + Laplace approximation).

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [nonlinear-regression](techniques/nonlinear-regression) (Levenberg-Marquardt NLS with numerical Jacobian) | 5.13 | ✅ | ✅ | N/A |
| 2 | [iv-2sls](techniques/iv-2sls) (two-stage least squares + weak-instrument F) | 5.22 | ✅ | ✅ | N/A |
| 3 | [heckman-selection](techniques/heckman-selection) (two-step probit + Mills-ratio-corrected outcome eq) | 5.21 | ✅ | ✅ | N/A |
| 4 | [gam](techniques/gam) (penalized cubic-spline GAM with GCV smoothing) | 5.14 | ✅ | ✅ | N/A |
| 5 | [splines-regression](techniques/splines-regression) (cubic + natural + B-spline basis regression) | 5.12 | ✅ | ✅ | N/A |
| 6 | [bayesian-optimization](techniques/bayesian-optimization) (GP surrogate + Expected-Improvement acquisition) | 14.28 | ✅ | ✅ | N/A |
| 7 | [kernel-pca](techniques/kernel-pca) (centered kernel-matrix eigendecomposition for nonlinear DR) | 9.10 | ✅ | ✅ | N/A |
| 8 | [independent-components](techniques/independent-components) (FastICA blind-source separation) | 9.9 | ✅ | ✅ | N/A |
| 9 | [sandwich-robust-se](techniques/sandwich-robust-se) (HC0/HC1/HC3 + cluster-robust SEs) | 5.7, 5.8 | ✅ | ✅ | N/A |
| 10 | [fine-gray](techniques/fine-gray) (subdistribution hazards for competing risks) | 11.9 | ✅ | ✅ | N/A |
| 11 | [latent-growth-mixture](techniques/latent-growth-mixture) (EM over K linear latent trajectories) | 12.13 | ✅ | ✅ | N/A |
| 12 | [laplace-approximation](techniques/laplace-approximation) (Gaussian Laplace posterior + INLA notes) | 14.29 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 17** — all are single-node inferential procedures. Distribute by key when needed.

### Batch 18 — Catch-up: another 12 across Chapters 3/5/10/11/12/14/15/16/18

Twelve more spread across inference, regression, resampling, survival, panel/DiD, Bayesian, and design of experiments.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [multiple-testing-corrections](techniques/multiple-testing-corrections) (Bonferroni + Holm + Hochberg + BH + BY + Storey q) | 3.30, 4.24 | ✅ | ✅ | N/A |
| 2 | [truncated-regression](techniques/truncated-regression) (truncated-normal MLE for selected samples) | 5.18 | ✅ | ✅ | N/A |
| 3 | [fixed-effects-panel](techniques/fixed-effects-panel) (within + between + RE + Hausman FE-vs-RE test) | 12.31, 12.32 | ✅ | ✅ | N/A |
| 4 | [diff-in-diff](techniques/diff-in-diff) (2x2 DID + two-way FE + staggered-adoption caveats) | 15.4 | ✅ | ✅ | N/A |
| 5 | [conformal-prediction](techniques/conformal-prediction) (split-conformal intervals with any base learner) | 10.19 | ✅ | ✅ | N/A |
| 6 | [harrell-c-index](techniques/harrell-c-index) (Harrell C + Uno IPCW C-index for survival discrimination) | 11.6 | ✅ | ✅ | N/A |
| 7 | [multiple-imputation](techniques/multiple-imputation) (MICE chained equations + Rubin combining rules) | 18.6 | ✅ | ✅ | N/A |
| 8 | [dirichlet-process-mixture](techniques/dirichlet-process-mixture) (CRP Gibbs on DP Gaussian mixture) | 14.31 | ✅ | ✅ | N/A |
| 9 | [gaussian-process-regression](techniques/gaussian-process-regression) (RBF GP + marginal-likelihood hyperparameters) | 14.32 | ✅ | ✅ | N/A |
| 10 | [response-surface](techniques/response-surface) (CCD + BBD + quadratic fit + stationary-point analysis) | 16.11 | ✅ | ✅ | N/A |
| 11 | [latin-square-design](techniques/latin-square-design) (Latin-square randomization + row/col-blocked ANOVA) | 16.6 | ✅ | ✅ | N/A |
| 12 | [mars](techniques/mars) (Multivariate Adaptive Regression Splines: hinge functions + GCV pruning) | 5.28 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 18** — all single-node inferential procedures. Distribute by key when needed.

### Batch 19 — Catch-up: heavy on causal inference (Ch 15) plus post-hoc / non-parametric picks

Twelve more, this one weighted toward Chapter 15 (Causal Inference) which had been untouched before Batch 18's diff-in-diff.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [tukey-hsd](techniques/tukey-hsd) (Tukey HSD + Dunnett + Scheffé post-hoc contrasts) | 6.9 | ✅ | ✅ | N/A |
| 2 | [non-inferiority-test](techniques/non-inferiority-test) (means + Farrington-Manning proportions) | 17.7 | ✅ | ✅ | N/A |
| 3 | [propensity-score-matching](techniques/propensity-score-matching) (1:1 NN PSM + ATT + SMD balance) | 15.6 | ✅ | ✅ | N/A |
| 4 | [inverse-probability-weighting](techniques/inverse-probability-weighting) (IPTW + Hajek + AIPW) | 15.7 | ✅ | ✅ | N/A |
| 5 | [regression-discontinuity](techniques/regression-discontinuity) (sharp + fuzzy local-linear RDD) | 15.9 | ✅ | ✅ | N/A |
| 6 | [mediation-analysis](techniques/mediation-analysis) (Baron-Kenny + natural direct/indirect + bootstrap) | 15.15 | ✅ | ✅ | N/A |
| 7 | [synthetic-control](techniques/synthetic-control) (Abadie-Diamond-Hainmueller simplex-weighted counterfactual) | 15.10 | ✅ | ✅ | N/A |
| 8 | [isotonic-regression](techniques/isotonic-regression) (Pool-Adjacent-Violators monotone regression) | 5.29 | ✅ | ✅ | N/A |
| 9 | [cochran-q](techniques/cochran-q) (Cochran's Q for repeated binary + post-hoc McNemar) | 8.10 | ✅ | ✅ | N/A |
| 10 | [meta-analysis](techniques/meta-analysis) (fixed + DerSimonian-Laird random effects + I²) | 20.1 | ✅ | ✅ | N/A |
| 11 | [runs-test](techniques/runs-test) (Wald-Wolfowitz + continuous-median dichotomization) | 7.15 | ✅ | ✅ | N/A |
| 12 | [tmle-doubly-robust](techniques/tmle-doubly-robust) (TMLE + AIPW ATE + IC-based SE) | 15.11 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 19** — all single-node inferential procedures. Distribute by key when needed.

### Batch 20 — Catch-up: SEM starters + ROC + sensitivity + factorial + more

Twelve techniques spanning SEM (CFA, path), diagnostic tests (ROC), causal sensitivity (E-value), experimental design (fractional factorial), ordinal effect size (Cliff's delta), high-p regression (PLS), covariate-adjusted ANOVA, embeddings (t-SNE/UMAP), fractional response, multivariate multiple regression, and Bayesian A/B.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [cfa-confirmatory-factor](techniques/cfa-confirmatory-factor) (ML CFA + chi2 + CFI + RMSEA + SRMR) | 19.5 | ✅ | ✅ | N/A |
| 2 | [path-analysis](techniques/path-analysis) (observed-variable SEM + total-effect calculator) | 19.4 | ✅ | ✅ | N/A |
| 3 | [roc-auc-analysis](techniques/roc-auc-analysis) (ROC + Mann-Whitney AUC + Hanley-McNeil CI + Youden J + partial AUC) | 21.5 | ✅ | ✅ | N/A |
| 4 | [sensitivity-e-value](techniques/sensitivity-e-value) (VanderWeele-Ding E-value + Rosenbaum bounds notes) | 15.14 | ✅ | ✅ | N/A |
| 5 | [fractional-factorial](techniques/fractional-factorial) (2^(k-p) generators + alias structure + resolution) | 16.4 | ✅ | ✅ | N/A |
| 6 | [cliff-delta](techniques/cliff-delta) (nonparametric ordinal effect size + Cliff 1993 CI) | 7.16 | ✅ | ✅ | N/A |
| 7 | [partial-least-squares](techniques/partial-least-squares) (NIPALS PLS1 + CV component selection) | 5.31 | ✅ | ✅ | N/A |
| 8 | [ancova](techniques/ancova) (ANCOVA + parallel-slopes test + adjusted means) | 6.16 | ✅ | ✅ | N/A |
| 9 | [tsne-umap](techniques/tsne-umap) (nonlinear DR for visualization) | 26.5 | ✅ | ✅ | N/A |
| 10 | [fractional-logit](techniques/fractional-logit) (Papke-Wooldridge quasi-MLE + HC0 SEs) | 5.26 | ✅ | ✅ | N/A |
| 11 | [multivariate-multiple-regression](techniques/multivariate-multiple-regression) (joint OLS + Wilks Λ) | 9.20 | ✅ | ✅ | N/A |
| 12 | [bayesian-ab-testing](techniques/bayesian-ab-testing) (Beta-Binomial + P(B>A) + expected loss) | 14.33 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 20** — all single-node inferential procedures. Distribute by key when needed.

### Batch 21 — Chapter 26: Machine Learning basics

Twelve foundational ML techniques.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [decision-tree](techniques/decision-tree) (CART regression + classification) | 26.6 | ✅ | ✅ | N/A |
| 2 | [random-forest](techniques/random-forest) (bagged trees + OOB) | 26.7 | ✅ | ✅ | N/A |
| 3 | [gradient-boosting](techniques/gradient-boosting) (L2 GBM + shrinkage) | 26.8 | ✅ | ✅ | N/A |
| 4 | [svm-classifier](techniques/svm-classifier) (linear Pegasos + RBF via sklearn) | 26.9 | ✅ | ✅ | N/A |
| 5 | [naive-bayes](techniques/naive-bayes) (Gaussian + Multinomial + Laplace) | 26.10 | ✅ | ✅ | N/A |
| 6 | [knn-classifier](techniques/knn-classifier) (kNN with CV k selection) | 26.11 | ✅ | ✅ | N/A |
| 7 | [neural-network-mlp](techniques/neural-network-mlp) (MLP + backprop from scratch) | 27.1 | ✅ | ✅ | N/A |
| 8 | [model-stacking](techniques/model-stacking) (K-fold OOF + meta learner) | 26.14 | ✅ | ✅ | N/A |
| 9 | [calibration-scaling](techniques/calibration-scaling) (reliability + Platt + isotonic + Brier / ECE) | 26.15 | ✅ | ✅ | N/A |
| 10 | [feature-importance](techniques/feature-importance) (permutation + PDP + ICE) | 26.16 | ✅ | ✅ | N/A |
| 11 | [class-imbalance](techniques/class-imbalance) (SMOTE + class weighting + threshold tuning) | 26.17 | ✅ | ✅ | N/A |
| 12 | [isolation-forest-anomaly](techniques/isolation-forest-anomaly) (Isolation Forest + OC-SVM + Elliptic Envelope) | 26.18 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 21** — single-node from-scratch. MLlib has distributed versions of decision-tree / RF / GBM / kNN / logistic / NB — swap in `pyspark.ml` for scale.

### Batch 22 — Chapter 22: IRT / Psychometrics

Twelve psychometric techniques.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [rasch-model](techniques/rasch-model) (1PL joint MLE) | 22.5 | ✅ | ✅ | N/A |
| 2 | [two-three-pl-irt](techniques/two-three-pl-irt) (2PL / 3PL Marginal MLE + Gauss-Hermite + EAP theta) | 22.6 | ✅ | ✅ | N/A |
| 3 | [graded-response-model](techniques/graded-response-model) (Samejima GRM MML) | 22.7 | ✅ | ✅ | N/A |
| 4 | [partial-credit-model](techniques/partial-credit-model) (Masters PCM + Muraki GPCM) | 22.8 | ✅ | ✅ | N/A |
| 5 | [cronbach-alpha](techniques/cronbach-alpha) (alpha + standardized alpha + omega + alpha-if-deleted) | 22.3 | ✅ | ✅ | N/A |
| 6 | [spearman-brown](techniques/spearman-brown) (split-half + Spearman-Brown prophecy) | 22.4 | ✅ | ✅ | N/A |
| 7 | [generalizability-theory](techniques/generalizability-theory) (G-study + D-study + G/Phi coefficients) | 22.10 | ✅ | ✅ | N/A |
| 8 | [item-analysis](techniques/item-analysis) (difficulty + discrimination + point-biserial) | 22.2 | ✅ | ✅ | N/A |
| 9 | [dif-mantel-haenszel](techniques/dif-mantel-haenszel) (MH + logistic uniform / non-uniform DIF) | 22.11 | ✅ | ✅ | N/A |
| 10 | [test-equating](techniques/test-equating) (mean + linear + equipercentile) | 22.12 | ✅ | ✅ | N/A |
| 11 | [person-fit-statistics](techniques/person-fit-statistics) (l_z aberrant-pattern detection) | 22.13 | ✅ | ✅ | N/A |
| 12 | [item-response-info](techniques/item-response-info) (item + test information + adaptive next-item) | 22.14 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 22** — psychometric fits are single-node maximum-likelihood; use Spark to distribute simulation studies or cross-fold validation.

### Batch 23 — Chapter 23: Spatial Statistics

Twelve spatial-statistics techniques covering areal, geostatistical, and point-pattern methods.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [spatial-weights-matrix](techniques/spatial-weights-matrix) (contiguity + distance-band + kNN + kernel W; row-standardization) | 23.1 | ✅ | ✅ | N/A |
| 2 | [morans-i-gearys-c](techniques/morans-i-gearys-c) (global Moran's I + Geary's C + permutation p) | 23.2, 23.3 | ✅ | ✅ | N/A |
| 3 | [local-moran-lisa](techniques/local-moran-lisa) (Anselin LISA + HH/LL/HL/LH typology + permutation p) | 23.4 | ✅ | ✅ | N/A |
| 4 | [variogram-modeling](techniques/variogram-modeling) (empirical semivariogram + spherical / exp / Gaussian fits) | 23.5, 23.6 | ✅ | ✅ | N/A |
| 5 | [ordinary-kriging](techniques/ordinary-kriging) (BLUP via kriging system with variogram model) | 23.7 | ✅ | ✅ | N/A |
| 6 | [inverse-distance-weighting](techniques/inverse-distance-weighting) (Shepard IDW + power/k options) | 23.8 | ✅ | ✅ | N/A |
| 7 | [spatial-autoregressive-sar](techniques/spatial-autoregressive-sar) (SAR lag + SAR error MLE via concentrated log-lik) | 23.9 | ✅ | ✅ | N/A |
| 8 | [conditional-autoregressive-car](techniques/conditional-autoregressive-car) (CAR precision matrix + ICAR penalty; BYM notes) | 23.10 | ✅ | ✅ | N/A |
| 9 | [geographically-weighted-regression](techniques/geographically-weighted-regression) (GWR Gaussian kernel + LOO CV bandwidth) | 23.11 | ✅ | ✅ | N/A |
| 10 | [ripleys-k-point-pattern](techniques/ripleys-k-point-pattern) (K̂ / L̂ with border edge correction + CSR envelope) | 23.12 | ✅ | ✅ | N/A |
| 11 | [spatial-scan-cluster](techniques/spatial-scan-cluster) (Kulldorff Poisson scan + MC p-value) | 23.13 | ✅ | ✅ | N/A |
| 12 | [kernel-intensity-2d](techniques/kernel-intensity-2d) (2D Gaussian intensity + Diggle edge correction) | 23.14 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 23** — spatial estimators need pairwise distance / neighbour structure that isn't well served by row-parallel MLlib operators; use Spark to distribute per-region fits (GWR one-per-location) but keep the estimator on the driver.

### Batch 24 — Chapter 24: Network / Graph Analysis

Twelve techniques covering descriptive, generative, and inferential graph methods.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [graph-descriptives](techniques/graph-descriptives) (density, degree, clustering, transitivity, assortativity, path length, components) | 24.1 | ✅ | ✅ | N/A |
| 2 | [centrality-measures](techniques/centrality-measures) (degree, closeness, betweenness (Brandes), eigenvector, Katz, PageRank) | 24.2 | ✅ | ✅ | N/A |
| 3 | [community-detection](techniques/community-detection) (modularity + greedy agglomerative + spectral 2-way) | 24.3 | ✅ | ✅ | N/A |
| 4 | [random-graph-models](techniques/random-graph-models) (Erdős-Rényi + Watts-Strogatz + Barabási-Albert) | 24.4 | ✅ | ✅ | N/A |
| 5 | [ergm-exponential-random-graph](techniques/ergm-exponential-random-graph) (pseudo-likelihood MLE for edges + triangles) | 24.5 | ✅ | ✅ | N/A |
| 6 | [stochastic-block-model](techniques/stochastic-block-model) (SBM hard EM + spectral warm start; block recovery) | 24.6 | ✅ | ✅ | N/A |
| 7 | [link-prediction](techniques/link-prediction) (common / Jaccard / Adamic-Adar / RA / preferential + AUC eval) | 24.7 | ✅ | ✅ | N/A |
| 8 | [network-diffusion](techniques/network-diffusion) (SI / SIR / independent cascade / linear threshold) | 24.8 | ✅ | ✅ | N/A |
| 9 | [graph-embedding-spectral](techniques/graph-embedding-spectral) (Laplacian eigenmaps + adjacency spectral embedding) | 24.9 | ✅ | ✅ | N/A |
| 10 | [bipartite-projection](techniques/bipartite-projection) (weighted + Newman projections + Barber bipartite modularity) | 24.10 | ✅ | ✅ | N/A |
| 11 | [network-motifs](techniques/network-motifs) (3-node census + Z-scores vs degree-preserving null) | 24.11 | ✅ | ✅ | N/A |
| 12 | [graph-comparison](techniques/graph-comparison) (spectral distance + feature signature + DeltaCon) | 24.12 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 24** — graph analytics on Spark uses GraphFrames / GraphX, a separate API optimised for pregel-style aggregations. For classical descriptive / inferential graph tasks the single-node NetworkX / igraph implementations are the right tool; use Spark to distribute per-graph fits across many small graphs, not per-node aggregations on a single large graph.

### Batch 25 — Cross-chapter cleanup (Chs 21–24)

Twelve subsections that Batches 21–24 skipped — three per chapter.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [one-class-svm](techniques/one-class-svm) (Schölkopf ν-OC-SVM + KDE-baseline anomaly detector) | 21 | ✅ | ✅ | N/A |
| 2 | [shap-values](techniques/shap-values) (exact Shapley enumeration + linear-model analytical check) | 21 | ✅ | ✅ | N/A |
| 3 | [online-learning-sgd](techniques/online-learning-sgd) (squared / log / hinge / passive-aggressive streaming) | 21 | ✅ | ✅ | N/A |
| 4 | [mirt-multidimensional-irt](techniques/mirt-multidimensional-irt) (compensatory M2PL via PCA warm-start + EAP quadrature) | 22 | ✅ | ✅ | N/A |
| 5 | [bayesian-irt](techniques/bayesian-irt) (2PL MAP with priors, coordinate-ascent Newton) | 22 | ✅ | ✅ | N/A |
| 6 | [nominal-response-model](techniques/nominal-response-model) (Bock NRM for unordered polytomous items via MML + quadrature) | 22 | ✅ | ✅ | N/A |
| 7 | [universal-kriging](techniques/universal-kriging) (drift kriging with trend covariates; augmented kriging system) | 23 | ✅ | ✅ | N/A |
| 8 | [getis-ord-g-statistic](techniques/getis-ord-g-statistic) (Gi* hot-spot statistic + asymptotic z + permutation p) | 23 | ✅ | ✅ | N/A |
| 9 | [spatial-glm](techniques/spatial-glm) (Poisson-CAR with PIRLS + CAR quadratic penalty) | 23 | ✅ | ✅ | N/A |
| 10 | [hits-authority-hub](techniques/hits-authority-hub) (Kleinberg HITS via power iteration) | 24 | ✅ | ✅ | N/A |
| 11 | [k-core-decomposition](techniques/k-core-decomposition) (Batagelj-Zaversnik coreness + k-truss) | 24 | ✅ | ✅ | N/A |
| 12 | [temporal-networks](techniques/temporal-networks) (snapshots + earliest-arrival BFS + temporal reachability) | 24 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 25** — mirrors the parents (single-node inferential procedures or specialised graph algorithms).

### Batch 26 — Cross-chapter cleanup (Chs 3, 13, 15, 16, 17, 18, 19, 20)

Twelve techniques closing the substantive gaps in causal inference,
SEM, diagnostic modelling, missing-data sensitivity, robust regression,
experimental design, sampling, and functional TS.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [mendelian-randomization](techniques/mendelian-randomization) (IVW + MR-Egger + weighted median) | 15 | ✅ | ✅ | N/A |
| 2 | [front-door-criterion](techniques/front-door-criterion) (Pearl front-door adjustment for unmeasured confounding via a mediator) | 15 | ✅ | ✅ | N/A |
| 3 | [pattern-mixture-model](techniques/pattern-mixture-model) (MNAR delta-adjustment MI for tipping-point sensitivity) | 16 | ✅ | ✅ | N/A |
| 4 | [mm-estimators-robust](techniques/mm-estimators-robust) (Yohai MM via FAST-S subset init + biweight M-step) | 17 | ✅ | ✅ | N/A |
| 5 | [split-plot-design](techniques/split-plot-design) (balanced split-plot ANOVA with two error strata) | 18 | ✅ | ✅ | N/A |
| 6 | [crossover-design](techniques/crossover-design) (Grizzle 2×2 crossover: treatment / period / carryover) | 18 | ✅ | ✅ | N/A |
| 7 | [measurement-invariance](techniques/measurement-invariance) (configural vs strict CFA across groups; nested LR test) | 19 | ✅ | ✅ | N/A |
| 8 | [latent-class-analysis](techniques/latent-class-analysis) (LCA EM + BIC / AIC selection of K) | 19 | ✅ | ✅ | N/A |
| 9 | [nri-idi](techniques/nri-idi) (Net Reclassification + Integrated Discrimination Improvement) | 20 | ✅ | ✅ | N/A |
| 10 | [decision-curve-analysis](techniques/decision-curve-analysis) (Vickers-Elkin net-benefit across thresholds) | 20 | ✅ | ✅ | N/A |
| 11 | [two-stage-cluster-sampling](techniques/two-stage-cluster-sampling) (two-stage SRS mean + variance decomposition + DEFF) | 3 | ✅ | ✅ | N/A |
| 12 | [functional-time-series](techniques/functional-time-series) (FPCA + AR on scores; Hyndman-Ullah 2007) | 13 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 26** — mirrors the parent chapters.

### Batch 27 — Chapter 25: Text Analytics / NLP

Twelve classical text-analytics techniques covering preprocessing,
representation, modelling, and evaluation.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [text-preprocessing](techniques/text-preprocessing) (tokenise + stopwords + stem + lemma) | 25.1 | ✅ | ✅ | N/A |
| 2 | [tfidf-bm25](techniques/tfidf-bm25) (TF-IDF vectoriser + Okapi BM25 ranking) | 25.2 | ✅ | ✅ | N/A |
| 3 | [word-embeddings](techniques/word-embeddings) (Skip-Gram with Negative Sampling from scratch) | 25.3 | ✅ | ✅ | N/A |
| 4 | [topic-modeling-lda](techniques/topic-modeling-lda) (LDA via collapsed Gibbs sampling) | 25.4 | ✅ | ✅ | N/A |
| 5 | [document-clustering](techniques/document-clustering) (spherical k-means on TF-IDF + purity / NMI / ARI) | 25.5 | ✅ | ✅ | N/A |
| 6 | [text-classification](techniques/text-classification) (multinomial NB + softmax logistic on TF-IDF) | 25.6 | ✅ | ✅ | N/A |
| 7 | [sentiment-analysis](techniques/sentiment-analysis) (VADER-style lexicon + supervised LR) | 25.7 | ✅ | ✅ | N/A |
| 8 | [named-entity-recognition](techniques/named-entity-recognition) (HMM NER with Viterbi + entity-span F1) | 25.8 | ✅ | ✅ | N/A |
| 9 | [string-similarity](techniques/string-similarity) (Levenshtein / Damerau / Jaro / JW / Jaccard / cosine) | 25.9 | ✅ | ✅ | N/A |
| 10 | [language-detection](techniques/language-detection) (Cavnar-Trenkle char-n-gram out-of-place) | 25.10 | ✅ | ✅ | N/A |
| 11 | [topic-coherence-eval](techniques/topic-coherence-eval) (UMass + UCI-PMI + perplexity) | 25.11 | ✅ | ✅ | N/A |
| 12 | [textrank-summarization](techniques/textrank-summarization) (PageRank on sentence-similarity graph) | 25.12 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 27** — text analytics on Spark uses MLlib's `Tokenizer`, `StopWordsRemover`, `HashingTF / IDF`, `Word2Vec`, `LDA`, `NaiveBayes`, and `CountVectorizer`; swap those in for corpus-scale pipelines. This batch's from-scratch demos live on the driver and are algorithmically identical.

### Batch 28 — Chapter 27: Deep Learning

Twelve neural-network architectures and training techniques, implemented
from scratch in numpy for pedagogy.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [deep-mlp-backprop](techniques/deep-mlp-backprop) (deep ReLU MLP + softmax-CE + manual backprop) | 27.1 | ✅ | ✅ | N/A |
| 2 | [convolutional-nn](techniques/convolutional-nn) (2-D Conv + max-pool + backward pass) | 27.2 | ✅ | ✅ | N/A |
| 3 | [recurrent-nn](techniques/recurrent-nn) (Elman RNN + BPTT + gradient clipping) | 27.3 | ✅ | ✅ | N/A |
| 4 | [lstm-gru](techniques/lstm-gru) (LSTM + GRU forward; contrast with RNN vanishing) | 27.4 | ✅ | ✅ | N/A |
| 5 | [attention-mechanism](techniques/attention-mechanism) (scaled dot-product + multi-head + causal mask) | 27.5 | ✅ | ✅ | N/A |
| 6 | [transformer-encoder](techniques/transformer-encoder) (pre-norm block + sinusoidal PE + FFN) | 27.6 | ✅ | ✅ | N/A |
| 7 | [autoencoder](techniques/autoencoder) (vanilla + denoising AE with manual backprop) | 27.7 | ✅ | ✅ | N/A |
| 8 | [variational-autoencoder](techniques/variational-autoencoder) (Kingma-Welling VAE + reparameterisation) | 27.8 | ✅ | ✅ | N/A |
| 9 | [gan-training](techniques/gan-training) (Goodfellow GAN on 2-D data) | 27.9 | ✅ | ✅ | N/A |
| 10 | [dropout-batchnorm](techniques/dropout-batchnorm) (inverted-dropout + BN with train/eval semantics) | 27.10 | ✅ | ✅ | N/A |
| 11 | [adam-optimizer](techniques/adam-optimizer) (SGD / Momentum / RMSProp / Adam / AdamW) | 27.11 | ✅ | ✅ | N/A |
| 12 | [embedding-layers](techniques/embedding-layers) (entity embeddings for categorical inputs) | 27.12 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 28** — neural-net training in production uses PyTorch / TensorFlow / JAX on GPU; MLlib has no full deep-learning stack. For very large-scale distributed training use Ray Train, TorchElastic, or Horovod (all outside Spark). This batch's numpy demos live on a single CPU for pedagogy.

### Batch 29 — Cross-chapter cleanup (Chs 25 + 27)

Twelve techniques closing gaps in Text Analytics and Deep Learning.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [pos-tagging](techniques/pos-tagging) (HMM POS tagger with Viterbi decoding) | 25 | ✅ | ✅ | N/A |
| 2 | [ngram-language-model](techniques/ngram-language-model) (Laplace + Kneser-Ney n-gram LM with perplexity) | 25 | ✅ | ✅ | N/A |
| 3 | [bleu-rouge-eval](techniques/bleu-rouge-eval) (BLEU-n + ROUGE-L for generation evaluation) | 25 | ✅ | ✅ | N/A |
| 4 | [sentence-similarity](techniques/sentence-similarity) (bag-of-embeddings + cosine STS baseline) | 25 | ✅ | ✅ | N/A |
| 5 | [word-alignment](techniques/word-alignment) (IBM Model 1 EM for parallel corpora) | 25 | ✅ | ✅ | N/A |
| 6 | [syntactic-parsing-cky](techniques/syntactic-parsing-cky) (Viterbi CKY chart parser for PCFG) | 25 | ✅ | ✅ | N/A |
| 7 | [residual-connections](techniques/residual-connections) (ResNet-style skip; gradient-flow demo) | 27 | ✅ | ✅ | N/A |
| 8 | [lr-schedules](techniques/lr-schedules) (constant / step / cosine / warmup+cosine / one-cycle) | 27 | ✅ | ✅ | N/A |
| 9 | [transfer-learning](techniques/transfer-learning) (feature extraction vs fine-tuning) | 27 | ✅ | ✅ | N/A |
| 10 | [graph-neural-network](techniques/graph-neural-network) (2-layer GCN; semi-supervised node classification) | 27 | ✅ | ✅ | N/A |
| 11 | [contrastive-learning](techniques/contrastive-learning) (SimCLR-style NT-Xent) | 27 | ✅ | ✅ | N/A |
| 12 | [diffusion-model](techniques/diffusion-model) (DDPM forward + reverse denoising) | 27 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 29** — mirrors the parents.

### Batch 30 — Cross-chapter cleanup (Chs 25 + 27)

Twelve more techniques closing remaining gaps.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [masked-language-modeling](techniques/masked-language-modeling) (BERT-style MLM objective) | 25 | ✅ | ✅ | N/A |
| 2 | [text-generation-decoding](techniques/text-generation-decoding) (greedy / beam / top-k / nucleus / temperature) | 25 | ✅ | ✅ | N/A |
| 3 | [word-sense-disambiguation](techniques/word-sense-disambiguation) (Lesk + embedding-based WSD) | 25 | ✅ | ✅ | N/A |
| 4 | [transformer-decoder](techniques/transformer-decoder) (causal + cross-attention decoder block) | 25 | ✅ | ✅ | N/A |
| 5 | [coreference-resolution](techniques/coreference-resolution) (mention-pair scoring; greedy clustering) | 25 | ✅ | ✅ | N/A |
| 6 | [bertscore-chrf-metrics](techniques/bertscore-chrf-metrics) (chrF + BERTScore surrogate) | 25 | ✅ | ✅ | N/A |
| 7 | [reinforcement-learning-basics](techniques/reinforcement-learning-basics) (tabular Q-learning + REINFORCE) | 27 | ✅ | ✅ | N/A |
| 8 | [normalizing-flows](techniques/normalizing-flows) (RealNVP; bijection + change-of-variables) | 27 | ✅ | ✅ | N/A |
| 9 | [knowledge-distillation](techniques/knowledge-distillation) (temperature-softened KL teacher → student) | 27 | ✅ | ✅ | N/A |
| 10 | [mixture-of-experts](techniques/mixture-of-experts) (sparse top-k gating + load-balance loss) | 27 | ✅ | ✅ | N/A |
| 11 | [quantization-pruning](techniques/quantization-pruning) (int8 quantise + magnitude / structured prune) | 27 | ✅ | ✅ | N/A |
| 12 | [vision-transformer](techniques/vision-transformer) (patch tokenise + [CLS] + transformer encoder) | 27 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 30** — mirrors the parents.

### Batch 31 — Chapter 28: Reinforcement Learning

Twelve RL techniques deepening beyond the single one in Batch 30, from
tabular exact planning to modern policy-gradient and offline RL.

| # | Technique | Ref §| R | Python | PySpark |
|---|-----------|------|---|--------|---------|
| 1 | [multi-armed-bandits](techniques/multi-armed-bandits) (ε-greedy + UCB1 + Thompson sampling) | 28.1 | ✅ | ✅ | N/A |
| 2 | [mdp-value-iteration](techniques/mdp-value-iteration) (VI + PI on finite MDPs) | 28.2 | ✅ | ✅ | N/A |
| 3 | [dqn-deep-q-network](techniques/dqn-deep-q-network) (NN approximator + replay + target network) | 28.3 | ✅ | ✅ | N/A |
| 4 | [actor-critic-a2c](techniques/actor-critic-a2c) (TD advantage + softmax policy) | 28.4 | ✅ | ✅ | N/A |
| 5 | [ppo-clipped](techniques/ppo-clipped) (clipped-surrogate policy update) | 28.5 | ✅ | ✅ | N/A |
| 6 | [monte-carlo-tree-search](techniques/monte-carlo-tree-search) (MCTS-UCT) | 28.6 | ✅ | ✅ | N/A |
| 7 | [model-based-rl](techniques/model-based-rl) (Dyna-Q + deep-MBRL notes) | 28.7 | ✅ | ✅ | N/A |
| 8 | [imitation-learning](techniques/imitation-learning) (behavioural cloning + DAgger) | 28.8 | ✅ | ✅ | N/A |
| 9 | [offline-rl](techniques/offline-rl) (CQL-style pessimism penalty) | 28.9 | ✅ | ✅ | N/A |
| 10 | [rlhf-preferences](techniques/rlhf-preferences) (Bradley-Terry reward model + DPO) | 28.10 | ✅ | ✅ | N/A |
| 11 | [exploration-strategies](techniques/exploration-strategies) (ε-greedy / Boltzmann / UCB / count-based intrinsic) | 28.11 | ✅ | ✅ | N/A |
| 12 | [gae-advantage-estimation](techniques/gae-advantage-estimation) (GAE(λ) for policy-gradient variance reduction) | 28.12 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 31** — RL training uses PyTorch / JAX + gymnasium; distributed RL uses Ray RLlib / Acme rather than Spark.

### Batch 32 — Cross-chapter cleanup (Chs 25, 27, 28)

Twelve techniques closing remaining gaps: 4 RL, 4 deep learning, 4 text.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [sac-soft-actor-critic](techniques/sac-soft-actor-critic) (soft policy iteration; entropy-regularised RL) | 28 | ✅ | ✅ | N/A |
| 2 | [ddpg-td3](techniques/ddpg-td3) (deterministic policy gradient + twin-Q + delayed updates) | 28 | ✅ | ✅ | N/A |
| 3 | [prioritized-experience-replay](techniques/prioritized-experience-replay) (TD-weighted sampling + IS correction) | 28 | ✅ | ✅ | N/A |
| 4 | [hierarchical-rl-options](techniques/hierarchical-rl-options) (SMDP-Q-learning over temporally-extended options) | 28 | ✅ | ✅ | N/A |
| 5 | [state-space-models](techniques/state-space-models) (S4 / Mamba scan + convolution kernel) | 27 | ✅ | ✅ | N/A |
| 6 | [neural-ode](techniques/neural-ode) (continuous-depth NN via Euler / RK4 solvers) | 27 | ✅ | ✅ | N/A |
| 7 | [energy-based-models](techniques/energy-based-models) (contrastive divergence + Langevin sampling) | 27 | ✅ | ✅ | N/A |
| 8 | [meta-learning-maml](techniques/meta-learning-maml) (first-order MAML for few-shot regression) | 27 | ✅ | ✅ | N/A |
| 9 | [question-answering](techniques/question-answering) (IDF-weighted sentence-retrieval baseline) | 25 | ✅ | ✅ | N/A |
| 10 | [abstractive-summarization](techniques/abstractive-summarization) (extract-then-simplify TextRank baseline) | 25 | ✅ | ✅ | N/A |
| 11 | [relation-extraction](techniques/relation-extraction) (regex patterns for `founder_of`, `ceo_of`, `born_in`, …) | 25 | ✅ | ✅ | N/A |
| 12 | [entity-linking](techniques/entity-linking) (alias-index + context-cosine disambiguation) | 25 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 32** — mirrors the parents.

### Batch 33 — Uncertainty Quantification (Ch 29)

Twelve techniques covering the modern UQ toolbox: ensembles, Bayesian
neural nets, conformal prediction, evidential heads, OOD detection,
selective prediction, covariate-shift adaptation, and the
epistemic-vs-aleatoric decomposition.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [deep-ensembles](techniques/deep-ensembles) (K MLPs + Gaussian-NLL head; aleatoric + epistemic split) | 29 | ✅ | ✅ | N/A |
| 2 | [mc-dropout](techniques/mc-dropout) (Gal-Ghahramani 2016: dropout stays on at inference for MC posterior samples) | 29 | ✅ | ✅ | N/A |
| 3 | [bayesian-neural-network](techniques/bayesian-neural-network) (mean-field VI + Bayes-by-Backprop) | 29 | ✅ | ✅ | N/A |
| 4 | [swag](techniques/swag) (SWA-Gaussian: low-rank + diagonal posterior from SGD iterates) | 29 | ✅ | ✅ | N/A |
| 5 | [last-layer-bayesian](techniques/last-layer-bayesian) (exact Bayesian LR on frozen penultimate features) | 29 | ✅ | ✅ | N/A |
| 6 | [conformal-classification](techniques/conformal-classification) (APS / RAPS: coverage-guaranteed prediction sets) | 29 | ✅ | ✅ | N/A |
| 7 | [jackknife-plus](techniques/jackknife-plus) (Barber 2021: LOO prediction intervals with 1−2α guarantee) | 29 | ✅ | ✅ | N/A |
| 8 | [evidential-deep-learning](techniques/evidential-deep-learning) (Dirichlet head + Sensoy loss; NIG for regression) | 29 | ✅ | ✅ | N/A |
| 9 | [ood-detection](techniques/ood-detection) (MSP + Energy + Mahalanobis; AUROC / FPR@95%TPR) | 29 | ✅ | ✅ | N/A |
| 10 | [selective-prediction](techniques/selective-prediction) (risk-coverage curve + AURC + coverage-at-risk) | 29 | ✅ | ✅ | N/A |
| 11 | [covariate-shift-adaptation](techniques/covariate-shift-adaptation) (density-ratio importance weighting; Shimodaira) | 29 | ✅ | ✅ | N/A |
| 12 | [epistemic-aleatoric](techniques/epistemic-aleatoric) (variance and entropy decompositions; BALD) | 29 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 33** — modern UQ pipelines run in PyTorch / JAX / TensorFlow with `uncertainty-toolbox`, `laplace-torch`, `mapie`, `pytorch-ood`; Spark ML has no first-class Bayesian / conformal support.

### Batch 34 — Robustness (Ch 30, ML side)

Twelve techniques covering modern ML robustness — adversarial attacks
and defences, smoothness regularisation, distributionally robust
optimisation. Complements the classical robust-statistics techniques
already covered (`mm-estimators-robust`, `robust-regression`,
`sandwich-robust-se`, `multivariate-outlier-detection`, `outlier-tests`,
`robust-location-scale`, `tmle-doubly-robust`).

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [fgsm-adversarial](techniques/fgsm-adversarial) (Goodfellow 2014 single-step L_∞ attack) | 30 | ✅ | ✅ | N/A |
| 2 | [pgd-adversarial-training](techniques/pgd-adversarial-training) (Madry 2018 saddle-point objective) | 30 | ✅ | ✅ | N/A |
| 3 | [trades-adversarial](techniques/trades-adversarial) (Zhang 2019 KL-based clean/robust trade-off) | 30 | ✅ | ✅ | N/A |
| 4 | [randomized-smoothing](techniques/randomized-smoothing) (Cohen 2019 certified L2 radius) | 30 | ✅ | ✅ | N/A |
| 5 | [label-smoothing](techniques/label-smoothing) (Szegedy 2016 soft-target CE) | 30 | ✅ | ✅ | N/A |
| 6 | [mixup](techniques/mixup) (Zhang 2018 Beta-convex-combination) | 30 | ✅ | ✅ | N/A |
| 7 | [cutmix](techniques/cutmix) (Yun 2019 rectangular patch-swap augmentation) | 30 | ✅ | ✅ | N/A |
| 8 | [distributionally-robust-optimization](techniques/distributionally-robust-optimization) (Group-DRO, Sagawa 2020) | 30 | ✅ | ✅ | N/A |
| 9 | [spectral-normalization](techniques/spectral-normalization) (Miyato 2018 per-layer Lipschitz cap) | 30 | ✅ | ✅ | N/A |
| 10 | [jacobian-regularization](techniques/jacobian-regularization) (Hoffman 2019 Frobenius Jacobian penalty) | 30 | ✅ | ✅ | N/A |
| 11 | [gradient-clipping](techniques/gradient-clipping) (Pascanu 2013 norm / value clipping) | 30 | ✅ | ✅ | N/A |
| 12 | [feature-squeezing](techniques/feature-squeezing) (Xu 2018 bit-depth + median-filter defence) | 30 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 34** — adversarial attacks and defences run in PyTorch / TensorFlow / JAX (`foolbox`, `cleverhans`, `advertorch`, `torchattacks`, `robustness`); Spark ML has no first-class adversarial-robustness support.

### Batch 35 — Fairness / Bias (Ch 31)

Twelve techniques covering the modern algorithmic-fairness toolbox:
four metrics (DP, EO, EOpp, calibration parity + the DI legal test),
three pre-processing / in-training / post-processing families
(reweighing, adversarial debiasing, EO postprocessing), plus fair
representations, counterfactual fairness, the fairlearn reduction
approach, and Dwork's individual fairness.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [demographic-parity](techniques/demographic-parity) (statistical parity + four-fifths rule) | 31 | ✅ | ✅ | N/A |
| 2 | [equalized-odds](techniques/equalized-odds) (Hardt 2016 equal TPR + FPR) | 31 | ✅ | ✅ | N/A |
| 3 | [equal-opportunity](techniques/equal-opportunity) (Hardt 2016 equal TPR only) | 31 | ✅ | ✅ | N/A |
| 4 | [calibration-parity](techniques/calibration-parity) (Chouldechova 2017 PPV / calibration by group) | 31 | ✅ | ✅ | N/A |
| 5 | [disparate-impact](techniques/disparate-impact) (EEOC 80 % rule + two-proportion CI) | 31 | ✅ | ✅ | N/A |
| 6 | [reweighing-preprocessing](techniques/reweighing-preprocessing) (Kamiran-Calders 2012 sample weights) | 31 | ✅ | ✅ | N/A |
| 7 | [adversarial-debiasing](techniques/adversarial-debiasing) (Zhang 2018 predictor vs adversary game) | 31 | ✅ | ✅ | N/A |
| 8 | [equalized-odds-postprocessing](techniques/equalized-odds-postprocessing) (Hardt 2016 group ROC hulls + randomised decision) | 31 | ✅ | ✅ | N/A |
| 9 | [fair-representations-lfr](techniques/fair-representations-lfr) (Zemel 2013 + Ravfogel 2020 INLP projection) | 31 | ✅ | ✅ | N/A |
| 10 | [counterfactual-fairness](techniques/counterfactual-fairness) (Kusner 2017 SCM-based Level-2 predictor) | 31 | ✅ | ✅ | N/A |
| 11 | [exponentiated-gradient-reduction](techniques/exponentiated-gradient-reduction) (Agarwal 2018 fairlearn reduction) | 31 | ✅ | ✅ | N/A |
| 12 | [individual-fairness](techniques/individual-fairness) (Dwork 2012 Lipschitz criterion + IF-loss) | 31 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 35** — fairness auditing and mitigation runs in Python (`fairlearn`, `aif360`, `fairtorch`, `concept-erasure`) or R (`fairness`, `fairml`, `fairmodels`, `mlr3fairness`); Spark ML has no first-class fairness constraints or metrics.

### Batch 36 — MLOps (Ch 32)

Twelve techniques covering the production ML lifecycle: drift
detection (data + concept), monitoring + alerts, shadow + canary
deployment, feature store with point-in-time joins, experiment
tracking + model registry, reproducibility, latency profiling, lineage
DAG, and model cards.

| # | Technique | Chapter | R | Python | PySpark |
|---|-----------|---------|---|--------|---------|
| 1 | [data-drift-detection](techniques/data-drift-detection) (PSI + KS + Wasserstein per feature) | 32 | ✅ | ✅ | N/A |
| 2 | [concept-drift-adwin](techniques/concept-drift-adwin) (Bifet 2007 ADWIN + Gama 2004 DDM online) | 32 | ✅ | ✅ | N/A |
| 3 | [model-monitoring-metrics](techniques/model-monitoring-metrics) (rolling metrics + EWMA baseline + alerts) | 32 | ✅ | ✅ | N/A |
| 4 | [shadow-deployment](techniques/shadow-deployment) (dual-scoring log-only comparison) | 32 | ✅ | ✅ | N/A |
| 5 | [canary-deployment](techniques/canary-deployment) (progressive rollout + SLO rollback) | 32 | ✅ | ✅ | N/A |
| 6 | [feature-store](techniques/feature-store) (point-in-time join + train/serve skew detection) | 32 | ✅ | ✅ | N/A |
| 7 | [experiment-tracking](techniques/experiment-tracking) (params / metrics / SHA-256 content-addressed artifacts) | 32 | ✅ | ✅ | N/A |
| 8 | [model-registry-versioning](techniques/model-registry-versioning) (semver + staging + rollback) | 32 | ✅ | ✅ | N/A |
| 9 | [reproducibility-seeds](techniques/reproducibility-seeds) (seed_everything + provenance manifest + hashes) | 32 | ✅ | ✅ | N/A |
| 10 | [inference-latency-profiling](techniques/inference-latency-profiling) (p50/p95/p99 + batch throughput sweep) | 32 | ✅ | ✅ | N/A |
| 11 | [model-lineage-provenance](techniques/model-lineage-provenance) (data→model→prediction DAG + up/downstream queries) | 32 | ✅ | ✅ | N/A |
| 12 | [model-cards](techniques/model-cards) (Mitchell 2019 9-section template + validator + markdown export) | 32 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 36** — MLOps tooling lives in Python (`mlflow`, `evidently`, `alibi-detect`, `whylogs`, `feast`, `openlineage`, `torch.profiler`, `argo-rollouts`) or Kubernetes (`seldon-core`, `kserve`); Spark ML has no first-class registry / monitoring / lineage support.

### Batch 37 — Semiparametric & Distribution-Free Methods (Ch 33)

Twelve techniques covering Chapter 33 subsections beyond the already-
covered `quantile-regression`, `isotonic-regression`, and
`sandwich-robust-se`: three quantile-regression extensions, two
distributional-regression families, three semiparametric estimator
theory pieces (efficiency, EIF, empirical likelihood), and three
smooth / shape-constrained models.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [bayesian-quantile-regression](techniques/bayesian-quantile-regression) (Yu-Moyeed 2001, asymmetric Laplace + Metropolis) | 33.2 | ✅ | ✅ | N/A |
| 2 | [censored-quantile-regression](techniques/censored-quantile-regression) (Powell 1986 subgradient solver) | 33.3 | ✅ | ✅ | N/A |
| 3 | [semiparametric-efficiency](techniques/semiparametric-efficiency) (Bickel-Klaassen bound; IPW vs AIPW 500-trial demo) | 33.4 | ✅ | ✅ | N/A |
| 4 | [empirical-likelihood](techniques/empirical-likelihood) (Owen 1988, Newton on Lagrangian + bisection CI) | 33.5 | ✅ | ✅ | N/A |
| 5 | [gamlss](techniques/gamlss) (Rigby-Stasinopoulos 2005, mean + log-sd both regressed) | 33.6 | ✅ | ✅ | N/A |
| 6 | [single-index-model](techniques/single-index-model) (Ichimura 1993, Nadaraya-Watson link inside SLS) | 33.8 | ✅ | ✅ | N/A |
| 7 | [varying-coefficient-model](techniques/varying-coefficient-model) (Hastie-Tibshirani 1993, local kernel WLS) | 33.9 | ✅ | ✅ | N/A |
| 8 | [influence-functions-eif](techniques/influence-functions-eif) (mean/median/variance IFs + SE derivation + outlier demo) | 33.11 | ✅ | ✅ | N/A |
| 9 | [distributional-regression](techniques/distributional-regression) (multinomial-bin CDF model + per-x quantile band) | 33.12 | ✅ | ✅ | N/A |
| 10 | [additive-quantile-regression](techniques/additive-quantile-regression) (B-spline basis + subgradient descent, three τ) | 33.13 | ✅ | ✅ | N/A |
| 11 | [shape-constrained-regression](techniques/shape-constrained-regression) (PAV monotone + SLSQP convex QP) | 33.14 | ✅ | ✅ | N/A |
| 12 | [expectile-regression](techniques/expectile-regression) (Newey-Powell 1987, IRLS asymmetric squared loss) | 33 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 37** — semiparametric estimators run in R (`quantreg`, `gamlss`, `mgcv`, `expectreg`, `emplik`, `tmle`) or Python (`statsmodels`, `pyGAM`, `ngboost`, `econml`); Spark ML has no first-class semiparametric-QR / EIF support.

### Batch 38 — Cross-chapter cleanup (Ch 25 Dim-Reduction + Ch 32 High-Dim / Sparse)

Twelve techniques filling gaps in previously-covered chapters: six
dimension-reduction methods that go beyond PCA, and six
high-dimensional / sparse-regression tools around the LASSO family.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [ica](techniques/ica) (Hyvarinen FastICA on mixed sinusoid + square) | 25.1 | ✅ | ✅ | N/A |
| 2 | [nmf](techniques/nmf) (Lee-Seung multiplicative updates; topic-word demo) | 25.2 | ✅ | ✅ | N/A |
| 3 | [sir-sufficient-dim-reduction](techniques/sir-sufficient-dim-reduction) (Li 1991 SIR + SAVE) | 25.3 | ✅ | ✅ | N/A |
| 4 | [sparse-pca](techniques/sparse-pca) (Zou-Hastie-Tibshirani 2006 alternating soft-threshold + SVD) | 25.8 | ✅ | ✅ | N/A |
| 5 | [isomap](techniques/isomap) (Tenenbaum 2000 geodesic MDS; Swiss roll) | 25.14 | ✅ | ✅ | N/A |
| 6 | [lle-locally-linear-embedding](techniques/lle-locally-linear-embedding) (Roweis-Saul 2000) | 25.15 | ✅ | ✅ | N/A |
| 7 | [scad-mcp-penalties](techniques/scad-mcp-penalties) (Fan-Li 2001 + Zhang 2010 nonconvex + LLA) | 32.2 | ✅ | ✅ | N/A |
| 8 | [debiased-lasso](techniques/debiased-lasso) (Zhang-Zhang 2014 / van de Geer 2014 CIs via node-wise LASSO) | 32.4 | ✅ | ✅ | N/A |
| 9 | [model-x-knockoffs](techniques/model-x-knockoffs) (Candes-Fan-Janson-Lv 2018 FDR-controlled selection) | 32.5 | ✅ | ✅ | N/A |
| 10 | [stability-selection](techniques/stability-selection) (Meinshausen-Buhlmann 2010 subsample-frequency) | 32.6 | ✅ | ✅ | N/A |
| 11 | [adaptive-lasso](techniques/adaptive-lasso) (Zou 2006 weighted-LASSO with oracle property) | 32.12 | ✅ | ✅ | N/A |
| 12 | [fused-lasso](techniques/fused-lasso) (Tibshirani 2005 / Chambolle 2004 TV denoising) | 32.13 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 38** — advanced dim-reduction + high-dim inference are R (`fastICA`, `NMF`, `dr`, `elasticnet`, `RDRToolbox`, `lle`, `ncvreg`, `hdi`, `knockoff`, `stabs`, `glmnet`, `genlasso`) or Python (`sklearn.decomposition`, `sklearn.manifold`, `celer`, `knockpy`, `skimage.restoration`); Spark ML has no first-class debiased-LASSO / knockoff support.

### Batch 39 — Cross-chapter cleanup (Ch 31 FDA + Ch 30 Networks)

Twelve techniques bringing Chapter 31 (Functional Data Analysis) from
one covered subsection to seven, and filling six substantial gaps in
Chapter 30 (Network Analysis) including generative models,
network-regression inference, precision-matrix networks, and healthcare
subtyping.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [functional-pca](techniques/functional-pca) (SVD-based; recovers amplitude + phase modes) | 31.2 | ✅ | ✅ | N/A |
| 2 | [functional-regression](techniques/functional-regression) (scalar-on-function via FPC scores; corr β̂,β=1.000) | 31.3 | ✅ | ✅ | N/A |
| 3 | [functional-anova](techniques/functional-anova) (pointwise F + permutation p; sup F 160 vs 6.7) | 31.4 | ✅ | ✅ | N/A |
| 4 | [functional-clustering](techniques/functional-clustering) (FPC scores + k-means; purity 1.00) | 31.6 | ✅ | ✅ | N/A |
| 5 | [curve-registration](techniques/curve-registration) (landmark PL-warp; 93.5% variance reduction) | 31.11 | ✅ | ✅ | N/A |
| 6 | [functional-depth](techniques/functional-depth) (Lopez-Pintado MBD + functional boxplot) | 31.9 | ✅ | ✅ | N/A |
| 7 | [stochastic-block-model](techniques/stochastic-block-model) (Nowicki-Snijders variational EM + spectral warm start; 1.00 accuracy) | 30.4 | ✅ | ✅ | N/A |
| 8 | [latent-space-network](techniques/latent-space-network) (Hoff-Raftery-Handcock 2002 MLE + Procrustes) | 30.5 | ✅ | ✅ | N/A |
| 9 | [qap-network-regression](techniques/qap-network-regression) (Krackhardt 1988 permutation; friendship p=0.000) | 30.6 | ✅ | ✅ | N/A |
| 10 | [gaussian-graphical-model](techniques/gaussian-graphical-model) (Friedman-Hastie-Tibshirani glasso; TP 4/4 FP 0) | 30.8 | ✅ | ✅ | N/A |
| 11 | [node2vec-deepwalk](techniques/node2vec-deepwalk) (Skip-gram / shifted-PMI equivalence; 1.92× cluster separation) | 30.19 | ✅ | ✅ | N/A |
| 12 | [patient-similarity-network](techniques/patient-similarity-network) (Gaussian similarity + k-NN + label propagation; purity 0.967) | 30.25 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 39** — FDA runs in R (`fda`, `fda.usc`, `refund`, `funHDDC`, `roahd`) or Python (`scikit-fda`, `fdasrsf`); network methods run in R (`sbm`, `latentnet`, `sna`, `glasso`, `qgraph`, `SNFtool`) or Python (`graspologic`, `graph-tool`, `sklearn.covariance`, `node2vec`, `snfpy`); Spark ML has no first-class FDA / latent-space / GGM support.

### Batch 40 — Cross-chapter cleanup (Ch 25/30/31/32)

Twelve techniques spread across the four largest cleanup targets:
four dim-reduction gaps (Ch 25), four high-dim / sparse gaps (Ch 32),
two more network sub-sections (Ch 30), and two more FDA sub-sections
(Ch 31).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [probabilistic-pca](techniques/probabilistic-pca) (Tipping-Bishop 1999 closed-form MLE + EM for missing data) | 25.10 | ✅ | ✅ | N/A |
| 2 | [random-projections](techniques/random-projections) (Johnson-Lindenstrauss Gaussian + Achlioptas; distortion 12→4% at k=20→200) | 25.12 | ✅ | ✅ | N/A |
| 3 | [robust-pca](techniques/robust-pca) (Candes-Li-Ma-Wright PCP via ADMM; perfect low-rank + sparse recovery) | 25.13 | ✅ | ✅ | N/A |
| 4 | [diffusion-maps](techniques/diffusion-maps) (Coifman-Lafon 2006 Markov spectral embedding) | 25.16 | ✅ | ✅ | N/A |
| 5 | [sure-independence-screening](techniques/sure-independence-screening) (Fan-Lv 2008 marginal-correlation screening; 104× speedup) | 32.7 | ✅ | ✅ | N/A |
| 6 | [post-selection-inference](techniques/post-selection-inference) (data-split PoSI vs naive; coverage 0.95 vs 0.86) | 32.8 | ✅ | ✅ | N/A |
| 7 | [group-lasso](techniques/group-lasso) (Yuan-Lin 2006 block-coordinate descent; exact group support) | 32.9 | ✅ | ✅ | N/A |
| 8 | [covariance-estimation-highdim](techniques/covariance-estimation-highdim) (Ledoit-Wolf shrinkage + banded; both beat sample cov) | 32.11 | ✅ | ✅ | N/A |
| 9 | [small-world-scale-free](techniques/small-world-scale-free) (Watts-Strogatz + Barabasi-Albert generators + signatures) | 30.12 | ✅ | ✅ | N/A |
| 10 | [homophily-assortativity](techniques/homophily-assortativity) (Newman 2003 attribute + degree assortativity coefficients) | 30.17 | ✅ | ✅ | N/A |
| 11 | [functional-basis-smoothing](techniques/functional-basis-smoothing) (P-spline penalised least squares + LOO-CV; 93% MSE reduction) | 31.1 | ✅ | ✅ | N/A |
| 12 | [functional-linear-model](techniques/functional-linear-model) (function-on-scalar with P-spline coefficient smoothing) | 31.7 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 40** — high-dim + FDA + network methods run in R (`pcaMethods`, `rpca`, `RandPro`, `SIS`, `selectiveInference`, `grplasso`, `corpcor`, `igraph`, `poweRlaw`, `fda`, `refund`) or Python (`sklearn.covariance`, `sklearn.random_projection`, `celer`, `networkx`, `powerlaw`, `scikit-fda`, `pyDiffMap`); Spark ML has no first-class support for any of these.

### Batch 41 — Information Theory & Statistics (Ch 34)

Twelve techniques covering the info-theoretic toolbox from entropy and
KL through model-selection criteria, transfer entropy, information
bottleneck, conditional-MI CI tests, and Fisher-Rao geometry.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [shannon-entropy](techniques/shannon-entropy) (discrete + histogram + Kozachenko-Leonenko k-NN estimator) | 34.1 | ✅ | ✅ | N/A |
| 2 | [kl-divergence](techniques/kl-divergence) (discrete + Gaussian closed form + MC + Jensen-Shannon) | 34.3 | ✅ | ✅ | N/A |
| 3 | [fisher-information](techniques/fisher-information) (analytic + Cramér-Rao vs MLE empirical variance) | 34.4 | ✅ | ✅ | N/A |
| 4 | [information-criteria](techniques/information-criteria) (AIC / AICc / BIC / DIC / WAIC; polynomial sweep BIC picks true order) | 34.5 | ✅ | ✅ | N/A |
| 5 | [cross-entropy-log-loss](techniques/cross-entropy-log-loss) (softmax gradient identity + proper-scoring recovery) | 34.6 | ✅ | ✅ | N/A |
| 6 | [f-divergences](techniques/f-divergences) (KL, χ², Hellinger, TV, Rényi + Pinsker inequality) | 34.7 | ✅ | ✅ | N/A |
| 7 | [minimum-description-length](techniques/minimum-description-length) (two-part MDL; recovers true polynomial order) | 34.8 | ✅ | ✅ | N/A |
| 8 | [maximum-entropy](techniques/maximum-entropy) (Jaynes; loaded-die + Gaussian is MaxEnt under (mean, var)) | 34.9 | ✅ | ✅ | N/A |
| 9 | [transfer-entropy](techniques/transfer-entropy) (Schreiber 2000; X drives Y ⇒ TE(X→Y)=0.50, TE(Y→X)≈0) | 34.11 | ✅ | ✅ | N/A |
| 10 | [information-bottleneck](techniques/information-bottleneck) (Tishby 1999 Blahut-Arimoto discrete IB) | 34.12 | ✅ | ✅ | N/A |
| 11 | [conditional-mutual-info](techniques/conditional-mutual-info) (CMI + strata-preserving permutation CI test) | 34.13 | ✅ | ✅ | N/A |
| 12 | [information-geometry](techniques/information-geometry) (KL≈Fisher-quadratic + natural gradient 4 iters vs vanilla 50) | 34.15 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 41** — info-theoretic tools run in R (`entropy`, `infotheo`, `FNN`, `philentropy`, `RTransferEntropy`, `bnlearn`) or Python (`scipy`, `NPEET`, `sklearn`, `torch`, `IDTxl`, `PyIF`); Spark ML has no first-class information-theoretic primitives.

### Batch 42 — Panel Data & Econometric Methods (Ch 35)

Twelve techniques rounding out Chapter 35: RE-vs-FE testing, dynamic
panel GMM, general GMM, nonlinear LS, seemingly-unrelated regression,
modern staggered / synthetic DiD, event-studies, HAC / cluster-robust
SEs, Oaxaca-Blinder decomposition, stochastic-frontier efficiency,
and panel cointegration.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [hausman-test](techniques/hausman-test) (Wooldridge auxiliary-reg form; A p=0.085 accept, B p<0.001 reject) | 35.2 | ✅ | ✅ | N/A |
| 2 | [arellano-bond-gmm](techniques/arellano-bond-gmm) (difference GMM with lagged-level instruments) | 35.3 | ✅ | ✅ | N/A |
| 3 | [gmm-general](techniques/gmm-general) (Hansen two-step; Hansen J test on over-identifying moments) | 35.5 | ✅ | ✅ | N/A |
| 4 | [nonlinear-least-squares](techniques/nonlinear-least-squares) (Levenberg-Marquardt; MM + 4-PL recovered) | 35.6 | ✅ | ✅ | N/A |
| 5 | [sur-regression](techniques/sur-regression) (Zellner FGLS; Σ̂ close to true 0.8 off-diagonal) | 35.7 | ✅ | ✅ | N/A |
| 6 | [staggered-did](techniques/staggered-did) (Callaway-Sant'Anna group-time ATTs; event-time recovery) | 35.19 | ✅ | ✅ | N/A |
| 7 | [synthetic-did](techniques/synthetic-did) (Arkhangelsky 2021; SDID = 2.46 vs plain DiD 1.12 vs truth 2.0) | 35.10 | ✅ | ✅ | N/A |
| 8 | [event-study](techniques/event-study) (classical + TWFE pathology demo under staggered adoption) | 35.11 | ✅ | ✅ | N/A |
| 9 | [newey-west-hac](techniques/newey-west-hac) (HAC intercept SE 0.12 vs OLS 0.07 under AR(1)) | 35.15 | ✅ | ✅ | N/A |
| 10 | [oaxaca-blinder](techniques/oaxaca-blinder) (threefold + twofold gap decomposition; sums exactly to gap) | 35.21 | ✅ | ✅ | N/A |
| 11 | [stochastic-frontier](techniques/stochastic-frontier) (Aigner-Lovell-Schmidt MLE + Jondrow TE scores) | 35.22 | ✅ | ✅ | N/A |
| 12 | [panel-cointegration](techniques/panel-cointegration) (Pedroni residual ADF; group-mean t=-8.19 cointegrated vs -1.93 not) | 35.26 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 42** — panel econometrics tools live in R (`plm`, `fixest`, `did`, `synthdid`, `systemfit`, `frontier`, `oaxaca`, `sandwich`, `punitroots`) or Python (`linearmodels`, `pyfixest`, `differences`, `statsmodels`, `pysfa`); Spark ML has no first-class panel-econometrics support.

### Batch 43 — Statistical Process Control & Sequential Methods (Ch 37)

Twelve techniques covering the core SPC toolbox: Shewhart charts with
Western Electric rules, CUSUM, EWMA, multivariate Hotelling T²,
Wald SPRT, capability indices, acceptance sampling, Six Sigma
DPMO ↔ σ ↔ yield, risk-adjusted (VLAD + Steiner CUSUM), rare-event
(G-chart + Bernoulli CUSUM), multi-vari variance decomposition, and
Pareto charts.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [shewhart-control-charts](techniques/shewhart-control-charts) (X-bar/R + Western Electric run rules; Phase I baseline; flags at subgroup 26) | 37.1 | ✅ | ✅ | N/A |
| 2 | [cusum-charts](techniques/cusum-charts) (tabular CUSUM k=0.5, h=6; detects 1σ shift at t=100 with delay 16) | 37.2 | ✅ | ✅ | N/A |
| 3 | [ewma-charts](techniques/ewma-charts) (λ ∈ {0.1, 0.2, 0.4}, L=3.5; λ=0.1/0.2 detect faster than λ=0.4) | 37.3 | ✅ | ✅ | N/A |
| 4 | [multivariate-control-charts](techniques/multivariate-control-charts) (Hotelling T² with F-UCL(α=0.005)=14.07; 3/10 shifts flagged) | 37.4 | ✅ | ✅ | N/A |
| 5 | [sequential-analysis](techniques/sequential-analysis) (Wald SPRT Bernoulli 0.5→0.7; avg n=34 vs fixed 63) | 37.5 | ✅ | ✅ | N/A |
| 6 | [process-capability-indices](techniques/process-capability-indices) (Cp/Cpk/Pp/Ppk/Cpm; A centered Cpk=1.08, B off-centre 0.57) | 37.6 | ✅ | ✅ | N/A |
| 7 | [acceptance-sampling](techniques/acceptance-sampling) (OC + AOQ + AOQL + ATI; producer's risk α=0.014 at AQL=0.01) | 37.7 | ✅ | ✅ | N/A |
| 8 | [six-sigma-methods](techniques/six-sigma-methods) (DPMO ↔ σ ↔ yield with 1.5-σ shift + DMAIC checklist) | 37.9 | ✅ | ✅ | N/A |
| 9 | [risk-adjusted-control-charts](techniques/risk-adjusted-control-charts) (VLAD +2.71→−10.02 + Steiner CUSUM signals at t=169) | 37.10 | ✅ | ✅ | N/A |
| 10 | [rare-event-control-charts](techniques/rare-event-control-charts) (G-chart + Bernoulli CUSUM; detects 0.005→0.02 at t=1555) | 37.11 | ✅ | ✅ | N/A |
| 11 | [multi-vari-charts](techniques/multi-vari-charts) (within 7.7% / between-piece 39.1% / between-time 53.2%) | 37.12 | ✅ | ✅ | N/A |
| 12 | [pareto-charts](techniques/pareto-charts) (10 defect types, N=401; 4 vital-few cover 83%) | 37.14 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 43** — SPC / sequential tools run in R (`qcc`, `qicharts2`, `SixSigma`, `MSQC`, `AcceptanceSampling`, `spc`, `vlad`) or Python (`pyspc`, `scipy.stats`, custom); Spark ML has no first-class SPC / sequential-analysis / acceptance-sampling primitives.

### Batch 44 — Additional Specialized Topics (Ch 38)

Twelve techniques covering specialised data types (extreme values,
compositional, circular), record linkage, measurement-error models,
change-point detection, copulas, capture-recapture, shrinkage /
tolerance intervals, agreement metrics beyond kappa, and the
immortal-time bias.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [extreme-value-theory](techniques/extreme-value-theory) (GEV + POT/GPD; 100-yr return level 174 (BM) / 223 (POT)) | 38.1 | ✅ | ✅ | N/A |
| 2 | [compositional-data](techniques/compositional-data) (Aitchison CLR/ALR/ILR + Aitchison distance; d_A=2.30 vs Euclid 0.39) | 38.2 | ✅ | ✅ | N/A |
| 3 | [circular-statistics](techniques/circular-statistics) (von Mises κ MLE via Bessel inversion; recovers μ=10.7°, κ=3.95) | 38.3 | ✅ | ✅ | N/A |
| 4 | [record-linkage](techniques/record-linkage) (Fellegi-Sunter + EM (m, u, π); 99/100 true matches recovered) | 38.6 | ✅ | ✅ | N/A |
| 5 | [measurement-error-models](techniques/measurement-error-models) (regression calibration β̂ 1.06→1.59 vs truth 1.5 + SIMEX) | 38.7 | ✅ | ✅ | N/A |
| 6 | [change-point-detection](techniques/change-point-detection) (binary segmentation + PELT; both recover 3/3 breaks in n=300 series) | 38.8 | ✅ | ✅ | N/A |
| 7 | [copulas](techniques/copulas) (Sklar + Gaussian/Clayton/Gumbel MLE; AIC picks Clayton, θ̂=1.85 vs truth 2.0) | 38.9 | ✅ | ✅ | N/A |
| 8 | [capture-recapture](techniques/capture-recapture) (Lincoln-Petersen + Chapman + Schnabel; N̂=424.5 vs truth 400) | 38.11 | ✅ | ✅ | N/A |
| 9 | [james-stein-shrinkage](techniques/james-stein-shrinkage) (dominates MLE for p≥3; JS/MLE ratio drops to 0.47 at p=25) | 38.15 | ✅ | ✅ | N/A |
| 10 | [tolerance-intervals](techniques/tolerance-intervals) (Howe 1969 normal + Wilks 1941 nonparametric; sim conf 0.946 vs target 0.95) | 38.16 | ✅ | ✅ | N/A |
| 11 | [agreement-beyond-kappa](techniques/agreement-beyond-kappa) (PABAK + Gwet AC1 + Krippendorff α; κ=−0.03 vs AC1=0.95 on prevalence paradox) | 38.19 | ✅ | ✅ | N/A |
| 12 | [immortal-time-bias](techniques/immortal-time-bias) (naive HR=0.12 vs time-varying HR=1.13 vs truth 1.0) | 38.25 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 44** — specialised methods run in R (`extRemes`, `compositions`, `circular`, `fastLink`, `simex`, `changepoint`, `copula`, `Rcapture`, `tolerance`, `irrCAC`, `survival::tmerge`) or Python (`scipy.stats`, `skbio`, `pycircstat`, `recordlinkage`, `scipy.odr`, `ruptures`, `copulas`, `sklearn.covariance`, `krippendorff`, `lifelines`); Spark ML has no first-class support for these specialised data types.

### Batch 45 — Clinical Prediction Modeling & Validation (Ch 39)

Twelve techniques covering the Steyerberg-Harrell workflow for
clinical prediction: prediction-vs-inference framing, Harrell's
full-model + shrinkage strategy, nomograms and integer-point risk
scores, bootstrap optimism correction, external validation,
recalibration, discrimination + calibration metrics + plots,
prediction vs confidence intervals, IECV multi-site validation, and
penalised regression for low-EPV settings.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [prediction-vs-inference](techniques/prediction-vs-inference) (change-in-est vs CV MSE; z-confounder decision agrees) | 39.1 | ✅ | ✅ | N/A |
| 2 | [multivariable-model-building](techniques/multivariable-model-building) (full vs stepwise vs vH-LC shrinkage; s=0.94) | 39.2 | ✅ | ✅ | N/A |
| 3 | [nomograms](techniques/nomograms) (points-based scoring from logistic; patient 1 P=0.003, patient 2 P=0.080) | 39.3 | ✅ | ✅ | N/A |
| 4 | [bootstrap-optimism-correction](techniques/bootstrap-optimism-correction) (apparent AUC 0.752 → corrected 0.651 at EPV=3.8) | 39.4/39.16 | ✅ | ✅ | N/A |
| 5 | [external-validation](techniques/external-validation) (AUC preserved 0.77 → 0.77; slope drift 1.00 → 0.88) | 39.5 | ✅ | ✅ | N/A |
| 6 | [model-recalibration](techniques/model-recalibration) (intercept + logistic recalibration; CITL −0.098 → 0.000) | 39.6 | ✅ | ✅ | N/A |
| 7 | [penalized-clinical-prediction](techniques/penalized-clinical-prediction) (unpenalised vs ridge vs lasso at EPV=5.9; lasso 1 nonzero) | 39.9 | ✅ | ✅ | N/A |
| 8 | [clinical-risk-scores](techniques/clinical-risk-scores) (Sullivan integer points; patient 20 pts → P=0.67) | 39.12 | ✅ | ✅ | N/A |
| 9 | [prediction-intervals](techniques/prediction-intervals) (PI ~8× wider than CI; sim coverage 0.947 target 0.95) | 39.14 | ✅ | ✅ | N/A |
| 10 | [discrimination-calibration](techniques/discrimination-calibration) (same AUC 0.705, ICI 0.040 vs 0.136) | 39.17 | ✅ | ✅ | N/A |
| 11 | [calibration-plots](techniques/calibration-plots) (decile + LOESS + ICI/E-max/E-90; well-cal 0.015 vs over-conf 0.082) | 39.19 | ✅ | ✅ | N/A |
| 12 | [iecv-multisite](techniques/iecv-multisite) (leave-one-site-out; K=5 sites, per-site AUC 0.64-0.73) | 39.25 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 45** — clinical prediction workflow lives in R (`rms`, `caret`, `tidymodels`, `riskRegression`, `predtools`, `CalibrationCurves`, `pmcalibration`, `metamisc`, `glmnet`) or Python (`sklearn.linear_model`, `sklearn.calibration`, `sklearn.metrics`, `statsmodels`, `lifelines`, `mapie`); Spark ML supports elements of the underlying model classes but has no first-class clinical-prediction validation / calibration / nomogram tooling.

### Batch 46 — Genomic and High-Throughput Statistical Methods (Ch 40)

Twelve techniques covering the core statistical genomics toolkit:
GWAS + PRS, differential expression + GSEA + WGCNA, ComBat batch
correction, eQTL, LD score regression + heritability, and the
population-genetics QC quartet (HWE, LD, haplotype phasing, F_ST).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [gwas](techniques/gwas) (per-SNP regression + genomic-inflation λ; SNP 50 p=1e-15) | 40.1/40.23 | ✅ | ✅ | N/A |
| 2 | [polygenic-risk-scores](techniques/polygenic-risk-scores) (P+T at 5 thresholds; ΔR² peaks at p<0.1) | 40.2/40.15 | ✅ | ✅ | N/A |
| 3 | [differential-expression](techniques/differential-expression) (limma-style moderated t; 30/30 DE recovered w/ FP=2) | 40.3 | ✅ | ✅ | N/A |
| 4 | [gsea](techniques/gsea) (KS-style enrichment score + label permutation; top-biased set p=0.006) | 40.4/40.18 | ✅ | ✅ | N/A |
| 5 | [wgcna-coexpression](techniques/wgcna-coexpression) (soft-threshold + TOM + hclust; 36-gene module corr −0.93 with trait) | 40.6 | ✅ | ✅ | N/A |
| 6 | [batch-effect-combat](techniques/batch-effect-combat) (EB batch correction; cross-batch var 0.70→0.04) | 40.11/40.14 | ✅ | ✅ | N/A |
| 7 | [eqtl](techniques/eqtl) (MatrixEQTL-style cis scan; SNP 10→gene 12 p=8e-10) | 40.16 | ✅ | ✅ | N/A |
| 8 | [ld-score-regression](techniques/ld-score-regression) (LDSC; recovers h²=0.30 exactly) | 40.17 | ✅ | ✅ | N/A |
| 9 | [hardy-weinberg](techniques/hardy-weinberg) (χ² + Wigginton exact; excess-het p=0.021) | 40.25 | ✅ | ✅ | N/A |
| 10 | [linkage-disequilibrium](techniques/linkage-disequilibrium) (D, D′, r² + block LD decay) | 40.26 | ✅ | ✅ | N/A |
| 11 | [haplotype-phasing](techniques/haplotype-phasing) (Excoffier-Slatkin 2-SNP EM; recovers truth to 3%) | 40.27 | ✅ | ✅ | N/A |
| 12 | [population-genetics-fst](techniques/population-genetics-fst) (Weir-Cockerham F_ST; panmixia 0.001 vs differentiated 0.23) | 40.20 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 46** — genomics workflows live in R (`limma`, `DESeq2`, `edgeR`, `sva`, `MatrixEQTL`, `WGCNA`, `qqman`, `bigsnpr`, `HardyWeinberg`, `hierfstat`, `haplo.stats`) or Python (`scipy`, `pydeseq2`, `scanpy`, `scikit-allel`, `hail`, `tensorqtl`, `pandas-plink`, `neuroCombat`, `PyWGCNA`, `ldsc`); Spark ML has no first-class support for these bioinformatics-specific pipelines. Hail is Spark-native for GWAS/eQTL and is the closest fit.

### Batch 47 — Data Transformations and Preprocessing (Ch 41)

Twelve techniques covering the applied preprocessing toolkit:
distribution transforms (Box-Cox, Yeo-Johnson, INT), scaling and
Winsorization, categorical encodings (dummy / contrast, target,
hashing), binning, multicollinearity diagnostics, the missing-
indicator method, and tidy long/wide reshaping.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [box-cox-transformation](techniques/box-cox-transformation) (MLE λ; log-normal Shapiro p 3e-20→0.35) | 41.1 | ✅ | ✅ | N/A |
| 2 | [yeo-johnson-transformation](techniques/yeo-johnson-transformation) (signed-y power transform; skew 3.85→0.34) | 41.2 | ✅ | ✅ | N/A |
| 3 | [inverse-normal-transformation](techniques/inverse-normal-transformation) (Blom/Tukey/vdW/Rankit; Shapiro p → 1.0) | 41.3 | ✅ | ✅ | N/A |
| 4 | [standardization-scaling](techniques/standardization-scaling) (z / min-max / robust / Gelman /2SD + group-mean centering) | 41.4 | ✅ | ✅ | N/A |
| 5 | [winsorization](techniques/winsorization) (Winsor vs trim at 5/10/20%; robust mean recovery) | 41.5 | ✅ | ✅ | N/A |
| 6 | [dummy-contrast-coding](techniques/dummy-contrast-coding) (dummy + effect + Helmert; same fit, different β) | 41.6 | ✅ | ✅ | N/A |
| 7 | [discretization-binning](techniques/discretization-binning) (equal-width, equal-freq, Fayyad-Irani entropy; recovers cut 2.50) | 41.7 | ✅ | ✅ | N/A |
| 8 | [multicollinearity-vif](techniques/multicollinearity-vif) (VIF + cond number; 441→1 after dropping collinear) | 41.8 | ✅ | ✅ | N/A |
| 9 | [feature-hashing](techniques/feature-hashing) (signed Weinberger hashing; RSS drops 17904→612 as d grows) | 41.10 | ✅ | ✅ | N/A |
| 10 | [target-encoding](techniques/target-encoding) (Micci-Barreca smoothed + LOO + WOE for binary y) | 41.11 | ✅ | ✅ | N/A |
| 11 | [missing-indicator-method](techniques/missing-indicator-method) (MNAR demo; RMSE 1.59→1.26 with indicators) | 41.12 | ✅ | ✅ | N/A |
| 12 | [tidy-data-reshape](techniques/tidy-data-reshape) (wide↔long round-trip on repeated-measures SBP) | 41.15 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 47** — preprocessing lives in R (`MASS`, `car`, `bestNormalize`, `RNOmni`, `recipes`, `DescTools`, `arules`, `Hmisc`, `car::vif`, `FeatureHashing`, `vtreat`, `tidyr`) or Python (`scipy.stats`, `sklearn.preprocessing`, `category_encoders`, `statsmodels`, `pandas`, custom); Spark ML's `pyspark.ml.feature` has scalers, OHE, VectorAssembler, and hashing but lacks the full preprocessing surface these techniques cover.

### Batch 48 — Text Mining & NLP (Ch 42)

Twelve techniques covering NLP methods not yet in the repo: clinical
concept + negation extraction, structural topic models with
covariates, document embedding + BM25, preprocessing pipelines,
LSA, LIWC-style dictionaries, Dunning keyness, readability indices,
manual content coding + reliability, Wordfish scaling, KWIC
concordances, and collocation statistics.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [clinical-nlp](techniques/clinical-nlp) (NegEx-style negation + sentence-restricted context; 4 notes correctly labelled) | 42.5 | ✅ | ✅ | N/A |
| 2 | [structural-topic-model](techniques/structural-topic-model) (STM with per-group Dirichlet prevalence; recovers 0.84/0.18 vs 0.16/0.82) | 42.7 | ✅ | ✅ | N/A |
| 3 | [document-embedding-similarity](techniques/document-embedding-similarity) (TF-IDF cosine + BM25; "aspirin chest pain" query ranks correctly) | 42.9 | ✅ | ✅ | N/A |
| 4 | [text-preprocessing-pipeline](techniques/text-preprocessing-pipeline) (tokenise → stop-words → Porter stem + lemma dict) | 42.11 | ✅ | ✅ | N/A |
| 5 | [lsa-latent-semantic](techniques/lsa-latent-semantic) (truncated SVD of TDM + query projection; recovers 2 topics from 6 short docs) | 42.13 | ✅ | ✅ | N/A |
| 6 | [dictionary-methods](techniques/dictionary-methods) (LIWC-style 5-category scoring on 4 clinical/everyday sentences) | 42.14 | ✅ | ✅ | N/A |
| 7 | [keyness-analysis](techniques/keyness-analysis) (Dunning G² + log-ratio; "pneumonia" G²=8.21) | 42.15 | ✅ | ✅ | N/A |
| 8 | [readability-measures](techniques/readability-measures) (Flesch/F-K/Fog/SMOG/Coleman-Liau; simple 110 vs complex −121) | 42.16 | ✅ | ✅ | N/A |
| 9 | [content-analysis-coding](techniques/content-analysis-coding) (κ=0.74, α=0.74 on 12 items × 2-3 raters) | 42.17 | ✅ | ✅ | N/A |
| 10 | [wordfish-scaling](techniques/wordfish-scaling) (Poisson unsupervised 1-D scaling; recovers left/right/centrist ω) | 42.18 | ✅ | ✅ | N/A |
| 11 | [kwic-concordance](techniques/kwic-concordance) (regex KWIC with 4-token window; 5 aspirin mentions aligned) | 42.20 | ✅ | ✅ | N/A |
| 12 | [collocation-pmi](techniques/collocation-pmi) (PMI + Dunning G² over bigrams; "left arm" identified) | 42.12 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 48** — NLP techniques live in R (`quanteda`, `tidytext`, `text2vec`, `tm`, `stm`, `clinspacy`, `irr`, `koRpus`, `SnowballC`) or Python (`nltk`, `spacy`, `scispacy`, `medspacy`, `gensim`, `sentence-transformers`, `sklearn.feature_extraction.text`, `textstat`, `krippendorff`, custom); Spark ML has some feature-hashing / word2vec primitives but no first-class NLP corpus-analysis surface.

### Batch 49 — Pharmacoepidemiology & Drug Safety (Ch 43)

Twelve techniques covering the pharmacoepi toolkit: disproportionality
signal detection, self-controlled case series, high-dimensional
propensity scores, PSSA, exposure-crossover for DDIs, drug
utilization / adherence, negative outcome controls, benefit-risk
MCDA, target-trial emulation, and the confounding / new-user /
time-window design triad.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [disproportionality-signal-detection](techniques/disproportionality-signal-detection) (PRR/ROR/IC + χ² signal criteria) | 43.1 | ✅ | ✅ | N/A |
| 2 | [sccs-self-controlled](techniques/sccs-self-controlled) (2-window IRR; recovers 3.0 with CI (2.01, 3.38)) | 43.2 | ✅ | ✅ | N/A |
| 3 | [hdps-high-dim-propensity](techniques/hdps-high-dim-propensity) (Bross-multiplier ranking + IPTW; 5/5 true confounders in top 10) | 43.4 | ✅ | ✅ | N/A |
| 4 | [prescription-sequence-symmetry](techniques/prescription-sequence-symmetry) (Hallas 1996 PSSA; SR=2.15, p=2e-13) | 43.5/43.13 | ✅ | ✅ | N/A |
| 5 | [exposure-crossover](techniques/exposure-crossover) (Rothman RERI for DDI; recovers RR_A=1.45, RR_B=1.33, RERI=0.94) | 43.6 | ✅ | ✅ | N/A |
| 6 | [drug-utilization-adherence](techniques/drug-utilization-adherence) (MPR/PDC/persistence + DDD) | 43.7 | ✅ | ✅ | N/A |
| 7 | [negative-outcome-controls](techniques/negative-outcome-controls) (Schuemie empirical calibration; p 8.6e-4 → 0.47) | 43.9 | ✅ | ✅ | N/A |
| 8 | [benefit-risk-mcda](techniques/benefit-risk-mcda) (weighted-sum MCDA + NNT/NNH/LHH; drug B wins) | 43.10 | ✅ | ✅ | N/A |
| 9 | [target-trial-emulation](techniques/target-trial-emulation) (Hernán-Robins 7-element emulation; naive 0.42 → IPTW 0.30 vs truth 0.30) | 43.11 | ✅ | ✅ | N/A |
| 10 | [confounding-by-indication](techniques/confounding-by-indication) (naive +0.76 → adjusted −0.30 vs truth −0.30) | 43.12 | ✅ | ✅ | N/A |
| 11 | [new-user-active-comparator](techniques/new-user-active-comparator) (ACNU design; washout drops prevalent users) | 43.14 | ✅ | ✅ | N/A |
| 12 | [time-window-bias](techniques/time-window-bias) (Suissa 2012; naive OR 5.31 → common-window OR 0.91) | 43.15 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 49** — pharmacoepi tools live in R (`PhViD`, `SCCS`, `MatchIt`, `WeightIt`, `AdhereR`, `EmpiricalCalibration`, `TrialEmulation`, `CohortMethod`/`Cyclops` OHDSI) or Python (`vigipy`, `zepid`, `causalinference`, `dowhy`, `pandas`, `sklearn`, custom); Spark ML has no first-class pharmacoepi surface. OHDSI provides Spark-native OMOP CDM data access but the analytic layer runs in R.

### Batch 50 — A/B Testing & Online Experimentation (Ch 44)

Twelve techniques covering the online-experimentation toolkit
beyond what's already in the repo (bayesian-ab-testing, multi-
armed-bandits, delta-method, sequential-analysis): fundamentals,
MDE, CUPED variance reduction, always-valid / anytime-valid
inference, FDR across metric families, interference / cluster
randomisation, HTE / uplift, ratio-metric delta method, platform
primitives, triggered analysis, surrogate index, guardrail
monitoring.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [ab-test-fundamentals](techniques/ab-test-fundamentals) (two-prop z + Welch t; revenue lift 7% p=1e-6) | 44.1 | ✅ | ✅ | N/A |
| 2 | [mde-sample-size](techniques/mde-sample-size) (closed-form n + reverse MDE; p=0.05 MDE=0.005 needs n≈31k) | 44.2 | ✅ | ✅ | N/A |
| 3 | [cuped-variance-reduction](techniques/cuped-variance-reduction) (Deng 2013; 64% variance reduction, 2.75× effective n) | 44.3 | ✅ | ✅ | N/A |
| 4 | [always-valid-inference](techniques/always-valid-inference) (mSPRT + Howard-Ramdas CS; peek-safe rejection rates) | 44.4/44.12 | ✅ | ✅ | N/A |
| 5 | [multiple-metrics-fdr](techniques/multiple-metrics-fdr) (BH vs Bonferroni vs hierarchical; 5 true / 15 null) | 44.5 | ✅ | ✅ | N/A |
| 6 | [interference-cluster](techniques/interference-cluster) (cluster-mean t-test; naive p 3e-27 vs correct 0.08) | 44.6 | ✅ | ✅ | N/A |
| 7 | [hte-uplift](techniques/hte-uplift) (T- / S-learner + Qini; ρ=0.98, Qini 0.20 vs random 0.05) | 44.7 | ✅ | ✅ | N/A |
| 8 | [ratio-metrics-abtest](techniques/ratio-metrics-abtest) (delta-method SE for CTR ratios) | 44.10 | ✅ | ✅ | N/A |
| 9 | [experimentation-platform](techniques/experimentation-platform) (deterministic hash assignment + SRM χ² check) | 44.11 | ✅ | ✅ | N/A |
| 10 | [triggered-analysis](techniques/triggered-analysis) (ITT 0.14 vs triggered 0.49 vs truth 0.50) | 44.13 | ✅ | ✅ | N/A |
| 11 | [surrogate-index](techniques/surrogate-index) (Athey-Chetty-Imbens; 3 short-term proxies → long-term index) | 44.14 | ✅ | ✅ | N/A |
| 12 | [guardrail-monitoring](techniques/guardrail-monitoring) (streaming Wilson-CI alarm; alerts at n=3000 on 4× regression) | 44.15 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 50** — A/B analytics live in R (`pwr`, `stats`, `gsDesign`, `rpact`, `inferference`, `grf`, `msm`, `bayesAB`) or Python (`scipy.stats`, `statsmodels`, `causalml`, `econml`, `planout`, custom + commercial SDKs like `eppo-sdk` / `statsig` / `growthbook`); Spark provides the scalable metric backend but the statistical layer runs elsewhere.

### Batch 51 — Cleanup (Ch 36 gaps + high-value gaps in earlier chapters)

Twelve targeted gap fillers: 4 remaining Ch 36 mixture / latent methods
(LPA, mixture regression, conjoint/DCE, factor mixture) plus 8 standout
gaps identified across Chs 14/17/21/22/45 (DeLong AUC, meta-regression,
trim-fill, HMC/NUTS, importance sampling, Taguchi, D-optimal, Latin
hypercube).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [latent-profile-analysis](techniques/latent-profile-analysis) (Gaussian mixture on continuous items; BIC picks K=3) | 36.2 | ✅ | ✅ | N/A |
| 2 | [mixture-regression](techniques/mixture-regression) (EM for 2-class opposite-slope mixture; 90% label recovery) | 36.3 | ✅ | ✅ | N/A |
| 3 | [conjoint-choice](techniques/conjoint-choice) (MNL from scratch; McFadden pseudo-R² 0.44) | 36.7 | ✅ | ✅ | N/A |
| 4 | [factor-mixture-model](techniques/factor-mixture-model) (FMM via GMM + SVD loadings; recovered to 0.05) | 36.10 | ✅ | ✅ | N/A |
| 5 | [delong-auc-test](techniques/delong-auc-test) (placement values + DeLong SE; diff 0.079 p=0.02) | 21.3 | ✅ | ✅ | N/A |
| 6 | [meta-regression](techniques/meta-regression) (REML + moderator; 87% heterogeneity explained) | 22.5 | ✅ | ✅ | N/A |
| 7 | [trim-fill](techniques/trim-fill) (Duval-Tweedie L₀ + mirror imputation; adjusts naive 0.378→0.355) | 22.4 | ✅ | ✅ | N/A |
| 8 | [hmc-nuts](techniques/hmc-nuts) (leapfrog HMC on 2-D correlated Gaussian; recovers mean and Σ) | 14.4 | ✅ | ✅ | N/A |
| 9 | [importance-sampling](techniques/importance-sampling) (SNIS + ESS; good vs bad proposal contrast) | 45.6 | ✅ | ✅ | N/A |
| 10 | [taguchi-methods](techniques/taguchi-methods) (L9 orthogonal array + SNR; per-level SNR ranking) | 17.15 | ✅ | ✅ | N/A |
| 11 | [d-optimal-design](techniques/d-optimal-design) (Fedorov exchange; 9-run design at 158% D-eff per run) | 17.19 | ✅ | ✅ | N/A |
| 12 | [latin-hypercube-sampling](techniques/latin-hypercube-sampling) (LHS + Maximin; min dist 0.325 vs random 0.230) | 45.7 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 51** — these are R (`mclust`, `flexmix`, `mlogit`, `OpenMx`, `pROC`, `metafor`, `rstan`, `DoE.base`, `AlgDesign`, `lhs`) or Python (`sklearn.mixture`, `scipy.stats`, `pyDOE2`, `xlogit`, `numpyro`, custom) — Spark ML has no first-class support for these classical / Bayesian / DOE tools.

### Batch 52 — Cleanup (Ch 15 causal-inference workhorses + earlier gaps)

Twelve more gap fillers, primarily filling out Ch 15 causal-inference
methods (IPTW, AIPW, g-computation, MSM, Rosenbaum bounds, DML,
entropy balancing, CEM) plus a Cox-with-time-varying-covariates,
network meta-analysis, 2PL/3PL IRT, and a normality-tests roundup.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [iptw](techniques/iptw) (ATE/ATT/ATC/ATO weighting; recovers true ATE 0.5) | 15.6 | ✅ | ✅ | N/A |
| 2 | [aipw-doubly-robust](techniques/aipw-doubly-robust) (Robins-Rotnitzky-Zhao; g-formula + IPTW + AIPW side-by-side) | 15.8 | ✅ | ✅ | N/A |
| 3 | [g-computation](techniques/g-computation) (parametric g-formula + bootstrap SE; recovers ATE 0.5) | 15.20 | ✅ | ✅ | N/A |
| 4 | [marginal-structural-model](techniques/marginal-structural-model) (stabilised IPTW 2-time-point; recovers period effects) | 15.19 | ✅ | ✅ | N/A |
| 5 | [rosenbaum-bounds](techniques/rosenbaum-bounds) (matched-pair sensitivity; critical Γ ≈ 1.8 for 45/60 discordant) | 15.23 | ✅ | ✅ | N/A |
| 6 | [dml-double-ml](techniques/dml-double-ml) (Chernozhukov DML-PLR with GBM nuisances; recovers θ=0.60) | 15.25 | ✅ | ✅ | N/A |
| 7 | [entropy-balancing](techniques/entropy-balancing) (Hainmueller Newton dual; exact moment matching, ESS 912/2000) | 15.55 | ✅ | ✅ | N/A |
| 8 | [coarsened-exact-matching](techniques/coarsened-exact-matching) (Iacus-King-Porro CEM; ATT 0.59 vs truth 0.5) | 15.10 | ✅ | ✅ | N/A |
| 9 | [cox-time-varying](techniques/cox-time-varying) (counting-process partial likelihood; recovers HR=1.7) | 11.10 | ✅ | ✅ | N/A |
| 10 | [network-meta-analysis](techniques/network-meta-analysis) (contrast-based RE-NMA; recovers ranking B>A>C) | 22.15 | ✅ | ✅ | N/A |
| 11 | [irt-2pl-3pl](techniques/irt-2pl-3pl) (marginal MLE via GH quadrature; 2PL b recovered to 0.2) | 20.4/20.5 | ✅ | ✅ | N/A |
| 12 | [normality-tests](techniques/normality-tests) (Shapiro/A-D/JB/KS; correctly reject non-normal cases) | 3.24/3.25 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 52** — R (`WeightIt`, `AIPW`, `stdReg`, `ipw`, `rbounds`, `DoubleML`, `ebal`, `MatchIt`, `survival`, `netmeta`, `mirt`, `nortest`) or Python (`causalinference`, `zepid`, `DoubleML`, `econml`, `lifelines`, `scipy.stats`, custom) — Spark ML has no first-class causal-inference / IRT / meta-analysis surface.

### Batch 53 — Cleanup (Ch 15 causal-inference frontier + cross-chapter gaps)

Twelve more gap fillers, mostly in Ch 15 (overlap weighting, CACE,
quantile treatment effects, Manski bounds, PC causal discovery,
case-crossover, causal forest, natural mediation effects,
transportability) plus Ch 5 fractional polynomials, Ch 11 random
survival forest and Ch 11 accelerated failure time.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [overlap-weighting](techniques/overlap-weighting) (Li-Morgan-Zaslavsky ATO; SMD 0.00 vs IPTW-ATE 0.05, weights bounded ≤ 0.5) | 15.31 | ✅ | ✅ | N/A |
| 2 | [principal-stratification-cace](techniques/principal-stratification-cace) (Wald / 2SLS CACE; recovers 1.91 vs truth 2.00 under 40 % compliance) | 15.32 | ✅ | ✅ | N/A |
| 3 | [quantile-treatment-effects](techniques/quantile-treatment-effects) (Firpo IPW QTE; recovers rank-shift 0.63→2.31 across τ=0.1→0.9) | 15.33 | ✅ | ✅ | N/A |
| 4 | [manski-bounds](techniques/manski-bounds) (worst-case + MTR + MTS; ATE ∈ [−3.64,+6.36], MTR tightens LB to 0) | 15.34 | ✅ | ✅ | N/A |
| 5 | [causal-discovery-pc](techniques/causal-discovery-pc) (Spirtes-Glymour-Scheines PC; recovers X0→X2←X1 v-structure exactly) | 15.35 | ✅ | ✅ | N/A |
| 6 | [case-crossover](techniques/case-crossover) (Maclure 1:1 & 1:M MH; recovers OR 2.81 & 2.32 vs truth 2.5) | 15.36 | ✅ | ✅ | N/A |
| 7 | [causal-forest](techniques/causal-forest) (Athey-Wager GRF R-forest; τ̂ monotone in x0 across grid) | 15.37 | ✅ | ✅ | N/A |
| 8 | [mediation-natural-effects](techniques/mediation-natural-effects) (VanderWeele closed form; NDE 0.51/NIE 0.62 vs truth 0.50/0.64) | 15.38 | ✅ | ✅ | N/A |
| 9 | [transportability-generalizability](techniques/transportability-generalizability) (Cole-Stuart IOW; transported ATE 2.59 vs truth 2.50) | 15.39 | ✅ | ✅ | N/A |
| 10 | [fractional-polynomials](techniques/fractional-polynomials) (Royston-Altman FP2 closed test; recovers powers (−0.5, +0.5)) | 5.14 | ✅ | ✅ | N/A |
| 11 | [random-survival-forest](techniques/random-survival-forest) (Ishwaran log-rank splits + NA CHF; Harrell C = 0.742) | 11.24 | ✅ | ✅ | N/A |
| 12 | [accelerated-failure-time](techniques/accelerated-failure-time) (Weibull/log-normal MLE; recovers β=(+0.51,−0.82), σ=0.39) | 11.25 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 53** — R (`PSweight`, `ivreg`, `Counterfactual`, `bounds`, `pcalg`, `survival::clogit`, `grf`, `mediation`, `generalize`, `mfp`, `randomForestSRC`, `survival::survreg`) or Python (`linearmodels`, `causal-learn`, `econml`, `lifelines`, `sksurv`, `causallib`, custom) — Spark ML has no first-class causal-inference / partial-identification / classical-parametric survival surface.

### Batch 54 — Cleanup (Ch 15/24/46/47 – weak IVs, learning theory, emerging LM techniques)

Twelve more gap fillers, mixing remaining Ch 15 workhorses (weak-IV
robust CIs, path-specific effects), Ch 24 disease-mapping empirical
Bayes, Ch 46 learning-theory bounds (U-statistics, Rademacher, VC,
Efron-Stein), and Ch 47 emerging LM techniques (RAG, in-context
learning, speculative decoding, Mamba SSMs, JEPA).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [weak-instruments-anderson-rubin](techniques/weak-instruments-anderson-rubin) (Cragg-Donald F + AR CI; strong-IV CI narrow, weak-IV CI honestly wide) | 15.40 | ✅ | ✅ | N/A |
| 2 | [path-specific-effects](techniques/path-specific-effects) (Avin-Shpitser-Pearl / VanderWeele-Chiba; recovers 4-path decomposition to 3 decimals) | 15.41 | ✅ | ✅ | N/A |
| 3 | [u-statistics](techniques/u-statistics) (Hoeffding 1948 + projection var; Gini 1.120 vs truth 1.128, SE within 7 %) | 46.9 | ✅ | ✅ | N/A |
| 4 | [rademacher-complexity](techniques/rademacher-complexity) (Bartlett-Mendelson; linear-ball R̂ 0.066 matches B/√n) | 46.10 | ✅ | ✅ | N/A |
| 5 | [vc-dimension](techniques/vc-dimension) (empirical shatter lower bound; half-planes in ℝᵖ recover p+1 exactly) | 46.11 | ✅ | ✅ | N/A |
| 6 | [efron-stein-inequality](techniques/efron-stein-inequality) (resample-i variance bound; ES ≥ Var for mean, variance, median) | 46.12 | ✅ | ✅ | N/A |
| 7 | [retrieval-augmented-generation](techniques/retrieval-augmented-generation) (TF-IDF retriever + template LM; recovers 3 survival docs from 6-doc corpus) | 47.19 | ✅ | ✅ | N/A |
| 8 | [in-context-learning](techniques/in-context-learning) (OLS-as-transformer proxy; MSE 8.1 → 0.002 as K rises 0 → 32) | 47.20 | ✅ | ✅ | N/A |
| 9 | [speculative-decoding](techniques/speculative-decoding) (Leviathan rejection sampler; 4.10× speedup with aligned draft) | 47.21 | ✅ | ✅ | N/A |
| 10 | [mamba-state-space-transformer](techniques/mamba-state-space-transformer) (LTI SSM; recurrent = convolutional to 10⁻¹⁷, L=500 stable) | 47.22 | ✅ | ✅ | N/A |
| 11 | [jepa-self-supervised](techniques/jepa-self-supervised) (I-JEPA-flavour toy; EMA target + loss 1.79 → 0.25 over 400 epochs) | 47.23 | ✅ | ✅ | N/A |
| 12 | [poisson-gamma-empirical-bayes](techniques/poisson-gamma-empirical-bayes) (Clayton-Kaldor 1987 shrinkage; MAE ↓ 34 % overall, 49 % in small areas) | 24.19 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 54** — R (`ivmodel`, `paths`, `Ustat`, `DCluster`, `SpatialEpi`, `INLA`, `ellmer`, `chattr`) or Python (`linearmodels`, `causal-learn`, `causallib`, `mamba-ssm`, `state-spaces`, LangChain, `pymc`, `transformers`, `vLLM`, custom) — Spark ML has no first-class causal-inference, LM-inference, or learning-theory surface.

### Batch 55 — Cleanup (SMC, small-area, theory, survey, meta / bootstrap gaps)

Twelve more gap fillers: sequential Monte Carlo, small-area
estimation, composite-likelihood family (Ch 46), simulated-moments
family (Ch 45), theory bundle (concentration, PAC-Bayes), Fisher /
Cauchy p-value combiners, wild cluster bootstrap, complex survey
design, expert prior elicitation, and cross-classified random-effects.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [particle-filter-smc](techniques/particle-filter-smc) (Gordon-Salmond-Smith bootstrap; PF RMSE 0.582 ≈ Kalman 0.583) | 18.10 | ✅ | ✅ | N/A |
| 2 | [fay-herriot-small-area](techniques/fay-herriot-small-area) (Fay-Herriot 1979 EBLUP; MAE ↓ 61 % vs direct estimator) | 27.6 | ✅ | ✅ | N/A |
| 3 | [composite-likelihood](techniques/composite-likelihood) (Lindsay / Varin-Reid-Firth; pairwise CL matches full MLE on Gaussian) | 46.13 | ✅ | ✅ | N/A |
| 4 | [method-of-simulated-moments](techniques/method-of-simulated-moments) (McFadden / Pakes-Pollard; log-normal (μ, σ) 0.295, 0.396 vs truth 0.3, 0.4) | 45.7 | ✅ | ✅ | N/A |
| 5 | [indirect-inference](techniques/indirect-inference) (Gouriéroux-Monfort-Renault MA(1) via AR(2); θ̂=0.57 vs truth 0.6) | 45.8 | ✅ | ✅ | N/A |
| 6 | [concentration-inequalities](techniques/concentration-inequalities) (Markov/Chebyshev/Hoeffding/Bernstein; empirical ≤ bound across t) | 46.14 | ✅ | ✅ | N/A |
| 7 | [pac-bayes-bounds](techniques/pac-bayes-bounds) (McAllester + Catoni; bound tight when KL(Q‖P) small) | 46.15 | ✅ | ✅ | N/A |
| 8 | [fisher-combine-pvalues](techniques/fisher-combine-pvalues) (Fisher / Stouffer / Cauchy; 10 mildly-sig p → 2.5e−7) | 22.16 | ✅ | ✅ | N/A |
| 9 | [wild-cluster-bootstrap](techniques/wild-cluster-bootstrap) (Cameron-Gelbach-Miller G=12; 95 % CI [0.461, 0.589] covers truth 0.5) | 12.15 | ✅ | ✅ | N/A |
| 10 | [complex-survey-design](techniques/complex-survey-design) (Kish-Cochran HT + jackknife; naive 2.96 vs HT 3.94 with unequal weights) | 27.7 | ✅ | ✅ | N/A |
| 11 | [elicitation-of-priors](techniques/elicitation-of-priors) (Kadane / O'Hagan SHELF; Beta(4.5, 6.6) & N(0.5, 0.24) matched to expert quantiles) | 23.20 | ✅ | ✅ | N/A |
| 12 | [cross-classified-random-effects](techniques/cross-classified-random-effects) (Raudenbush-Bryk CCREM; σ̂² recovers school + neigh + resid vs mis-spec) | 8.19 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 55** — R (`pomp`, `sae`, `CompRandFld`, `gmm`, `indirectInference`, `metap`, `poolr`, `fwildclusterboot`, `survey`, `SHELF`, `lme4`) or Python (`particles`, `samplics`, `pyblp`, `scipy.stats`, `wildboottest`, `statsmodels.MixedLM`, `pymer4`, custom) — Spark ML has no first-class SMC, small-area, learning-theory, survey-design or mixed-effects surface.

### Batch 56 — Cleanup (epi / survival / MCMC / trials / kernels / OT gaps)

Twelve more gap fillers: two epi study designs, two survival methods
(piecewise-exp + DeepSurv), two Bayesian samplers (RJ-MCMC + bridge
sampling), a smoother (Nadaraya-Watson), optimal transport, MRP,
two adaptive-trial designs (RAR + platform), a panel model (RI-CLPM),
and PROCOVA covariate adjustment.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [nested-case-control](techniques/nested-case-control) (Thomas 1977 1:K risk-set matching; HR 3.21 (1:1) → 3.27 (1:4) vs truth 3) | 15.42 | ✅ | ✅ | N/A |
| 2 | [piecewise-exponential-model](techniques/piecewise-exponential-model) (Friedman split-and-Poisson MLE; λ̂ = (0.20, 0.50, 0.81) vs (0.20, 0.50, 1.00), ĤR = 2.09) | 11.26 | ✅ | ✅ | N/A |
| 3 | [deep-survival-network](techniques/deep-survival-network) (Katzman DeepSurv; C = 0.739 vs linear Cox 0.622 on nonlinear hazard) | 11.27 | ✅ | ✅ | N/A |
| 4 | [nadaraya-watson-kernel-regression](techniques/nadaraya-watson-kernel-regression) (LOO-CV bandwidth; RMSE 0.07 vs Silverman 0.25 on `sin(1.5x) + 0.3x`) | 5.15 | ✅ | ✅ | N/A |
| 5 | [optimal-transport-wasserstein](techniques/optimal-transport-wasserstein) (1-D W₁ + Sinkhorn; matches analytic W₁ = 1 on N(0,1) vs N(1,1)) | 46.16 | ✅ | ✅ | N/A |
| 6 | [reversible-jump-mcmc](techniques/reversible-jump-mcmc) (Green 1995 mixture-order; posterior P(k=2) = 0.997 on 2-comp mixture) | 25.6 | ✅ | ✅ | N/A |
| 7 | [bridge-sampling-evidence](techniques/bridge-sampling-evidence) (Meng-Wong iterative; Ẑ = 5.04 vs truth 5.00, harmonic-mean fails at 0.46) | 25.7 | ✅ | ✅ | N/A |
| 8 | [mrp-poststratification](techniques/mrp-poststratification) (Gelman-Little / Park-Gelman-Bafumi; matches poststrat mean 0.373 vs truth 0.379) | 27.8 | ✅ | ✅ | N/A |
| 9 | [response-adaptive-randomization](techniques/response-adaptive-randomization) (Wei-Durham + Thompson BAR; Thompson allocates 87 % to better arm) | 44.13 | ✅ | ✅ | N/A |
| 10 | [platform-trial-design](techniques/platform-trial-design) (Woodcock-LaVange; winner graduates 93 %, harmful arm hit for futility 71 %) | 44.14 | ✅ | ✅ | N/A |
| 11 | [ri-clpm-random-intercept-cross-lagged](techniques/ri-clpm-random-intercept-cross-lagged) (Hamaker 2015; recovers φ_xy = +0.14, classical CLPM misses at 0) | 20.30 | ✅ | ✅ | N/A |
| 12 | [prognostic-score-covariate-adjustment](techniques/prognostic-score-covariate-adjustment) (Hansen 2008 / Schuler PROCOVA; SE 0.10 vs 0.18 → 3× ESS) | 44.15 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 56** — R (`Epi`, `survival`, `survivalmodels`, `KernSmooth`, `transport`, `rjmcmc`, `bridgesampling`, `rstanarm`, `brms`, `adaptr`, `pipe`, `lavaan`, `RATES`) or Python (`lifelines`, `sksurv`, `pycox`, `statsmodels`, `POT`, `pymc`, `semopy`, `procova`, custom) — Spark ML has no first-class survival-ML, MCMC, small-area, adaptive-trial, or panel-SEM surface.

### Batch 57 — Cleanup (score matching / DRE / QMC / MC variance reduction / trend / agreement gaps)

Twelve more gap fillers: two density-estimation methods (score
matching, DRE), a sampling design (RSS), agreement (Bland-Altman),
two Bayesian samplers (adaptive Metropolis, SGLD), QMC, a trend test
(Cochran-Armitage), a survival model (first-hitting-time / IG),
BIBD, and two classical MC variance-reduction techniques (antithetic
/ control variates, Rao-Blackwellisation).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [score-matching](techniques/score-matching) (Hyvarinen 2005; Gaussian recovers μ, σ without touching normalising constant) | 46.17 | ✅ | ✅ | N/A |
| 2 | [density-ratio-estimation](techniques/density-ratio-estimation) (Sugiyama uLSIF; matches analytic ratio N(0,1)/N(1,1.5)) | 46.18 | ✅ | ✅ | N/A |
| 3 | [ranked-set-sampling](techniques/ranked-set-sampling) (McIntyre 1952; up to 3.4× variance reduction over SRS with m=8) | 27.9 | ✅ | ✅ | N/A |
| 4 | [bland-altman-agreement](techniques/bland-altman-agreement) (Bland-Altman 1986; LoA ±6.1 reveals disagreement corr=0.977 hides) | 13.16 | ✅ | ✅ | N/A |
| 5 | [adaptive-metropolis-haario](techniques/adaptive-metropolis-haario) (Haario 2001; drops lag-1 autocorr 0.99 → 0.76 on anisotropic 2-D) | 25.8 | ✅ | ✅ | N/A |
| 6 | [stochastic-gradient-mcmc](techniques/stochastic-gradient-mcmc) (Welling-Teh SGLD; Bayesian linreg posterior matches analytic mean) | 25.9 | ✅ | ✅ | N/A |
| 7 | [quasi-monte-carlo-sobol](techniques/quasi-monte-carlo-sobol) (Owen-scrambled Sobol; 158× RMSE reduction over MC, d=3 N=4096) | 45.9 | ✅ | ✅ | N/A |
| 8 | [cochran-armitage-trend](techniques/cochran-armitage-trend) (Cochran 1954 / Armitage 1955; Z = +5.36 on true linear dose-response) | 4.16 | ✅ | ✅ | N/A |
| 9 | [first-hitting-time-model](techniques/first-hitting-time-model) (Whitmore IG survival; recovers (μ, λ) = (3.04, 9.82) at 42% censoring) | 11.28 | ✅ | ✅ | N/A |
| 10 | [balanced-incomplete-block-design](techniques/balanced-incomplete-block-design) (Yates / Fisher Fano-plane BIBD; recovers treatment effects with block structure) | 32.9 | ✅ | ✅ | N/A |
| 11 | [antithetic-control-variates](techniques/antithetic-control-variates) (Hammersley-Morton; 28× (antithetic) / 54× (CV) variance reduction on E[exp(U)]) | 45.10 | ✅ | ✅ | N/A |
| 12 | [rao-blackwellization](techniques/rao-blackwellization) (Rao 1945 / Blackwell 1947; 1.5× variance reduction via conditional on latent) | 45.11 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 57** — R (`densratio`, `RSSampling`, `blandr`, `adaptMCMC`, `SGmcmc`, `randtoolbox`, `DescTools`, `threg`, `AlgDesign`, `crossdes`) or Python (`densratio`, `pyCompare`, `pymc`, `tfp`, `scipy.stats.qmc`, `scipy.stats.invgauss`, `pyDOE2`, custom) — Spark ML has no first-class score-matching, MCMC, QMC, or classical-design surface.

### Batch 58 — Cleanup (nested sampling / slice / LM fine-tuning / theory / classical inference gaps)

Twelve more gap fillers: two Bayesian samplers (nested sampling,
slice sampler), three LM training / prompting techniques (LoRA, DPO,
CoT + self-consistency), NN theory (NTK), a hurdle model, a
one-outlier test (Grubbs / Rosner), OR-homogeneity (Woolf), active
learning, model soups, and a structural-break test (Chow / QLR).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [nested-sampling](techniques/nested-sampling) (Skilling 2006; recovers Z = 0.10 on 1-D Gaussian target) | 25.10 | ✅ | ✅ | N/A |
| 2 | [slice-sampler](techniques/slice-sampler) (Neal 2003 stepping-out + shrinkage; matches analytic P(X<0) on bimodal) | 25.11 | ✅ | ✅ | N/A |
| 3 | [lora-peft](techniques/lora-peft) (Hu 2021 low-rank adapter; 12–47 % trainable params on synthetic linear layer) | 47.24 | ✅ | ✅ | N/A |
| 4 | [dpo-direct-preference-optimization](techniques/dpo-direct-preference-optimization) (Rafailov 2023; β=0.5 → 74 % mass on argmax action) | 47.25 | ✅ | ✅ | N/A |
| 5 | [chain-of-thought-reasoning](techniques/chain-of-thought-reasoning) (Wei 2022 + Wang 2022 self-consistency; K=41 → 0.74 accuracy from 0.55 single) | 47.26 | ✅ | ✅ | N/A |
| 6 | [neural-tangent-kernel](techniques/neural-tangent-kernel) (Jacot 2018; NTK regression vs wide MLP corr 0.9957 on test) | 46.19 | ✅ | ✅ | N/A |
| 7 | [hurdle-model](techniques/hurdle-model) (Mullahy 1986; recovers gate + count betas; plain Poisson biased) | 7.20 | ✅ | ✅ | N/A |
| 8 | [grubbs-outlier-test](techniques/grubbs-outlier-test) (Grubbs + Rosner ESD; single test flags G=4.92, ESD detects 3 injected outliers) | 3.26 | ✅ | ✅ | N/A |
| 9 | [woolf-homogeneity-of-or](techniques/woolf-homogeneity-of-or) (Woolf 1955; correctly non-sig for common OR, p=3e-6 for interaction) | 4.17 | ✅ | ✅ | N/A |
| 10 | [active-learning-query-strategies](techniques/active-learning-query-strategies) (Settles survey; uncertainty / margin / random pool baselines) | 47.27 | ✅ | ✅ | N/A |
| 11 | [model-soups-fine-tune-averaging](techniques/model-soups-fine-tune-averaging) (Wortsman 2022; soup ≈ ensemble w/ one forward pass) | 47.28 | ✅ | ✅ | N/A |
| 12 | [chow-test-structural-break](techniques/chow-test-structural-break) (Chow 1960 + Andrews QLR; QLR picks true τ*=100 with supF=336) | 12.16 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 58** — R (`nestedmodels`, `mcmc::slice.sample`, `pscl`, `outliers`, `EnvStats`, `DescTools`, `strucchange`, `ALEval`) or Python (`dynesty`, `nestle`, `pymc`, `peft`, `trl`, `neural-tangents`, `statsmodels`, `modAL`, `mergekit`, `scipy`, custom) — Spark ML has no first-class NS / slice / LM-fine-tuning / NN-theory / structural-break surface.

### Batch 59 — Cleanup (Kalman variants / health economics / classical CIs / explainability / weighted logrank / GPLVM)

Twelve more gap fillers: three Kalman-family filters (EnKF, EKF,
UKF), health-economic CEA, two classical CIs (Wilson score for
proportion, Fieller for ratio), actuarial Bühlmann credibility,
three XAI methods (LIME, integrated gradients, EBM), weighted
log-rank (Fleming-Harrington), and a nonlinear dim-red (GPLVM).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [ensemble-kalman-filter](techniques/ensemble-kalman-filter) (Evensen 1994; RMSE 0.46 vs obs RMSE 1.06 on nonlinear 2-D) | 18.11 | ✅ | ✅ | N/A |
| 2 | [extended-kalman-filter](techniques/extended-kalman-filter) (Jazwinski / Anderson-Moore; RMSE 0.26 on 1-D quadratic-obs system) | 18.12 | ✅ | ✅ | N/A |
| 3 | [unscented-kalman-filter](techniques/unscented-kalman-filter) (Julier-Uhlmann sigma points; RMSE 0.27 matches EKF w/o Jacobian) | 18.13 | ✅ | ✅ | N/A |
| 4 | [cost-effectiveness-analysis](techniques/cost-effectiveness-analysis) (Drummond; ICER ≈ 39k, CEAC 85 % CE at $50k WTP) | 44.16 | ✅ | ✅ | N/A |
| 5 | [wilson-score-interval-proportion](techniques/wilson-score-interval-proportion) (Wilson 1927 / Agresti-Coull / Jeffreys / CP; Wald degenerate at x=0) | 4.18 | ✅ | ✅ | N/A |
| 6 | [fieller-interval-ratio](techniques/fieller-interval-ratio) (Fieller 1954; unbounded CI correctly flags weak denominator) | 3.27 | ✅ | ✅ | N/A |
| 7 | [buhlmann-credibility](techniques/buhlmann-credibility) (Bühlmann 1967; 35 % MSE reduction over raw group means) | 24.20 | ✅ | ✅ | N/A |
| 8 | [lime-local-explanations](techniques/lime-local-explanations) (Ribeiro 2016; local coefs match analytic gradients at x*) | 47.29 | ✅ | ✅ | N/A |
| 9 | [integrated-gradients](techniques/integrated-gradients) (Sundararajan 2017; completeness Σ IG = Δf exactly) | 47.30 | ✅ | ✅ | N/A |
| 10 | [explainable-boosting-machine](techniques/explainable-boosting-machine) (Nori 2019 / Lou 2013; R² 0.92, recovers tanh shape) | 47.31 | ✅ | ✅ | N/A |
| 11 | [fleming-harrington-weighted-logrank](techniques/fleming-harrington-weighted-logrank) (G(0,1) p=0.018 vs classical p=0.073 for delayed benefit) | 11.29 | ✅ | ✅ | N/A |
| 12 | [gaussian-process-latent-variable-model](techniques/gaussian-process-latent-variable-model) (Lawrence 2004; PCA-init RBF-GPLVM on 6-D projected swiss roll) | 6.16 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 59** — R (`dartR`, `KFAS`, `mvKf`, `BCEA`, `heemod`, `binom`, `PropCIs`, `mratios`, `actuar`, `lime`, `interpret`, `survival`, `kergp`) or Python (`filterpy`, `dapper`, `scipy.stats`, `statsmodels.stats.proportion`, `chainladder`, `lime`, `captum`, `interpret`, `lifelines`, GPy, gpflow, custom) — Spark ML has no first-class filtering, health-economics, explainability, weighted-survival, or GP-latent-variable surface.

### Batch 60 — Cleanup (XAI II / BART / credibility / LLM architecture / cutpoints / SSL gaps)

Twelve more gap fillers: four XAI methods (anchor, counterfactual,
PDP+ICE, ALE), BART Bayesian ensemble, Bühlmann-Straub credibility,
three LLM architecture components (RoPE, GQA, FlashAttention), two
classifier-cutpoint methods (F1-optimal, Youden J), and semi-
supervised pseudo-labelling.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [anchor-explanations](techniques/anchor-explanations) (Ribeiro 2018; greedy anchor recovers logical rule with precision 1.00) | 47.32 | ✅ | ✅ | N/A |
| 2 | [counterfactual-explanations](techniques/counterfactual-explanations) (Wachter 2017; sparse Δincome flips loan-approval prediction) | 47.33 | ✅ | ✅ | N/A |
| 3 | [pdp-ice-plots](techniques/pdp-ice-plots) (Friedman + Goldstein; ICE fans reveal x0·x1 interaction PDP averages away) | 47.34 | ✅ | ✅ | N/A |
| 4 | [ale-accumulated-local-effects](techniques/ale-accumulated-local-effects) (Apley 2020; recovers tanh(x0) under ρ=0.9 correlation, PDP would fail) | 47.35 | ✅ | ✅ | N/A |
| 5 | [bart-bayesian-additive-regression-trees](techniques/bart-bayesian-additive-regression-trees) (Chipman 2010; stumps surrogate R² 0.83 on nonlinear task) | 5.16 | ✅ | ✅ | N/A |
| 6 | [buhlmann-straub-credibility](techniques/buhlmann-straub-credibility) (per-group exposure; 93 % MSE reduction with variable exposures) | 24.21 | ✅ | ✅ | N/A |
| 7 | [rope-rotary-position-embedding](techniques/rope-rotary-position-embedding) (Su 2021; shift invariance verified to 10⁻¹⁵) | 47.36 | ✅ | ✅ | N/A |
| 8 | [grouped-query-attention](techniques/grouped-query-attention) (Ainslie 2023; 4× KV-cache reduction w/ n_kv_heads = n_q_heads/4) | 47.37 | ✅ | ✅ | N/A |
| 9 | [flash-attention](techniques/flash-attention) (Dao 2022 tiled online softmax; matches naive to machine precision) | 47.38 | ✅ | ✅ | N/A |
| 10 | [f1-optimal-threshold](techniques/f1-optimal-threshold) (5 % prevalence; t*=0.77 lifts F1 from 0.32 → 0.49 vs t=0.5 baseline) | 26.9 | ✅ | ✅ | N/A |
| 11 | [youden-optimal-cutpoint](techniques/youden-optimal-cutpoint) (Youden 1950; recovers ROC-optimal cut vs over-sensitive t=0.5) | 26.10 | ✅ | ✅ | N/A |
| 12 | [semi-supervised-pseudo-labeling](techniques/semi-supervised-pseudo-labeling) (Lee 2013; illustrates the confirmation-bias caveat) | 47.39 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 60** — R (`iml`, `pdp`, `ICEbox`, `ALEPlot`, `BART`, `dbarts`, `actuar`, `cutpointr`, `OptimalCutpoints`, `RSSL`) or Python (`alibi`, `DiCE`, `pymc-bart`, `chainladder`, `rotary-embedding-torch`, `flash-attn`, `sklearn`, custom) — Spark ML has no first-class XAI / BART / credibility / LLM-architecture / cutpoint / SSL surface.

### Batch 61 — Cleanup (long-tail: multi-break / MCMC / survival / econometrics / SVR / XAI / sparse)

Twelve more gap fillers: multi-break testing, elliptical-slice MCMC,
interval-censored + multi-state + Buckley-James survival, tensor
decomposition, INLA, SVR, spatial DiD, quantile IV, SHAP interactions,
Dantzig selector.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [bai-perron-multiple-breaks](techniques/bai-perron-multiple-breaks) (Bai-Perron 1998; DP + BIC picks m=2 breaks at 69, 139 vs truth 70, 140) | 12.17 | ✅ | ✅ | N/A |
| 2 | [elliptical-slice-sampling](techniques/elliptical-slice-sampling) (Murray-Adams-MacKay 2010; ESS mean corr 0.999 with analytic GP posterior) | 25.12 | ✅ | ✅ | N/A |
| 3 | [turnbull-interval-censored](techniques/turnbull-interval-censored) (Turnbull 1976 NPMLE; S(10)=0.41 vs truth 0.37 under check-up censoring) | 11.30 | ✅ | ✅ | N/A |
| 4 | [illness-death-model](techniques/illness-death-model) (3-state Markov; exact-MLE q̂ within 5 % of truth; P(dead \| t=10) 0.55) | 11.31 | ✅ | ✅ | N/A |
| 5 | [buckley-james-aft](techniques/buckley-james-aft) (Buckley-James 1979 semi-param AFT; β̂=(1.76, 0.49) at 32 % censoring) | 11.32 | ✅ | ✅ | N/A |
| 6 | [tensor-decomposition-tucker-cp](techniques/tensor-decomposition-tucker-cp) (Tucker + CP-ALS; recovers rank-2 tensor to relative error 0.21) | 6.17 | ✅ | ✅ | N/A |
| 7 | [inla-integrated-nested-laplace](techniques/inla-integrated-nested-laplace) (Rue-Martino-Chopin 2009; grid-Laplace over precision matches oracle Bayes) | 25.13 | ✅ | ✅ | N/A |
| 8 | [support-vector-regression](techniques/support-vector-regression) (Drucker 1996 / Smola-Scholkopf; RBF SVR MSE 0.035 recovering sin(x)) | 5.17 | ✅ | ✅ | N/A |
| 9 | [spatial-diff-in-diff](techniques/spatial-diff-in-diff) (Delgado-Florax SLX-DiD; δ̂=+2.38, θ̂=+0.62 vs truth 2.0, 0.8) | 15.43 | ✅ | ✅ | N/A |
| 10 | [quantile-iv-regression](techniques/quantile-iv-regression) (Chernozhukov-Hansen; α̂=1.00 corrects naive-QR bias 0.84) | 15.44 | ✅ | ✅ | N/A |
| 11 | [shap-interactions](techniques/shap-interactions) (Lundberg 2018; exact enumeration reveals (0, 2) as the true interaction pair) | 47.40 | ✅ | ✅ | N/A |
| 12 | [dantzig-selector](techniques/dantzig-selector) (Candes-Tao 2007 LP; recovers 3-sparse support with (1.45, −0.90, 0.84)) | 6.18 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 61** — R (`strucchange`, `ess`, `icenReg`, `msm`, `mstate`, `bujar`, `rTensor`, `INLA`, `e1071`, `spatialreg`, `quantreg`, `flare`) or Python (`ruptures`, `scipy.linalg.expm`, `lifelines`, `tensorly`, `sklearn.svm`, `PySAL`, `econml`, `shap`, `scipy.optimize.linprog`, custom) — Spark ML has no first-class multi-break / MCMC / interval-survival / multi-state / tensor / INLA / SVR / spatial-DiD / QIV / SHAP-interaction / Dantzig surface.

### Batch 62 — Cleanup (point-process / risk / matrix-completion / MC / RIF / semi-Markov / SEM / LLM / SSL / XAI)

Twelve more long-tail fillers: self-exciting temporal processes,
coherent tail risk, low-rank matrix completion, scalable coresets,
classical MC sampling, RIF-based unconditional quantile regression,
sojourn-time multi-state survival, spatial-error econometrics, LLM
context extension, contrastive-free SSL, ICE-curve CATE clustering,
and attribution-stability diagnostics.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [hawkes-process](techniques/hawkes-process) (Hawkes 1971; Ogata thinning + Ozaki MLE, (μ̂, α̂, β̂)=(0.53, 0.45, 1.65) vs truth (0.5, 0.6, 1.5)) | 47.41 | ✅ | ✅ | N/A |
| 2 | [cvar-expected-shortfall](techniques/cvar-expected-shortfall) (Rockafellar-Uryasev 2000; RU-CVaR matches tail-mean ES for t₃ losses across α∈{0.90, 0.95, 0.99}) | 47.42 | ✅ | ✅ | N/A |
| 3 | [matrix-completion-svt](techniques/matrix-completion-svt) (Cai-Candès-Shen 2010; 50×40 rank-3 truth, 62 % observed → exact recovery, rel err 0.0000) | 47.43 | ✅ | ✅ | N/A |
| 4 | [coreset-selection](techniques/coreset-selection) (Feldman-Langberg 2011 sensitivity; k-means cost error 24 %→0.4 % as coreset grows 50→500) | 47.44 | ✅ | ✅ | N/A |
| 5 | [rejection-sampling](techniques/rejection-sampling) (von Neumann 1951; truncN(0,1) on [0,3] via Exp(1) proposal, KS-stat 0.0045 on 20 000 samples) | 47.45 | ✅ | ✅ | N/A |
| 6 | [rif-regression-firpo](techniques/rif-regression-firpo) (Firpo-Fortin-Lemieux 2009; RIF slopes diverge from cond QR under HTE at τ=0.10 / 0.90) | 47.46 | ✅ | ✅ | N/A |
| 7 | [semi-markov-multistate](techniques/semi-markov-multistate) (Foucher 2010 Weibull sojourn; MLE (1.53, 3.10, 0.71) vs truth (1.5, 3.0, 0.7)) | 47.47 | ✅ | ✅ | N/A |
| 8 | [spatial-error-model-sem](techniques/spatial-error-model-sem) (Anselin 1988; SEM MLE recovers β=(0.96, 1.98, −0.46) vs truth (1, 2, −0.5)) | 47.48 | ✅ | ✅ | N/A |
| 9 | [yarn-context-extension](techniques/yarn-context-extension) (Peng-Quesnelle-Sharkey-Chan 2023; scale 16× preserves 34 % high freqs, attn temp 1.28) | 47.49 | ✅ | ✅ | N/A |
| 10 | [byol-simsiam](techniques/byol-simsiam) (Grill 2020; Chen-He 2021; loss 1.26 → 0.25 with EMA + stop-grad, feature std 0.25 -> no collapse) | 47.50 | ✅ | ✅ | N/A |
| 11 | [cate-clustering-ice](techniques/cate-clustering-ice) (Zhao-Hastie 2021; ARI(cluster, true moderator) = 1.000) | 47.51 | ✅ | ✅ | N/A |
| 12 | [attribution-stability](techniques/attribution-stability) (Yeh 2019; correlated features drop bootstrap top-3 Jaccard 1.00 → 0.43) | 47.52 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 62** — R (`hawkes`, `PerformanceAnalytics`, `softImpute`, `submodlib` [Py], `ars`, `dineq`, `SemiMarkov`, `spatialreg`, `ICEbox`, `iml`) or Python (`tick`, `empyrical`, `fancyimpute`, `submodlib`, `scipy.stats`, `statsmodels`, `pysal`, `transformers`/`vllm`, `lightly`, `PyCEbox`, `shap`/`captum`) — Spark ML has no first-class point-process / CVaR / matrix-completion / coreset / rejection / RIF / semi-Markov / SEM / LLM-context / SSL-image / ICE-clustering / attribution-stability surface.

### Batch 63 — Cleanup (rankings / choice / causal-DAG / kernel-ind / SDE / bagging / Bayes-factor / bandit / classical variance / mixtures / DoE / MR)

Twelve more long-tail gap fillers spanning ranking / choice
modelling, continuous DAG learning, kernel-based independence
testing, stochastic differential equations, ensemble variance
reduction, Bayesian hypothesis testing, adaptive bandits, robust
variance tests, environmental exposure mixtures, DoE screening,
and Mendelian-randomization pleiotropy checks.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [plackett-luce-ranking](techniques/plackett-luce-ranking) (Plackett 1975 / Luce 1959; MM-alg recovers (1.91, 1.42, 0.97, 0.47, 0.23) vs truth (1.90, 1.43, 0.95, 0.48, 0.24)) | 47.53 | ✅ | ✅ | N/A |
| 2 | [mixed-logit-mnl](techniques/mixed-logit-mnl) (McFadden-Train 2000; sim-MLE μ̂=(0.96, −0.49), σ̂=(0.23, 0.33) vs truth (1, −0.5) / (0.4, 0.3) via 200 Halton draws) | 47.54 | ✅ | ✅ | N/A |
| 3 | [notears-dag-learning](techniques/notears-dag-learning) (Zheng-Aragam-Ravikumar-Xing 2018; exact DAG recovery on d=4 truth, SHD 0) | 47.55 | ✅ | ✅ | N/A |
| 4 | [hsic-independence](techniques/hsic-independence) (Gretton et al 2005; RBF-HSIC detects independent (p≈1), linear, quadratic, sinusoidal (all p<10⁻³)) | 47.56 | ✅ | ✅ | N/A |
| 5 | [euler-maruyama-sde](techniques/euler-maruyama-sde) (Maruyama 1955; GBM E[S_T]=104.8 vs 105.1, OU stat var 0.0268 vs 0.0267) | 47.57 | ✅ | ✅ | N/A |
| 6 | [bagging-oob](techniques/bagging-oob) (Breiman 1996; single-tree CV R² 0.49 → bagged B=200 OOB R² 0.71; coverage → 1.0) | 47.58 | ✅ | ✅ | N/A |
| 7 | [savage-dickey-bf](techniques/savage-dickey-bf) (Dickey 1971; BF₀₁=9.4 for true μ=0.0, BF₁₀=1.2×10⁷ for true μ=1.0) | 47.59 | ✅ | ✅ | N/A |
| 8 | [thompson-sampling](techniques/thompson-sampling) (Thompson 1933; 5-arm bandit T=5000 regret 85 vs UCB1 180) | 47.60 | ✅ | ✅ | N/A |
| 9 | [levene-brown-forsythe](techniques/levene-brown-forsythe) (Levene 1960; Brown-Forsythe 1974; equal p=0.58, σ=(1,2,3.5) p<10⁻⁶, robust to t₃) | 47.61 | ✅ | ✅ | N/A |
| 10 | [weighted-quantile-sum-wqs](techniques/weighted-quantile-sum-wqs) (Carrico et al 2015; w̄[0]=0.44, w̄[3]=0.45 for truly bad exposures; β̂₁=0.46 vs truth 0.40) | 47.62 | ✅ | ✅ | N/A |
| 11 | [plackett-burman-screening](techniques/plackett-burman-screening) (Plackett-Burman 1946; 8 factors in N=12 runs; top-3 |effect| ranking matches truth (1, 3, 6)) | 47.63 | ✅ | ✅ | N/A |
| 12 | [mr-egger-pleiotropy](techniques/mr-egger-pleiotropy) (Bowden-Davey Smith-Burgess 2015; IVW β̂=0.50 biased, MR-Egger β̂=0.38 consistent under α₀=0.02) | 47.64 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 63** — R (`PlackettLuce`, `mlogit`, `notearsC`, `dHSIC`, `Sim.DiffProc`, `ipred`, `BayesFactor`, `contextual`, `car`, `gWQS`, `FrF2`, `MendelianRandomization`) or Python (`choix`, `xlogit`, `causalnex`/`dagma`, `hyppo`, `sdeint`/`diffrax`, `sklearn.BaggingRegressor`, `pymc`, `mabwiser`, `scipy.stats`, `wqspy`, `pyDOE`, custom) — Spark ML has no first-class ranking / mixed-logit / DAG / kernel-independence / SDE / OOB-bagging / Bayes-factor / Thompson / robust-variance / WQS / DoE / MR-Egger surface.

### Batch 64 — Cleanup (kernel-2-sample / rating / lasso-solvers / boosting / KAN / sketches / cluster-val / HF-vol / LSH)

Twelve more long-tail fillers: multivariate distribution testing,
online player ratings, coordinate-descent / ADMM lasso solvers,
classical AdaBoost, Kolmogorov-Arnold Networks, streaming
cardinality / membership sketches, cluster-validation
diagnostics, high-frequency realized volatility, and MinHash-LSH
near-neighbour retrieval.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [mmd-two-sample-test](techniques/mmd-two-sample-test) (Gretton et al 2012; detects mean / variance / mixture shifts (all p ≈ 0.005) missed by univariate tests) | 47.65 | ✅ | ✅ | N/A |
| 2 | [elo-glicko-rating](techniques/elo-glicko-rating) (Elo 1978 / Glickman 1999; Corr(final Elo, true skill) = 0.987 after 4 000 games) | 47.66 | ✅ | ✅ | N/A |
| 3 | [coordinate-descent-lasso](techniques/coordinate-descent-lasso) (Friedman-Hastie-Tibshirani 2010; 6 sweeps to converge at λ=0.05, exact support recovery) | 47.67 | ✅ | ✅ | N/A |
| 4 | [admm-consensus](techniques/admm-consensus) (Boyd et al 2011; consensus ADMM recovers β to 0.003 across 5 agents in 15 outer iters) | 47.68 | ✅ | ✅ | N/A |
| 5 | [adaboost-classifier](techniques/adaboost-classifier) (Freund-Schapire 1997; single stump 0.63 → AdaBoost T=200 0.91 on concentric rings) | 47.69 | ✅ | ✅ | N/A |
| 6 | [kolmogorov-arnold-network](techniques/kolmogorov-arnold-network) (Liu et al 2024; 1-layer additive KAN R² 0.997 vs MLP 0.970 vs linear 0.033) | 47.70 | ✅ | ✅ | N/A |
| 7 | [hyperloglog-cardinality](techniques/hyperloglog-cardinality) (Flajolet et al 2007; b=14 (64 KB) → 0.1–1 % error on 10³–10⁶ distinct items) | 47.71 | ✅ | ✅ | N/A |
| 8 | [bloom-filter-membership](techniques/bloom-filter-membership) (Bloom 1970; empirical FP 0.0102 vs target 0.010 at 11.7 KB for n=10 000; 0 false negatives) | 47.72 | ✅ | ✅ | N/A |
| 9 | [hopkins-clusterability](techniques/hopkins-clusterability) (Hopkins-Skellam 1954; uniform H=0.50, cluster H=0.97, grid H=0.27 avg over 20 seeds) | 47.73 | ✅ | ✅ | N/A |
| 10 | [silhouette-clusters](techniques/silhouette-clusters) (Rousseeuw 1987; k=3 wins for both separated (0.82) and overlapping (0.45) 3-cluster data) | 47.74 | ✅ | ✅ | N/A |
| 11 | [realized-volatility-hf](techniques/realized-volatility-hf) (Andersen-Bollerslev 1998; sqrt(RV)→0.020 as M grows, BV isolates jumps, two-scales fixes noise) | 47.75 | ✅ | ✅ | N/A |
| 12 | [min-hash-lsh](techniques/min-hash-lsh) (Broder 1997; LSH K=128, b=32, r=4 recovers 5/5 planted duplicate pairs among 200 sets, only 14 candidates) | 47.76 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 64** — R (`kernlab`, `PlayerRatings`, `glmnet`, `ADMM`, `adabag`, no KAN, `hll`, `bloomfilter`, `clustertend`, `cluster`, `highfrequency`, `LSHR`) or Python (`hyppo`, `skelo`/`trueskill`, `sklearn`/`celer`, `cvxpy`, `sklearn.AdaBoostClassifier`, `pykan`, `datasketch`, `pybloom`, `pyclustertend`, `sklearn.metrics`, `arch`, `datasketch.MinHash`) — Spark ML has no first-class MMD / rating-system / ADMM-lasso / KAN / HLL-native (HLL++ available via `approx_count_distinct`) / cluster-validation-index / realized-vol / MinHash-LSH surface (MinHashLSH does exist in `pyspark.ml.feature` for Jaccard).

### Batch 65 — Cleanup (EM / clustering algorithms + validation / BPE / PQ / HNSW / RFF / SVGD / CatBoost)

Twelve more long-tail fillers: EM for finite mixtures, non-convex
and density-varying clustering algorithms, gap / Davies-Bouldin
K-selection, subword tokenisation, vector-search compression and
graph-based ANN, kernel approximation, Stein-variational Bayesian
inference, and CatBoost's ordered-target-statistic remedy for
leakage.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [em-algorithm-mixture](techniques/em-algorithm-mixture) (Dempster-Laird-Rubin 1977; 1-D GMM recovers π̂/μ̂/σ̂ within 3 %; BIC picks K=3) | 47.77 | ✅ | ✅ | N/A |
| 2 | [spectral-clustering](techniques/spectral-clustering) (Ng-Jordan-Weiss 2001; k-NN Laplacian ARI 1.000 on moons & rings vs k-means 0.32 / ~0) | 47.78 | ✅ | ✅ | N/A |
| 3 | [hdbscan-clustering](techniques/hdbscan-clustering) (Campello-Moulavi-Sander 2013; 3-density clusters + noise ARI 0.775 vs k-means 0.743 with noise absorption) | 47.79 | ✅ | ✅ | N/A |
| 4 | [affinity-propagation](techniques/affinity-propagation) (Frey-Dueck 2007; 4-corner blobs recovered with ARI 1.000 across 4 preference settings) | 47.80 | ✅ | ✅ | N/A |
| 5 | [gap-statistic-cluster](techniques/gap-statistic-cluster) (Tibshirani-Walther-Hastie 2001; picks K=3 (Gap +2.53) on 3-cluster and K=1 on uniform) | 47.81 | ✅ | ✅ | N/A |
| 6 | [davies-bouldin-index](techniques/davies-bouldin-index) (Davies-Bouldin 1979; min at K=3 (DB 0.16 / 0.59) for both separated and overlapping clusters) | 47.82 | ✅ | ✅ | N/A |
| 7 | [byte-pair-encoding-bpe](techniques/byte-pair-encoding-bpe) (Sennrich et al 2016; learns morphemes `low`, `er`, `est`; OOV `lowering` -> subwords) | 47.83 | ✅ | ✅ | N/A |
| 8 | [product-quantization-pq](techniques/product-quantization-pq) (Jegou-Douze-Schmid 2011; M=16 K=256 -> 16 B/vec (16x compress), Recall@10 = 0.55) | 47.84 | ✅ | ✅ | N/A |
| 9 | [hnsw-ann-search](techniques/hnsw-ann-search) (Malkov-Yashunin 2018; fallback proximity graph Recall@10 0.50 → 0.93 as ef 20 → 400) | 47.85 | ✅ | ✅ | N/A |
| 10 | [random-fourier-features](techniques/random-fourier-features) (Rahimi-Recht 2007; RFF ridge D=2000 MSE 0.42 vs exact kernel-ridge 0.39) | 47.86 | ✅ | ✅ | N/A |
| 11 | [stein-variational-gradient](techniques/stein-variational-gradient) (Liu-Wang 2016; 30 particles hit both modes of ½N(±2, 1) with std 2.16 vs truth √5) | 47.87 | ✅ | ✅ | N/A |
| 12 | [catboost-ordered-boosting](techniques/catboost-ordered-boosting) (Prokhorenkova et al 2018; leaky enc train MSE 0.066 vs ordered 0.134 exposes leakage) | 47.88 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 65** — R (`mclust`, `kernlab`, `dbscan`, `apcluster`, `cluster`, `fpc`, `text2vec`, `RcppFaiss`, `RcppHNSW`, no RFF, no SVGD, `catboost`) or Python (`sklearn.mixture`, `sklearn.cluster`, `sklearn.cluster.HDBSCAN`, `sklearn.cluster.AffinityPropagation`, `gap-statistic`, `sklearn.metrics`, `tokenizers`, `faiss`, `hnswlib`, `sklearn.kernel_approximation`, `pyro`, `catboost`) — Spark ML has some clustering (KMeans, GMM, BisectingKMeans) but nothing matching this batch's HDBSCAN / AP / gap / DB / BPE / PQ / HNSW / RFF / SVGD / ordered-boosting scope.

### Batch 66 — Cleanup (sparse GP / Hopfield / LLM prompting / EMD / Welch PSD / interaction / IPP / HNN / noisy labels / CPC / Frank-Wolfe)

Twelve more long-tail fillers spanning scalable Gaussian processes,
associative memory, LLM reasoning patterns (self-consistency and
tree-of-thoughts), classical signal decomposition and spectral
analysis, black-box interaction detection, inhomogeneous point
processes, physics-inspired networks, robust training under label
noise, self-supervised representation learning on sequences, and a
projection-free convex-optimisation solver.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [sparse-gaussian-process](techniques/sparse-gaussian-process) (Snelson-Ghahramani 2005 / Titsias 2009; FITC M=30 -> rel err ~0 vs full GP on n=500) | 47.89 | ✅ | ✅ | N/A |
| 2 | [hopfield-network](techniques/hopfield-network) (Hopfield 1982; N=100 Hebbian storage, capacity limit 0.14 N reproduced; P=5 30 % flip -> 5/5 recovery, P=20 -> 1/20) | 47.90 | ✅ | ✅ | N/A |
| 3 | [self-consistency-prompting](techniques/self-consistency-prompting) (Wang et al 2022; K=1 acc 0.23 -> K=40 acc 0.89 via majority-vote over 4-step chains) | 47.91 | ✅ | ✅ | N/A |
| 4 | [tree-of-thoughts](techniques/tree-of-thoughts) (Yao et al 2023; Game-of-24 CoT (100 chains) 0.35 vs ToT BFS beam=32 -> 0.78) | 47.92 | ✅ | ✅ | N/A |
| 5 | [empirical-mode-decomposition](techniques/empirical-mode-decomposition) (Huang et al 1998; IMFs isolate 1 Hz / 3 Hz components, residue tracks linear trend) | 47.93 | ✅ | ✅ | N/A |
| 6 | [welch-power-spectral-density](techniques/welch-power-spectral-density) (Welch 1967; nperseg=2048 pinpoints 60 & 220 Hz, P(60)=0.50 vs truth 0.50, Parseval holds) | 47.94 | ✅ | ✅ | N/A |
| 7 | [friedmans-h-statistic](techniques/friedmans-h-statistic) (Friedman-Popescu 2008; H(0,2)=0.83 flags true pair vs ≤0.20 elsewhere) | 47.95 | ✅ | ✅ | N/A |
| 8 | [poisson-point-process-inhomog](techniques/poisson-point-process-inhomog) (Cox 1955 / Ogata 1981; MLE α̂=0.36 β̂=+0.31 γ̂=1.64 τ̂=14.3 vs truth (0.4, 0.3, 2, 20)) | 47.96 | ✅ | ✅ | N/A |
| 9 | [hnn-hamiltonian-neural-networks](techniques/hnn-hamiltonian-neural-networks) (Greydanus et al 2019; HNN drift 0.0000 vs MLP drift 0.30 on 20-s pendulum roll-out) | 47.97 | ✅ | ✅ | N/A |
| 10 | [noisy-label-cotraining](techniques/noisy-label-cotraining) (Han et al 2018; +3-4 pts over noisy-tree baseline at 10-40 % label noise) | 47.98 | ✅ | ✅ | N/A |
| 11 | [contrastive-predictive-coding](techniques/contrastive-predictive-coding) (van den Oord et al 2018; linear InfoNCE retrieval acc 0.65 vs 0.50 chance on 12 sinusoids) | 47.99 | ✅ | ✅ | N/A |
| 12 | [frank-wolfe-conditional-gradient](techniques/frank-wolfe-conditional-gradient) (Frank-Wolfe 1956; L1-ball radius=10 recovers true 3-sparse (1.56, -1.01, 0.79)) | 47.100 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 66** — R (limited SGP, no Hopfield, no LLM tooling, `Rlibeemd`, `stats::spectrum`/`spectral`, `iml`, `spatstat`, no HNN, no co-teach, no CPC, `CVXR`) or Python (`GPflow`/`gpytorch`, custom Hopfield/`hflayers`, `litellm`/`langchain`/`vllm`, `PyEMD`, `scipy.signal`, `sklearn.inspection`, `tick`/`NHPoisson`, `torchdyn`/`hamiltonian-nn`, `cleanlab`, `cpc-audio`/`torchaudio`, `cvxpy`) — Spark ML has no first-class scalable-GP / associative-memory / LLM-reasoning / EMD / Welch-PSD / interaction-detection / IPP-MLE / physics-NN / co-teaching / CPC / Frank-Wolfe surface.

### Batch 67 — Cleanup (GoF trio / XGBoost / LightGBM / RMSprop / NAG / cGAN / WGAN / top-k+top-p / prompt tuning / TabNet)

Twelve more long-tail fillers spanning three classical EDF /
moment-based goodness-of-fit tests, two industrial gradient-boosting
frameworks, two accelerated / adaptive first-order optimisers, two
generative-adversarial variants, and three modern LLM / tabular-DL
techniques.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [anderson-darling-test](techniques/anderson-darling-test) (Anderson-Darling 1954; N(0,1) p=0.29 vs t3 / uniform / outliers p<10^-4) | 47.101 | ✅ | ✅ | N/A |
| 2 | [cramer-von-mises-test](techniques/cramer-von-mises-test) (Cramér 1928 / von Mises 1931; W^2=0.045 (N) vs 0.72-1.01 (uniform/Laplace/t3) with p≤0.011) | 47.102 | ✅ | ✅ | N/A |
| 3 | [jarque-bera-test](techniques/jarque-bera-test) (Jarque-Bera 1980; log-normal JB=14 945 vs N(0,1) JB=6.6) | 47.103 | ✅ | ✅ | N/A |
| 4 | [xgboost-boosting](techniques/xgboost-boosting) (Chen-Guestrin 2016 KDD; from-scratch Newton boost T=500 test MSE 0.82) | 47.104 | ✅ | ✅ | N/A |
| 5 | [lightgbm-histogram-boosting](techniques/lightgbm-histogram-boosting) (Ke et al 2017 NeurIPS; hist-GBM 65x faster than exact-split at similar/better MSE) | 47.105 | ✅ | ✅ | N/A |
| 6 | [rmsprop-optimizer](techniques/rmsprop-optimizer) (Tieleman-Hinton 2012; RMSprop reaches ‖x‖ 0.35 vs SGD 1.02 on κ=100 quadratic) | 47.106 | ✅ | ✅ | N/A |
| 7 | [nesterov-accelerated-gradient](techniques/nesterov-accelerated-gradient) (Nesterov 1983; NAG 162 iters to 1e-3 on κ=100 vs GD >200) | 47.107 | ✅ | ✅ | N/A |
| 8 | [conditional-gan-cgan](techniques/conditional-gan-cgan) (Mirza-Osindero 2014; equilibrium D acc 0.51 -> 0.50 with per-class optimum G) | 47.108 | ✅ | ✅ | N/A |
| 9 | [wgan-wasserstein-gan](techniques/wgan-wasserstein-gan) (Arjovsky-Chintala-Bottou 2017; W_1 grows linearly with shift while JS saturates at log 2) | 47.109 | ✅ | ✅ | N/A |
| 10 | [topk-topp-nucleus-sampling](techniques/topk-topp-nucleus-sampling) (Fan 2018 / Holtzman 2020; top-p 0.99 -> 45 unique tokens vs greedy K=1 -> 1) | 47.110 | ✅ | ✅ | N/A |
| 11 | [prefix-prompt-tuning](techniques/prefix-prompt-tuning) (Li-Liang 2021 / Lester 2021; 2-16 tunable prompt params match full-FT acc on toy) | 47.111 | ✅ | ✅ | N/A |
| 12 | [tabnet-tabular-attention](techniques/tabnet-tabular-attention) (Arik-Pfister 2021; sequential sparsemax mask recovers {0, 3, 7} true active features) | 47.112 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 67** — R (`nortest`, `goftest`, `tseries`, `xgboost`, `lightgbm`, `torch`, `torch`, no GAN, no WGAN, no decoding, no prompt-tuning, no TabNet) or Python (`scipy.stats`, `scipy.stats`, `scipy.stats`, `xgboost`, `lightgbm`/`sklearn.HistGBM`, `torch.optim.RMSprop`, `torch.optim.SGD(nesterov)`, `torchGAN`, `torchGAN`, `transformers.generation`, `peft`, `pytorch-tabnet`) — Spark ML has classical GLM / GBT / RF but nothing matching this batch's tail-focused GoF / XGBoost-native / LightGBM / adaptive-optimizer / GAN / decoding / prompt-tuning / TabNet surface.

### Batch 68 — Cleanup (Welch t / vision-XAI / distributional-RL / Prophet / word2vec / ALS / NCF / EP / loopy-BP / LARS / OMP)

Twelve more long-tail fillers spanning classical Welch's test, two
gradient-based saliency methods for CNNs, distributional
reinforcement learning, additive time-series forecasting, foundational
word / user-item embeddings, two collaborative-filtering baselines,
two graphical-model approximate-inference methods, and two classical
greedy sparse-regression algorithms.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [welchs-t-test](techniques/welchs-t-test) (Welch 1947; Type-I ~0.05 (target) vs Student 0.003 (deflated) under unequal var + unbalanced n) | 47.113 | ✅ | ✅ | N/A |
| 2 | [grad-cam-saliency](techniques/grad-cam-saliency) (Selvaraju et al 2017 ICCV; ReLU-weighted heatmap peaks inside true 3×3 signal region) | 47.114 | ✅ | ✅ | N/A |
| 3 | [smoothgrad-saliency](techniques/smoothgrad-saliency) (Smilkov et al 2017; denoises raw gradients via Gaussian noise averaging) | 47.115 | ✅ | ✅ | N/A |
| 4 | [prophet-forecasting](techniques/prophet-forecasting) (Taylor-Letham 2018; Prophet RMSE 0.38 vs seasonal-naive 1.05, persistence 2.75) | 47.116 | ✅ | ✅ | N/A |
| 5 | [word2vec-skipgram](techniques/word2vec-skipgram) (Mikolov et al 2013; within-cluster cosine 0.63-0.69 vs cross-cluster 0.13-0.31) | 47.117 | ✅ | ✅ | N/A |
| 6 | [matrix-factorization-als-recsys](techniques/matrix-factorization-als-recsys) (Hu-Koren-Volinsky 2008 ICDM; hit@10 = 100 % vs popularity baseline 59 %) | 47.118 | ✅ | ✅ | N/A |
| 7 | [neural-collaborative-filtering](techniques/neural-collaborative-filtering) (He et al 2017 WWW; MLP head captures nonlinear u-v interaction linear GMF misses) | 47.119 | ✅ | ✅ | N/A |
| 8 | [expectation-propagation-ep](techniques/expectation-propagation-ep) (Minka 2001 UAI; probit-regression EP mean 1.33 matches grid mean 1.34) | 47.120 | ✅ | ✅ | N/A |
| 9 | [belief-propagation-loopy](techniques/belief-propagation-loopy) (Pearl 1988 / Yedidia et al 2005; 3×3 Ising loopy-BP matches exact enumeration to 2 dp) | 47.121 | ✅ | ✅ | N/A |
| 10 | [lars-least-angle-regression](techniques/lars-least-angle-regression) (Efron et al 2004; step 5 recovers true 4-sparse support (1.66, -1.16, 0.71, -0.53)) | 47.122 | ✅ | ✅ | N/A |
| 11 | [orthogonal-matching-pursuit](techniques/orthogonal-matching-pursuit) (Pati et al 1993; K=4 OMP recovers β̂=(1.49, -0.99, 0.77, -0.59) matches truth to 3 dp) | 47.123 | ✅ | ✅ | N/A |
| 12 | [distributional-rl-c51](techniques/distributional-rl-c51) (Bellemare-Dabney-Munos 2017 ICML; C51 recovers bimodal Var[Z]=1.0, P(z<0)=0.50 for the two-mode return) | 47.124 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 68** — R (`stats::t.test`, no Grad-CAM, no SmoothGrad, `prophet`, `text2vec`/`word2vec`, `recosystem`, no NCF, `EPGLM`, `gRain`/`bnlearn`, `lars`, no OMP, no C51) or Python (`scipy.stats`, `pytorch-grad-cam`, `captum.NoiseTunnel`, `prophet`/`neuralprophet`, `gensim`, `implicit`/`LightFM`, `recommenders`, `GPy`+custom, `pgmpy`/`libDAI`, `sklearn.Lars`, `sklearn.OrthogonalMatchingPursuit`, `dopamine`/`torchrl`) — Spark ML has KMeans / ALS (implicit) but nothing matching this batch's classical Welch / vision-XAI / distributional-RL / Prophet / word2vec / NCF / EP / loopy-BP / LARS / OMP / C51 surface.

### Batch 69 — Cleanup (SINDy / DMD / RTS smoother / FM / LambdaMART / kernel-SHAP / RRF / beta-binomial / Hoeffding-D / Reptile / LODA / KTA)

Twelve more long-tail fillers spanning data-driven dynamical
systems, state-space smoothing, factorization / ranking / recsys
models, model-agnostic Shapley explanations, retrieval fusion,
hierarchical proportions, nonlinear-independence testing,
first-order meta-learning, streaming anomaly detection, and
kernel-selection scoring.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [sindy-sparse-dynamics](techniques/sindy-sparse-dynamics) (Brunton-Proctor-Kutz 2016 PNAS; exact recovery of Van der Pol: dx1/dt = -x0 + x1 - x0^2 x1) | 47.125 | ✅ | ✅ | N/A |
| 2 | [dynamic-mode-decomposition-dmd](techniques/dynamic-mode-decomposition-dmd) (Schmid 2010; dominant DMD freqs [2.0, 3.0] recovered; rel-recon-err 0.0000) | 47.126 | ✅ | ✅ | N/A |
| 3 | [rts-kalman-smoother](techniques/rts-kalman-smoother) (Rauch-Tung-Striebel 1965; KF RMSE 0.33 → RTS RMSE 0.18, variance tightened 49 %) | 47.127 | ✅ | ✅ | N/A |
| 4 | [factorization-machines](techniques/factorization-machines) (Rendle 2010 ICDM; k=4 FM MSE 0.026 vs Ridge 0.035 on sparse binary + true (0, 3) interaction) | 47.128 | ✅ | ✅ | N/A |
| 5 | [learning-to-rank-lambdamart](techniques/learning-to-rank-lambdamart) (Burges 2010 MSR-TR; LambdaMART-lite NDCG@10 = 0.996 vs random 0.691) | 47.129 | ✅ | ✅ | N/A |
| 6 | [shapley-permutation-explainer](techniques/shapley-permutation-explainer) (Strumbelj-Kononenko 2010 / Lundberg-Lee 2017; perm-exact / perm-MC / Kernel SHAP all sum to +2.500 (efficiency)) | 47.130 | ✅ | ✅ | N/A |
| 7 | [reciprocal-rank-fusion](techniques/reciprocal-rank-fusion) (Cormack-Clarke-Buttcher 2009 SIGIR; RRF (k=1) NDCG@10 = 0.936 vs individual ~0.9) | 47.131 | ✅ | ✅ | N/A |
| 8 | [beta-binomial-hierarchical](techniques/beta-binomial-hierarchical) (Efron-Morris 1975; EB shrinkage MSE 0.0010 vs MLE 0.0036 (71 % lift) on baseball data) | 47.132 | ✅ | ✅ | N/A |
| 9 | [hoeffding-d-independence](techniques/hoeffding-d-independence) (Hoeffding 1948; D detects quadratic (p<0.005, Spearman -0.15) & sinusoid) | 47.133 | ✅ | ✅ | N/A |
| 10 | [reptile-meta-learning](techniques/reptile-meta-learning) (Nichol-Achiam-Schulman 2018; meta-init 3-shot MSE 1.43 vs wide-random-init 2.44 at K=20) | 47.134 | ✅ | ✅ | N/A |
| 11 | [loda-anomaly-detection](techniques/loda-anomaly-detection) (Pevny 2016; K=100 random projections → LODA top-20 precision 1.000 = IsolationForest) | 47.135 | ✅ | ✅ | N/A |
| 12 | [kernel-target-alignment](techniques/kernel-target-alignment) (Cristianini et al 2001; RBF alignment peaks at σ ~ 1–3 (A ≈ 0.49) for sign-boundary target) | 47.136 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 69** — R (`sindyr`, no DMD, `KFAS`/`dlm`, `libFMR`, `xgboost`+`lightgbm`, `iml`/`fastshap`, no RRF, `VGAM`, `Hmisc`, no Reptile, no LODA, `kernlab`) or Python (`pysindy`, `pydmd`, `filterpy`, `xLearn`/`pywFM`, `lightgbm.LGBMRanker`, `shap`, `ranx`, `pymc`, `hyppo`, `learn2learn`, `pyod`, `sklearn`+custom) — Spark ML has some tabular / ALS support but nothing matching this batch's dynamics / smoother / FM / LTR-lambda / SHAP / RRF / beta-binomial / Hoeffding / Reptile / LODA / KTA surface.

### Batch 70 — Cleanup (SEM / CMA-ES / PSO / SA / GA / Savitzky-Golay / Hodrick-Prescott / Butterworth / Rainbow-DQN / NoisyNets / OpenAI-ES / GCN)

Twelve more long-tail fillers covering confirmatory latent-variable
modeling, five population-based / stochastic global optimisers,
three classical signal-processing filters, three deep-RL
improvements (Rainbow / NoisyNet exploration / OpenAI ES), and
graph-convolutional semi-supervised node classification.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [structural-equation-modeling](techniques/structural-equation-modeling) (Joreskog 1970; one-factor CFA n=400, p=5: chi^2=7.27, df=5, CFI≈0.985) | 47.137 | ✅ | ✅ | N/A |
| 2 | [cma-es-evolution-strategy](techniques/cma-es-evolution-strategy) (Hansen-Ostermeier 2001; 5-D Rosenbrock @ 500 iters: f=0.000000, ‖x−1‖∞=0.0000) | 47.138 | ✅ | ✅ | N/A |
| 3 | [particle-swarm-optimization](techniques/particle-swarm-optimization) (Kennedy-Eberhart 1995; 5-D Rastrigin @ 500 iters: f=0.0000, swarm converges) | 47.139 | ✅ | ✅ | N/A |
| 4 | [simulated-annealing](techniques/simulated-annealing) (Kirkpatrick-Gelatt-Vecchi 1983; 1-D multi-well x*=0.000; 15-city TSP 3.77 vs random 8.48) | 47.140 | ✅ | ✅ | N/A |
| 5 | [genetic-algorithm](techniques/genetic-algorithm) (Holland 1975; Goldberg 1989; 30-item knapsack 500 gens value=310 vs greedy 305) | 47.141 | ✅ | ✅ | N/A |
| 6 | [savitzky-golay-filter](techniques/savitzky-golay-filter) (Savitzky-Golay 1964; sin+3sin-3t signal RMSE 0.299 → 0.074 at w=51,p=3) | 47.142 | ✅ | ✅ | N/A |
| 7 | [hodrick-prescott-filter](techniques/hodrick-prescott-filter) (Hodrick-Prescott 1997; λ=1600 trend RMSE 0.30, cycle RMSE 0.54 on 200-pt series) | 47.143 | ✅ | ✅ | N/A |
| 8 | [butterworth-bandpass](techniques/butterworth-bandpass) (Butterworth 1930; 30-70 Hz bandpass retains 78.5% energy, dominant peak 50 Hz) | 47.144 | ✅ | ✅ | N/A |
| 9 | [rainbow-dqn](techniques/rainbow-dqn) (Hessel et al 2018 AAAI; tabular Double-Q + PER + n-step on 7-chain: greedy policy [1,1,1,1,1,1,0]) | 47.145 | ✅ | ✅ | N/A |
| 10 | [noisy-networks-exploration](techniques/noisy-networks-exploration) (Fortunato et al 2018 ICLR; contextual-bandit greedy acc 95.3% vs chance 25%) | 47.146 | ✅ | ✅ | N/A |
| 11 | [evolution-strategies-openai](techniques/evolution-strategies-openai) (Salimans et al 2017; 5-D step function f=0 at 50 iters, gradient methods fail) | 47.147 | ✅ | ✅ | N/A |
| 12 | [gcn-graph-convolutional](techniques/gcn-graph-convolutional) (Kipf-Welling 2017 ICLR; 3-community SBM GCN test acc 93.3% vs LR baseline 37.1%) | 47.148 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 70** — R (`lavaan`/`sem`, `cmaes`, `pso`, `GenSA`, `GA`, `signal`, `mFilter`, `signal`, no Rainbow, no NoisyNet, custom-ES, custom-GCN) or Python (`semopy`/`factor_analyzer`, `cmaes`/`pycma`, `pyswarm`/`pyswarms`, `scipy.optimize.dual_annealing`, `DEAP`/`pymoo`, `scipy.signal.savgol_filter`, `statsmodels.tsa.filters.hp_filter`, `scipy.signal.butter`, `stable_baselines3`/`cleanrl`, custom, custom, `pytorch-geometric`/`dgl`) — Spark ML has no SEM, no global-optimiser suite, no signal-processing / DSP module, and no deep-RL / GNN support; this cleanup batch is squarely single-node.

### Batch 71 — Cleanup (GraphSAGE / GAT / Triplet / Siamese / SimCLR / Barlow Twins / Viterbi / Baum-Welch / CRF / Beam / BPE / ViT)

Twelve more long-tail fillers covering two inductive / attention-
based GNN architectures, four metric-learning and self-supervised
representation-learning methods, four classical sequence-model
inference algorithms (Viterbi decoding, Baum-Welch EM, linear-chain
CRF, beam-search decoding), sub-word BPE tokenisation, and the
vision Transformer.

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [graphsage-inductive-gnn](techniques/graphsage-inductive-gnn) (Hamilton-Ying-Leskovec 2017 NeurIPS; 3-community SBM test acc 0.66 vs LR 0.37) | 47.149 | ✅ | ✅ | N/A |
| 2 | [graph-attention-networks-gat](techniques/graph-attention-networks-gat) (Velickovic et al 2018 ICLR; 4-head GAT test acc 0.82 vs LR baseline 0.37) | 47.150 | ✅ | ✅ | N/A |
| 3 | [deep-metric-learning-triplet](techniques/deep-metric-learning-triplet) (Schroff-Kalenichenko-Philbin 2015 FaceNet; noisy-synth kNN acc 1.00 vs PCA 0.44) | 47.151 | ✅ | ✅ | N/A |
| 4 | [siamese-networks](techniques/siamese-networks) (Bromley 1994; Koch et al 2015; noisy-synth 4-way one-shot 1.00 vs raw 0.34) | 47.152 | ✅ | ✅ | N/A |
| 5 | [simclr-contrastive](techniques/simclr-contrastive) (Chen-Kornblith-Norouzi-Hinton 2020 ICML; SSL 16-D LR acc 0.883 vs random 0.794) | 47.153 | ✅ | ✅ | N/A |
| 6 | [barlow-twins](techniques/barlow-twins) (Zbontar-Jing-Misra-LeCun-Deny 2021 ICML; SSL 16-D LR acc 0.828 vs random 0.794) | 47.154 | ✅ | ✅ | N/A |
| 7 | [viterbi-algorithm](techniques/viterbi-algorithm) (Viterbi 1967 IEEE-IT; fair/loaded die HMM decoding acc 0.65 on T=300 rolls) | 47.155 | ✅ | ✅ | N/A |
| 8 | [baum-welch-hmm](techniques/baum-welch-hmm) (Baum-Petrie 1966; Baum et al 1970; T=800 EM lifts LL -1922 -> -1408, aligns loaded state) | 47.156 | ✅ | ✅ | N/A |
| 9 | [crf-conditional-random-field](techniques/crf-conditional-random-field) (Lafferty-McCallum-Pereira 2001 ICML; token acc 0.670 vs no-trans 0.625) | 47.157 | ✅ | ✅ | N/A |
| 10 | [beam-search-decoding](techniques/beam-search-decoding) (Reddy 1977; Lowerre 1976; B=1 log-prob -23.13 vs B>=2 -2.18 on toy bigram LM) | 47.158 | ✅ | ✅ | N/A |
| 11 | [bpe-tokenization](techniques/bpe-tokenization) (Gage 1994; Sennrich et al 2016 ACL; 50 merges give 84.5% token reduction, generalises to unseen 'narrowly') | 47.159 | ✅ | ✅ | N/A |
| 12 | [vision-transformer-vit](techniques/vision-transformer-vit) (Dosovitskiy et al 2021 ICLR; toy 8x8 shapes 3-class ViT [CLS]+LR acc 1.000) | 47.160 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 71** — R (`reticulate` to Python for all four GNN / metric / SSL / ViT; `HMM::viterbi`, `HMM::baumWelch`, `crfsuite`, pure-R beam search, `tokenizers.bpe`) or Python (`pytorch-geometric`/`dgl` for GNNs, `pytorch-metric-learning`/`tensorflow-similarity` for triplet / Siamese, `lightly`/`solo-learn` for SimCLR / Barlow, `hmmlearn` for HMM / BW, `sklearn_crfsuite`/`pystruct` for CRF, `transformers.generate` for beam, `sentencepiece`/`huggingface tokenizers`, `timm`/`transformers.ViTModel`) — Spark ML has no GNN / metric-learning / SSL / HMM / CRF / neural-tokeniser / ViT support; this cleanup batch is squarely single-node deep-learning territory.

### Batch 72 — Cleanup (Lottery Ticket / Double Descent / SAM / Lookahead / RAdam / One-Cycle / SWA / MP-Train / Grad-Ckpt / ALiBi / RMSNorm / WS)

Twelve more long-tail fillers on modern deep-net **training and
architecture** techniques: two phenomena of over-parametrised
learning (lottery ticket + double descent), five optimiser wrappers
and learning-rate schedules (SAM, Lookahead, RAdam, One-Cycle, SWA),
two memory / compute optimisations (mixed precision, gradient
checkpointing), and three architecture tricks (ALiBi position bias,
RMSNorm, weight standardisation).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [lottery-ticket-hypothesis](techniques/lottery-ticket-hypothesis) (Frankle-Carbin 2019 ICLR; 94% sparse ticket MSE 0.090 vs dense 0.073) | 47.161 | ✅ | ✅ | N/A |
| 2 | [double-descent](techniques/double-descent) (Belkin et al 2019 PNAS; Nakkiran et al 2020 ICLR; peak test MSE 19.58 at p=n=40, descends to ~4 as p>>n) | 47.162 | ✅ | ✅ | N/A |
| 3 | [sam-sharpness-aware-minimization](techniques/sam-sharpness-aware-minimization) (Foret et al 2021 ICLR; moons+noise MLP: SAM 0.771/flatness 0.003 vs SGD 0.757/0.010) | 47.163 | ✅ | ✅ | N/A |
| 4 | [lookahead-optimizer](techniques/lookahead-optimizer) (Zhang-Lucas-Ba-Hinton 2019 NeurIPS; more robust at extreme lr, +1 pt at lr=1.0) | 47.164 | ✅ | ✅ | N/A |
| 5 | [rectified-adam-radam](techniques/rectified-adam-radam) (Liu et al 2020 ICLR; early-iter loss 15 vs Adam 34 at step 20 under high noise) | 47.165 | ✅ | ✅ | N/A |
| 6 | [one-cycle-super-convergence](techniques/one-cycle-super-convergence) (Smith 2018; triangular lr sweep with reversed momentum, 5-10x faster on non-convex nets) | 47.166 | ✅ | ✅ | N/A |
| 7 | [stochastic-weight-averaging-swa](techniques/stochastic-weight-averaging-swa) (Izmailov et al 2018 UAI; SWA test acc 0.710 vs SGD endpoint 0.698) | 47.167 | ✅ | ✅ | N/A |
| 8 | [mixed-precision-training](techniques/mixed-precision-training) (Micikevicius et al 2018 ICLR; FP16 recovers 1e-8 via loss scale 2^15 vs underflow to 0) | 47.168 | ✅ | ✅ | N/A |
| 9 | [gradient-checkpointing](techniques/gradient-checkpointing) (Chen et al 2016; 25-layer MLP: 4.3x memory reduction, gradients bit-exact) | 47.169 | ✅ | ✅ | N/A |
| 10 | [alibi-linear-attention-bias](techniques/alibi-linear-attention-bias) (Press-Smith-Lewis 2022 ICLR; 4-head slopes give per-head attention radius 3->100 at T=512) | 47.170 | ✅ | ✅ | N/A |
| 11 | [rmsnorm-normalization](techniques/rmsnorm-normalization) (Zhang-Sennrich 2019 NeurIPS; 56.6% faster than LayerNorm on 128x512 tensor) | 47.171 | ✅ | ✅ | N/A |
| 12 | [weight-standardization](techniques/weight-standardization) (Qiao et al 2019; per-channel weight normalisation, pairs with GroupNorm for micro-batch training) | 47.172 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 72** — R (`reticulate` to Python for most; pure-R double-descent / one-cycle / RMSNorm / weight-standardisation demos) or Python (`torch.nn.utils.prune`/OpenLTH for LTH, `sam-optimizer` for SAM, `pytorch_optimizer` for Lookahead / RAdam, `torch.optim.lr_scheduler.OneCycleLR` for one-cycle, `torch.optim.swa_utils` for SWA, `torch.amp` for MP training, `torch.utils.checkpoint` for grad-ckpt, `transformers` Bloom/MPT/OPT for ALiBi, `torch.nn.RMSNorm`/LlamaRMSNorm for RMSNorm, `kornia`/`timm` for weight standardisation) — Spark ML has no deep-net training-toolbox support; this cleanup batch is squarely single-node deep-learning training territory.

### Batch 73 — Cleanup (PagedAttention / KV-quant / TP / PP / FSDP / ZeRO / bitsandbytes / GPTQ / AWQ / FLAN / CAI / ToT)

Twelve more long-tail fillers on modern LLM SYSTEMS and ALIGNMENT
techniques: two LLM-inference memory optimisations (PagedAttention,
KV-cache quant), four parallelism / sharding strategies (tensor
parallel, pipeline parallel, FSDP, ZeRO), three post-training
weight-quantisation methods (LLM.int8, GPTQ, AWQ), and three
alignment / reasoning methods (FLAN instruction tuning,
Constitutional AI, Tree of Thoughts).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [paged-attention-vllm](techniques/paged-attention-vllm) (Kwon et al 2023 SOSP; block allocator + copy-on-write fork, shares 3/5 blocks across sibling sequences) | 47.173 | ✅ | ✅ | N/A |
| 2 | [kv-cache-quantization](techniques/kv-cache-quantization) (Sheng 2023 FlexGen; Liu 2023 KIVI; INT8 K per-channel rel-err 0.0017, INT4 V per-group 0.066, 56% memory saved) | 47.174 | ✅ | ✅ | N/A |
| 3 | [tensor-parallelism](techniques/tensor-parallelism) (Shoeybi 2020 Megatron-LM; TP block bit-exact to single-device (rel-err 4e-16), 1/n_shards memory) | 47.175 | ✅ | ✅ | N/A |
| 4 | [pipeline-parallelism](techniques/pipeline-parallelism) (Huang 2019 GPipe; K=8, M=32 → 82.1% pipeline efficiency, bubble = (K-1)/(M+K-1)) | 47.176 | ✅ | ✅ | N/A |
| 5 | [fsdp-fully-sharded-data-parallel](techniques/fsdp-fully-sharded-data-parallel) (Zhao 2023 VLDB; 7B LLM 112GB→14GB per rank at N=8, 87.5% memory reduction) | 47.177 | ✅ | ✅ | N/A |
| 6 | [zero-redundancy-optimizer](techniques/zero-redundancy-optimizer) (Rajbhandari 2020 SC; 30B model 480→30GB per rank (stage 3), 7.5GB with offload) | 47.178 | ✅ | ✅ | N/A |
| 7 | [bitsandbytes-int8-llm](techniques/bitsandbytes-int8-llm) (Dettmers 2022 NeurIPS; outlier-decomposed INT8 rel-err 0.003 vs naive 0.05, 15x lower error) | 47.179 | ✅ | ✅ | N/A |
| 8 | [gptq-quantization](techniques/gptq-quantization) (Frantar 2023 ICLR; Hessian-guided INT4 rel-err 0.116 vs RTN 0.162, 4x storage saving) | 47.180 | ✅ | ✅ | N/A |
| 9 | [awq-quantization](techniques/awq-quantization) (Lin 2024 MLSys; activation-aware INT4 rel-err 0.081 vs RTN 0.104, α* search) | 47.181 | ✅ | ✅ | N/A |
| 10 | [instruction-tuning-flan](techniques/instruction-tuning-flan) (Wei 2022 ICLR; multi-task natural-language instructions enable zero-shot task generalisation) | 47.182 | ✅ | ✅ | N/A |
| 11 | [constitutional-ai](techniques/constitutional-ai) (Bai 2022 Anthropic; critique-and-revise loop lifts preference score -1.67 → +1.67 on toy harmfulness) | 47.183 | ✅ | ✅ | N/A |
| 12 | [tree-of-thoughts-reasoning](techniques/tree-of-thoughts-reasoning) (Yao 2023 NeurIPS; Game-of-24 CoT-greedy 2/7 vs ToT beam=8 4/7 solved) | 47.184 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 73** — R (`reticulate` to Python for all twelve; these are LLM systems / alignment techniques with no R ecosystem) or Python (`vllm` for PagedAttention, `vllm`/`FlexGen`/`llama.cpp` for KV quant, `megatron-lm`/`colossalai` for TP, `torch.distributed.pipeline`/`deepspeed` for PP, `torch.distributed.fsdp` for FSDP, `deepspeed` for ZeRO, `bitsandbytes` for LLM.int8, `auto-gptq`/`optimum` for GPTQ, `llm-awq`/`autoawq`/`vllm.awq` for AWQ, `transformers.Trainer`+FLAN-T5/Tulu for FLAN, `trl`/`trlx` for CAI, `princeton-nlp/tree-of-thought-llm`/`langgraph`/`lmql` for ToT) — Spark ML has no LLM inference / alignment / large-model training support; this cleanup batch is squarely modern-LLM systems territory.

### Batch 74 — Cleanup (Medusa / MQA / SWA / DPR / ColBERT / Reranker / HyDE / Self-RAG / RECOMP / MoD / PRM / BoN)

Twelve more long-tail fillers on modern LLM INFERENCE, RETRIEVAL,
and REASONING techniques: one speculative-decoding variant
(Medusa), two attention efficiencies (MQA/GQA, sliding-window),
four retrieval / RAG methods (DPR, ColBERT, cross-encoder,
HyDE), two RAG-pipeline extensions (Self-RAG, RECOMP), one
mixture-of-experts across depth (MoD), and two reasoning /
selection methods (PRM, best-of-N).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [medusa-speculative-heads](techniques/medusa-speculative-heads) (Cai et al 2024; 2 heads + tree verify → 2.58 tokens/step vs 1 baseline) | 47.185 | ✅ | ✅ | N/A |
| 2 | [multi-query-attention](techniques/multi-query-attention) (Shazeer 2019, Ainslie 2023 GQA; MQA cuts KV memory 32× vs MHA on 32-head model) | 47.186 | ✅ | ✅ | N/A |
| 3 | [sliding-window-attention](techniques/sliding-window-attention) (Beltagy 2020 Longformer, Jiang 2023 Mistral; W=64 rel-diff 0.24 vs full at T=128) | 47.187 | ✅ | ✅ | N/A |
| 4 | [dense-passage-retrieval-dpr](techniques/dense-passage-retrieval-dpr) (Karpukhin et al 2020 EMNLP; DPR R@1 1.00 vs BM25 0.75 on 8-pair QA) | 47.188 | ✅ | ✅ | N/A |
| 5 | [colbert-late-interaction](techniques/colbert-late-interaction) (Khattab-Zaharia 2020 SIGIR; MaxSim per-token retrieval, 3/3 on rare-term queries) | 47.189 | ✅ | ✅ | N/A |
| 6 | [cross-encoder-reranker](techniques/cross-encoder-reranker) (Nogueira-Cho 2019; second-stage BERT([q;SEP;p]) reranking on top-k) | 47.190 | ✅ | ✅ | N/A |
| 7 | [hyde-hypothetical-doc](techniques/hyde-hypothetical-doc) (Gao et al 2022 ACL; HyDE R@1 3/5 vs plain-DPR 0/5 on paraphrased queries) | 47.191 | ✅ | ✅ | N/A |
| 8 | [self-rag](techniques/self-rag) (Asai et al 2024 ICLR; reflection tokens IsRel/IsSup/IsUse for retrieve-when-needed critique) | 47.192 | ✅ | ✅ | N/A |
| 9 | [context-compression-recomp](techniques/context-compression-recomp) (Xu et al 2024 ICLR; extractive top-p compresses 80→15-29 tokens, preserves answer) | 47.193 | ✅ | ✅ | N/A |
| 10 | [mixture-of-depths](techniques/mixture-of-depths) (Raposo et al 2024 DeepMind; per-token per-block router cuts compute 50% at keep_frac=0.5) | 47.194 | ✅ | ✅ | N/A |
| 11 | [process-reward-model-prm](techniques/process-reward-model-prm) (Lightman et al 2023 OpenAI PRM800K; PRM downgrades 'lucky' chain 1.0 ORM → 0.30 PRM) | 47.195 | ✅ | ✅ | N/A |
| 12 | [best-of-n-sampling](techniques/best-of-n-sampling) (Cobbe 2021, Nakano 2021 WebGPT; N=256 reduces expected error 74× on toy) | 47.196 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 74** — R (`reticulate` to Python for all; these are LLM / retrieval / reasoning-time methods with no R ecosystem) or Python (`FasterDecoding/Medusa` for Medusa, `transformers.LlamaAttention` for MQA/GQA, `transformers.Mistral`/`Longformer` for SWA, `sentence-transformers`/`haystack` for DPR, `colbert-ai`/`ragatouille` for ColBERT, `sentence-transformers.CrossEncoder`/`MonoT5` for reranker, `llama-index`/`langchain` HyDE for HyDE, `AkariAsai/self-rag` for Self-RAG, `carriex/recomp`/`llmlingua` for RECOMP, DeepMind JAX reference for MoD, `openai/prm800k`/Math-Shepherd for PRM, `transformers.generate(num_return_sequences=…)` for best-of-N) — Spark ML has no LLM inference / retrieval / reasoning support; this cleanup batch is squarely modern-LLM territory.

### Batch 75 — Cleanup (KTO / ORPO / SimPO / SPIN / Reflexion / ReAct / Toolformer / Function-Call / Verifier / Debate / AutoGen / CrewAI)

Twelve more long-tail fillers on modern LLM ALIGNMENT and AGENTIC
techniques: four preference-optimisation variants beyond DPO
(KTO, ORPO, SimPO, SPIN), four reasoning / tool-use methods
(Reflexion, ReAct, Toolformer, function calling), one verifier-
guided search, and three multi-agent orchestration patterns
(debate, AutoGen, CrewAI).

| # | Technique | Ref | R | Python | PySpark |
|---|-----------|-----|---|--------|---------|
| 1 | [kto-kahneman-tversky](techniques/kto-kahneman-tversky) (Ethayarajh et al 2024; unpaired thumbs labels shift liked mass 70%→93%, disliked 30%→7%) | 47.197 | ✅ | ✅ | N/A |
| 2 | [orpo-odds-ratio](techniques/orpo-odds-ratio) (Hong-Lee-Thorne 2024 EMNLP; SFT+OR loss, no ref model, chosen prob → 0.989 in 150 iters) | 47.198 | ✅ | ✅ | N/A |
| 3 | [simpo-simple-preference](techniques/simpo-simple-preference) (Meng-Xia-Chen 2024; length-normalised reference-free preference with target margin γ) | 47.199 | ✅ | ✅ | N/A |
| 4 | [spin-self-play-fine-tuning](techniques/spin-self-play-fine-tuning) (Chen et al 2024 ICML; KL(π‖human) 0.30→0.075 over 5 self-play rounds using human samples only) | 47.200 | ✅ | ✅ | N/A |
| 5 | [reflexion-self-critique](techniques/reflexion-self-critique) (Shinn et al 2023 NeurIPS; episodic memory of failed actions: 5/6 vs baseline 4/6 on arithmetic puzzles) | 47.201 | ✅ | ✅ | N/A |
| 6 | [react-reasoning-acting](techniques/react-reasoning-acting) (Yao et al 2023 ICLR; Thought/Action/Observation loop: 3/3 vs CoT-only 0/3 on QA + calc questions) | 47.202 | ✅ | ✅ | N/A |
| 7 | [toolformer-tool-use](techniques/toolformer-tool-use) (Schick et al 2023 NeurIPS; perplexity-filtered self-supervised API insertions, KEEP only useful ones) | 47.203 | ✅ | ✅ | N/A |
| 8 | [function-calling-openai](techniques/function-calling-openai) (OpenAI 2023; JSON-schema-validated structured tool calls for get_weather + calc) | 47.204 | ✅ | ✅ | N/A |
| 9 | [verifier-guided-search](techniques/verifier-guided-search) (Uesato 2022; Lightman 2023; beam=3 verifier: 4/5 vs greedy-N=20 1/5 on arithmetic-target task) | 47.205 | ✅ | ✅ | N/A |
| 10 | [multi-agent-debate](techniques/multi-agent-debate) (Du et al 2023; 3-agent 3-round debate: 93% vs single-agent 55% on arithmetic QA) | 47.206 | ✅ | ✅ | N/A |
| 11 | [autogen-multi-agent](techniques/autogen-multi-agent) (Wu et al 2023 Microsoft; planner+coder+reviewer 4-message conversation to TASK COMPLETE) | 47.207 | ✅ | ✅ | N/A |
| 12 | [crewai-hierarchical-agents](techniques/crewai-hierarchical-agents) (Moura 2024; sequential + manager-hierarchical processes on researcher/writer/editor task DAG) | 47.208 | ✅ | ✅ | N/A |

**PySpark N/A across Batch 75** — R (`reticulate` to Python for all; these are LLM alignment / agentic methods with no R ecosystem) or Python (`trl.KTOTrainer`/`ORPOTrainer`/`CPOTrainer` for preference variants, `uclaml/SPIN` for self-play, `noahshinn024/reflexion`/`langgraph` for Reflexion, `langchain.ReActAgent`/`llama-index` for ReAct, `conceptofmind/toolformer` for Toolformer, `openai`/`anthropic` SDKs for function calling, `llm-reasoners`/`princeton-nlp/tree-of-thought-llm` for verifier search, `composable-models/llm-multiagent-debate` for debate, `pyautogen` for AutoGen, `crewai` for CrewAI) — Spark ML has no alignment / agent / tool-use support; this cleanup batch is squarely modern-LLM territory.

Later batches: any remaining chapters.

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
