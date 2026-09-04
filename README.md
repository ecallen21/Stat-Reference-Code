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
