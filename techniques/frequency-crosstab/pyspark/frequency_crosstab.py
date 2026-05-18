"""Frequency distributions and cross-tabulations on a Spark DataFrame (Reference §1.7).

  - frequency table   : groupBy(col).count(), then a window for cumulative totals
  - cross-tabulation  : DataFrame.crosstab(rowCol, colCol)  (built in)
  - row / col / total %: derive from the count crosstab with simple column arithmetic

Run:  python frequency_crosstab.py
"""
from __future__ import annotations    # stdlib: postpone type-hint evaluation (lets us write int | None)

from pyspark.sql import SparkSession, Window    # SparkSession: entry point;  Window: window-function specifications
from pyspark.sql import functions as F    # Spark DataFrame column functions (F.col, F.mean, F.sum, F.when, ...)


def frequency_table(df, col: str):
    n = df.count()
    w = Window.orderBy(col).rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return (df.groupBy(F.col(col).alias("category")).agg(F.count(F.lit(1)).alias("count"))
            .withColumn("rel_freq", F.col("count") / F.lit(n))
            .withColumn("cum_count", F.sum("count").over(w))
            .withColumn("cum_rel", F.col("cum_count") / F.lit(n))
            .orderBy("category"))


def crosstab_counts(df, row_col: str, col_col: str):
    """Built-in contingency table; first column is the row category, rest are column levels."""
    return df.crosstab(row_col, col_col)


def crosstab_row_pct(df, row_col: str, col_col: str):
    ct = df.crosstab(row_col, col_col)
    cat_col = ct.columns[0]
    level_cols = ct.columns[1:]
    total = sum(F.col(c) for c in level_cols)
    return ct.select(cat_col, *[(F.col(c) / total * 100).alias(c) for c in level_cols])


if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("frequency-crosstab").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    try:
        rows = [
            ("North", "Yes"), ("North", "No"), ("North", "No"), ("South", "Yes"),
            ("South", "No"), ("East", "Yes"), ("East", "No"), ("North", "Yes"),
            ("South", "No"), ("East", "No"), ("North", "No"), ("South", "Yes"),
        ]
        df = spark.createDataFrame(rows, ["region", "outcome"])
        print("frequency table (region):")
        frequency_table(df, "region").show()
        print("crosstab counts:")
        crosstab_counts(df, "region", "outcome").show()
        print("crosstab row %:")
        crosstab_row_pct(df, "region", "outcome").show()
    finally:
        spark.stop()
