# HyperLogLog cardinality estimator (Reference Sec 47.71)
# Native R via hll / RcppHash; Python via datasketch / hyperloglog.
# Run with:  Rscript hyperloglog_cardinality.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  hll::add / cardinality       -- R wrapper around HLL++ (C++)\n")
  cat("  RcppHash                     -- MurmurHash3 for HLL implementations\n")
  cat("  dbplyr + Redshift/BQ         -- server-side APPROX_COUNT_DISTINCT\n")
  cat("Python:\n")
  cat("  datasketch.HyperLogLog       -- HLL / HLL++ + serialisation\n")
  cat("  hyperloglog                  -- pure-Python HLL\n")
  cat("  pyspark.sql.functions.approx_count_distinct  -- Spark HLL++ at scale\n")
  cat("  from-scratch                 -- see hyperloglog_cardinality.py\n")
  cat("Refs: Flajolet, Fusy, Gandouet & Meunier (2007) AofA; Heule, Nunkesser &\n")
  cat("      Hall (2013) 'HLL++ in practice', EDBT.\n")
}
