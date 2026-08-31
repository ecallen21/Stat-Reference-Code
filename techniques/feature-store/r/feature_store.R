# Feature store (Reference Ch 32 MLOps)
# R via pins / vetiver + database backends; hosted feature stores in Python.
# Run with:  Rscript feature_store.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pins                        -- versioned tabular asset storage (RStudio Connect)\n")
  cat("  vetiver                     -- model + feature endpoint bundling\n")
  cat("  DBI + duckdb / arrow        -- offline store on parquet / DuckDB\n")
  cat("Python:\n")
  cat("  feast                       -- open-source reference feature store\n")
  cat("  tecton                      -- managed feature-store service\n")
  cat("  sagemaker-feature-store     -- AWS; online (DynamoDB) + offline (S3)\n")
  cat("  vertex-ai-feature-store     -- Google Cloud; BigQuery offline + Bigtable online\n")
  cat("Refs: Chip Huyen (2022) 'Designing Machine Learning Systems', O'Reilly,\n")
  cat("      ch. 7 'Feature Engineering'; Uber Michelangelo Palette (2017).\n")
}
