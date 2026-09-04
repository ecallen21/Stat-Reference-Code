# Record linkage / entity resolution (Reference Sec 38.6)
# Native R via fastLink / RecordLinkage; Python recordlinkage + custom.
# Run with:  Rscript record_linkage.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  fastLink                       -- FS + EM + posterior probs (Enamorado 2019)\n")
  cat("  RecordLinkage                  -- deterministic + probabilistic linkage\n")
  cat("  reclin2                        -- pipeline API for FS linkage\n")
  cat("Python:\n")
  cat("  recordlinkage                  -- FS + blocking + comparison metrics\n")
  cat("  dedupe                         -- ML-based deduplication\n")
  cat("  splink                         -- Spark-scale FS linkage\n")
  cat("Refs: Fellegi, I.P. & Sunter, A.B. (1969) 'A theory for record linkage', JASA;\n")
  cat("      Herzog, Scheuren & Winkler (2007) Data Quality and Record Linkage Techniques,\n")
  cat("      Springer; Winkler, W.E. (1988) 'Using the EM algorithm for weight computation'.\n")
}
