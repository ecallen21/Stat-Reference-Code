# Readability measures (Reference Sec 42.16)
# Native R via quanteda / koRpus; Python textstat + custom.
# Run with:  Rscript readability_measures.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quanteda::textstat_readability  -- 47+ indices in one call\n")
  cat("  koRpus                           -- Flesch, F-K, SMOG, Coleman-Liau, ARI...\n")
  cat("Python:\n")
  cat("  textstat (flesch_reading_ease, gunning_fog, smog_index, coleman_liau_index)\n")
  cat("  readability                       -- alternate implementation\n")
  cat("Refs: Flesch (1948) 'A new readability yardstick', J Appl Psych;\n")
  cat("      Crossley, Greenfield & McNamara (2008) 'Assessing text readability using\n")
  cat("      cognitively based indices', TESOL Quarterly.\n")
}
