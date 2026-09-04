# Polygenic risk scores (Reference Sec 40.2, 40.15)
# External PRSice2 / bigsnpr canonical; Python custom.
# Run with:  Rscript polygenic_risk_scores.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  bigsnpr::snp_PRS               -- fast PRS computation over large cohorts\n")
  cat("  lassosum::lassosum.pipeline    -- LD-aware PRS via lasso\n")
  cat("  PRSice2 (external)             -- pipeline for P+T PRS\n")
  cat("Python:\n")
  cat("  ldpred                         -- LD-adjusted PRS\n")
  cat("  pandas-plink + custom          -- P+T pipeline\n")
  cat("  PRSice2 (external)\n")
  cat("Refs: Choi, Mak & O'Reilly (2020) 'Tutorial: PRS analyses', Nat Protocols;\n")
  cat("      Wand et al. (2021) 'Improving reporting standards for polygenic scores',\n")
  cat("      Nature.\n")
}
