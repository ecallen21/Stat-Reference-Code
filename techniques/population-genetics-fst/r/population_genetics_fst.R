# Population-genetics F_ST (Reference Sec 40.20)
# Native R via hierfstat / pegas; Python scikit-allel + custom.
# Run with:  Rscript population_genetics_fst.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  hierfstat::pairwise.WCfst / basic.stats -- Weir-Cockerham F-statistics\n")
  cat("  pegas::Fst                       -- classical F-statistics\n")
  cat("  adegenet::dapc                   -- discriminant analysis of populations\n")
  cat("  LEA::snmf                        -- admixture proportions\n")
  cat("Python:\n")
  cat("  scikit-allel::weir_cockerham_fst / hudson_fst / patterson_f3\n")
  cat("  hail::hl.hardy_weinberg_test etc.\n")
  cat("Refs: Weir & Cockerham (1984) 'Estimating F-statistics for the analysis of\n")
  cat("      population structure', Evolution; Pritchard, Stephens & Donnelly (2000)\n")
  cat("      'Inference of population structure using multilocus genotype data', Genetics.\n")
}
