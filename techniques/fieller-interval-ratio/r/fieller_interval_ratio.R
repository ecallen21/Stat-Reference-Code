# Fieller confidence interval for a ratio (Reference Sec 3.27)
# Native R via mratios / ratio; Python via from-scratch scipy.
# Run with:  Rscript fieller_interval_ratio.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mratios                     -- multiple ratios with Fieller CIs\n")
  cat("  MBESS::ci.pi                 -- Fieller in psychometrics (item indices)\n")
  cat("  car::deltaMethod / Fieller.wrapping in specific packages\n")
  cat("  drc::EDcomp                  -- Fieller CI for dose ratios\n")
  cat("Python:\n")
  cat("  From-scratch scipy (see fieller_interval_ratio.py)\n")
  cat("  linearmodels + Wald test wrappers (delta baseline)\n")
  cat("Refs: Fieller, E.C. (1954) 'Some problems in interval estimation',\n")
  cat("      JRSS-B 16(2): 175-185; Buonaccorsi, J.P. (2005) Fieller's Theorem\n")
  cat("      chapter in Encyclopedia of Biostatistics.\n")
}
