# Change-point detection (Reference Sec 38.8)
# Native R via changepoint / ecp; Python ruptures + custom.
# Run with:  Rscript change_point_detection.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  changepoint (cpt.mean, cpt.var, cpt.meanvar) -- PELT, BinSeg, SegNeigh, AMOC\n")
  cat("  ecp                            -- nonparametric multivariate change points\n")
  cat("  bcp                            -- Bayesian change-point analysis\n")
  cat("Python:\n")
  cat("  ruptures (Binseg, Pelt, BottomUp, KernelCPD)\n")
  cat("  bayesian_changepoint_detection -- online BOCPD (Adams-MacKay)\n")
  cat("  custom                         -- binary segmentation + PELT reference\n")
  cat("Refs: Killick, Fearnhead & Eckley (2012) 'Optimal detection of change points\n")
  cat("      with a linear computational cost', JASA; Chen & Gupta (2012) Parametric\n")
  cat("      Statistical Change Point Analysis, Birkhauser.\n")
}
