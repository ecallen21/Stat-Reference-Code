# Subgradient Method (Reference Sec 47.338)
# Native R via quantreg / conquer for L1 regression; Python via cvxpy / from-scratch.
# Run with:  Rscript subgradient_method.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  quantreg::rq (tau=0.5)         -- LAD via LP (usually better than subgrad)\n")
  cat("  conquer                         -- fast smoothed quantile regression\n")
  cat("  base R custom loop              -- subgradient in 10 lines\n")
  cat("Python:\n")
  cat("  cvxpy                           -- specify LAD as LP\n")
  cat("  statsmodels QuantReg\n")
  cat("  From-scratch (see subgradient_method.py)\n")
  cat("Refs: Shor, N.Z. (1985) 'Minimization Methods for Non-differentiable\n")
  cat("      Functions', Springer;  Boyd, S., Xiao, L. & Mutapcic, A. (2004)\n")
  cat("      'Subgradient Methods', Stanford EE392o Lecture Notes.\n")
}
