# L-BFGS (Reference Sec 47.319)
# Native R via stats::optim(method="L-BFGS-B"); Python via scipy.optimize / from-scratch.
# Run with:  Rscript lbfgs_quasi_newton.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  stats::optim(method='L-BFGS-B')  -- box-constrained L-BFGS\n")
  cat("  lbfgs (R port)                    -- Nocedal implementation\n")
  cat("  optimParallel                     -- L-BFGS with parallel gradient\n")
  cat("Python:\n")
  cat("  scipy.optimize.minimize(method='L-BFGS-B')\n")
  cat("  scipy.optimize.fmin_l_bfgs_b\n")
  cat("  torch.optim.LBFGS (mini-batch style)\n")
  cat("  From-scratch two-loop (see lbfgs_quasi_newton.py)\n")
  cat("Refs: Liu, D.C. & Nocedal, J. (1989) 'On the limited memory BFGS method\n")
  cat("      for large scale optimization', Math Programming 45;\n")
  cat("      Byrd, R.H., Lu, P., Nocedal, J. & Zhu, C. (1995) 'A limited memory\n")
  cat("      algorithm for bound constrained optimization', SIAM J Sci Comput 16.\n")
}
