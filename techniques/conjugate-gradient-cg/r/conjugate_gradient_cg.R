# Conjugate Gradient (Reference Sec 47.320)
# Native R via Matrix / RSpectra / rARPACK; Python via scipy.sparse / from-scratch.
# Run with:  Rscript conjugate_gradient_cg.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Matrix::CHMfactor / rspectra    -- SPD linear solvers\n")
  cat("  RcppEigen (Eigen ConjugateGradient)\n")
  cat("  cg (older R port)\n")
  cat("Python:\n")
  cat("  scipy.sparse.linalg.cg / cgs / bicg / bicgstab / minres / gmres\n")
  cat("  torch.linalg.solve for direct alternative\n")
  cat("  From-scratch (see conjugate_gradient_cg.py)\n")
  cat("Refs: Hestenes, M.R. & Stiefel, E. (1952) 'Methods of conjugate\n")
  cat("      gradients for solving linear systems', J Res NBS 49;\n")
  cat("      Shewchuk, J.R. (1994) 'An Introduction to the Conjugate Gradient\n")
  cat("      Method Without the Agonizing Pain', CMU technical note.\n")
}
