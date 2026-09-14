# Mirror Descent (Reference Sec 47.339)
# Native R via base R custom loops; Python via cvxpy / from-scratch.
# Run with:  Rscript mirror_descent.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  base R custom exp-grad update  -- 5-line implementation\n")
  cat("  gtools::rdirichlet + custom     -- simplex-projected iterates\n")
  cat("Python:\n")
  cat("  cvxpy (declarative)\n")
  cat("  Custom torch autograd + softmax step\n")
  cat("  From-scratch (see mirror_descent.py)\n")
  cat("Refs: Nemirovsky, A.S. & Yudin, D.B. (1983) 'Problem Complexity and\n")
  cat("      Method Efficiency in Optimization', Wiley;\n")
  cat("      Beck, A. & Teboulle, M. (2003) 'Mirror descent and nonlinear\n")
  cat("      projected subgradient methods for convex optimization',\n")
  cat("      Oper Res Lett 31(3).\n")
}
