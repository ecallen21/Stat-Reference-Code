# Frank-Wolfe (Reference Sec 47.100)
# Native R via CVXR / cvxopt; Python via cvxpy / pymanopt.
# Run with:  Rscript frank_wolfe_conditional_gradient.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  CVXR                     -- disciplined convex programming; ships FW-style\n")
  cat("  cvxopt                   -- interior-point + FW options\n")
  cat("  Rmosek / gurobi          -- commercial LP/QP solvers for LMO steps\n")
  cat("Python:\n")
  cat("  cvxpy                    -- CVX-style modelling; auto solver selection\n")
  cat("  pymanopt                 -- Riemannian FW variants\n")
  cat("  chocolate                -- Frank-Wolfe research code (Jaggi group)\n")
  cat("  from-scratch             -- see frank_wolfe_conditional_gradient.py\n")
  cat("Refs: Frank & Wolfe (1956) Naval Research Logistics 3(1-2); Jaggi (2013)\n")
  cat("      'Revisiting Frank-Wolfe: Projection-free sparse convex optimization',\n")
  cat("      ICML.\n")
}
