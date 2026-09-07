# Causal discovery -- PC algorithm (Reference Sec 15.35)
# Native R via pcalg / bnlearn; Python via causal-learn / cdt.
# Run with:  Rscript causal_discovery_pc.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  pcalg::pc                 -- PC algorithm with Fisher-z / disCI test\n")
  cat("  pcalg::fci                -- FCI for latent-confounder-robust discovery\n")
  cat("  bnlearn::pc.stable        -- order-independent PC variant\n")
  cat("  bnlearn::gs / iamb        -- Grow-Shrink / IAMB alternatives\n")
  cat("Python:\n")
  cat("  causal-learn (CMU)        -- pip install causal-learn\n")
  cat("  cdt (Causal Discovery Toolbox)\n")
  cat("Refs: Spirtes, P., Glymour, C. & Scheines, R. (2000) Causation, Prediction\n")
  cat("      and Search, 2nd ed., MIT Press; Kalisch, M. & Buhlmann, P. (2007)\n")
  cat("      'Estimating high-dimensional DAGs with the PC-algorithm', JMLR.\n")
}
