# NOTEARS DAG learning (Reference Sec 47.55)
# Native R via notearsC / pcalg; Python via causalnex / dagma.
# Run with:  Rscript notears_dag_learning.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  notearsC             -- C++ port of the original NOTEARS algorithm\n")
  cat("  pcalg::gds / ges     -- score-based DAG search alternatives\n")
  cat("  bnlearn::hc          -- hill-climbing greedy DAG learner\n")
  cat("Python:\n")
  cat("  causalnex.notears    -- linear + MLP nonlinear NOTEARS\n")
  cat("  dagma                -- log-det acyclicity, faster / better optima\n")
  cat("  gCastle              -- suite of DAG learners incl. NOTEARS variants\n")
  cat("  from-scratch         -- see notears_dag_learning.py\n")
  cat("Refs: Zheng, Aragam, Ravikumar & Xing (2018) 'DAGs with NO TEARS', NeurIPS;\n")
  cat("      Bello, Aragam & Ravikumar (2022) 'DAGMA', NeurIPS.\n")
}
