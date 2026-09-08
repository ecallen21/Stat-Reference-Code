# Loopy belief propagation (Reference Sec 47.121)
# Native R via gRain / bnlearn; Python via pgmpy / libDAI.
# Run with:  Rscript belief_propagation_loopy.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  gRain::propagate            -- exact junction-tree propagation\n")
  cat("  bnlearn                     -- Bayesian-network learning + inference\n")
  cat("  igraph::cluster_leading_eigen -- spectral BP variant\n")
  cat("Python:\n")
  cat("  pgmpy.inference.BeliefPropagation  -- exact + approximate BP\n")
  cat("  libDAI                             -- C++ loopy BP + generalised BP\n")
  cat("  pyMRF / pystruct                    -- structured-prediction with BP\n")
  cat("  from-scratch                        -- see belief_propagation_loopy.py\n")
  cat("Refs: Pearl (1988) 'Probabilistic Reasoning in Intelligent Systems',\n")
  cat("      Morgan Kaufmann; Yedidia, Freeman & Weiss (2005) IEEE TIT 51(7).\n")
}
