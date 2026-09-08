# Mixed logit / random-coefficient MNL (Reference Sec 47.54)
# Native R via mlogit / gmnl / apollo; Python via xlogit / pylogit.
# Run with:  Rscript mixed_logit_mnl.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  mlogit::mlogit(rpar = ...)   -- random-parameters MNL, Halton draws\n")
  cat("  gmnl                         -- generalised MNL: scale + preference het\n")
  cat("  apollo                       -- flexible choice model + Bayesian option\n")
  cat("Python:\n")
  cat("  xlogit                       -- GPU-accelerated mixed logit simulation\n")
  cat("  pylogit                      -- classical + mixed logit\n")
  cat("  from-scratch                 -- see mixed_logit_mnl.py\n")
  cat("Refs: McFadden & Train (2000) J Appl Econ 15(5); Train (2009) 'Discrete\n")
  cat("      Choice Methods with Simulation', 2nd ed, Cambridge.\n")
}
