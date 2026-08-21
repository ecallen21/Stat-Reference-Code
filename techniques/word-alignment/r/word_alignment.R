# IBM Model 1 word alignment (Reference §25.x extra)
# R has limited native SMT tooling; use reticulate + Python or run fastalign / eflomal externally.
# Run with:  Rscript word_alignment.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  Rmoses, tmt — limited coverage; classical SMT ecosystem is in C++ (Moses).\n")
  cat("  reticulate + Python (eflomal, awesome-align, or fastalign) for production alignment.\n")
  cat("External:\n")
  cat("  fastalign (Dyer et al. 2013)  -- fastest IBM-2 style aligner\n")
  cat("  GIZA++ (Och-Ney 2003)          -- IBM models 1-5 + HMM aligner\n")
  cat("  eflomal (Ostling-Tiedemann)    -- Gibbs-sampled variational alignment\n")
  cat("  awesome-align                    -- transformer-based cross-lingual alignment\n")
  cat("Modern MT: no explicit alignment step; attention weights in a transformer serve as soft alignment.\n")
}
