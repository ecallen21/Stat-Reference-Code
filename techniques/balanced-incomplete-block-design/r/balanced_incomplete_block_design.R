# Balanced incomplete block design (BIBD) (Reference Sec 32.9)
# Native R via AlgDesign / crossdes / ibd; Python via pyDOE2 / custom.
# Run with:  Rscript balanced_incomplete_block_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  AlgDesign::optBlock       -- algorithmic BIBD / general block designs\n")
  cat("  crossdes::isBIB / find.BIB -- BIBD constructors / validators\n")
  cat("  ibd                        -- specialised BIBD packages\n")
  cat("  agricolae::design.bib      -- agriculture-flavoured BIBD constructor\n")
  cat("  lme4::lmer + intra-block   -- inter/intra-block combined analysis\n")
  cat("Python:\n")
  cat("  pyDOE2 (community) -- experimental designs (LHS, factorial, BIBD helpers)\n")
  cat("  From-scratch numpy (see balanced_incomplete_block_design.py)\n")
  cat("Refs: Yates, F. (1936) 'A new method of arranging variety trials\n")
  cat("      involving a large number of varieties', J Agric Sci 26; Fisher,\n")
  cat("      R.A. (1935) The Design of Experiments; Cochran & Cox (1957)\n")
  cat("      Experimental Designs, Wiley.\n")
}
