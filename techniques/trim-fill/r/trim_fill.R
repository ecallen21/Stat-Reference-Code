# Trim-and-fill (Reference Sec 22.4)
# Native R via metafor::trimfill; Python custom.
# Run with:  Rscript trim_fill.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  metafor::trimfill                 -- Duval-Tweedie trim-fill\n")
  cat("  meta::trimfill                    -- alternative implementation\n")
  cat("  metabias                          -- Egger's + Begg's tests\n")
  cat("Python:\n")
  cat("  pymare                            -- meta-analysis toolkit\n")
  cat("  custom (R^0 / L_0 estimators)\n")
  cat("Refs: Duval & Tweedie (2000) 'A nonparametric trim and fill method of\n")
  cat("      accounting for publication bias in meta-analysis', JASA; Sterne, Egger &\n")
  cat("      Smith (2001) 'Investigating and dealing with publication and other biases\n")
  cat("      in meta-analysis', BMJ.\n")
}
