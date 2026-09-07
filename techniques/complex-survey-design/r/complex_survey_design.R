# Complex survey design analysis (Reference Sec 27.7)
# Native R via survey (Lumley); Python via samplics.
# Run with:  Rscript complex_survey_design.R

if (sys.nframe() == 0) {
  cat("R packages:\n")
  cat("  survey::svydesign / svymean / svyglm  -- reference design-based analysis\n")
  cat("  survey::as.svrepdesign                 -- replicate weights (BRR, JK, boot)\n")
  cat("  srvyr                                   -- dplyr-style survey wrapper\n")
  cat("  sampling                                 -- design creation helpers\n")
  cat("Python:\n")
  cat("  samplics                                 -- SAS-comparable survey suite\n")
  cat("  statsmodels.stats.weightstats            -- weighted means and Ttests\n")
  cat("  from-scratch numpy for jackknife / linearisation (see .py)\n")
  cat("Refs: Kish, L. (1965) Survey Sampling, Wiley; Cochran, W.G. (1977)\n")
  cat("      Sampling Techniques, 3rd ed., Wiley; Lumley, T. (2010) Complex\n")
  cat("      Surveys, Wiley.\n")
}
