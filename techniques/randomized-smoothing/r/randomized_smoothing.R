# Randomised smoothing certified L2 robustness (Reference Ch 30 Robustness)
# R via reticulate + Python; the sampling loop is easy in native R too.
# Run with:  Rscript randomized_smoothing.R

if (sys.nframe() == 0) {
  cat("R packages: no first-class R support; the Monte-Carlo loop is trivial.\n")
  cat("  DescTools::BinomCI            -- Clopper-Pearson bounds for p_A / p_B\n")
  cat("Python:\n")
  cat("  smoothing-cohen (Cohen 2019)  -- reference implementation\n")
  cat("  certified-robustness (pip)    -- unified certified-robustness benchmark\n")
  cat("  torchcp                        -- includes randomised-smoothing helpers\n")
  cat("Refs: Cohen, J., Rosenfeld, E. & Kolter, Z. (2019) 'Certified Adversarial\n")
  cat("      Robustness via Randomized Smoothing', ICML;\n")
  cat("      Salman, H. et al. (2019) 'Provably Robust Deep Learning via Adversarially\n")
  cat("      Trained Smoothed Classifiers', NeurIPS.\n")
}
