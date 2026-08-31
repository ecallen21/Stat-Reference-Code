# FGSM adversarial attack (Reference Ch 30 Robustness)
# R via reticulate + Python; no first-class R attack library.
# Run with:  Rscript fgsm_adversarial.R

if (sys.nframe() == 0) {
  cat("R packages: no first-class R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  foolbox                    -- FGSM/PGD/CW/DeepFool + robustness benchmark\n")
  cat("  cleverhans (JAX / PyTorch) -- classic attack + defence library\n")
  cat("  advertorch                 -- PyTorch attack + defence toolkit\n")
  cat("  torchattacks               -- lightweight FGSM/PGD/AutoAttack for pytorch\n")
  cat("Refs: Goodfellow, I., Shlens, J. & Szegedy, C. (2014) 'Explaining and\n")
  cat("      Harnessing Adversarial Examples', ICLR 2015.\n")
}
