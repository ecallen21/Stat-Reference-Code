# TRADES adversarial training (Reference Ch 30 Robustness)
# R via reticulate + Python; no first-class native R support.
# Run with:  Rscript trades_adversarial.R

if (sys.nframe() == 0) {
  cat("R packages: no first-class R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  trades-pytorch (yaodongyu)  -- reference TRADES implementation for CIFAR-10\n")
  cat("  advertorch                  -- pytorch attack + defence toolkit incl. TRADES helpers\n")
  cat("  torchattacks                -- KL-based PGD variants + AutoAttack\n")
  cat("Refs: Zhang, H., Yu, Y., Jiang, J., Xing, E., El Ghaoui, L. & Jordan, M. (2019)\n")
  cat("      'Theoretically Principled Trade-off between Robustness and Accuracy',\n")
  cat("      ICML.\n")
}
