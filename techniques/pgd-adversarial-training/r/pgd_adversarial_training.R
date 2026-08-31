# PGD attack + adversarial training (Reference Ch 30 Robustness)
# R via reticulate + Python; no first-class native R support.
# Run with:  Rscript pgd_adversarial_training.R

if (sys.nframe() == 0) {
  cat("R packages: no first-class R support; use reticulate + Python.\n")
  cat("Python:\n")
  cat("  foolbox                    -- PGD/FGSM/CW/AutoAttack + adversarial-training helpers\n")
  cat("  cleverhans (JAX / PyTorch) -- Madry-style PGD adversarial training\n")
  cat("  advertorch                 -- pytorch attack + adversarial training loop\n")
  cat("  torchattacks               -- lightweight PGD + AutoAttack for pytorch\n")
  cat("  robustness (madry-lab)     -- Madry-style AT reference implementation for CIFAR/ImageNet\n")
  cat("Refs: Madry, A., Makelov, A., Schmidt, L., Tsipras, D. & Vladu, A. (2018)\n")
  cat("      'Towards Deep Learning Models Resistant to Adversarial Attacks', ICLR.\n")
}
