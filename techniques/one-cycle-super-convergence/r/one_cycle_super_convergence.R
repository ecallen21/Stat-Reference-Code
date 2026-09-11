# One-Cycle / Super-Convergence (Reference Sec 47.166).
#
# Smith 2018. Triangular learning-rate cycle with momentum inverse
# to lr; enables very-fast training on non-convex nets.

one_cycle_lr <- function(step, total, lr_max, lr_min_ratio = 25, lr_final_ratio = 1e4) {
    lr_min <- lr_max / lr_min_ratio
    lr_end <- lr_max / lr_final_ratio
    half <- total %/% 2
    if (step < half) lr_min + (lr_max - lr_min) * step / half
    else if (step < 2 * half) lr_max + (lr_min - lr_max) * (step - half) / half
    else lr_min + (lr_end - lr_min) * (step - 2 * half) / max(total - 2 * half, 1)
}

cat("=== One-Cycle schedule (Smith 2018) ===\n")
total <- 500
for (s in c(0, 125, 250, 375, 500)) {
    cat(sprintf("  step = %3d, lr = %.4f\n", s, one_cycle_lr(s, total, 0.3)))
}
cat("\nFor real use in R: torch.optim.lr_scheduler.OneCycleLR via reticulate,\n")
cat("or the fastai R port (fastai/rstudio wrapper).\n")
