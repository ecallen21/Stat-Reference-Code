# Response-surface methodology (Reference §16.11)
# R via rsm::rsm and rsm::steepest.
# Run with:  Rscript response_surface.R

if (sys.nframe() == 0) {
  set.seed(0)
  if (requireNamespace("rsm", quietly = TRUE)) {
    # CCD in 2 factors
    ccd <- rsm::ccd(2, n0 = 3, alpha = "rotatable", coding = list(x1 ~ z1, x2 ~ z2),
                    randomize = FALSE)
    ccd$y <- 5 + 2 * ccd$x1 + 3 * ccd$x2 - ccd$x1^2 - 2 * ccd$x2^2 + rnorm(nrow(ccd), 0, 0.1)
    cat("=== rsm::rsm second-order fit ===\n")
    fit <- rsm::rsm(y ~ SO(x1, x2), data = ccd)
    print(summary(fit))
    cat("\n=== steepest ascent path ===\n")
    print(rsm::steepest(fit, dist = seq(0, 5, by = 0.5)))
  }
}
