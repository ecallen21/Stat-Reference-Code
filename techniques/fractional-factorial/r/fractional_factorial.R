# Fractional factorial designs (Reference §16.4)
# R via FrF2 (Groemping) or DoE.base.
# Run with:  Rscript fractional_factorial.R

if (sys.nframe() == 0) {
  if (requireNamespace("FrF2", quietly = TRUE)) {
    cat("=== FrF2::FrF2 for 2^(5-1) Res V ===\n")
    print(FrF2::FrF2(nruns = 16, nfactors = 5, generators = "ABCD"))
    cat("\n=== 2^(7-2) Res IV ===\n")
    print(FrF2::FrF2(nruns = 32, nfactors = 7, generators = c("ABC", "BCD")))
  }
}
