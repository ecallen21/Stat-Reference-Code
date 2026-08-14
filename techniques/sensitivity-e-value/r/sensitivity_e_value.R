# Sensitivity analysis: E-value + Rosenbaum bounds (Reference §15.14)
# R via EValue::evalue and sensitivitymv::senmv or rbounds::psens.
# Run with:  Rscript sensitivity_e_value.R

if (sys.nframe() == 0) {
  if (requireNamespace("EValue", quietly = TRUE)) {
    cat("=== EValue::evalue for RR = 2.0 (CI 1.4-2.8) ===\n")
    print(EValue::evalue(EValue::RR(2.0), lo = 1.4, hi = 2.8))
  } else {
    e_from_rr <- function(rr) { if (rr < 1) rr <- 1/rr; rr + sqrt(rr * (rr - 1)) }
    cat(sprintf("E-value(RR = 2.0)     = %.3f\n", e_from_rr(2.0)))
    cat(sprintf("E-value(CI lower 1.4) = %.3f\n", e_from_rr(1.4)))
  }
}
