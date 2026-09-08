# Butterworth Bandpass (Reference Sec 47.144).
#
# Butterworth 1930. Maximally flat digital IIR bandpass.
#
# Uses `signal::butter` + `signal::filtfilt` (zero-phase).

if (!requireNamespace("signal", quietly = TRUE)) install.packages("signal")
library(signal)                                       # signal-processing filters

set.seed(0)
fs <- 1000
tt <- seq(0, 2, by = 1 / fs)
x <- sin(2 * pi * 2 * tt) + sin(2 * pi * 50 * tt) + 0.7 * sin(2 * pi * 300 * tt) +
     0.1 * rnorm(length(tt))

bf <- butter(4, c(30, 70) / (fs / 2), type = "pass")
y <- filtfilt(bf, x)
# Estimate dominant freq via periodogram
per <- spec.pgram(y, plot = FALSE, taper = 0, fast = FALSE)
peak <- per$freq[which.max(per$spec)] * fs
cat("=== Butterworth 30-70 Hz bandpass (Butterworth 1930) ===\n")
cat(sprintf("  Peak frequency after bandpass = %.2f Hz (truth: 50)\n", peak))
