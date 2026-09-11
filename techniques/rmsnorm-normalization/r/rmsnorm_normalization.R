# RMSNorm (Reference Sec 47.171).
#
# Zhang & Sennrich 2019. Drops the mean subtraction from LayerNorm;
# 7-64% faster, same quality on Transformers.

rms_norm <- function(x, gamma, eps = 1e-6) {
    rms <- sqrt(mean(x^2) + eps)
    x / rms * gamma
}

layer_norm <- function(x, gamma, beta, eps = 1e-6) {
    (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
}

set.seed(0)
x <- rnorm(512, mean = 0.5, sd = 2.0)
gamma <- rep(1, 512); beta <- rep(0, 512)

y_ln <- layer_norm(x, gamma, beta)
y_rms <- rms_norm(x, gamma)

cat("=== RMSNorm vs LayerNorm (Zhang-Sennrich 2019) ===\n")
cat(sprintf("Input:  mean=%.3f, sd=%.3f\n", mean(x), sd(x)))
cat(sprintf("LayerNorm: mean=%.3f, sd=%.3f\n", mean(y_ln), sd(y_ln)))
cat(sprintf("RMSNorm:   mean=%.3f, sd=%.3f\n", mean(y_rms), sd(y_rms)))
cat("\nR ports of RMSNorm are trivial; use it inside a torch-in-R model,\n")
cat("or bridge to torch.nn.RMSNorm / transformers LlamaRMSNorm via reticulate.\n")
