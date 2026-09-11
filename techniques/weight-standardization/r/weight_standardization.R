# Weight Standardization (Reference Sec 47.172).
#
# Qiao et al 2019. Standardises weight rows (output channels)
# instead of activations; complements Group Normalization at
# micro-batch sizes.

weight_standardize <- function(W, eps = 1e-5) {
    mu <- rowMeans(W)
    sigma <- apply(W, 1, sd)
    sweep(sweep(W, 1, mu, "-"), 1, sigma + eps, "/")
}

set.seed(0)
W <- matrix(rnorm(64, sd = 3), 4, 16)
W_ws <- weight_standardize(W)
cat("=== Weight Standardisation (Qiao et al 2019) ===\n")
cat(sprintf("Row-wise mean of W    : %s\n", toString(round(rowMeans(W), 3))))
cat(sprintf("Row-wise mean of W_ws : %s\n", toString(round(rowMeans(W_ws), 3))))
cat(sprintf("Row-wise sd of W_ws   : %s\n",
             toString(round(apply(W_ws, 1, sd), 3))))
cat("\nFor deep-net use: torch.nn.utils.weight_norm-inspired ops in torch-for-R,\n")
cat("or reticulate to timm's Std_Conv2d / kornia weight_std wrappers.\n")
