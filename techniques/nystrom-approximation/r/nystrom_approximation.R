# Nystrom kernel approximation (Williams-Seeger 2001)
# R: `kernlab::kkmeans`, `bigKRLS` (large kernel ridge)
# Python: `sklearn.kernel_approximation.Nystroem`,
#         `sklearn.pipeline.make_pipeline(Nystroem, ...)`,
#         from-scratch
#
# library(kernlab)
# fit <- kkmeans(as.kernelMatrix(kernelMatrix(rbfdot(0.1), X)),
#                centers = 3)
