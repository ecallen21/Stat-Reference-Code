# Chambolle-Pock primal-dual (Chambolle-Pock 2011)
# R: `flare` (TV-related solvers with primal-dual internals),
#    `CVXR` (uses splitting), `imager::denoise_TV` (2D image).
# Python: `pyproximal.primaldual` (LOAD-BALANCE style),
#         `scikit-image.restoration.denoise_tv_chambolle`,
#         from-scratch
#
# library(imager)
# denoised <- denoise_tv_chambolle(noisy_image, weight = 0.5)
