# Adafactor (Shazeer-Stern 2018)
# R: no first-class R package; access via reticulate to
#    `transformers.optimization.Adafactor` or
#    `optax.adafactor`.
# Python: `transformers.optimization.Adafactor`,
#         `optax.adafactor`, `keras.optimizers.Adafactor`,
#         from-scratch
#
# reticulate::py_install(c("torch", "transformers"))
# adafactor <- reticulate::import("transformers")$Adafactor
