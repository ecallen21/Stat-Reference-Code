# Convolutional Neural Network (Reference §27.2)

Neural network built from **2-D convolution** layers rather than dense
matrix products. Exploits translation invariance and local structure in
images (also 1-D convolutions for audio / genomics; 3-D for MRI / video).

## Convolution layer

For each output channel `c` and spatial position `(i, j)`:

```
Y[c, i, j] = Σ_{c', p, q} X[c', i+p, j+q] · K[c, c', p, q] + b[c]
```

- `K`: `(C_out, C_in, kH, kW)` learnable filters.
- Output size: `(H − kH + 1) × (W − kW + 1)` with stride 1 and no padding.
- **Padding** `same` keeps output size = input; `valid` shrinks.
- **Stride** > 1 downsamples spatially.
- **Dilation** inserts holes between kernel elements — increases receptive field cheaply.

## Typical block

```
Conv → BatchNorm → ReLU → (MaxPool)     ×  N
Flatten → Linear → Softmax
```

Standard architectures: LeNet-5, AlexNet, VGG, ResNet, EfficientNet.
Modern practice: ResNet-50 as baseline; ViT (vision transformer) for
strong performance with enough compute + data.

## When to use

- **Images, audio spectrograms, video, MRI, genomics tracks** — any signal with strong local structure and translation invariance.
- **Feature extractor** for transfer learning (see `transfer-learning` note in `deep-mlp-backprop`).
- **1-D convolution** for time-series with local patterns; competitive with LSTMs at a fraction of the parameters.

## Files

- `python/convolutional_nn.py` — from-scratch numpy Conv2D forward + backward (explicit `einsum` accumulation) + 2×2 max-pool + linear head + softmax-CE. Demo (240 8×8 binary images of vertical vs horizontal bars): train accuracy 0.87 in 120 epochs with a single 3×3 conv layer of 4 output channels and a linear classifier. Cross-check with a torch `nn.Conv2d + MaxPool2d + Linear` module.
- `r/convolutional_nn.R` — `torch::nn_conv2d`, `keras3::layer_conv_2d`.

## Assumptions & caveats

- **Naive convolution is slow** — `O(N · C_out · C_in · H · W · kH · kW)` per layer. Real code uses **im2col + BLAS**, **FFT** for large kernels, or **Winograd** for small kernels. Frameworks call cuDNN.
- **Receptive field grows linearly with depth** — needed for global context; alternatives include dilated / stride > 1 convolutions.
- **Padding + stride interplay** — mind the output-size formula `⌊(H + 2P − kH) / S⌋ + 1`.
- **Weight sharing** is what buys translation invariance — the same filter slides across spatial positions.
- **Pooling** was a design decision; modern networks often use strided convolutions instead.
- **Data augmentation** (crops, flips, rotations, mixup) matters more than architecture past a certain baseline.

## Related in this repo

- `deep-mlp-backprop`, `dropout-batchnorm`, `adam-optimizer` — training add-ons.
- `attention-mechanism`, `transformer-encoder` — ViT-style alternative.
- `autoencoder`, `variational-autoencoder`, `gan-training` — often use convolutional encoder / decoder.
- `feature-importance`, `shap-values` — model-explanation companions.

## Run

```
python techniques/convolutional-nn/python/convolutional_nn.py
Rscript techniques/convolutional-nn/r/convolutional_nn.R
```

**Refs:** LeCun, Y. et al. "Gradient-based learning applied to document recognition." *Proc. IEEE* 86(11), 2278–2324, 1998; Krizhevsky, A., Sutskever, I. & Hinton, G.E. "ImageNet classification with deep convolutional neural networks." *NeurIPS*, 2012; He, K. et al. "Deep residual learning for image recognition." *CVPR*, 2016.

---

## Author

Elisabeth F. Callen, Ph.D., PStat®
Biostatistician and applied health data researcher

[LinkedIn](https://www.linkedin.com/in/your-profile) · [ORCID](https://orcid.org/your-id) · elisabeth.f.callen@gmail.com

## Acknowledgments

**AI tooling.** This codebase was developed with the support of AI coding assistants (Claude Code). Methodology, statistical approach, validation logic, and interpretation of results are my own. AI tooling was used to accelerate code drafting, refactor for readability, and assist with documentation. All code was reviewed, tested, and validated against expected outputs before committing.

No protected health information was ever provided to AI coding assistants. All development and testing was conducted against synthetic data.

## License

[MIT](../../LICENSE)
