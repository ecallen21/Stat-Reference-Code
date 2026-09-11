"""SAM - Segment Anything Model (Reference Sec 47.229).

Kirillov, Mintun, Ravi, Mao, Rolland, Gustafson, Xiao, Whitehead,
Berg, Lo, Dollar & Girshick 2023 'Segment Anything', ICCV. Three
components:

    1. IMAGE ENCODER (ViT-H) computes an image embedding.
    2. PROMPT ENCODER embeds point / box / mask / text prompts.
    3. MASK DECODER (light Transformer) predicts N masks + confidences
       from (image_embed, prompt_embed).

Trained on 1.1 B masks in SA-1B. Zero-shot segments almost anything
with a click, box, or text prompt.
"""
from __future__ import annotations    # stdlib

import numpy as np    # numerical arrays


def image_encoder_toy(img):
    """Toy encoder: identity + downsample."""
    return img[::2, ::2]                                          # 2x downsample


def prompt_encoder_point(point, image_shape):
    """Encode a click as a Gaussian bump at the point (scaled to embedding grid)."""
    H, W = image_shape[0] // 2, image_shape[1] // 2
    py, px = point[0] // 2, point[1] // 2
    grid = np.zeros((H, W))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    grid = np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / 8)
    return grid


def prompt_encoder_box(box, image_shape):
    """Encode a bounding box as a binary mask on the embedding grid."""
    H, W = image_shape[0] // 2, image_shape[1] // 2
    y0, x0, y1, x1 = box[0] // 2, box[1] // 2, box[2] // 2, box[3] // 2
    grid = np.zeros((H, W))
    grid[y0:y1, x0:x1] = 1
    return grid


def mask_decoder_toy(img_embed, prompt_embed):
    """Toy decoder: use prompt as attention prior over image similarity."""
    # Compute local similarity of image_embed to a 'foreground' template
    # from the prompted region (weighted mean)
    weights = prompt_embed / (prompt_embed.sum() + 1e-8)
    template = (img_embed * weights).sum()
    similarity = 1 - np.abs(img_embed - template) / max(abs(template), 1e-6)
    similarity = np.clip(similarity, 0, 1)
    # Refine: multiply by prompt (foreground bias)
    mask = similarity * (0.5 + 0.5 * prompt_embed)
    return (mask > 0.5).astype(int)


def upsample_mask(mask, image_shape):
    """Nearest-neighbour 2x upsample."""
    return np.repeat(np.repeat(mask, 2, axis=0), 2, axis=1)[:image_shape[0], :image_shape[1]]


if __name__ == "__main__":
    print("=== SAM - Segment Anything (Kirillov et al 2023 ICCV) ===\n")
    rng = np.random.default_rng(0)

    # Synthetic image: 3 blobs of different intensities
    H = W = 48
    img = np.zeros((H, W))
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    for (cy, cx, val, r) in [(12, 12, 1.0, 6), (30, 30, 0.6, 5), (10, 35, 0.3, 4)]:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = val

    img_embed = image_encoder_toy(img)
    print(f"  Image: {H}x{W}, encoded to {img_embed.shape}")

    # Point prompt on each blob
    for (name, click) in [("bright blob", (12, 12)), ("mid blob", (30, 30)),
                            ("dim blob", (10, 35))]:
        prompt = prompt_encoder_point(click, img.shape)
        mask = mask_decoder_toy(img_embed, prompt)
        mask_full = upsample_mask(mask, img.shape)
        print(f"  Point click on {name} at {click}: mask covers "
              f"{int(mask_full.sum())} px out of {H * W}")

    # Box prompt
    box = (5, 5, 20, 20)                                          # (y0, x0, y1, x1)
    prompt = prompt_encoder_box(box, img.shape)
    mask = mask_decoder_toy(img_embed, prompt)
    print(f"  Box prompt {box}: mask covers {int(upsample_mask(mask, img.shape).sum())} px")

    print("\n  Real SAM handles points / boxes / masks / text prompts; the ViT-H image")
    print("  encoder runs once per image, mask decoder is a few ms per prompt.")

    print("\n--- library cross-check (facebookresearch/segment-anything; SAM2; MobileSAM) ---")
