"""
transforms.py — Controlled image transformations for provenance dataset generation.

Supports 12 realistic transformation types with randomized parameter sampling:
  1. crop
  2. resize
  3. jpeg
  4. blur
  5. brightness
  6. contrast
  7. sharpen
  8. rotation
  9. color
 10. noise
 11. perspective
 12. quality_degradation

Each transformation accepts a PIL.Image and a parameters dictionary, returning a new PIL.Image.
"""

import io
import math
import random
from typing import Tuple, Dict, Any, List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps

# ── Transformation Vocabulary ──────────────────────────────────────────────────
TRANSFORMATION_NAMES = [
    "crop",
    "resize",
    "jpeg",
    "blur",
    "brightness",
    "contrast",
    "sharpen",
    "rotation",
    "color",
    "noise",
    "perspective",
    "quality_degradation",
]

N_TRANSFORMS = len(TRANSFORMATION_NAMES)

# ── Helper functions ──────────────────────────────────────────────────────────

def _wh(img: Image.Image) -> Tuple[int, int]:
    return img.size  # (width, height)


# ── 1. Crop ────────────────────────────────────────────────────────────────────

def apply_crop(img: Image.Image, params: dict) -> Image.Image:
    """
    Crop a region from the image.
    params:
      mode: 'random', 'center', or 'rectangular'
      frac_w: fraction of width to crop (0.05 - 0.35)
      frac_h: fraction of height to crop (0.05 - 0.35)
      resize_back: bool (whether to resize back to original dimensions)
    """
    w, h = _wh(img)
    mode = params.get("mode", "random")
    frac_w = params.get("frac_w", 0.15)
    frac_h = params.get("frac_h", 0.15)
    resize_back = params.get("resize_back", True)

    crop_w = int(w * (1.0 - frac_w))
    crop_h = int(h * (1.0 - frac_h))
    crop_w = max(10, crop_w)
    crop_h = max(10, crop_h)

    if mode == "center":
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
    elif mode == "rectangular":
        left = int(params.get("off_x", 0.0) * (w - crop_w))
        top = int(params.get("off_y", 0.0) * (h - crop_h))
    else:  # random
        left = int(params.get("off_x", 0.5) * (w - crop_w))
        top = int(params.get("off_y", 0.5) * (h - crop_h))

    left = max(0, min(w - crop_w, left))
    top = max(0, min(h - crop_h, top))
    box = (left, top, left + crop_w, top + crop_h)

    cropped = img.crop(box)
    if resize_back:
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    return cropped


def sample_crop_params(rng: random.Random, cfg: dict = None) -> dict:
    mode = rng.choice(["random", "center", "rectangular"])
    frac_w = round(rng.uniform(0.08, 0.30), 4)
    frac_h = round(rng.uniform(0.08, 0.30), 4)
    off_x = round(rng.uniform(0.0, 1.0), 4)
    off_y = round(rng.uniform(0.0, 1.0), 4)
    resize_back = rng.choice([True, False])
    return {
        "mode": mode,
        "frac_w": frac_w,
        "frac_h": frac_h,
        "off_x": off_x,
        "off_y": off_y,
        "resize_back": resize_back,
    }


# ── 2. Resize ──────────────────────────────────────────────────────────────────

def apply_resize(img: Image.Image, params: dict) -> Image.Image:
    """
    Scale image dimensions by scale factor using specified interpolation.
    """
    w, h = _wh(img)
    scale = params.get("scale", 0.7)
    interp_str = params.get("interpolation", "LANCZOS")
    resize_back = params.get("resize_back", True)

    interp_map = {
        "LANCZOS": Image.Resampling.LANCZOS,
        "BICUBIC": Image.Resampling.BICUBIC,
        "BILINEAR": Image.Resampling.BILINEAR,
        "NEAREST": Image.Resampling.NEAREST,
    }
    interp = interp_map.get(interp_str, Image.Resampling.LANCZOS)

    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))

    resized = img.resize((new_w, new_h), interp)
    if resize_back:
        return resized.resize((w, h), interp)
    return resized


def sample_resize_params(rng: random.Random, cfg: dict = None) -> dict:
    scale = round(rng.uniform(0.35, 1.40), 4)
    if 0.95 <= scale <= 1.05:
        scale = 0.75
    interp = rng.choice(["LANCZOS", "BICUBIC", "BILINEAR", "NEAREST"])
    resize_back = rng.choice([True, False])
    return {"scale": scale, "interpolation": interp, "resize_back": resize_back}


# ── 3. JPEG Compression ───────────────────────────────────────────────────────

def apply_jpeg(img: Image.Image, params: dict) -> Image.Image:
    quality = int(params.get("quality", 65))
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def sample_jpeg_params(rng: random.Random, cfg: dict = None) -> dict:
    quality = rng.randint(30, 95)
    return {"quality": quality}


# ── 4. Gaussian Blur ──────────────────────────────────────────────────────────

def apply_blur(img: Image.Image, params: dict) -> Image.Image:
    radius = float(params.get("radius", 2.0))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def sample_blur_params(rng: random.Random, cfg: dict = None) -> dict:
    radius = round(rng.uniform(0.8, 4.5), 2)
    return {"radius": radius}


# ── 5. Brightness Adjustment ──────────────────────────────────────────────────

def apply_brightness(img: Image.Image, params: dict) -> Image.Image:
    factor = float(params.get("factor", 1.2))
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def sample_brightness_params(rng: random.Random, cfg: dict = None) -> dict:
    factor = round(rng.uniform(0.55, 1.55), 4)
    return {"factor": factor}


# ── 6. Contrast Adjustment ────────────────────────────────────────────────────

def apply_contrast(img: Image.Image, params: dict) -> Image.Image:
    factor = float(params.get("factor", 1.2))
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def sample_contrast_params(rng: random.Random, cfg: dict = None) -> dict:
    factor = round(rng.uniform(0.55, 1.60), 4)
    return {"factor": factor}


# ── 7. Sharpening ─────────────────────────────────────────────────────────────

def apply_sharpen(img: Image.Image, params: dict) -> Image.Image:
    factor = float(params.get("factor", 1.8))
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(factor)


def sample_sharpen_params(rng: random.Random, cfg: dict = None) -> dict:
    factor = round(rng.uniform(1.4, 3.2), 4)
    return {"factor": factor}


# ── 8. Rotation ───────────────────────────────────────────────────────────────

def apply_rotation(img: Image.Image, params: dict) -> Image.Image:
    angle = float(params.get("angle", 15.0))
    expand = bool(params.get("expand", False))
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=expand)


def sample_rotation_params(rng: random.Random, cfg: dict = None) -> dict:
    angle = round(rng.uniform(-25.0, 25.0), 2)
    if abs(angle) < 1.0:
        angle = 12.5
    expand = rng.choice([False, True])
    return {"angle": angle, "expand": expand}


# ── 9. Color / Saturation Adjustment ──────────────────────────────────────────

def apply_color(img: Image.Image, params: dict) -> Image.Image:
    factor = float(params.get("factor", 1.3))
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def sample_color_params(rng: random.Random, cfg: dict = None) -> dict:
    factor = round(rng.uniform(0.40, 1.80), 4)
    return {"factor": factor}


# ── 10. Gaussian Noise ────────────────────────────────────────────────────────

def apply_noise(img: Image.Image, params: dict) -> Image.Image:
    sigma = float(params.get("sigma", 15.0))
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def sample_noise_params(rng: random.Random, cfg: dict = None) -> dict:
    sigma = round(rng.uniform(6.0, 28.0), 2)
    return {"sigma": sigma}


# ── 11. Perspective Transformation ───────────────────────────────────────────

def apply_perspective(img: Image.Image, params: dict) -> Image.Image:
    w, h = _wh(img)
    distortion = float(params.get("distortion", 0.1))
    
    # Calculate perturbed corner points
    dx1 = int(w * distortion * params.get("p1", 0.5))
    dy1 = int(h * distortion * params.get("p2", 0.5))
    dx2 = int(w * distortion * params.get("p3", -0.5))
    dy2 = int(h * distortion * params.get("p4", 0.5))
    
    coeffs = (
        1 + distortion * params.get("a", 0.05),
        distortion * params.get("b", 0.02),
        dx1,
        distortion * params.get("c", -0.02),
        1 + distortion * params.get("d", 0.05),
        dy1,
        distortion * 0.0001,
        distortion * 0.0001,
    )
    return img.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def sample_perspective_params(rng: random.Random, cfg: dict = None) -> dict:
    return {
        "distortion": round(rng.uniform(0.05, 0.18), 4),
        "p1": round(rng.uniform(-1, 1), 2),
        "p2": round(rng.uniform(-1, 1), 2),
        "p3": round(rng.uniform(-1, 1), 2),
        "p4": round(rng.uniform(-1, 1), 2),
        "a": round(rng.uniform(-0.1, 0.1), 4),
        "b": round(rng.uniform(-0.1, 0.1), 4),
        "c": round(rng.uniform(-0.1, 0.1), 4),
        "d": round(rng.uniform(-0.1, 0.1), 4),
    }


# ── 12. Quality Degradation ───────────────────────────────────────────────────

def apply_quality_degradation(img: Image.Image, params: dict) -> Image.Image:
    """
    Combined multi-degradation: downscaling + Gaussian noise + low JPEG compression.
    """
    downscale = float(params.get("downscale", 0.6))
    sigma = float(params.get("sigma", 10.0))
    jpeg_quality = int(params.get("jpeg_quality", 45))

    w, h = _wh(img)
    small_w = max(16, int(w * downscale))
    small_h = max(16, int(h * downscale))
    res = img.resize((small_w, small_h), Image.Resampling.BILINEAR)

    # Add noise
    arr = np.array(res.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr)

    # Re-upscale to original size
    upscaled = noisy_img.resize((w, h), Image.Resampling.BICUBIC)

    # Apply JPEG
    buf = io.BytesIO()
    upscaled.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    return Image.open(buf).copy()


def sample_quality_degradation_params(rng: random.Random, cfg: dict = None) -> dict:
    return {
        "downscale": round(rng.uniform(0.40, 0.75), 4),
        "sigma": round(rng.uniform(8.0, 20.0), 2),
        "jpeg_quality": rng.randint(30, 60),
    }


# ── Transformation Dispatcher Maps ──────────────────────────────────────────

APPLY_FN = {
    "crop": apply_crop,
    "resize": apply_resize,
    "jpeg": apply_jpeg,
    "blur": apply_blur,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "sharpen": apply_sharpen,
    "rotation": apply_rotation,
    "color": apply_color,
    "noise": apply_noise,
    "perspective": apply_perspective,
    "quality_degradation": apply_quality_degradation,
}

SAMPLE_FN = {
    "crop": sample_crop_params,
    "resize": sample_resize_params,
    "jpeg": sample_jpeg_params,
    "blur": sample_blur_params,
    "brightness": sample_brightness_params,
    "contrast": sample_contrast_params,
    "sharpen": sample_sharpen_params,
    "rotation": sample_rotation_params,
    "color": sample_color_params,
    "noise": sample_noise_params,
    "perspective": sample_perspective_params,
    "quality_degradation": sample_quality_degradation_params,
}


def apply_transform(img: Image.Image, name: str, params: dict) -> Image.Image:
    """Apply a named transformation with the given params dict."""
    if name not in APPLY_FN:
        raise ValueError(f"Unknown transformation: {name!r}")
    return APPLY_FN[name](img, params)


def sample_transform_params(name: str, rng: random.Random, cfg: dict = None) -> dict:
    """Draw random parameters for *name* using *rng*."""
    if name not in SAMPLE_FN:
        raise ValueError(f"Unknown transformation: {name!r}")
    return SAMPLE_FN[name](rng, cfg)
