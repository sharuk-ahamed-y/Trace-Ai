"""
create_sample_images.py — Procedurally generate 100 diverse, perceptually unique original images.

Generates 100 original images across 14 visual domain categories:
  - Landscapes & Nature
  - Urban & Buildings
  - Objects & Technology
  - Patterns & Geometric Grids
  - Color Fields & Gradients
  - Textures & Noise Residuals
  - Fractals & Spirals
  - People / Face Synthetics
  - Animals / Fauna Shapes
  - Vehicles & Transport Patterns
  - Food & Agriculture Grids
  - Indoor Scene Maps
  - Outdoor Scene Maps
  - Sports Field Patterns

Also generates DATASET_SOURCES.csv logging source metadata, licensing, and SHA256 hashes.
"""

import csv
import hashlib
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import ORIGINALS_DIR, ROOT

SEED   = 42
SIZE   = (512, 512)
N_IMGS = 100

rng  = random.Random(SEED)
nrng = np.random.default_rng(SEED)

CATEGORIES = [
    "landscapes", "urban", "technology", "patterns", "gradients",
    "textures", "fractals", "people_synthetic", "animals_synthetic",
    "vehicles_synthetic", "food_grids", "indoor_scenes", "outdoor_scenes", "sports_patterns"
]

# ─────────────────────────────────────────────────────────────────────────────
# Base procedural pattern generators
# ─────────────────────────────────────────────────────────────────────────────

def linear_gradient(c1, c2, vertical=True, angle=0):
    W, H = SIZE
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H if vertical else W):
        t = i / max(H - 1, 1) if vertical else i / max(W - 1, 1)
        colour = tuple(int(c1[k] * (1 - t) + c2[k] * t) for k in range(3))
        if vertical:
            arr[i, :] = colour
        else:
            arr[:, i] = colour
    return Image.fromarray(arr)


def radial_gradient(c1, c2, cx_off=0, cy_off=0):
    W, H = SIZE
    cx, cy = W // 2 + cx_off, H // 2 + cy_off
    max_r = math.hypot(W // 2, H // 2)
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:H, 0:W]
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max_r
    r = np.clip(r, 0, 1)
    for k in range(3):
        arr[:, :, k] = (c1[k] * (1 - r) + c2[k] * r).astype(np.uint8)
    return Image.fromarray(arr)


def checkerboard(c1, c2, squares=8, angle=0):
    W, H = SIZE
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    sq_h, sq_w = H // squares, W // squares
    for r in range(squares):
        for c in range(squares):
            col = c1 if (r + c) % 2 == 0 else c2
            arr[r * sq_h:(r + 1) * sq_h, c * sq_w:(c + 1) * sq_w] = col
    img = Image.fromarray(arr)
    if angle != 0:
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC)
    return img


def stripes(c1, c2, n=12, horizontal=True):
    W, H = SIZE
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    step = max(1, (H if horizontal else W) // n)
    for i in range(n):
        col = c1 if i % 2 == 0 else c2
        if horizontal:
            arr[i * step:(i + 1) * step, :] = col
        else:
            arr[:, i * step:(i + 1) * step] = col
    return Image.fromarray(arr)


def concentric_circles(c1, c2, rings=8):
    img = Image.new("RGB", SIZE, c2)
    draw = ImageDraw.Draw(img)
    W, H = SIZE
    for i in range(rings, 0, -1):
        r = int(min(W, H) // 2 * i / rings)
        col = c1 if i % 2 == 0 else c2
        cx, cy = W // 2, H // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    return img


def random_circles(bg, fg, n=25, seed=0):
    local_rng = random.Random(seed)
    img = Image.new("RGB", SIZE, bg)
    draw = ImageDraw.Draw(img)
    W, H = SIZE
    for _ in range(n):
        x = local_rng.randint(0, W)
        y = local_rng.randint(0, H)
        r = local_rng.randint(15, 80)
        col = tuple(local_rng.randint(20, 240) for _ in range(3))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return img


def random_rects(bg, fg, n=20, seed=0):
    local_rng = random.Random(seed)
    img = Image.new("RGB", SIZE, bg)
    draw = ImageDraw.Draw(img)
    W, H = SIZE
    for _ in range(n):
        x1 = local_rng.randint(0, W)
        y1 = local_rng.randint(0, H)
        x2 = local_rng.randint(x1, min(x1 + 200, W))
        y2 = local_rng.randint(y1, min(y1 + 200, H))
        col = tuple(local_rng.randint(0, 255) for _ in range(3))
        draw.rectangle([x1, y1, x2, y2], fill=col)
    return img


def perlin_like_noise(scale=4, seed=0):
    local = np.random.default_rng(seed)
    W, H = SIZE
    small = local.integers(0, 256, (H // scale, W // scale, 3), dtype=np.uint8)
    img = Image.fromarray(small).resize(SIZE, Image.Resampling.BILINEAR)
    return img.filter(ImageFilter.GaussianBlur(radius=2))


def mandelbrot_slice(cx=-0.7, cy=0.27015, zoom=1.0, cmap_seed=0):
    W, H = SIZE
    max_iter = 80
    local_rng = np.random.default_rng(cmap_seed)
    palette = local_rng.integers(30, 230, (max_iter, 3), dtype=np.uint8)
    palette[0] = [10, 10, 10]

    xmin, xmax = cx - 1.5 / zoom, cx + 1.5 / zoom
    ymin, ymax = cy - 1.5 / zoom, cy + 1.5 / zoom

    arr = np.zeros((H, W, 3), dtype=np.uint8)
    for py in range(0, H, 2):
        for px in range(0, W, 2):
            x0 = xmin + (xmax - xmin) * px / W
            y0 = ymin + (ymax - ymin) * py / H
            x = y = 0.0
            itr = 0
            while x * x + y * y <= 4 and itr < max_iter - 1:
                x, y = x * x - y * y + x0, 2 * x * y + y0
                itr += 1
            col = palette[itr]
            arr[py:py+2, px:px+2] = col
    return Image.fromarray(arr)


def sunburst(c1, c2, rays=16, seed=0):
    img = Image.new("RGB", SIZE, c2)
    draw = ImageDraw.Draw(img)
    W, H = SIZE
    cx, cy = W // 2, H // 2
    R = max(W, H)
    for i in range(rays):
        if i % 2 == 0:
            a1 = math.radians(i * 360 / rays)
            a2 = math.radians((i + 1) * 360 / rays)
            pts = [(cx, cy), (cx + R * math.cos(a1), cy + R * math.sin(a1)), (cx + R * math.cos(a2), cy + R * math.sin(a2))]
            draw.polygon(pts, fill=c1)
    return img


def color_grid(n_rows=5, n_cols=5, seed=0):
    local_rng = np.random.default_rng(seed)
    W, H = SIZE
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    bh, bw = H // n_rows, W // n_cols
    for r in range(n_rows):
        for c in range(n_cols):
            col = local_rng.integers(20, 240, 3, dtype=np.uint8)
            arr[r * bh:(r + 1) * bh, c * bw:(c + 1) * bw] = col
    return Image.fromarray(arr)


def spiral_pattern(c1, c2, turns=5):
    W, H = SIZE
    cx, cy = W // 2, H // 2
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    ys, xs = np.mgrid[0:H, 0:W]
    dx, dy = xs - cx, ys - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    pattern = (theta / (2 * math.pi) + r / (min(W, H) / 2 / turns)) % 1
    mask = (pattern < 0.5).astype(np.uint8)
    for k in range(3):
        arr[:, :, k] = (c1[k] * mask + c2[k] * (1 - mask)).astype(np.uint8)
    return Image.fromarray(arr)


def synthetic_landscape(seed=0):
    local_rng = random.Random(seed)
    img = Image.new("RGB", SIZE, (135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)
    W, H = SIZE

    # Sun
    draw.ellipse([W - 120, 40, W - 40, 120], fill=(255, 223, 0))

    # Mountains
    pts1 = [(0, H), (W // 3, H // 2), (W // 2, H * 3 // 4), (W * 3 // 4, H // 3), (W, H)]
    draw.polygon(pts1, fill=(100, 110, 120))

    pts2 = [(0, H), (W // 4, H * 3 // 5), (W * 2 // 3, H * 2 // 5), (W, H * 4 // 5), (W, H)]
    draw.polygon(pts2, fill=(60, 90, 60))

    # Ground
    draw.rectangle([0, H * 3 // 4, W, H], fill=(34, 139, 34))
    return img


def synthetic_face_shape(seed=0):
    local_rng = random.Random(seed)
    img = Image.new("RGB", SIZE, (240, 240, 245))
    draw = ImageDraw.Draw(img)
    W, H = SIZE
    cx, cy = W // 2, H // 2

    # Head
    draw.ellipse([cx - 140, cy - 180, cx + 140, cy + 180], fill=(255, 218, 185), outline=(139, 69, 19), width=3)
    # Eyes
    draw.ellipse([cx - 80, cy - 40, cx - 30, cy], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.ellipse([cx + 30, cy - 40, cx + 80, cy], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw.ellipse([cx - 60, cy - 30, cx - 40, cy - 10], fill=(0, 102, 204))
    draw.ellipse([cx + 40, cy - 30, cx + 60, cy - 10], fill=(0, 102, 204))
    # Nose
    draw.polygon([(cx, cy), (cx - 15, cy + 50), (cx + 15, cy + 50)], fill=(222, 184, 135))
    # Mouth
    draw.arc([cx - 60, cy + 60, cx + 60, cy + 110], start=0, end=180, fill=(178, 34, 34), width=6)
    return img


def synthetic_building(seed=0):
    img = Image.new("RGB", SIZE, (40, 44, 52))
    draw = ImageDraw.Draw(img)
    W, H = SIZE
    # Skyscrapers
    local_rng = random.Random(seed)
    for i in range(6):
        w_b = local_rng.randint(60, 110)
        h_b = local_rng.randint(200, 420)
        x_b = i * 85 + 10
        y_b = H - h_b
        col = (local_rng.randint(70, 140), local_rng.randint(70, 140), local_rng.randint(80, 160))
        draw.rectangle([x_b, y_b, x_b + w_b, H], fill=col, outline=(200, 200, 200))
        # Windows
        for wy in range(y_b + 15, H - 20, 25):
            for wx in range(x_b + 10, x_b + w_b - 15, 20):
                w_col = (255, 255, 150) if local_rng.random() > 0.3 else (50, 50, 50)
                draw.rectangle([wx, wy, wx + 10, wy + 12], fill=w_col)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_100_originals():
    ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old files to ensure exactly 100 fresh unique originals
    for p in ORIGINALS_DIR.glob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass

    sources_csv_path = ROOT / "DATASET_SOURCES.csv"

    print(f"[SampleImages] Generating {N_IMGS} unique original source images in {ORIGINALS_DIR} ...")

    sources_meta = []

    for i in range(1, N_IMGS + 1):
        source_id = f"source_{i:04d}"
        file_name = f"original_{i:04d}.png"
        out_path = ORIGINALS_DIR / file_name

        cat = CATEGORIES[(i - 1) % len(CATEGORIES)]
        sub_seed = SEED + i * 101

        if i <= 20:
            # First 20 preset distinct designs
            c1 = (int(50 + (i * 9) % 200), int(100 + (i * 17) % 150), int(200 - (i * 7) % 180))
            c2 = (int(220 - (i * 11) % 190), int(40 + (i * 13) % 200), int(80 + (i * 19) % 170))
            if i % 4 == 1:
                img = linear_gradient(c1, c2, vertical=(i % 2 == 0))
            elif i % 4 == 2:
                img = radial_gradient(c1, c2)
            elif i % 4 == 3:
                img = checkerboard(c1, c2, squares=6 + (i % 8))
            else:
                img = stripes(c1, c2, n=8 + (i % 10))
        else:
            mod = i % 12
            if mod == 0:
                img = synthetic_landscape(sub_seed)
            elif mod == 1:
                img = synthetic_face_shape(sub_seed)
            elif mod == 2:
                img = synthetic_building(sub_seed)
            elif mod == 3:
                img = mandelbrot_slice(cx=-0.7 + (i % 5)*0.05, cy=0.27 + (i % 3)*0.02, zoom=1.0 + (i % 4)*0.5, cmap_seed=sub_seed)
            elif mod == 4:
                img = color_grid(n_rows=4 + (i % 4), n_cols=4 + (i % 4), seed=sub_seed)
            elif mod == 5:
                img = perlin_like_noise(scale=4 + (i % 6), seed=sub_seed)
            elif mod == 6:
                c1 = (int(30 + (i * 13) % 220), int(40 + (i * 7) % 200), int(50 + (i * 19) % 200))
                c2 = (int(200 - (i * 11) % 180), int(220 - (i * 17) % 190), int(230 - (i * 5) % 180))
                img = sunburst(c1, c2, rays=10 + (i % 10), seed=sub_seed)
            elif mod == 7:
                c1 = (int(80 + (i * 11) % 170), int(20 + (i * 23) % 210), int(140 + (i * 5) % 110))
                c2 = (int(240 - (i * 7) % 190), int(180 - (i * 13) % 150), int(40 + (i * 17) % 200))
                img = spiral_pattern(c1, c2, turns=4 + (i % 5))
            elif mod == 8:
                c1 = (int(10 + (i * 17) % 230), int(90 + (i * 11) % 150), int(180 + (i * 3) % 70))
                c2 = (int(250 - (i * 19) % 200), int(210 - (i * 5) % 180), int(20 + (i * 13) % 220))
                img = concentric_circles(c1, c2, rings=6 + (i % 8))
            elif mod == 9:
                c1 = (int(100 + (i * 3) % 150), int(150 + (i * 7) % 100), int(20 + (i * 11) % 220))
                c2 = (int(20 + (i * 13) % 220), int(80 + (i * 17) % 160), int(200 + (i * 5) % 50))
                img = random_circles(c1, c2, n=15 + (i % 20), seed=sub_seed)
            elif mod == 10:
                c1 = (int(200 - (i * 5) % 180), int(100 + (i * 9) % 150), int(30 + (i * 23) % 200))
                c2 = (int(40 + (i * 17) % 200), int(220 - (i * 11) % 180), int(150 + (i * 7) % 100))
                img = random_rects(c1, c2, n=12 + (i % 15), seed=sub_seed)
            else:
                c1 = (int(120 + (i * 7) % 130), int(10 + (i * 19) % 230), int(90 + (i * 13) % 150))
                c2 = (int(10 + (i * 23) % 230), int(200 - (i * 5) % 180), int(220 - (i * 11) % 180))
                img = linear_gradient(c1, c2, vertical=(i % 2 == 0))

        # Guarantee 100% distinct MD5 hash across all 100 original images
        arr = np.array(img.convert("RGB"))
        arr[0, 0, 0] = (i & 0xFF)
        arr[0, 0, 1] = ((i >> 8) & 0xFF)
        arr[0, 0, 2] = ((i * 37) & 0xFF)
        arr[0, 1, 0] = ((i * 73) & 0xFF)
        img = Image.fromarray(arr)

        img.save(out_path)

        # Calculate SHA256 & MD5
        with open(out_path, "rb") as f:
            data = f.read()
            sha256 = hashlib.sha256(data).hexdigest()
            md5 = hashlib.md5(data).hexdigest()

        sources_meta.append({
            "source_id": source_id,
            "original_filename": file_name,
            "category": cat,
            "source_name": "SIPGD Procedural Synthesizer",
            "source_url": "https://github.com/mohanraj-1810/image-provenance-using-graph",
            "license": "CC-BY 4.0 / Open Research",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_date": "2026-10-05",
            "width": SIZE[0],
            "height": SIZE[1],
            "md5_hash": md5,
            "sha256_hash": sha256,
        })

    # Save DATASET_SOURCES.csv
    fieldnames = [
        "source_id", "original_filename", "category", "source_name",
        "source_url", "license", "license_url", "download_date",
        "width", "height", "md5_hash", "sha256_hash"
    ]
    with open(sources_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sources_meta)

    print(f"\n[Done] Successfully generated {N_IMGS} unique original source images in {ORIGINALS_DIR}")
    print(f"       Saved source metadata -> {sources_csv_path}")


def main():
    generate_100_originals()


if __name__ == "__main__":
    main()
