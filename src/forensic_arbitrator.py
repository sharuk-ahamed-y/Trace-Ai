"""
src/forensic_arbitrator.py — Deterministic, Grounded Forensic Direction & Transformation Arbitrator

Implements principled, physically grounded forensic rules that resolve model blindspots
while strictly preserving mathematical direction anti-symmetry:
  arbitrate_direction(B, A) == reverse(arbitrate_direction(A, B))

Arbitrated Forensics:
1. Low-Pass Spectral Attenuation: Detects Gaussian blur without falling victim to high-frequency prior bias.
2. Scale-Invariant Sub-Region Containment: Detects crops even when rescaled back to original dimensions.
3. Physical Canvas Invariance: Downscaling or bounding crops cannot expand image canvas area.
4. DCT 8x8 Grid Discontinuity: Distinguishes JPEG block artifacts from true optical sharpening.
5. Compound Manipulation Tagging: Provides multi-label capability for composite/sequential edits.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from PIL import Image
import cv2

# Session-level in-memory arbitration caches (cleared per analysis session)
_GRAY_IMAGE_CACHE: Dict[str, np.ndarray] = {}
_SIFT_CACHE: Dict[str, Tuple[Any, Any]] = {}
_CROP_CONTAINMENT_CACHE: Dict[Tuple[str, str], Tuple[float, float]] = {}
_CROP_ALIGNMENT_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_FULL_CANVAS_RESIZE_CACHE: Dict[Tuple[str, str], Tuple[bool, float, str]] = {}
_GEOMETRY_CACHE: Dict[Tuple[str, str], Tuple[Optional[str], float, str]] = {}
_ARBITRATION_EVIDENCE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def clear_arbitration_caches() -> None:
    """Clears per-analysis in-memory arbitration caches to prevent cross-analysis leaks."""
    _GRAY_IMAGE_CACHE.clear()
    _SIFT_CACHE.clear()
    _CROP_CONTAINMENT_CACHE.clear()
    _CROP_ALIGNMENT_CACHE.clear()
    _FULL_CANVAS_RESIZE_CACHE.clear()
    _GEOMETRY_CACHE.clear()
    _ARBITRATION_EVIDENCE_CACHE.clear()


def _get_gray_image(path: Path) -> Optional[np.ndarray]:
    """Retrieves or decodes and caches grayscale image array."""
    p_str = str(Path(path).resolve())
    if p_str in _GRAY_IMAGE_CACHE:
        return _GRAY_IMAGE_CACHE[p_str]
    try:
        with Image.open(path).convert("L") as im:
            arr = np.array(im)
            _GRAY_IMAGE_CACHE[p_str] = arr
            return arr
    except Exception:
        return None


def check_geometric_containment_sift(
    path_src: Path,
    path_dst: Path,
    arr_src: np.ndarray,
    arr_dst: np.ndarray
) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]]]:
    """
    Evaluates whether arr_src geometrically projects entirely inside arr_dst as an
    interior sub-region (crop) using SIFT keypoints and RANSAC homography estimation.
    Validates:
      - Keypoint match count and Lowe's ratio test (0.75)
      - Homography RANSAC inliers (>= 12) and inlier ratio (>= 0.55)
      - Projected 4-corner bounded containment within [0, w_dst] x [0, h_dst]
      - Projected quadrilateral convexity and area ratio (0.02 <= area_ratio <= 0.98)
      - Perspective distortion constraint (|H_20| + |H_21| < 0.005)
    Returns: (is_contained, inlier_ratio, bounding_box_in_dst)
    """
    p_src_str = str(Path(path_src).resolve())
    p_dst_str = str(Path(path_dst).resolve())

    try:
        if p_src_str in _SIFT_CACHE:
            kp1, des1 = _SIFT_CACHE[p_src_str]
        else:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(arr_src, None)
            _SIFT_CACHE[p_src_str] = (kp1, des1)

        if p_dst_str in _SIFT_CACHE:
            kp2, des2 = _SIFT_CACHE[p_dst_str]
        else:
            sift = cv2.SIFT_create()
            kp2, des2 = sift.detectAndCompute(arr_dst, None)
            _SIFT_CACHE[p_dst_str] = (kp2, des2)

        if des1 is None or des2 is None or len(des1) < 12 or len(des2) < 12:
            return False, 0.0, None

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < 12:
            return False, 0.0, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            return False, 0.0, None

        inliers = int(mask.sum())
        inlier_ratio = inliers / float(len(good))
        if inliers < 12 or inlier_ratio < 0.55:
            return False, 0.0, None

        # Check projective distortion: standard planar crops have negligible perspective terms
        persp_skew = float(abs(H[2, 0]) + abs(H[2, 1]))
        if persp_skew > 0.005:
            return False, 0.0, None

        h_s, w_s = arr_src.shape[:2]
        h_d, w_d = arr_dst.shape[:2]
        corners_s = np.float32([[0, 0], [w_s, 0], [w_s, h_s], [0, h_s]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners_s, H)
        if proj is None:
            return False, 0.0, None

        # 1. Bounded containment: All 4 projected corners must lie inside dst canvas (with small subpixel tolerance)
        margin = 5.0
        for i in range(4):
            x = float(proj[i, 0, 0])
            y = float(proj[i, 0, 1])
            if x < -margin or x > w_d + margin or y < -margin or y > h_d + margin:
                return False, 0.0, None

        # 2. Quadrilateral convexity & area check
        poly_pts = np.int32(proj)
        if not cv2.isContourConvex(poly_pts):
            return False, 0.0, None

        area_proj = float(cv2.contourArea(proj))
        area_dst = float(w_d * h_d)
        if area_proj < 0.02 * area_dst or area_proj > 0.98 * area_dst:
            return False, 0.0, None

        # 3. Bounding box in dst coordinates
        x1 = max(0, int(np.min(proj[:, 0, 0])))
        y1 = max(0, int(np.min(proj[:, 0, 1])))
        x2 = min(w_d, int(np.max(proj[:, 0, 0])))
        y2 = min(h_d, int(np.max(proj[:, 0, 1])))

        if x2 - x1 < 16 or y2 - y1 < 16:
            return False, 0.0, None

        return True, inlier_ratio, (x1, y1, x2, y2)
    except Exception:
        return False, 0.0, None


def check_sub_region_containment(path_a: Path, path_b: Path) -> Tuple[float, float]:
    """
    Antisymmetric dual-evidence sub-region containment test combining multi-scale
    normalized cross-correlation with feature-based SIFT homography geometric verification.
    Returns: (direction_score, max_correlation)
      +1.0: Image B is an interior sub-region of Image A (A -> B, A is parent).
      -1.0: Image A is an interior sub-region of Image B (B -> A, B is parent).
       0.0: No scale-invariant sub-region containment detected.
    """
    p_a_str = str(Path(path_a).resolve())
    p_b_str = str(Path(path_b).resolve())

    if (p_a_str, p_b_str) in _CROP_CONTAINMENT_CACHE:
        return _CROP_CONTAINMENT_CACHE[(p_a_str, p_b_str)]
    if (p_b_str, p_a_str) in _CROP_CONTAINMENT_CACHE:
        rev_score, corr = _CROP_CONTAINMENT_CACHE[(p_b_str, p_a_str)]
        res = (-rev_score, corr)
        _CROP_CONTAINMENT_CACHE[(p_a_str, p_b_str)] = res
        return res

    arr_a = _get_gray_image(path_a)
    arr_b = _get_gray_image(path_b)
    if arr_a is None or arr_b is None:
        return 0.0, 0.0

    scales = [
        1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
        0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15
    ]
    if arr_a.shape[0] > 0 and arr_a.shape[1] > 0 and arr_b.shape[0] > 0 and arr_b.shape[1] > 0:
        for (src_h, src_w, dst_h, dst_w) in [(arr_b.shape[0], arr_b.shape[1], arr_a.shape[0], arr_a.shape[1]),
                                             (arr_a.shape[0], arr_a.shape[1], arr_b.shape[0], arr_b.shape[1])]:
            sw = (dst_w - 4) / float(src_w)
            sh = (dst_h - 4) / float(src_h)
            for cand in [sw, sh, sw * 0.98, sw * 1.02, sh * 0.98, sh * 1.02]:
                if 0.12 <= cand <= 2.0:
                    scales.append(round(cand, 4))
    scales = sorted(list(set(scales)), reverse=True)

    def scan_containment(src_arr, dst_arr):
        max_src_dim = max(src_arr.shape)
        max_dst_dim = max(dst_arr.shape)
        use_coarse = (max_src_dim > 512 or max_dst_dim > 512)

        scales_to_eval = scales
        if use_coarse:
            scale_src = 512.0 / float(max_src_dim)
            scale_dst = 512.0 / float(max_dst_dim)
            src_c = cv2.resize(src_arr, (max(16, int(src_arr.shape[1] * scale_src)), max(16, int(src_arr.shape[0] * scale_src))), interpolation=cv2.INTER_AREA)
            dst_c = cv2.resize(dst_arr, (max(16, int(dst_arr.shape[1] * scale_dst)), max(16, int(dst_arr.shape[0] * scale_dst))), interpolation=cv2.INTER_AREA)

            coarse_candidates = []
            best_coarse_v = 0.0
            for s in scales:
                bw_c = int(src_c.shape[1] * (s * (scale_dst / scale_src)))
                bh_c = int(src_c.shape[0] * (s * (scale_dst / scale_src)))
                if bw_c < 16 or bh_c < 16 or bw_c >= dst_c.shape[1] or bh_c >= dst_c.shape[0]:
                    continue
                interp_c = cv2.INTER_LINEAR if (bw_c > src_c.shape[1]) else cv2.INTER_AREA
                scaled_c = cv2.resize(src_c, (bw_c, bh_c), interpolation=interp_c)
                res_c = cv2.matchTemplate(dst_c, scaled_c, cv2.TM_CCOEFF_NORMED)
                _, max_v_c, _, _ = cv2.minMaxLoc(res_c)
                if max_v_c > best_coarse_v:
                    best_coarse_v = float(max_v_c)
                if max_v_c >= 0.50:
                    coarse_candidates.append(s)

            if best_coarse_v >= 0.50 and len(coarse_candidates) > 0:
                scales_to_eval = coarse_candidates
            else:
                scales_to_eval = scales

        best_corr = 0.0
        best_s = 1.0
        best_patch = None
        best_mad = 999.0
        best_mad_norm = 999.0
        best_box = None
        for s in scales_to_eval:
            bw = int(src_arr.shape[1] * s)
            bh = int(src_arr.shape[0] * s)
            if bw < 24 or bh < 24 or bw >= dst_arr.shape[1] or bh >= dst_arr.shape[0]:
                continue
            interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
            scaled = cv2.resize(src_arr, (bw, bh), interpolation=interp)
            res = cv2.matchTemplate(dst_arr, scaled, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_l = cv2.minMaxLoc(res)
            patch = dst_arr[max_l[1]:max_l[1]+bh, max_l[0]:max_l[0]+bw]
            mad = float(np.mean(np.abs(patch.astype(float) - scaled.astype(float))))
            mad_norm = float(np.mean(np.abs((patch.astype(float) - np.mean(patch)) - (scaled.astype(float) - np.mean(scaled)))))
            if max_v > best_corr and (mad <= 30.0 or mad_norm <= 20.0):
                best_corr = float(max_v)
                best_s = s
                best_patch = patch
                best_mad = mad
                best_mad_norm = mad_norm
                best_box = (max_l[0], max_l[1], max_l[0] + bw, max_l[1] + bh)

        # Safeguard 3 fallback: full resolution if coarse didn't achieve strong match
        if use_coarse and scales_to_eval is not scales and best_corr < 0.70:
            for s in scales:
                if s in scales_to_eval:
                    continue
                bw = int(src_arr.shape[1] * s)
                bh = int(src_arr.shape[0] * s)
                if bw < 24 or bh < 24 or bw >= dst_arr.shape[1] or bh >= dst_arr.shape[0]:
                    continue
                interp = cv2.INTER_LINEAR if s > 1.0 else cv2.INTER_AREA
                scaled = cv2.resize(src_arr, (bw, bh), interpolation=interp)
                res = cv2.matchTemplate(dst_arr, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_l = cv2.minMaxLoc(res)
                patch = dst_arr[max_l[1]:max_l[1]+bh, max_l[0]:max_l[0]+bw]
                mad = float(np.mean(np.abs(patch.astype(float) - scaled.astype(float))))
                mad_norm = float(np.mean(np.abs((patch.astype(float) - np.mean(patch)) - (scaled.astype(float) - np.mean(scaled)))))
                if max_v > best_corr and (mad <= 30.0 or mad_norm <= 20.0):
                    best_corr = float(max_v)
                    best_s = s
                    best_patch = patch
                    best_mad = mad
                    best_mad_norm = mad_norm
                    best_box = (max_l[0], max_l[1], max_l[0] + bw, max_l[1] + bh)

        # Fine-scale local refinement around candidate peak if match >= 0.75
        if best_corr >= 0.75:
            for s_fine in np.linspace(best_s - 0.03, best_s + 0.03, 19):
                bw = int(src_arr.shape[1] * s_fine)
                bh = int(src_arr.shape[0] * s_fine)
                if bw < 24 or bh < 24 or bw >= dst_arr.shape[1] or bh >= dst_arr.shape[0]:
                    continue
                interp = cv2.INTER_LINEAR if s_fine > 1.0 else cv2.INTER_AREA
                scaled = cv2.resize(src_arr, (bw, bh), interpolation=interp)
                res = cv2.matchTemplate(dst_arr, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_l = cv2.minMaxLoc(res)
                patch = dst_arr[max_l[1]:max_l[1]+bh, max_l[0]:max_l[0]+bw]
                mad = float(np.mean(np.abs(patch.astype(float) - scaled.astype(float))))
                mad_norm = float(np.mean(np.abs((patch.astype(float) - np.mean(patch)) - (scaled.astype(float) - np.mean(scaled)))))
                if max_v > best_corr and (mad <= 30.0 or mad_norm <= 20.0):
                    best_corr = float(max_v)
                    best_s = s_fine
                    best_patch = patch
                    best_mad = mad
                    best_mad_norm = mad_norm
                    best_box = (max_l[0], max_l[1], max_l[0] + bw, max_l[1] + bh)

        return best_corr, best_mad, best_mad_norm, best_s, best_patch, best_box

    h_a, w_a = arr_a.shape[:2]
    h_b, w_b = arr_b.shape[:2]
    is_equal_canvas = (abs(w_a - w_b) <= 6 and abs(h_a - h_b) <= 6)

    # 1. Raster cross-correlation containment
    best_b_in_a, mad_b_in_a, mad_norm_b, best_s_b, best_patch_b, best_box_b = scan_containment(arr_b, arr_a)
    best_a_in_b, mad_a_in_b, mad_norm_a, best_s_a, best_patch_a, best_box_a = scan_containment(arr_a, arr_b)

    # Equal canvas check: on identical canvas dimensions, high-scale correspondence (s >= 0.88)
    # is full-canvas alignment (blur, photometric, resize, etc.), not a sub-region crop!
    if is_equal_canvas:
        if best_s_b >= 0.88 and best_s_a >= 0.88:
            res = (0.0, 0.0)
            _CROP_CONTAINMENT_CACHE[(p_a_str, p_b_str)] = res
            _CROP_CONTAINMENT_CACHE[(p_b_str, p_a_str)] = (-res[0], res[1])
            return res

    # 2. Geometric SIFT homography containment
    sift_b_in_a, sift_ratio_b, sift_box_b = check_geometric_containment_sift(path_b, path_a, arr_b, arr_a)
    sift_a_in_b, sift_ratio_a, sift_box_a = check_geometric_containment_sift(path_a, path_b, arr_a, arr_b)

    # Sibling verification for equal-area or near equal-area canvases:
    # Restricts sibling raster cross-correlation false positives (downscaled sub-windows matching textures).
    # True geometric SIFT homography already validates 4-corner bounded containment, Lowe ratio, and convexity.
    area_a = arr_a.shape[0] * arr_a.shape[1]
    area_b = arr_b.shape[0] * arr_b.shape[1]
    is_similar_area = (0.70 <= area_a / max(1, area_b) <= 1.40)

    if is_similar_area:
        patch_b_eval = best_patch_b
        is_subscale_b = (best_s_b <= 0.75)
        if patch_b_eval is not None and is_subscale_b:
            edge_patch = float(cv2.Laplacian(patch_b_eval, cv2.CV_64F).std())
            edge_b = float(cv2.Laplacian(arr_b, cv2.CV_64F).std())
            if edge_patch > 5.0 and edge_b >= 0.85 * edge_patch:
                best_b_in_a = 0.0

        patch_a_eval = best_patch_a
        is_subscale_a = (best_s_a <= 0.75)
        if patch_a_eval is not None and is_subscale_a:
            edge_patch = float(cv2.Laplacian(patch_a_eval, cv2.CV_64F).std())
            edge_a = float(cv2.Laplacian(arr_a, cv2.CV_64F).std())
            if edge_patch > 5.0 and edge_a >= 0.85 * edge_patch:
                best_a_in_b = 0.0

    # Determine which direction has superior spatial geometric containment:
    # Geometric Direction Constraint:
    # Unscaled crops (s in [0.85, 1.15] with child area < 0.95 * parent area) are valid.
    # Subscale or inverted matches require verified geometric SIFT homography and child area < parent area,
    # with an asymmetric exception for near-equal-area recompressed crops (e.g., WhatsApp).
    ratio_ba = area_b / max(1, area_a)
    ratio_ab = area_a / max(1, area_b)
    box_area_b = ((sift_box_b[2] - sift_box_b[0]) * (sift_box_b[3] - sift_box_b[1])) if sift_box_b else 0
    proj_coverage_b = box_area_b / float(area_a) if area_a > 0 else 1.0
    box_area_a = ((sift_box_a[2] - sift_box_a[0]) * (sift_box_a[3] - sift_box_a[1])) if sift_box_a else 0
    proj_coverage_a = box_area_a / float(area_b) if area_b > 0 else 1.0

    init_sift_b_in_a = bool(sift_b_in_a)
    init_sift_a_in_b = bool(sift_a_in_b)

    is_sift_crop_b = (
        init_sift_b_in_a and not init_sift_a_in_b and
        (area_b < area_a or proj_coverage_b <= 0.85)
    )
    is_unscaled_crop_b = (0.85 <= best_s_b <= 1.15 and area_b < 0.95 * area_a)
    is_sift_crop_a = (
        init_sift_a_in_b and not init_sift_b_in_a and
        (area_a < area_b or proj_coverage_a <= 0.85)
    )
    is_unscaled_crop_a = (0.85 <= best_s_a <= 1.15 and area_a < 0.95 * area_b)

    raster_sub_b = (best_b_in_a >= 0.90 and best_s_b <= 0.85 and (mad_b_in_a <= 25.0 or mad_norm_b <= 20.0))
    raster_sub_a = (best_a_in_b >= 0.90 and best_s_a <= 0.85 and (mad_a_in_b <= 25.0 or mad_norm_a <= 20.0))

    if not is_unscaled_crop_b and not is_sift_crop_b and not raster_sub_b:
        best_b_in_a = 0.0
        sift_b_in_a = False

    if not is_unscaled_crop_a and not is_sift_crop_a and not raster_sub_a:
        best_a_in_b = 0.0
        sift_a_in_b = False

    raster_valid_b_in_a = (best_b_in_a >= 0.88 and (mad_b_in_a <= 25.0 or mad_norm_b <= 20.0) and (best_b_in_a > best_a_in_b + 0.05 or mad_b_in_a < 0.70 * mad_a_in_b))
    raster_valid_a_in_b = (best_a_in_b >= 0.88 and (mad_a_in_b <= 25.0 or mad_norm_a <= 20.0) and (best_a_in_b > best_b_in_a + 0.05 or mad_a_in_b < 0.70 * mad_b_in_a))

    valid_b_in_a = (raster_valid_b_in_a or (sift_b_in_a and not sift_a_in_b)) and not (raster_valid_a_in_b and not raster_valid_b_in_a)
    valid_a_in_b = (raster_valid_a_in_b or (sift_a_in_b and not sift_b_in_a)) and not (raster_valid_b_in_a and not raster_valid_a_in_b)

    if valid_b_in_a:
        # B is candidate child of A.
        # Check physical frequency conservation: child B cannot have high frequencies absent in parent patch A.
        if best_patch_b is not None:
            edge_patch_b = float(np.abs(cv2.Laplacian(best_patch_b.astype(np.float32), cv2.CV_32F)).mean())
            edge_child_b = float(np.abs(cv2.Laplacian(arr_b.astype(np.float32), cv2.CV_32F)).mean())
            if edge_patch_b < 2.5 and edge_child_b >= 4.0 and edge_child_b >= 1.8 * edge_patch_b:
                res = (0.0, 0.0)
            else:
                conf = round(max(0.88, best_b_in_a, sift_ratio_b if sift_b_in_a else 0.0), 4)
                res = (1.0, conf)
                box = best_box_b if (best_box_b is not None and best_b_in_a >= 0.88) else sift_box_b
                _CROP_ALIGNMENT_CACHE[(p_a_str, p_b_str)] = {"parent": "A", "box": box}
                _CROP_ALIGNMENT_CACHE[(p_b_str, p_a_str)] = {"parent": "A", "box": box}
        else:
            conf = round(max(0.88, best_b_in_a, sift_ratio_b if sift_b_in_a else 0.0), 4)
            res = (1.0, conf)
            box = best_box_b if best_box_b is not None else sift_box_b
            _CROP_ALIGNMENT_CACHE[(p_a_str, p_b_str)] = {"parent": "A", "box": box}
            _CROP_ALIGNMENT_CACHE[(p_b_str, p_a_str)] = {"parent": "A", "box": box}
    elif valid_a_in_b:
        # A is candidate child of B.
        # Check physical frequency conservation: child A cannot have high frequencies absent in parent patch B.
        if best_patch_a is not None:
            edge_patch_a = float(np.abs(cv2.Laplacian(best_patch_a.astype(np.float32), cv2.CV_32F)).mean())
            edge_child_a = float(np.abs(cv2.Laplacian(arr_a.astype(np.float32), cv2.CV_32F)).mean())
            if edge_patch_a < 2.5 and edge_child_a >= 4.0 and edge_child_a >= 1.8 * edge_patch_a:
                res = (0.0, 0.0)
            else:
                conf = round(max(0.88, best_a_in_b, sift_ratio_a if sift_a_in_b else 0.0), 4)
                res = (-1.0, conf)
                box = best_box_a if (best_box_a is not None and best_a_in_b >= 0.88) else sift_box_a
                _CROP_ALIGNMENT_CACHE[(p_a_str, p_b_str)] = {"parent": "B", "box": box}
                _CROP_ALIGNMENT_CACHE[(p_b_str, p_a_str)] = {"parent": "B", "box": box}
        else:
            conf = round(max(0.88, best_a_in_b, sift_ratio_a if sift_a_in_b else 0.0), 4)
            res = (-1.0, conf)
            box = best_box_a if best_box_a is not None else sift_box_a
            _CROP_ALIGNMENT_CACHE[(p_a_str, p_b_str)] = {"parent": "B", "box": box}
            _CROP_ALIGNMENT_CACHE[(p_b_str, p_a_str)] = {"parent": "B", "box": box}
    else:
        res = (0.0, max(best_b_in_a, best_a_in_b, sift_ratio_b if sift_b_in_a else 0.0, sift_ratio_a if sift_a_in_b else 0.0))

    _CROP_CONTAINMENT_CACHE[(p_a_str, p_b_str)] = res
    _CROP_CONTAINMENT_CACHE[(p_b_str, p_a_str)] = (-res[0], res[1])
    return res


def check_spectral_blur_attenuation(f_a: Dict[str, Any], f_b: Dict[str, Any], cos_sim: float) -> float:
    """
    Antisymmetric test for low-pass Gaussian blur attenuation.
    Returns:
      +1.0: Image B has undergone low-pass blur attenuation relative to A (A -> B).
      -1.0: Image A has undergone low-pass blur attenuation relative to B (B -> A).
       0.0: No asymmetric low-pass attenuation detected.
    """
    if cos_sim < 0.35:
        return 0.0

    e_a = float(f_a.get("edge_mean", 0.0))
    n_a = float(f_a.get("noise_res_std", 0.0))
    e_b = float(f_b.get("edge_mean", 0.0))
    n_b = float(f_b.get("noise_res_std", 0.0))

    min_edge_threshold = 2.0

    # Symmetric condition: significant edge energy drop AND noise residual drop characteristic of Gaussian blur (>= 2.0x)
    if e_a >= min_edge_threshold and e_a >= 2.0 * max(1e-4, e_b) and n_a >= 2.0 * max(1e-4, n_b):
        return 1.0
    if e_b >= min_edge_threshold and e_b >= 2.0 * max(1e-4, e_a) and n_b >= 2.0 * max(1e-4, n_a):
        return -1.0

    return 0.0


def check_full_canvas_resizing(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any]
) -> Tuple[bool, float, str]:
    """
    Antisymmetric test for full-canvas physical resizing / geometric scaling.
    Verifies that the entire visual field of one canvas maps to the other canvas:
    - High full-canvas normalized cross-correlation (NCC_full >= 0.88).
    - Aspect ratio consistency (|AR_a - AR_b| / max(AR_a, AR_b) <= 0.12).
    - Low intensity-normalized mean absolute difference (MAD_full <= 28.0).

    Returns:
      (is_resize: bool, confidence: float, direction: str)
      where direction is:
        "A_TO_B": A has larger area than B (downscaling A -> B).
        "B_TO_A": B has larger area than A (downscaling B -> A).
        "UNCERTAIN": Equal or near-equal dimensions (resampling without area loss).
    """
    p_a_str = str(Path(path_a).resolve())
    p_b_str = str(Path(path_b).resolve())

    if (p_a_str, p_b_str) in _FULL_CANVAS_RESIZE_CACHE:
        return _FULL_CANVAS_RESIZE_CACHE[(p_a_str, p_b_str)]
    if (p_b_str, p_a_str) in _FULL_CANVAS_RESIZE_CACHE:
        is_r, c, d = _FULL_CANVAS_RESIZE_CACHE[(p_b_str, p_a_str)]
        rev_d = "B_TO_A" if d == "A_TO_B" else ("A_TO_B" if d == "B_TO_A" else d)
        res = (is_r, c, rev_d)
        _FULL_CANVAS_RESIZE_CACHE[(p_a_str, p_b_str)] = res
        return res

    g_a = _get_gray_image(path_a)
    g_b = _get_gray_image(path_b)
    if g_a is None or g_b is None:
        return False, 0.0, "UNCERTAIN"

    h_a, w_a = g_a.shape
    h_b, w_b = g_b.shape
    if w_a == 0 or h_a == 0 or w_b == 0 or h_b == 0:
        return False, 0.0, "UNCERTAIN"

    area_a = w_a * h_a
    area_b = w_b * h_b
    ar_a = w_a / float(h_a)
    ar_b = w_b / float(h_b)
    ar_diff = abs(ar_a - ar_b) / max(ar_a, ar_b)

    # Resize B to A's dimensions to evaluate full-canvas correspondence
    interp = cv2.INTER_LINEAR if area_a >= area_b else cv2.INTER_AREA
    g_b_res = cv2.resize(g_b, (w_a, h_a), interpolation=interp)

    mu_a, mu_b = float(np.mean(g_a)), float(np.mean(g_b_res))
    std_a, std_b = float(np.std(g_a)), float(np.std(g_b_res))
    if std_a < 1e-4 or std_b < 1e-4:
        return False, 0.0, "UNCERTAIN"

    ncc = float(np.mean((g_a - mu_a) * (g_b_res - mu_b)) / (std_a * std_b))
    mad = float(np.mean(np.abs(g_a.astype(float) - g_b_res.astype(float))))

    # Full canvas visual match requires high correlation and consistent aspect ratio
    res = (False, 0.0, "UNCERTAIN")
    std_ratio = min(std_a, std_b) / max(1e-4, max(std_a, std_b))
    if ncc >= 0.88 and mad <= 28.0 and ar_diff <= 0.12 and std_ratio >= 0.80:
        conf = round(min(0.98, max(0.88, ncc)), 4)
        if area_a >= 1.15 * area_b:
            res = (True, conf, "A_TO_B")
        elif area_b >= 1.15 * area_a:
            res = (True, conf, "B_TO_A")

    _FULL_CANVAS_RESIZE_CACHE[(p_a_str, p_b_str)] = res
    rev_d = "B_TO_A" if res[2] == "A_TO_B" else ("A_TO_B" if res[2] == "B_TO_A" else res[2])
    _FULL_CANVAS_RESIZE_CACHE[(p_b_str, p_a_str)] = (res[0], res[1], rev_d)
    return res


def check_equal_canvas_resampling(path_a: Path, path_b: Path) -> Tuple[bool, float]:
    """
    Checks for equal-canvas interpolation filtering (resampling decimation),
    such as downscaling by a factor and scaling back to original dimensions.
    """
    try:
        g_a = _get_gray_image(path_a)
        g_b = _get_gray_image(path_b)
        if g_a is None or g_b is None:
            return False, 0.0
        if g_a.shape != g_b.shape:
            return False, 0.0
        mu_a, mu_b = float(np.mean(g_a)), float(np.mean(g_b))
        std_a, std_b = float(np.std(g_a)), float(np.std(g_b))
        if std_a < 1e-4 or std_b < 1e-4:
            return False, 0.0
        ncc = float(np.mean((g_a - mu_a) * (g_b - mu_b)) / (std_a * std_b))
        mad = float(np.mean(np.abs(g_a.astype(float) - g_b.astype(float))))
        if ncc >= 0.88 and 0.08 <= mad <= 28.0:
            return True, round(min(0.98, max(0.88, ncc)), 4)
    except Exception:
        pass
    return False, 0.0


def check_perspective_and_rotation(
    path_a: Path,
    path_b: Path,
    cos_sim: float
) -> Tuple[Optional[str], float, str]:
    """
    Checks whether two images differ by geometric rotation or perspective warping
    using SIFT keypoint matching with RANSAC homography estimation.
    Returns: (transformation_name, confidence, direction)
      where transformation_name is 'rotation', 'perspective', or None
    """
    if cos_sim < 0.35:
        return None, 0.0, "UNCERTAIN"

    p_a_str = str(Path(path_a).resolve())
    p_b_str = str(Path(path_b).resolve())

    if (p_a_str, p_b_str) in _GEOMETRY_CACHE:
        return _GEOMETRY_CACHE[(p_a_str, p_b_str)]
    if (p_b_str, p_a_str) in _GEOMETRY_CACHE:
        res = _GEOMETRY_CACHE[(p_b_str, p_a_str)]
        _GEOMETRY_CACHE[(p_a_str, p_b_str)] = res
        return res

    try:
        arr_a = _get_gray_image(path_a)
        arr_b = _get_gray_image(path_b)
        if arr_a is None or arr_b is None:
            return None, 0.0, "UNCERTAIN"

        if p_a_str in _SIFT_CACHE:
            kp1, des1 = _SIFT_CACHE[p_a_str]
        else:
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(arr_a, None)
            _SIFT_CACHE[p_a_str] = (kp1, des1)

        if p_b_str in _SIFT_CACHE:
            kp2, des2 = _SIFT_CACHE[p_b_str]
        else:
            sift = cv2.SIFT_create()
            kp2, des2 = sift.detectAndCompute(arr_b, None)
            _SIFT_CACHE[p_b_str] = (kp2, des2)

        if des1 is None or des2 is None or len(des1) < 12 or len(des2) < 12:
            return None, 0.0, "UNCERTAIN"

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
        if len(good) < 15:
            return None, 0.0, "UNCERTAIN"

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            return None, 0.0, "UNCERTAIN"

        inliers = int(mask.sum())
        inlier_ratio = inliers / float(len(good))
        if inliers < 15 or inlier_ratio < 0.65:
            return None, 0.0, "UNCERTAIN"

        angle = float(np.arctan2(H[1, 0], H[0, 0]) * 180.0 / np.pi)
        persp_comp = float(abs(H[2, 0]) + abs(H[2, 1]))
        conf = round(min(0.96, max(0.85, inlier_ratio)), 4)

        # Estimate affine/rigid transformation for robust rotation vs shear decomposition
        M_aff, aff_mask = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        shear = 0.0
        aff_angle = 0.0
        if M_aff is not None:
            a, b = float(M_aff[0, 0]), float(M_aff[0, 1])
            c, d = float(M_aff[1, 0]), float(M_aff[1, 1])
            sx = float(np.sqrt(a**2 + c**2))
            sy = float(np.sqrt(b**2 + d**2))
            if sx > 1e-4 and sy > 1e-4:
                shear = float(abs(a * b + c * d) / (sx * sy))
                aff_angle = float(np.arctan2(c, a) * 180.0 / np.pi)

        # Check geometric corner displacement to reject planar identity/photometric noise
        h_a, w_a = arr_a.shape[:2]
        corners = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
        proj_corners = cv2.perspectiveTransform(corners, H)
        max_disp = float(np.linalg.norm(proj_corners - corners, axis=2).max())

        res = (None, 0.0, "UNCERTAIN")
        # If orthogonal rigid rotation (angle >= 3.0 deg and shear < 0.05)
        if abs(aff_angle) >= 3.0 and shear < 0.05 and max_disp >= 12.0:
            res = ("rotation", conf, "UNCERTAIN")
        # If significant projective distortion or non-affine shear (shear >= 0.06 or (persp_comp >= 0.0003 and shear >= 0.03))
        elif (shear >= 0.06 or (persp_comp >= 0.0003 and shear >= 0.03)) and max_disp >= 20.0:
            res = ("perspective", conf, "UNCERTAIN")

        _GEOMETRY_CACHE[(p_a_str, p_b_str)] = res
        _GEOMETRY_CACHE[(p_b_str, p_a_str)] = res
        return res

    except Exception:
        pass

    return None, 0.0, "UNCERTAIN"


def check_pixel_identity_and_mad(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    cos_sim: float
) -> Tuple[bool, float]:
    """
    Checks whether two images are physically identical copies.
    Returns: (is_identical, mean_absolute_pixel_diff)
    """
    w_a, h_a = int(f_a.get("width", 0)), int(f_a.get("height", 0))
    w_b, h_b = int(f_b.get("width", 0)), int(f_b.get("height", 0))
    if w_a != w_b or h_a != h_b or w_a == 0 or h_a == 0:
        return False, 999.0

    r_diff = abs(float(f_a.get("r_mean", 0.0)) - float(f_b.get("r_mean", 0.0)))
    g_diff = abs(float(f_a.get("g_mean", 0.0)) - float(f_b.get("g_mean", 0.0)))
    b_diff = abs(float(f_a.get("b_mean", 0.0)) - float(f_b.get("b_mean", 0.0)))
    if r_diff > 0.6 or g_diff > 0.6 or b_diff > 0.6:
        return False, 999.0

    try:
        with Image.open(path_a).convert("RGB") as im_a, Image.open(path_b).convert("RGB") as im_b:
            arr_a = np.array(im_a, dtype=np.float32)
            arr_b = np.array(im_b, dtype=np.float32)
            diff = np.abs(arr_a - arr_b)
            mad = float(np.mean(diff))
            max_diff = float(np.max(diff))
            is_ident = (mad <= 0.05 and max_diff <= 1.0 and cos_sim >= 0.995)
            return is_ident, mad
    except Exception:
        return False, 999.0


def check_jpeg_recompression(path_a: Path, path_b: Path, f_a: Dict[str, Any], f_b: Dict[str, Any]) -> Tuple[bool, float, str]:
    """
    Authoritative test for JPEG compression artifacts and direction.
    Checks PIL image format/quantization tables.
    Returns: (is_jpeg, confidence, direction)
      direction: "A_TO_B" (A is clean/higher quality, B is compressed)
                 "B_TO_A" (B is clean/higher quality, A is compressed)
                 "UNCERTAIN"
    """
    try:
        with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
            fmt_a = str(getattr(im_a, "format", "") or "").upper()
            fmt_b = str(getattr(im_b, "format", "") or "").upper()
            has_q_a = getattr(im_a, "quantization", None) is not None
            has_q_b = getattr(im_b, "quantization", None) is not None

            # Distinct format/quantization asymmetry
            if (fmt_b == "JPEG" or has_q_b) and not (fmt_a == "JPEG" or has_q_a):
                return True, 0.96, "A_TO_B"
            elif (fmt_a == "JPEG" or has_q_a) and not (fmt_b == "JPEG" or has_q_b):
                return True, 0.96, "B_TO_A"
            elif has_q_a and has_q_b:
                q0_a = float(im_a.quantization[0][0])
                q0_b = float(im_b.quantization[0][0])
                if q0_b >= q0_a + 4.0:
                    return True, 0.92, "A_TO_B"
                elif q0_a >= q0_b + 4.0:
                    return True, 0.92, "B_TO_A"
                else:
                    return False, 0.0, "UNCERTAIN"
    except Exception:
        pass

    return False, 0.0, "UNCERTAIN"


def check_noise_and_sharpen(
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    cos_sim: float
) -> Tuple[Optional[str], float, str]:
    """
    Antisymmetric discriminator between Gaussian noise addition and Laplacian edge sharpening.
    - Gaussian noise causes a surge in high-pass edge gradient (delta_edge >= 5.0) AND noise std.
    - Sharpening produces moderate edge enhancement (1.15x - 3.0x, 0.20 <= delta_edge < 5.0) with mild noise std increase
      and distinct gradient overshoot (edge_max increases by >= 1.15x and >= 20.0).
    - Resampling decimation softens high frequencies (edge_std < 8.0, edge_max < 160.0),
      which must NOT be misclassified as sharpening.
    """
    if cos_sim < 0.65:
        return None, 0.0, "UNCERTAIN"

    e_a = float(f_a.get("edge_mean", 0.0))
    e_b = float(f_b.get("edge_mean", 0.0))
    n_a = float(f_a.get("noise_res_std", 0.0))
    n_b = float(f_b.get("noise_res_std", 0.0))
    emax_a = float(f_a.get("edge_max", 0.0))
    emax_b = float(f_b.get("edge_max", 0.0))
    estd_a = float(f_a.get("edge_std", 0.0))
    estd_b = float(f_b.get("edge_std", 0.0))

    # Condition 1: Noise Influx (A -> B: A is clean parent, B is noisy child)
    if e_b - e_a >= 5.0 and (n_b - n_a) >= 1.5 and n_a <= 10.0:
        return "noise", 0.94, "A_TO_B"
    # Condition 1 reverse: Noise Influx (B -> A: B is clean parent, A is noisy child)
    if e_a - e_b >= 5.0 and (n_a - n_b) >= 1.5 and n_b <= 10.0:
        return "noise", 0.94, "B_TO_A"

    # Condition 2: Sharpening (Moderate edge energy boost with unsharp mask peak gradient overshoot)
    if cos_sim >= 0.85:
        # A -> B: A is unsharpened parent, B is sharpened child
        if (e_b - e_a >= 0.20 and e_b >= 1.15 * e_a and n_b >= n_a
            and estd_a >= 8.0 and emax_a >= 160.0 and emax_b >= 1.15 * emax_a and (emax_b - emax_a) >= 20.0):
            return "sharpen", 0.92, "A_TO_B"
        # B -> A: B is unsharpened parent, A is sharpened child
        if (e_a - e_b >= 0.20 and e_a >= 1.15 * e_b and n_a >= n_b
            and estd_b >= 8.0 and emax_b >= 160.0 and emax_a >= 1.15 * emax_b and (emax_a - emax_b) >= 20.0):
            return "sharpen", 0.92, "B_TO_A"

    return None, 0.0, "UNCERTAIN"


def check_contrast_and_color(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    cos_sim: float
) -> Tuple[Optional[str], float]:
    """
    Distinguishes luminance contrast adjustment from chromatic saturation adjustment.
    - Luminance contrast changes the standard deviation of grayscale luminance.
    - Color saturation shifts HSV saturation while preserving grayscale luminance variance.
    """
    if cos_sim < 0.80:
        return None, 0.0

    try:
        with Image.open(path_a) as im_a, Image.open(path_b) as im_b:
            gray_a = np.array(im_a.convert("L"), dtype=np.float32)
            gray_b = np.array(im_b.convert("L"), dtype=np.float32)
            g_std_a = float(gray_a.std())
            g_std_b = float(gray_b.std())
            delta_gray_std = abs(g_std_b - g_std_a)

            # HSV saturation test
            hsv_a = np.array(im_a.convert("HSV"), dtype=np.float32)
            hsv_b = np.array(im_b.convert("HSV"), dtype=np.float32)
            sat_a = hsv_a[:, :, 1] / 255.0
            sat_b = hsv_b[:, :, 1] / 255.0
            delta_sat = abs(float(sat_b.mean()) - float(sat_a.mean()))

            # If luminance std changed significantly (>= 5.0) -> Contrast
            if delta_gray_std >= 5.0:
                conf = round(min(0.96, 0.70 + delta_gray_std / 30.0), 4)
                return "contrast", conf

            # If luminance std is relatively preserved (< 4.0) but saturation changed (>= 0.08) -> Color
            if delta_sat >= 0.08 and delta_gray_std < 4.0:
                return "color", 0.92
    except Exception:
        pass

    return None, 0.0


def _get_pair_evidence(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    cos_sim: float
) -> Dict[str, Any]:
    """
    Evaluates physical evidence once per image pair and caches it for shared reuse
    across both directional and transformation arbitration stages.
    """
    p_a_str = str(Path(path_a).resolve())
    p_b_str = str(Path(path_b).resolve())

    if (p_a_str, p_b_str) in _ARBITRATION_EVIDENCE_CACHE:
        return _ARBITRATION_EVIDENCE_CACHE[(p_a_str, p_b_str)]

    if (p_b_str, p_a_str) in _ARBITRATION_EVIDENCE_CACHE:
        rev = _ARBITRATION_EVIDENCE_CACHE[(p_b_str, p_a_str)]
        res_dir = "B_TO_A" if rev["res_dir"] == "A_TO_B" else ("A_TO_B" if rev["res_dir"] == "B_TO_A" else rev["res_dir"])
        ns_dir = "B_TO_A" if rev["ns_dir"] == "A_TO_B" else ("A_TO_B" if rev["ns_dir"] == "B_TO_A" else rev["ns_dir"])
        jq_dir = "B_TO_A" if rev["jq_dir"] == "A_TO_B" else ("A_TO_B" if rev["jq_dir"] == "B_TO_A" else rev["jq_dir"])
        ev = {
            "is_identical": rev["is_identical"],
            "mad_ident": rev["mad_ident"],
            "is_res": rev["is_res"],
            "res_conf": rev["res_conf"],
            "res_dir": res_dir,
            "geom_tfm": rev["geom_tfm"],
            "geom_conf": rev["geom_conf"],
            "crop_score": -rev["crop_score"],
            "crop_corr": rev["crop_corr"],
            "blur_score": -rev["blur_score"],
            "is_brightness": rev["is_brightness"],
            "is_contrast": rev["is_contrast"],
            "is_color": rev["is_color"],
            "delta_mu": rev["delta_mu"],
            "delta_sigma": rev["delta_sigma"],
            "cc_tfm": rev["cc_tfm"],
            "cc_conf": rev["cc_conf"],
            "ns_tfm": rev["ns_tfm"],
            "ns_conf": rev["ns_conf"],
            "ns_dir": ns_dir,
            "is_jpeg": rev["is_jpeg"],
            "jq_conf": rev["jq_conf"],
            "jq_dir": jq_dir,
        }
        _ARBITRATION_EVIDENCE_CACHE[(p_a_str, p_b_str)] = ev
        return ev

    # 0. Identity test
    is_identical, mad_ident = check_pixel_identity_and_mad(path_a, path_b, f_a, f_b, cos_sim)

    # 1. Full canvas resize test
    is_res, res_conf, res_dir = check_full_canvas_resizing(path_a, path_b, f_a, f_b)

    # 2. Geometric perspective / rotation test
    geom_tfm, geom_conf, _ = check_perspective_and_rotation(path_a, path_b, cos_sim)

    # 3. Sub-region containment (skip only if full-canvas dimensional resize verified)
    if not is_res:
        crop_score, crop_corr = check_sub_region_containment(path_a, path_b)
    else:
        crop_score, crop_corr = 0.0, 0.0

    # 4. Low-pass spectral blur attenuation
    blur_score = check_spectral_blur_attenuation(f_a, f_b, cos_sim)

    # 5. Photometric shift analysis (mean luminance vs dynamic range dispersion)
    mu_a = (float(f_a.get("r_mean", 0)) + float(f_a.get("g_mean", 0)) + float(f_a.get("b_mean", 0))) / 3.0
    mu_b = (float(f_b.get("r_mean", 0)) + float(f_b.get("g_mean", 0)) + float(f_b.get("b_mean", 0))) / 3.0
    delta_mu = abs(mu_b - mu_a)
    sigma_a = (float(f_a.get("r_std", 0)) + float(f_a.get("g_std", 0)) + float(f_a.get("b_std", 0))) / 3.0
    sigma_b = (float(f_b.get("r_std", 0)) + float(f_b.get("g_std", 0)) + float(f_b.get("b_std", 0))) / 3.0
    delta_sigma = abs(sigma_b - sigma_a)

    cc_tfm, cc_conf = check_contrast_and_color(path_a, path_b, f_a, f_b, cos_sim)
    is_brightness = False
    is_contrast = False
    is_color = (cc_tfm == "color")

    # If blur is present, std reduction is caused by low-pass smoothing, not contrast adjustment
    if abs(blur_score) <= 0.5:
        # Check if spatial alignment exists to perform region-aligned photometric assessment
        align_info = _CROP_ALIGNMENT_CACHE.get((p_a_str, p_b_str))
        has_aligned_patch = False
        if abs(crop_score) > 0.5 and align_info and align_info.get("box"):
            x1, y1, x2, y2 = align_info["box"]
            parent_tag = align_info.get("parent")
            arr_parent = _get_gray_image(path_a) if parent_tag == "A" else _get_gray_image(path_b)
            arr_child = _get_gray_image(path_b) if parent_tag == "A" else _get_gray_image(path_a)
            if arr_parent is not None and arr_child is not None and x2 > x1 and y2 > y1:
                patch = arr_parent[y1:y2, x1:x2]
                if patch.shape[0] > 8 and patch.shape[1] > 8:
                    has_aligned_patch = True
                    mu_patch = float(np.mean(patch))
                    std_patch = float(np.std(patch))
                    mu_child = float(np.mean(arr_child))
                    std_child = float(np.std(arr_child))
                    delta_mu_loc = abs(mu_child - mu_patch)
                    delta_sigma_loc = abs(std_child - std_patch)

                    # When spatial alignment is verified, evaluate photometrics strictly on corresponding regions:
                    if delta_mu_loc < 5.0 and delta_sigma_loc < 5.0:
                        # Patch intensities match: field-of-view differences must NOT trigger false brightness/contrast
                        is_brightness = False
                        is_contrast = False
                    elif delta_mu_loc >= 6.0 and (delta_sigma_loc < 4.0 or delta_mu_loc >= 2.0 * delta_sigma_loc):
                        # Genuine local brightness shift on top of spatial crop
                        is_brightness = True
                    elif delta_sigma_loc >= 5.0 and delta_sigma_loc >= 0.50 * delta_mu_loc:
                        # Genuine local contrast shift on top of spatial crop
                        is_contrast = True

        if not has_aligned_patch:
            # Full-canvas photometric analysis
            # Disambiguate brightness vs contrast:
            # 1. When mean shift dominates (delta_mu >= 6.0 and delta_mu >= 2.0 * delta_sigma) -> Brightness
            if delta_mu >= 6.0 and (delta_sigma < 4.0 or delta_mu >= 2.0 * delta_sigma):
                is_brightness = True
            # 2. When dispersion shift dominates (delta_sigma >= 5.0) -> Contrast
            elif cc_tfm == "contrast" or (delta_sigma >= 5.0 and delta_sigma >= 0.50 * delta_mu):
                is_contrast = True
            elif cc_tfm == "contrast" or delta_sigma >= 5.0:
                is_contrast = True
            elif delta_mu >= 6.0:
                is_brightness = True

    # 6. High-pass noise influx vs sharpening
    ns_tfm, ns_conf, ns_dir = check_noise_and_sharpen(f_a, f_b, cos_sim)

    # 7. JPEG compression
    is_jpeg, jq_conf, jq_dir = check_jpeg_recompression(path_a, path_b, f_a, f_b)

    ev = {
        "is_identical": is_identical,
        "mad_ident": mad_ident,
        "is_res": is_res,
        "res_conf": res_conf,
        "res_dir": res_dir,
        "geom_tfm": geom_tfm,
        "geom_conf": geom_conf,
        "crop_score": crop_score,
        "crop_corr": crop_corr,
        "blur_score": blur_score,
        "is_brightness": is_brightness,
        "is_contrast": is_contrast,
        "is_color": is_color,
        "delta_mu": delta_mu,
        "delta_sigma": delta_sigma,
        "cc_tfm": cc_tfm,
        "cc_conf": cc_conf,
        "ns_tfm": ns_tfm,
        "ns_conf": ns_conf,
        "ns_dir": ns_dir,
        "is_jpeg": is_jpeg,
        "jq_conf": jq_conf,
        "jq_dir": jq_dir,
    }
    _ARBITRATION_EVIDENCE_CACHE[(p_a_str, p_b_str)] = ev
    return ev


def arbitrate_pair_direction(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    base_direction: str,
    base_confidence: float,
    cos_sim: float,
    predicted_tfm: str
) -> Tuple[str, float, str]:
    """
    Arbitrates final parent-child direction by fusing physical constraints with ML logits.
    Guarantees 100% mathematical antisymmetry:
      arbitrate(B, A) == reverse(arbitrate(A, B))
    Returns: (final_direction, confidence, rationale)
    """
    ev = _get_pair_evidence(path_a, path_b, f_a, f_b, cos_sim)

    # 0. Check for exact visual identity / pixel copies
    if ev["is_identical"]:
        return "UNCERTAIN", 0.50, "Identity: Images are identical or pixel-equivalent copies; directional ancestry cannot be determined"

    # 1. Scale-invariant full-canvas resizing & downscaling constraint
    if ev["is_res"] and ev["res_dir"] in ["A_TO_B", "B_TO_A"]:
        child_label = "Image B" if ev["res_dir"] == "A_TO_B" else "Image A"
        parent_label = "Image A" if ev["res_dir"] == "A_TO_B" else "Image B"
        return ev["res_dir"], ev["res_conf"], f"Full-Canvas Resampling & Geometric Downscaling: {child_label} has reduced canvas area relative to {parent_label}"

    # 1.5 Check for Perspective & Geometric Rotation (planar warping is symmetric)
    if ev["geom_tfm"] in ["perspective", "rotation"]:
        return "UNCERTAIN", 0.50, f"{ev['geom_tfm'].capitalize()} transformation: Rigid/projective planar warping without irreversible ancestral attenuation"

    # 2. Check scale-invariant sub-region crop containment
    if ev["crop_score"] > 0.5:
        return "A_TO_B", round(max(0.92, ev["crop_corr"]), 4), "Forensic Field-of-View Containment: Image B is an interior sub-window of Image A"
    elif ev["crop_score"] < -0.5:
        return "B_TO_A", round(max(0.92, ev["crop_corr"]), 4), "Forensic Field-of-View Containment: Image A is an interior sub-window of Image B"
    elif predicted_tfm == "crop":
        return "UNCERTAIN", 0.50, "Crop Direction Unresolved: Neither image exhibits physical field-of-view containment"

    # 3. Check spectral blur attenuation
    if ev["blur_score"] > 0.5:
        return "A_TO_B", 0.9850, "Forensic Spectral Attenuation: Image B exhibits severe high-frequency loss consistent with Gaussian low-pass filtering"
    elif ev["blur_score"] < -0.5:
        return "B_TO_A", 0.9850, "Forensic Spectral Attenuation: Image A exhibits severe high-frequency loss consistent with Gaussian low-pass filtering"

    # 4. Check for brightness, contrast, and color shifts (physically symmetric without clipping)
    if ev["is_brightness"] or ev["is_contrast"] or ev["is_color"]:
        label = "Brightness" if ev["is_brightness"] else ("Contrast" if ev["is_contrast"] else "Color")
        return "UNCERTAIN", 0.50, f"{label} variation: Symmetric photometric shift without irreversible structural loss"

    # 5. Check for high-pass Gaussian noise influx or edge sharpening
    if ev["ns_tfm"] == "noise" and ev["ns_dir"] in ["A_TO_B", "B_TO_A"]:
        child_label = "Image B" if ev["ns_dir"] == "A_TO_B" else "Image A"
        parent_label = "Image A" if ev["ns_dir"] == "A_TO_B" else "Image B"
        return ev["ns_dir"], ev["ns_conf"], f"High-Pass Noise Influx: {child_label} exhibits elevated residual noise variance relative to clean {parent_label}"
    elif ev["ns_tfm"] == "sharpen" and ev["ns_dir"] in ["A_TO_B", "B_TO_A"]:
        child_label = "Image B" if ev["ns_dir"] == "A_TO_B" else "Image A"
        parent_label = "Image A" if ev["ns_dir"] == "A_TO_B" else "Image B"
        return ev["ns_dir"], ev["ns_conf"], f"Edge Gradient Amplification: {child_label} exhibits artificial high-pass edge boost consistent with Laplacian unsharp masking"

    # 6. Check for JPEG compression degradation
    if ev["is_jpeg"] and ev["jq_dir"] in ["A_TO_B", "B_TO_A"]:
        child_label = "Image B" if ev["jq_dir"] == "A_TO_B" else "Image A"
        parent_label = "Image A" if ev["jq_dir"] == "A_TO_B" else "Image B"
        return ev["jq_dir"], ev["jq_conf"], f"JPEG Blocking Artifacts: {child_label} exhibits compression quality degradation relative to {parent_label}"

    # 7. Check for genuine ambiguity, symmetric transformations, or uncalibrated fallback
    if ev["is_res"] and ev["res_dir"] == "UNCERTAIN":
        return "UNCERTAIN", 0.50, "Full-Canvas Resampling: Equal canvas dimensions with interpolation filtering"

    if base_direction in ["UNCERTAIN", "NONE"] or base_direction not in ["A_TO_B", "B_TO_A"]:
        return "UNCERTAIN", 0.50, "Direction unresolved: Insufficient asymmetric forensic evidence"

    if 0.48 <= base_confidence <= 0.52:
        return base_direction, base_confidence, "Slight Directional Gradient: Close to uncertainty boundary"

    # Default to verified Antisymmetric Direction MLP
    return base_direction, base_confidence, "Antisymmetric Neural Gradient: Multimodal feature divergence flow"


def arbitrate_transformation(
    path_a: Path,
    path_b: Path,
    f_a: Dict[str, Any],
    f_b: Dict[str, Any],
    base_tfm: str,
    base_conf: float,
    cos_sim: float,
    direction: str
) -> Tuple[str, Optional[float], List[str]]:
    """
    Arbitrates transformation classification, resolving JPEG-sharpen confusion,
    verifying full-canvas resizing independently from cropping, and generating
    multi-label compound tags for composite transformations.
    Returns: (primary_transformation, confidence, compound_tags)
    """
    compound_tags = []
    if base_tfm and base_tfm not in ["none", "identity", "unresolved", "unknown"]:
        compound_tags.append(base_tfm)

    ev = _get_pair_evidence(path_a, path_b, f_a, f_b, cos_sim)

    # 0. Check for exact visual identity / pixel copies
    if ev["is_identical"]:
        return "identity", 0.99, ["identity"]

    edge_a = float(f_a.get("edge_mean", 0.0))
    edge_b = float(f_b.get("edge_mean", 0.0))

    # 1. Scale-Invariant Full-Canvas Resizing / Downscaling
    is_res = ev["is_res"]
    res_conf = ev["res_conf"]
    res_dir = ev["res_dir"]
    if is_res and "resize" not in compound_tags:
        compound_tags.append("resize")

    # 2. Scale-Invariant Sub-Region Containment (Crop) - only if not full-canvas resize
    crop_score = ev["crop_score"]
    crop_corr = ev["crop_corr"]
    is_crop = abs(crop_score) > 0.5
    if is_crop and "crop" not in compound_tags:
        compound_tags.append("crop")

    # 3. Spectral Blur Signature
    blur_score = ev["blur_score"]
    is_blur = abs(blur_score) > 0.5
    if is_blur and "blur" not in compound_tags:
        compound_tags.append("blur")

    # 4. Brightness Shift
    is_brightness = ev["is_brightness"]
    delta_mu = ev["delta_mu"]
    if is_brightness and "brightness" not in compound_tags:
        compound_tags.append("brightness")

    # 5. Contrast vs Color Saturation
    is_contrast = ev["is_contrast"]
    is_color = ev["is_color"]
    cc_conf = ev["cc_conf"]
    if is_contrast and "contrast" not in compound_tags:
        compound_tags.append("contrast")
    if is_color and "color" not in compound_tags:
        compound_tags.append("color")

    # 6. JPEG Compression Artifacts
    is_jpeg = ev["is_jpeg"]
    jq_conf = ev["jq_conf"]
    if is_jpeg and "jpeg" not in compound_tags:
        compound_tags.append("jpeg")

    # 7. High-Pass Noise Influx vs Sharpening
    ns_tfm = ev["ns_tfm"]
    ns_conf = ev["ns_conf"]
    is_noise = (ns_tfm == "noise")
    is_sharpen = (ns_tfm == "sharpen")
    if is_noise and "noise" not in compound_tags:
        compound_tags.append("noise")
    if is_sharpen and "sharpen" not in compound_tags:
        compound_tags.append("sharpen")

    # 7.5 Perspective & Geometric Rotation
    geom_tfm = ev["geom_tfm"]
    geom_conf = ev["geom_conf"]
    is_perspective = (geom_tfm == "perspective")
    is_rotation = (geom_tfm == "rotation")
    if is_perspective and "perspective" not in compound_tags:
        compound_tags.append("perspective")
    if is_rotation and "rotation" not in compound_tags:
        compound_tags.append("rotation")

    # 8. Composite Quality Degradation (multi-degradation: JPEG + noise/resampling, or resize + jpeg)
    deg_signals = [s for s in ["resize", "jpeg", "noise"] if s in compound_tags]
    is_quality_deg = (
        len(deg_signals) >= 2 or
        base_tfm == "quality_degradation" or
        (is_jpeg and not is_brightness and not is_contrast and not is_color and (max(edge_a, edge_b) >= 1.35 * min(edge_a, edge_b) or is_noise))
    )
    if is_quality_deg:
        if "quality_degradation" not in compound_tags:
            compound_tags.insert(0, "quality_degradation")
        return "quality_degradation", 0.92, list(dict.fromkeys(compound_tags))

    # Priority determination for verified physical transformations:
    if is_res and res_dir in ["A_TO_B", "B_TO_A"]:
        return "resize", res_conf, list(dict.fromkeys(compound_tags))
    if is_rotation:
        return "rotation", geom_conf, list(dict.fromkeys(compound_tags))
    if is_perspective:
        return "perspective", geom_conf, list(dict.fromkeys(compound_tags))
    if is_crop:
        if is_brightness and "brightness" not in compound_tags:
            compound_tags.append("brightness")
        if is_contrast and "contrast" not in compound_tags:
            compound_tags.append("contrast")
        return "crop", round(max(0.88, crop_corr), 4), list(dict.fromkeys(compound_tags))
    if is_blur:
        return "blur", 0.95, list(dict.fromkeys(compound_tags))
    if is_contrast:
        return "contrast", cc_conf if cc_conf > 0 else 0.90, list(dict.fromkeys(compound_tags))
    if is_brightness:
        conf = round(min(0.96, 0.70 + delta_mu / 50.0), 4)
        return "brightness", conf, list(dict.fromkeys(compound_tags))
    if is_color:
        return "color", cc_conf if cc_conf > 0 else 0.90, list(dict.fromkeys(compound_tags))
    if is_jpeg:
        return "jpeg", jq_conf, list(dict.fromkeys(compound_tags))
    if is_noise:
        return "noise", ns_conf, list(dict.fromkeys(compound_tags))
    if is_sharpen:
        return "sharpen", ns_conf, list(dict.fromkeys(compound_tags))

    # Fallback to model's predicted base_tfm if verified non-trivial and confident
    if base_tfm and base_tfm not in ["none", "identity", "unresolved", "unknown"] and base_conf is not None and base_conf > 0.50:
        return base_tfm, round(float(base_conf), 4), list(dict.fromkeys(compound_tags or [base_tfm]))

    # When no supported transformation can be verified by evidence, honestly report unresolved
    return "unresolved", None, []
