"""
Tissue segmentation: extract foreground from downsampled sum-projections,
separate touching tissues using erosion + cross-slice guidance.

Strategy:
  1. Hybrid foreground extraction (intensity + local texture) with hole-filling.
     This captures both bright and dim tissues in a generalizable way.
  2. Analyse connected components in the foreground mask.
     a. If we already have >= num_tissues CCs, keep the top N by area
        and use watershed to assign remaining foreground pixels.
     b. If fewer CCs than needed (merged tissues), use iterative erosion
        to split them until enough distinct regions emerge.
  3. Evaluate quality (area balance, coverage) of each slice.
  4. Use the best-segmented slice as reference to guide all others
     via centroid-seeded watershed.
  5. Final hole-filling within each labelled tissue region.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.filters import threshold_multiotsu, threshold_otsu
from skimage.segmentation import watershed


# ---------------------------------------------------------------------------
# Foreground extraction
# ---------------------------------------------------------------------------

def extract_foreground(image: np.ndarray, fill_holes: bool = True) -> np.ndarray:
    """
    Hybrid foreground extraction that captures both bright and dim tissues.

    Pass 1 – Intensity:  multi-Otsu on the sum-projection (good for most
    tissues but misses dim ones).

    Pass 2 – Local texture: compute local standard deviation and threshold
    it.  Any region with spatial structure (edges, folds, cells) has high
    local-std even if its mean intensity is low.

    The two masks are OR-combined, then morphologically cleaned and
    hole-filled.
    """
    rng = np.random.default_rng(42)

    # ---- Pass 1: intensity-based ----
    flat = image.ravel()
    sample_size = min(1_000_000, len(flat))
    sample = flat[rng.choice(len(flat), size=sample_size, replace=False)]
    thresholds = threshold_multiotsu(sample.astype(np.float64), classes=3)
    thresh_int = thresholds[0] / 2
    binary_int = ((image > thresh_int) * 255).astype(np.uint8)
    print(f"  Intensity threshold: {thresh_int:.1f}")

    # ---- Pass 2: local-texture-based ----
    nonzero_vals = image[image > 0]
    if len(nonzero_vals) > 0:
        p99 = np.percentile(nonzero_vals, 99)
    else:
        p99 = 1.0
    norm = np.clip(image / max(p99, 1) * 255, 0, 255).astype(np.uint8)

    ksize = 31
    mean = cv2.blur(norm.astype(np.float32), (ksize, ksize))
    mean_sq = cv2.blur(norm.astype(np.float32) ** 2, (ksize, ksize))
    local_std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))

    # Threshold local-std: Otsu on nonzero values, then use half
    lstd_nz = local_std[local_std > 0].ravel()
    if len(lstd_nz) > 1_000_000:
        lstd_nz = lstd_nz[rng.choice(len(lstd_nz), 1_000_000, replace=False)]
    if len(lstd_nz) > 0:
        std_thresh = threshold_otsu(lstd_nz) * 0.5
    else:
        std_thresh = 1.0
    binary_std = ((local_std > std_thresh) * 255).astype(np.uint8)
    print(f"  Texture threshold: {std_thresh:.1f}")

    # ---- Combine ----
    combined = cv2.bitwise_or(binary_int, binary_std)

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)

    if fill_holes:
        combined = _fill_holes(combined)

    # Smooth out tile-boundary artifacts: opening removes thin rectangular
    # protrusions from tile stitching seams, then re-close and re-fill to
    # restore tissue contours.
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_smooth)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_smooth)
    if fill_holes:
        combined = _fill_holes(combined)

    fg_pixels = np.count_nonzero(combined)
    total_pixels = combined.size
    print(
        f"  Foreground: {fg_pixels} pixels "
        f"({100 * fg_pixels / total_pixels:.1f}% of image)"
    )
    return combined


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """Fill all internal holes via flood-fill from the image border."""
    h, w = binary.shape
    inv = cv2.bitwise_not(binary)
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = inv
    mask_ff = np.zeros((h + 4, w + 4), dtype=np.uint8)
    cv2.floodFill(padded, mask_ff, (0, 0), 255)
    filled_inv = padded[1:-1, 1:-1]
    result = binary | cv2.bitwise_not(filled_inv)

    n_filled = np.count_nonzero(result) - np.count_nonzero(binary)
    if n_filled > 0:
        print(f"  Hole-filling: filled {n_filled} pixels")
    return result


# ---------------------------------------------------------------------------
# Connected-component helpers
# ---------------------------------------------------------------------------

def _count_large_ccs(
    binary: np.ndarray, min_area: int
) -> Tuple[list, np.ndarray]:
    """Return (list_of_(label, area, centroid), label_image)."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, 8, cv2.CV_32S
    )
    large = [
        (i, stats[i, cv2.CC_STAT_AREA], centroids[i])
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] > min_area
    ]
    large.sort(key=lambda x: -x[1])
    return large, labels


# ---------------------------------------------------------------------------
# Erosion-based segmentation
# ---------------------------------------------------------------------------

def _segment_from_ccs(
    foreground: np.ndarray,
    num_tissues: int,
    min_area: int,
    avg_area: float,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Segment tissues by analysing connected components in the foreground.

    If enough CCs already exist (>= num_tissues), keep the top N by area
    and watershed-expand them to fill remaining foreground.

    If fewer CCs exist (merged tissues), use iterative erosion to find
    an erosion level where num_tissues CCs emerge, then watershed back.

    Returns:
        (label_image_or_None, quality_score)
    """
    large, labels = _count_large_ccs(foreground, min_area)
    n_initial = len(large)
    print(f"  Initial foreground CCs (area>{min_area}): {n_initial}")

    if n_initial >= num_tissues:
        return _segment_from_existing_ccs(
            foreground, large, labels, num_tissues, avg_area
        )

    print(f"  Only {n_initial} CCs, need {num_tissues} → using erosion to split")
    return _segment_by_erosion(
        foreground, num_tissues, min_area, avg_area
    )


def _segment_from_existing_ccs(
    foreground: np.ndarray,
    large_ccs: list,
    labels: np.ndarray,
    num_tissues: int,
    avg_area: float,
) -> Tuple[np.ndarray, float]:
    """
    When we already have >= num_tissues CCs, keep the top N by area
    and use watershed to assign stray foreground pixels to the nearest tissue.
    """
    top_ccs = large_ccs[:num_tissues]
    print(f"  Using top {num_tissues} CCs directly (no erosion needed):")

    markers = np.zeros_like(foreground, dtype=np.int32)
    for j, (cc_id, area, (cx, cy)) in enumerate(top_ccs):
        markers[labels == cc_id] = j + 1
        ratio = area / avg_area
        print(f"    CC{j}: area={area:,} ({ratio:.2f}×avg), "
              f"centroid=({cx:.0f},{cy:.0f})")

    dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    ws_labels = watershed(-dist, markers=markers, mask=foreground > 0)
    result = _filter_and_relabel(ws_labels, num_tissues)
    quality = _compute_quality(result, num_tissues, avg_area)
    return result, quality


def _segment_by_erosion(
    foreground: np.ndarray,
    num_tissues: int,
    min_area: int,
    avg_area: float,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Segment by iterative erosion: find erosion level with exactly num_tissues
    CCs and best area balance, then watershed back.

    Returns:
        (label_image_or_None, quality_score)
    """
    ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    best = None
    best_imbalance = float("inf")

    current = foreground.copy()
    for i in range(1, 80):
        current = cv2.erode(current, ek, iterations=1)
        large, labels = _count_large_ccs(current, min_area)
        n = len(large)

        if n == 0:
            break

        if n == num_tissues:
            imbalance = large[0][1] / avg_area
            if imbalance < best_imbalance:
                best_imbalance = imbalance
                best = (labels.copy(), [x for x in large], i)

        if n < num_tissues and best is not None:
            break

        if i % 10 == 0:
            print(f"    erode iter={i}: {n} CCs")

        if np.count_nonzero(current) < min_area * 2:
            break

    if best is None:
        return None, float("inf")

    seed_labels, seed_ccs, best_iter = best

    print(f"  Best erosion: iter={best_iter}, {num_tissues} CCs, "
          f"imbalance={best_imbalance:.2f}")

    markers = np.zeros_like(foreground, dtype=np.int32)
    for j, (cc_id, area, (cx, cy)) in enumerate(seed_ccs):
        markers[seed_labels == cc_id] = j + 1
        ratio = area / avg_area
        print(f"    CC{j}: area={area} ({ratio:.2f}×avg), "
              f"centroid=({cx:.0f},{cy:.0f})")

    dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
    ws_labels = watershed(-dist, markers=markers, mask=foreground > 0)
    result = _filter_and_relabel(ws_labels, num_tissues)
    quality = _compute_quality(result, num_tissues, avg_area)
    return result, quality


def _compute_quality(
    result: np.ndarray, num_tissues: int, avg_area: float
) -> float:
    """Score segmentation quality; lower is better."""
    unique = np.unique(result)
    unique = unique[unique > 0]
    areas = np.array([np.count_nonzero(result == lbl) for lbl in unique])

    if len(areas) < num_tissues:
        return float("inf")

    max_ratio = areas.max() / avg_area
    min_ratio = areas.min() / avg_area
    quality = max_ratio + max(0, 0.15 - min_ratio) * 10

    print(f"  Quality: {quality:.3f} (max_ratio={max_ratio:.2f}, "
          f"min_ratio={min_ratio:.2f})")
    return quality


# ---------------------------------------------------------------------------
# Centroid-guided segmentation
# ---------------------------------------------------------------------------

def _segment_guided(
    foreground: np.ndarray,
    guide_centroids: List[Tuple[float, float]],
    num_tissues: int,
) -> np.ndarray:
    """
    Segment using centroids from another slice as watershed seeds.

    Finds the local maximum of the distance transform nearest to each
    guide centroid so seeds land deep inside tissues, not at edges.
    """
    print(f"  Centroid-guided segmentation with {len(guide_centroids)} seeds")

    dist = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)

    all_peaks = peak_local_max(
        dist, min_distance=50, threshold_abs=20, exclude_border=False,
    )
    print(f"    Found {len(all_peaks)} DT local maxima")

    markers = np.zeros_like(foreground, dtype=np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    used_peaks = set()

    for idx, (cx, cy) in enumerate(guide_centroids):
        gy, gx = cy, cx

        y, x = None, None
        if len(all_peaks) > 0:
            peak_dists = np.sqrt(
                (all_peaks[:, 0] - gy) ** 2 + (all_peaks[:, 1] - gx) ** 2
            )
            for ui in used_peaks:
                peak_dists[ui] += 1e6

            best_peak_idx = int(np.argmin(peak_dists))
            best_peak_dist = peak_dists[best_peak_idx]

            if best_peak_dist < 800:
                py = int(all_peaks[best_peak_idx, 0])
                px = int(all_peaks[best_peak_idx, 1])
                if dist[py, px] > 0:
                    y, x = py, px
                    used_peaks.add(best_peak_idx)

        if y is None:
            y, x = _snap_to_foreground(
                foreground, int(round(gy)), int(round(gx))
            )

        dv = dist[y, x]
        if dv < 5:
            dv = 5
        iters = max(3, min(15, int(dv / 20)))
        pt = np.zeros_like(markers, dtype=np.uint8)
        pt[y, x] = 255
        pt = cv2.dilate(pt, kernel, iterations=iters)
        markers[pt > 0] = idx + 1
        print(f"    Seed {idx+1}: guide=({gx:.0f},{gy:.0f}) → "
              f"placed=({x},{y}), dist_val={dv:.1f}")

    ws = watershed(-dist, markers=markers, mask=foreground > 0)
    return ws


def _snap_to_foreground(
    foreground: np.ndarray, y: int, x: int, search_r: int = 1000
) -> Tuple[int, int]:
    """Find nearest foreground pixel to (y, x)."""
    y = max(0, min(foreground.shape[0] - 1, y))
    x = max(0, min(foreground.shape[1] - 1, x))
    if foreground[y, x] > 0:
        return y, x
    y1 = max(0, y - search_r)
    y2 = min(foreground.shape[0], y + search_r)
    x1 = max(0, x - search_r)
    x2 = min(foreground.shape[1], x + search_r)
    patch = foreground[y1:y2, x1:x2]
    fg_ys, fg_xs = np.where(patch > 0)
    if len(fg_ys) > 0:
        dists_sq = (fg_ys - (y - y1)) ** 2 + (fg_xs - (x - x1)) ** 2
        nearest = np.argmin(dists_sq)
        return fg_ys[nearest] + y1, fg_xs[nearest] + x1
    return y, x


# ---------------------------------------------------------------------------
# Public API: segment one slice
# ---------------------------------------------------------------------------

def segment_tissues(
    image: np.ndarray,
    num_tissues: int = 8,
    min_area_frac: float = 0.005,
    guide_centroids: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Segment individual tissues from a downsampled sum-projection.
    """
    print(f"Segmenting tissues (expecting {num_tissues})...")

    foreground = extract_foreground(image, fill_holes=True)
    total_fg = np.count_nonzero(foreground)
    min_area = int(total_fg * min_area_frac)
    avg_area = total_fg / num_tissues

    if guide_centroids is not None and len(guide_centroids) == num_tissues:
        ws = _segment_guided(foreground, guide_centroids, num_tissues)
        result = _filter_and_relabel(ws, num_tissues)
    else:
        result, _ = _segment_from_ccs(
            foreground, num_tissues, min_area, avg_area
        )
        if result is None:
            print("  FATAL: segmentation failed")
            return np.zeros_like(foreground, dtype=np.int32)

    result = _fill_label_holes(result, foreground)
    _report_results(result)
    return result


# ---------------------------------------------------------------------------
# Public API: segment all slices together
# ---------------------------------------------------------------------------

def segment_all_slices(
    images: List[np.ndarray],
    num_tissues: int = 8,
    min_area_frac: float = 0.005,
) -> List[np.ndarray]:
    """
    Segment tissues across all z-slices with cross-slice guidance.

    1. Extract foreground and segment each slice independently.
    2. Pick the best-quality slice as reference.
    3. For good-quality slices, relabel to match the reference
       (using centroid-based Hungarian matching).
    4. For poor-quality slices, re-segment using guided watershed.
    """
    from scipy.optimize import linear_sum_assignment

    n_slices = len(images)
    foregrounds = []
    independent_results = []
    had_enough_ccs = []

    for i, img in enumerate(images):
        print(f"\n{'=' * 60}")
        print(f"Slice {i}: extracting foreground")
        print(f"{'=' * 60}")
        fg = extract_foreground(img, fill_holes=True)
        foregrounds.append(fg)

        total_fg = np.count_nonzero(fg)
        min_area = int(total_fg * min_area_frac)
        avg_area = total_fg / num_tissues

        large, _ = _count_large_ccs(fg, min_area)
        enough = len(large) >= num_tissues
        had_enough_ccs.append(enough)

        print(f"\nSlice {i}: independent segmentation")
        labels, quality = _segment_from_ccs(
            fg, num_tissues, min_area, avg_area
        )
        independent_results.append((labels, quality))
        print(f"  Slice {i} quality: {quality:.3f}"
              f" (direct CCs: {'yes' if enough else 'no, used erosion'})")

    # Pick the reference from slices that had enough CCs (direct path).
    # Fall back to all slices if none had enough.
    direct_indices = [i for i, e in enumerate(had_enough_ccs) if e]
    if direct_indices:
        qualities_d = [(i, independent_results[i][1]) for i in direct_indices]
        ref_idx = min(qualities_d, key=lambda x: x[1])[0]
    else:
        qualities = [q for _, q in independent_results]
        ref_idx = int(np.argmin(qualities))

    ref_labels = independent_results[ref_idx][0]

    print(f"\n{'=' * 60}")
    print(f"Reference slice: {ref_idx} "
          f"(quality={independent_results[ref_idx][1]:.3f})")
    print(f"{'=' * 60}")

    if ref_labels is None:
        print("  ERROR: no slice segmented successfully!")
        return [np.zeros_like(fg, dtype=np.int32) for fg in foregrounds]

    ref_centroids = _get_centroids(ref_labels, num_tissues)
    print(f"  Reference centroids ({len(ref_centroids)}):")
    for idx, (cx, cy) in enumerate(ref_centroids):
        print(f"    T{idx+1}: ({cx:.0f}, {cy:.0f})")

    segmentations = [None] * n_slices

    ref_result = _fill_label_holes(ref_labels, foregrounds[ref_idx])
    segmentations[ref_idx] = ref_result
    print(f"\nSlice {ref_idx} (reference):")
    _report_results(ref_result)

    for i in range(n_slices):
        if i == ref_idx:
            continue

        ind_labels, ind_quality = independent_results[i]

        if had_enough_ccs[i] and ind_labels is not None:
            print(f"\n{'=' * 60}")
            print(f"Slice {i}: using independent segmentation "
                  f"(quality={ind_quality:.3f}), relabelling to match ref")
            print(f"{'=' * 60}")
            result = _relabel_to_match_reference(
                ind_labels, ref_centroids, num_tissues, linear_sum_assignment
            )
        else:
            print(f"\n{'=' * 60}")
            print(f"Slice {i}: guided segmentation "
                  f"(not enough direct CCs, using reference centroids)")
            print(f"{'=' * 60}")
            ws = _segment_guided(foregrounds[i], ref_centroids, num_tissues)
            result = _filter_and_relabel(ws, num_tissues)

        result = _fill_label_holes(result, foregrounds[i])
        segmentations[i] = result
        _report_results(result)

    return segmentations


def _get_centroids(
    labels: np.ndarray, num_tissues: int
) -> List[Tuple[float, float]]:
    """Extract centroids (cx, cy) for labels 1..num_tissues."""
    centroids = []
    for lbl in range(1, num_tissues + 1):
        ys, xs = np.where(labels == lbl)
        if len(ys) > 0:
            centroids.append((float(xs.mean()), float(ys.mean())))
        else:
            centroids.append((0.0, 0.0))
    return centroids


def _relabel_to_match_reference(
    labels: np.ndarray,
    ref_centroids: List[Tuple[float, float]],
    num_tissues: int,
    linear_sum_assignment,
) -> np.ndarray:
    """
    Relabel an independent segmentation so tissue IDs match the reference
    slice, using a centroid-distance cost matrix and the Hungarian algorithm.
    """
    src_centroids = _get_centroids(labels, num_tissues)

    cost = np.zeros((num_tissues, num_tissues))
    for i, (sx, sy) in enumerate(src_centroids):
        for j, (rx, ry) in enumerate(ref_centroids):
            cost[i, j] = np.sqrt((sx - rx) ** 2 + (sy - ry) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)

    new_labels = np.zeros_like(labels)
    for src_lbl_0, ref_lbl_0 in zip(row_ind, col_ind):
        src_lbl = src_lbl_0 + 1
        ref_lbl = ref_lbl_0 + 1
        new_labels[labels == src_lbl] = ref_lbl
        dist = cost[src_lbl_0, ref_lbl_0]
        print(f"    T{src_lbl} → T{ref_lbl} (centroid dist={dist:.0f}px)")

    return new_labels


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _filter_and_relabel(labels: np.ndarray, num_tissues: int) -> np.ndarray:
    """Keep the top *num_tissues* regions by area; relabel 1..N."""
    unique = np.unique(labels)
    unique = unique[unique > 0]
    areas = {lbl: np.count_nonzero(labels == lbl) for lbl in unique}
    top = sorted(areas, key=areas.get, reverse=True)[:num_tissues]
    result = np.zeros_like(labels)
    for new_id, old_id in enumerate(top, start=1):
        result[labels == old_id] = new_id
    return result


def _fill_label_holes(
    labels: np.ndarray, foreground: np.ndarray
) -> np.ndarray:
    """
    For each labelled tissue, fill internal holes within its contour.
    """
    result = labels.copy()
    unique = np.unique(labels)
    unique = unique[unique > 0]
    total_filled = 0

    for lbl in unique:
        mask = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 1, cv2.FILLED)
        holes = (filled > 0) & (labels == 0)
        n = np.count_nonzero(holes)
        if n > 0:
            result[holes] = lbl
            total_filled += n

    if total_filled > 0:
        print(f"  Label hole-filling: filled {total_filled} pixels")
    return result


def _report_results(labels: np.ndarray):
    unique = np.unique(labels)
    unique = unique[unique > 0]
    print(f"  Final segmentation: {len(unique)} tissues")
    for lbl in unique:
        area = np.count_nonzero(labels == lbl)
        ys, xs = np.where(labels == lbl)
        cy, cx = ys.mean(), xs.mean()
        print(f"    Tissue {lbl}: area={area}, centroid=({cx:.0f}, {cy:.0f})")
