"""
Cross-slice tissue matching: compute features for each tissue in each
z-slice and use the Hungarian algorithm to find optimal one-to-one
correspondences.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def compute_tissue_features(
    label_image: np.ndarray,
) -> Dict[int, Dict]:
    """
    Compute geometric features for each tissue in a label image.

    Args:
        label_image: 2D int array where 0=background, 1..N=tissue IDs

    Returns:
        Dict mapping tissue_id -> {
            'centroid': (cx, cy),
            'area': int,
            'hu_moments': np.ndarray (7,),
            'eccentricity': float,
            'solidity': float,
        }
    """
    features = {}
    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels > 0]

    for lbl in unique_labels:
        mask = (label_image == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Use largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)

        if area < 10:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Hu Moments (log-transformed for scale invariance)
        hu = cv2.HuMoments(M).flatten()
        hu_log = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

        # Eccentricity
        eccentricity = 0.0
        if len(cnt) >= 5:
            try:
                (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
                if ma > 0:
                    eccentricity = np.sqrt(1 - (min(MA, ma) / max(MA, ma)) ** 2)
            except cv2.error:
                pass

        # Solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0.0

        features[int(lbl)] = {
            "centroid": (cx, cy),
            "area": area,
            "hu_moments": hu_log,
            "eccentricity": eccentricity,
            "solidity": solidity,
        }

    return features


def match_tissues_across_slices(
    segmentations: List[np.ndarray],
    weight_centroid: float = 0.8,
    weight_area: float = 0.2,
) -> List[Dict[int, int]]:
    """
    Match tissues across z-slices using the Hungarian algorithm.

    The first slice is used as the reference. For each subsequent slice,
    tissues are matched to the reference by minimizing a weighted cost
    that combines centroid distance and area difference.

    Args:
        segmentations: List of label images (one per z-slice)
        weight_centroid: Weight for centroid distance in cost function
        weight_area: Weight for area difference in cost function

    Returns:
        List of dicts (one per z-slice), mapping local_tissue_id -> global_tissue_id.
        The first slice mapping is identity (reference).
    """
    print("Matching tissues across z-slices...")

    # Compute features for each slice
    all_features = []
    for i, seg in enumerate(segmentations):
        feats = compute_tissue_features(seg)
        all_features.append(feats)
        print(f"  Slice {i}: {len(feats)} tissues detected")

    # Reference is the first slice
    ref_features = all_features[0]
    ref_labels = sorted(ref_features.keys())
    num_tissues = len(ref_labels)

    # Identity mapping for reference
    ref_mapping = {lbl: lbl for lbl in ref_labels}
    mappings = [ref_mapping]

    for slice_idx in range(1, len(segmentations)):
        cur_features = all_features[slice_idx]
        cur_labels = sorted(cur_features.keys())

        if not cur_labels or not ref_labels:
            print(f"  WARNING: Slice {slice_idx} has no tissues to match!")
            mappings.append({})
            continue

        # Build cost matrix
        cost_matrix = _build_cost_matrix(
            ref_features, ref_labels,
            cur_features, cur_labels,
            weight_centroid, weight_area,
        )

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Build mapping: current label -> reference (global) label
        mapping = {}
        for r, c in zip(row_ind, col_ind):
            if r < len(ref_labels) and c < len(cur_labels):
                ref_lbl = ref_labels[r]
                cur_lbl = cur_labels[c]
                cost = cost_matrix[r, c]
                mapping[cur_lbl] = ref_lbl
                ref_cent = ref_features[ref_lbl]["centroid"]
                cur_cent = cur_features[cur_lbl]["centroid"]
                dist = np.sqrt(
                    (ref_cent[0] - cur_cent[0]) ** 2
                    + (ref_cent[1] - cur_cent[1]) ** 2
                )
                print(
                    f"  Slice {slice_idx}: tissue {cur_lbl} -> "
                    f"global {ref_lbl} (dist={dist:.0f}px, cost={cost:.3f})"
                )

        mappings.append(mapping)

    return mappings


def _build_cost_matrix(
    ref_features: Dict[int, Dict],
    ref_labels: List[int],
    cur_features: Dict[int, Dict],
    cur_labels: List[int],
    weight_centroid: float,
    weight_area: float,
) -> np.ndarray:
    """
    Build cost matrix for Hungarian assignment.

    Rows = reference tissues, Cols = current slice tissues.
    Cost = weighted combination of normalized centroid distance and
    normalized area difference.
    """
    n_ref = len(ref_labels)
    n_cur = len(cur_labels)

    # Centroid distance matrix
    ref_centroids = np.array([ref_features[l]["centroid"] for l in ref_labels])
    cur_centroids = np.array([cur_features[l]["centroid"] for l in cur_labels])
    centroid_dists = cdist(ref_centroids, cur_centroids, metric="euclidean")

    # Normalize centroid distances to [0, 1]
    max_dist = centroid_dists.max() if centroid_dists.max() > 0 else 1.0
    centroid_norm = centroid_dists / max_dist

    # Area difference matrix (relative)
    ref_areas = np.array([ref_features[l]["area"] for l in ref_labels])
    cur_areas = np.array([cur_features[l]["area"] for l in cur_labels])
    area_diffs = np.abs(ref_areas[:, None] - cur_areas[None, :])
    max_area = max(ref_areas.max(), cur_areas.max())
    area_norm = area_diffs / max_area if max_area > 0 else area_diffs

    # Combined cost
    cost = weight_centroid * centroid_norm + weight_area * area_norm

    return cost


def apply_matching(
    segmentations: List[np.ndarray],
    mappings: List[Dict[int, int]],
) -> List[np.ndarray]:
    """
    Apply tissue matching to relabel segmentation images so that
    corresponding tissues share the same label across slices.

    Args:
        segmentations: List of label images
        mappings: List of dicts mapping local_id -> global_id

    Returns:
        List of relabeled images
    """
    relabeled = []
    for seg, mapping in zip(segmentations, mappings):
        new_seg = np.zeros_like(seg)
        for local_lbl, global_lbl in mapping.items():
            new_seg[seg == local_lbl] = global_lbl
        relabeled.append(new_seg)
    return relabeled
