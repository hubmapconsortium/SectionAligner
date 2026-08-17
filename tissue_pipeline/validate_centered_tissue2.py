#!/usr/bin/env python3
"""Validate the centered crop fix on tissue 2 only.

This is a targeted driver for the new centered-mode logic in ``cropping.py``.
By default it uses the real tissue_pipeline segmentation path from cached
downsampled projections, then writes only tissue 2.

There is also a ``--from-qc-overlays`` debug mode that reconstructs T2 from
colored QC PNGs, but that is intentionally not the default because color
overlays are a lossy proxy for the true label images.

Outputs are written to a separate validation directory so the current
``new_dataset_output_v7/tissue_2_stacked.ome.tif`` is not overwritten:

  $SECTIONALIGNER_DATA_ROOT/
      new_dataset_output_v7_centered_t2_test/
          tissue_2_stacked.ome.tif
          qc/centered_t2_plan_summary.json

The resulting OME-TIFF should be compared to both:
  - new_dataset_output_v7/tissue_2_stacked.ome.tif
  - new_dataset_output_v7/tissue_2_stacked_recentered.ome.tif
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import tifffile
from PIL import Image

# Make sibling imports work when executed from either repo root or this folder.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cropping import (  # noqa: E402
    compute_bounding_boxes,
    compute_dilated_masks,
    crop_and_save_tissues,
)
from io_utils import (  # noqa: E402
    find_qptiff_files,
    load_channel_names,
    load_or_create_downsampled,
)
from matching import apply_matching, match_tissues_across_slices  # noqa: E402
from segmentation import segment_all_slices  # noqa: E402


# The dataset and its runs live outside this repo. Point SECTIONALIGNER_DATA_ROOT
# somewhere else to validate a different one.
ROOT = Path(
    os.environ.get(
        "SECTIONALIGNER_DATA_ROOT", "/hive/users/tedz/SectionAligner/SectionAligner"
    )
)
INPUT_DIR = ROOT / "new_dataset"
SOURCE_OUTPUT_DIR = ROOT / "new_dataset_output_v7"
QC_DIR = SOURCE_OUTPUT_DIR / "qc"
VALIDATION_DIR = ROOT / "new_dataset_output_v7_centered_t2_test"
VALIDATION_QC_DIR = VALIDATION_DIR / "qc"

N_TISSUES = 8
TISSUE_ID = 2
PADDING = 50


def _detect_tissue2_mask_from_overlay(rgb: np.ndarray) -> np.ndarray:
    """Detect T2 from the matched_segmentation overlay PNG.

    T2 is green in ``visualization.py``. The overlay is alpha-blended over a
    grayscale background, so exact color matching is not appropriate. A raw
    green-dominant mask also picks up other green-ish overlay pixels, so keep
    only the largest connected component. That gives ~2.0M downsampled pixels
    per z, which matches the actual T2 area in the existing stacked TIFF.
    """

    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)

    mask = (g > r + 25) & (g > b + 25) & (g > 60)

    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if n <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = 1 + int(np.argmax(areas))
    return labels == biggest


def load_matched_segmentations_from_qc() -> List[np.ndarray]:
    """Reconstruct T2-only downsampled label images from QC overlays."""

    segmentations: List[np.ndarray] = []
    for z_idx in range(17):
        path = QC_DIR / f"slice_{z_idx}_matched_segmentation.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing QC overlay: {path}")
        rgb = np.array(Image.open(path).convert("RGB"))
        labels = np.zeros(rgb.shape[:2], dtype=np.uint8)
        t2_mask = _detect_tissue2_mask_from_overlay(rgb)
        labels[t2_mask] = TISSUE_ID
        segmentations.append(labels)

        print(
            f"  z={z_idx:>2} labels shape={labels.shape} "
            f"T2={int(t2_mask.sum()):,} px"
        )
    return segmentations


def compute_matched_segmentations_from_pipeline(
    downsampled_images: List[np.ndarray],
) -> List[np.ndarray]:
    """Run the same segmentation + matching path as run_pipeline.py."""

    print("Running segment_all_slices on cached downsampled projections...")
    segmentations = segment_all_slices(
        downsampled_images,
        num_tissues=N_TISSUES,
    )

    print("Running match_tissues_across_slices + apply_matching...")
    mappings = match_tissues_across_slices(segmentations)
    matched = apply_matching(segmentations, mappings)

    for z_idx, labels in enumerate(matched):
        areas = {tid: int((labels == tid).sum()) for tid in range(1, 9)}
        print(
            f"  z={z_idx:>2} labels shape={labels.shape} "
            f"T2={areas[TISSUE_ID]:,} px all={areas}"
        )
    return matched


def get_full_res_shapes(qptiff_paths: List[Path]) -> List[tuple[int, int]]:
    """Read full-resolution (Y, X) page shapes without loading image data."""

    shapes = []
    for z_idx, qptiff_path in enumerate(qptiff_paths):
        with tifffile.TiffFile(str(qptiff_path)) as tif:
            shape = tif.pages[0].shape
        shapes.append(shape)
        print(f"  z={z_idx:>2} {qptiff_path.name}: full-res {shape[0]}x{shape[1]}")
    return shapes


def write_plan_summary(
    tissue_bbox: Dict,
    qptiff_paths: List[Path],
    output_path: Path,
) -> None:
    """Save a compact JSON summary of the centered crop plan."""

    per_z = []
    for z_idx, info in enumerate(tissue_bbox["per_slice"]):
        if info["y1"] is None:
            per_z.append({"z": z_idx, "empty": True})
            continue
        per_z.append(
            {
                "z": z_idx,
                "qptiff": qptiff_paths[z_idx].name,
                "bbox": [info["y1"], info["y2"], info["x1"], info["x2"]],
                "centroid": [info["centroid_y"], info["centroid_x"]],
                "paste_offset": [info["paste_oy"], info["paste_ox"]],
            }
        )

    summary = {
        "input_dir": str(INPUT_DIR),
        "source_qc_dir": str(QC_DIR),
        "output_path": str(output_path),
        "tissue_id": TISSUE_ID,
        "padding": PADDING,
        "canvas": [tissue_bbox["canvas_h"], tissue_bbox["canvas_w"]],
        "target_center": [
            tissue_bbox["target_center_y"],
            tissue_bbox["target_center_x"],
        ],
        "legacy_union_extent": [
            tissue_bbox["height"],
            tissue_bbox["width"],
        ],
        "per_z": per_z,
    }

    VALIDATION_QC_DIR.mkdir(parents=True, exist_ok=True)
    out_json = VALIDATION_QC_DIR / "centered_t2_plan_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved crop-plan summary: {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only build centered crop plan; do not write OME-TIFF.",
    )
    parser.add_argument(
        "--from-qc-overlays",
        action="store_true",
        help=(
            "Debug shortcut: reconstruct T2 from existing colored QC PNGs "
            "instead of using the real pipeline segmentation path."
        ),
    )
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_QC_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Discover raw qptiffs + channel names ===")
    qptiff_paths = find_qptiff_files(str(INPUT_DIR))
    channel_names = load_channel_names(str(INPUT_DIR))
    full_res_shapes = get_full_res_shapes(qptiff_paths)

    if args.from_qc_overlays:
        print("\n=== Reconstruct matched segmentations from existing QC overlays ===")
        segmentations = load_matched_segmentations_from_qc()
        scale_factor_x = scale_factor_y = 8
    else:
        print("\n=== Load cached downsampled projections ===")
        downsampled_images, scale_factor_x, scale_factor_y = load_or_create_downsampled(
            str(INPUT_DIR),
            qptiff_paths,
        )
        print("\n=== Compute matched segmentations via tissue_pipeline ===")
        segmentations = compute_matched_segmentations_from_pipeline(
            downsampled_images
        )

    print("\n=== Build dilated masks ===")
    all_masks = compute_dilated_masks(
        segmentations,
        num_tissues=N_TISSUES,
        dilation_px=15,
        exclusion_erode_px=3,
    )
    t2_masks = {TISSUE_ID: all_masks[TISSUE_ID]}

    print("\n=== Compute centered tissue-2 crop plan ===")
    tissue_bboxes = compute_bounding_boxes(
        t2_masks,
        scale_factor_x=scale_factor_x,
        scale_factor_y=scale_factor_y,
        padding=PADDING,
        full_res_shapes=full_res_shapes,
        centered=True,
    )
    t2_bbox = tissue_bboxes[TISSUE_ID]
    output_path = VALIDATION_DIR / "tissue_2_stacked.ome.tif"
    write_plan_summary(t2_bbox, qptiff_paths, output_path)

    print(
        "\nPlan:"
        f"\n  centered canvas = {t2_bbox['canvas_w']} x {t2_bbox['canvas_h']}"
        f"\n  legacy union    = {t2_bbox['width']} x {t2_bbox['height']}"
        f"\n  estimated raw   = "
        f"{len(qptiff_paths) * len(channel_names) * t2_bbox['canvas_h'] * t2_bbox['canvas_w'] / 1e9:.1f} GB"
    )

    if args.plan_only:
        print("\n--plan-only set; stopping before OME-TIFF write.")
        return

    print("\n=== Write centered tissue-2 stack ===")
    crop_and_save_tissues(
        qptiff_paths=qptiff_paths,
        matched_segmentations=segmentations,
        dilated_masks=t2_masks,
        tissue_bboxes=tissue_bboxes,
        num_channels=len(channel_names),
        scale_factor_x=scale_factor_x,
        scale_factor_y=scale_factor_y,
        num_tissues=N_TISSUES,
        output_dir=str(VALIDATION_DIR),
        channel_names=channel_names,
        compression="zlib",
        compression_level=1,
    )

    print(f"\nDone. Wrote validation output: {output_path}")
    if output_path.exists():
        print(f"Size: {os.path.getsize(output_path) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
