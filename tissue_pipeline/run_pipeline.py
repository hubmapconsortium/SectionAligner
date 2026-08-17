#!/usr/bin/env python3
"""
Tissue Segmentation Pipeline
=============================
Segments individual tissues from multichannel qptiff z-slices, matches
corresponding tissues across slices, and outputs per-tissue 3D
multichannel OME-TIFF volumes.

Usage:
    python run_pipeline.py --input /path/to/raw_data --output /path/to/output

    # Or with conda env:
    conda run -n bigstream python run_pipeline.py \
        --input /path/to/raw_data --output /path/to/output
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import (
    find_qptiff_files,
    load_channel_names,
    load_or_create_downsampled,
)
from segmentation import segment_tissues, segment_all_slices
from matching import (
    match_tissues_across_slices,
    apply_matching,
)
from cropping import (
    compute_dilated_masks,
    compute_bounding_boxes,
    crop_and_save_tissues,
)
from visualization import (
    generate_segmentation_overlay,
    generate_matching_comparison,
    generate_bbox_overlay,
    generate_tissue_crop_gifs,
    generate_tissue_crop_previews,
)


def main(
    input_dir: str,
    output_dir: str,
    num_tissues: int = 8,
    padding: int = 50,
    skip_crop: bool = False,
):
    """
    Main pipeline: segment, match, crop, stack, and save tissue volumes.

    Args:
        input_dir: Directory containing qptiff files and channelnames.txt
        output_dir: Directory to write output OME-TIFF files and QC images
        num_tissues: Expected number of tissues per image
        padding: Padding around bounding boxes in full-res pixels
        skip_crop: If True, only run segmentation + visualization (no cropping)
    """
    total_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    qc_dir = os.path.join(output_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    # =========================================================
    # Step 1: Discover input files
    # =========================================================
    print("=" * 60)
    print("STEP 1: Discovering input files")
    print("=" * 60)

    qptiff_paths = find_qptiff_files(input_dir)
    channel_names = load_channel_names(input_dir)
    num_channels = len(channel_names)
    num_slices = len(qptiff_paths)

    print(f"  {num_slices} z-slices, {num_channels} channels, "
          f"expecting {num_tissues} tissues")

    # =========================================================
    # Step 2: Load downsampled sum-projections
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 2: Loading downsampled images")
    print("=" * 60)

    downsampled_images, sfx, sfy = load_or_create_downsampled(
        input_dir, qptiff_paths
    )
    print(f"  Scale factors: X={sfx}, Y={sfy}")

    # =========================================================
    # Step 3: Segment tissues across all slices (with guidance)
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 3: Segmenting tissues (cross-slice guided)")
    print("=" * 60)

    segmentations = segment_all_slices(
        downsampled_images, num_tissues=num_tissues
    )

    # Save raw segmentation QC
    print("\nGenerating raw segmentation overlays...")
    generate_segmentation_overlay(
        downsampled_images, segmentations, qc_dir, matched=False
    )

    # =========================================================
    # Step 4: Match tissues across slices
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 4: Matching tissues across slices")
    print("=" * 60)

    # Since segment_all_slices uses the reference slice's labels
    # for guided segmentation, tissues already have corresponding
    # IDs. We still run matching for QC and to handle any edge cases.
    mappings = match_tissues_across_slices(segmentations)
    matched_segmentations = apply_matching(segmentations, mappings)

    # Save matched segmentation QC
    print("\nGenerating matched segmentation overlays...")
    generate_segmentation_overlay(
        downsampled_images, matched_segmentations, qc_dir, matched=True
    )
    generate_matching_comparison(
        downsampled_images, matched_segmentations, qc_dir
    )

    # =========================================================
    # Step 5: Compute dilated masks and bounding boxes
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 5: Computing dilated masks and bounding boxes")
    print("=" * 60)

    # Get full-resolution shapes
    import tifffile
    full_res_shapes = []
    for qp in qptiff_paths:
        with tifffile.TiffFile(str(qp)) as t:
            h, w = t.pages[0].shape
            full_res_shapes.append((h, w))
            print(f"  {qp.name}: full-res {w}x{h}")

    dilated_masks = compute_dilated_masks(
        matched_segmentations,
        num_tissues=num_tissues,
        dilation_px=15,
        exclusion_erode_px=3,
    )

    tissue_bboxes = compute_bounding_boxes(
        dilated_masks,
        scale_factor_x=sfx,
        scale_factor_y=sfy,
        padding=padding,
        full_res_shapes=full_res_shapes,
    )

    # Save bbox QC
    generate_bbox_overlay(
        downsampled_images, matched_segmentations,
        tissue_bboxes, sfx, sfy, qc_dir,
    )

    # =========================================================
    # Step 5b: Generate crop-preview QC (GIFs + montage PNGs)
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 5b: Generating crop-preview QC")
    print("=" * 60)

    print("\nGenerating per-tissue stack GIFs...")
    generate_tissue_crop_gifs(
        downsampled_images, dilated_masks, tissue_bboxes,
        sfx, sfy, qc_dir,
    )

    print("\nGenerating per-tissue preview montages...")
    generate_tissue_crop_previews(
        downsampled_images, dilated_masks, tissue_bboxes,
        sfx, sfy, qc_dir,
    )

    if skip_crop:
        print("\n*** skip_crop=True: stopping before full-res cropping ***")
        elapsed = time.time() - total_start
        print(f"\nSegmentation + matching completed in {elapsed:.1f}s")
        return

    # =========================================================
    # Step 6: Crop at full resolution and save to disk
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 6: Cropping at full resolution and saving to disk")
    print("=" * 60)

    output_paths = crop_and_save_tissues(
        qptiff_paths=qptiff_paths,
        matched_segmentations=matched_segmentations,
        dilated_masks=dilated_masks,
        tissue_bboxes=tissue_bboxes,
        num_channels=num_channels,
        scale_factor_x=sfx,
        scale_factor_y=sfy,
        num_tissues=num_tissues,
        output_dir=output_dir,
        channel_names=channel_names,
    )

    # =========================================================
    # Summary
    # =========================================================
    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output directory: {output_dir}")
    print(f"  Tissues saved: {len(output_paths)}")
    print(f"  QC images: {qc_dir}")
    for tissue_id, path in sorted(output_paths.items()):
        if os.path.exists(path):
            size_gb = os.path.getsize(path) / 1e9
            print(f"    tissue_{tissue_id}_stacked.ome.tif: {size_gb:.2f} GB")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tissue Segmentation Pipeline: segment, match, crop and "
                    "stack multichannel tissue sections from qptiff z-slices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing qptiff files and channelnames.txt",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for OME-TIFF files and QC images",
    )
    parser.add_argument(
        "--num-tissues", "-n",
        type=int,
        default=8,
        help="Expected number of tissues per image",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=50,
        help="Padding in full-res pixels around each tissue bounding box",
    )
    parser.add_argument(
        "--skip-crop",
        action="store_true",
        help="Only run segmentation + matching (no full-res cropping)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        input_dir=args.input,
        output_dir=args.output,
        num_tissues=args.num_tissues,
        padding=args.padding,
        skip_crop=args.skip_crop,
    )
