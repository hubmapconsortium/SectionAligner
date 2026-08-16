"""
Full-resolution cropping with dilated per-tissue masks.

Strategy (Option D, centered):
  For each tissue and each z-slice:
    1. Take the segmentation mask for that tissue at downsampled resolution.
    2. Dilate it generously to create an inclusion zone that fully contains
       the tissue with ample margin (no edge erosion).
    3. Subtract regions positively identified as OTHER tissues (with a small
       erosion buffer at shared boundaries) so neighbours don't leak in.
    4. Compute a per-slice TIGHT bbox + mask CENTROID for the tissue.
    5. Determine a uniform per-tissue canvas big enough to contain the
       tissue centered at the canvas center for every z (size driven by
       max half-radius from each z's centroid to its bbox edges, + padding).
  Then for each z:
    - Crop the per-slice tight bbox from the raw qptiff (always within
      slide bounds because the bbox was derived from data that exists
      on this slide -> NO clipping artifacts).
    - Paste into the uniform canvas with the tissue centroid landing at
      the canvas center (centroid-aligned rigid placement).
    - Apply the per-slice dilated mask within the placed region.

Why this is better than the previous union-bbox approach:
  Previously a SINGLE bbox was computed as the union of per-slice extents
  in absolute slide coordinates, then applied to every slide.  Because
  individual slides are mounted independently and have slightly different
  full-res dimensions (51120x32640, 50400x30720, etc.) and the tissue
  position on each slide differs, the union bbox routinely fell off the
  edge of smaller slides.  cv2 clipping then truncated the crop and
  offset the cropped data within the canvas, producing the appearance
  of "missing tissue" / "fragments" at z's where the slide happened to
  be small or the tissue happened to be positioned far from the slide
  center.  That bad layout in turn caused the downstream optical-flow
  alignment to attempt enormous (>=7000 px) registration shifts and
  introduce non-rigid warp artifacts.

  The per-slice tight crop + centroid-aligned placement keeps every z's
  tissue at the same canvas position by construction, so the alignment
  step (and everything downstream) sees a clean stack.

For backward compatibility the returned tissue_bboxes dict still carries
the old "y1, y2, x1, x2, height, width" union-extent fields so existing
visualization callers continue to work.  The new "per_slice", "canvas_h",
"canvas_w", and "target_center_y/x" fields drive the crop.
"""

import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Per-slice dilated tissue masks (at downsampled resolution)
# ---------------------------------------------------------------------------

def compute_dilated_masks(
    matched_segmentations: List[np.ndarray],
    num_tissues: int,
    dilation_px: int = 15,
    exclusion_erode_px: int = 3,
) -> Dict[int, List[np.ndarray]]:
    """
    Build a dilated inclusion mask for each tissue at each z-slice,
    all at the *downsampled* resolution.

    Args:
        matched_segmentations: list of label images (one per z-slice)
        num_tissues: number of tissue IDs (1..num_tissues)
        dilation_px: how many downsampled pixels to dilate the target
            tissue mask (controls the margin around the tissue)
        exclusion_erode_px: how many downsampled pixels to erode the
            other-tissue exclusion zones (safety buffer at boundaries)

    Returns:
        dict  tissue_id -> list[np.ndarray]  (one uint8 mask per z-slice,
        where 1 = keep, 0 = exclude)
    """
    print(f"Computing dilated masks (dilation={dilation_px}px, "
          f"exclusion_erode={exclusion_erode_px}px at downsampled res)...")

    dk = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_px + 1, 2 * dilation_px + 1),
    )
    ek = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * exclusion_erode_px + 1, 2 * exclusion_erode_px + 1),
    )

    masks = {}  # tissue_id -> [mask_z0, mask_z1, ...]

    for tissue_id in range(1, num_tissues + 1):
        tissue_masks = []
        for z_idx, seg in enumerate(matched_segmentations):
            target = (seg == tissue_id).astype(np.uint8)
            dilated = cv2.dilate(target, dk, iterations=1)

            others = np.zeros_like(seg, dtype=np.uint8)
            for lbl in np.unique(seg):
                if lbl == 0 or lbl == tissue_id:
                    continue
                others[seg == lbl] = 1
            if exclusion_erode_px > 0:
                others = cv2.erode(others, ek, iterations=1)

            keep = cv2.bitwise_and(dilated, 1 - others)
            tissue_masks.append(keep)

        masks[tissue_id] = tissue_masks

    return masks


# ---------------------------------------------------------------------------
# Bounding boxes from dilated masks
# ---------------------------------------------------------------------------

def compute_bounding_boxes(
    dilated_masks: Dict[int, List[np.ndarray]],
    scale_factor_x: int,
    scale_factor_y: int,
    padding: int = 50,
    full_res_shapes: Optional[List[Tuple[int, int]]] = None,
    centered: bool = True,
) -> Dict[int, Dict]:
    """
    Compute per-tissue cropping plan at full resolution from the dilated masks.

    centered=True (default, recommended):
      For each tissue, compute per-slice TIGHT bboxes and mask centroids
      in full-res coords, plus a single uniform target canvas that fits
      the tissue centered at the canvas center for every z.  Each z is
      then cropped from its OWN tight bbox (no off-slide clipping) and
      pasted at a centroid-aligned offset within the canvas.

    centered=False (legacy):
      Single union bbox per tissue.  Kept for reproducibility of older
      runs.  Suffers from off-slide clipping when slides have different
      dimensions or the tissue is at different positions per slide.

    Returns:
      dict[tissue_id] with keys:
        # backward-compat union-extent fields (still computed, for vis)
        y1, y2, x1, x2, height, width
        # new fields when centered=True
        canvas_h, canvas_w               # uniform per-z output dims
        target_center_y, target_center_x # where each centroid lands
        per_slice: list of dicts, one per z, each with:
          y1, y2, x1, x2  (per-slice tight bbox in full-res coords;
                            None if mask empty for that z)
          centroid_y, centroid_x  (full-res, float)
          paste_oy, paste_ox  (top-left of the per-slice crop within
                                  the canvas; computed so that
                                  centroid lands at target_center)
    """
    mode = "centered (per-slice tight crop)" if centered else "legacy union-bbox"
    print(f"Computing full-resolution bounding boxes ({mode})...")

    if full_res_shapes:
        max_h = max(s[0] for s in full_res_shapes)
        max_w = max(s[1] for s in full_res_shapes)
    else:
        max_h = max_w = None

    tissue_bboxes = {}

    for tissue_id, per_slice_masks in dilated_masks.items():
        per_slice_info = []
        u_y1_min, u_x1_min = float("inf"), float("inf")
        u_y2_max, u_x2_max = 0, 0

        for z_idx, mask in enumerate(per_slice_masks):
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                per_slice_info.append({
                    "y1": None, "y2": None, "x1": None, "x2": None,
                    "centroid_y": None, "centroid_x": None,
                })
                continue

            y1_ds, y2_ds = int(ys.min()), int(ys.max())
            x1_ds, x2_ds = int(xs.min()), int(xs.max())
            cy_ds = float(ys.mean())
            cx_ds = float(xs.mean())

            y1_full = int(y1_ds * scale_factor_y) - padding
            x1_full = int(x1_ds * scale_factor_x) - padding
            y2_full = int((y2_ds + 1) * scale_factor_y) + padding
            x2_full = int((x2_ds + 1) * scale_factor_x) + padding
            cy_full = cy_ds * scale_factor_y
            cx_full = cx_ds * scale_factor_x

            y1_full = max(0, y1_full)
            x1_full = max(0, x1_full)
            if max_h is not None:
                y2_full = min(max_h, y2_full)
            if max_w is not None:
                x2_full = min(max_w, x2_full)

            per_slice_info.append({
                "y1": y1_full, "y2": y2_full,
                "x1": x1_full, "x2": x2_full,
                "centroid_y": cy_full,
                "centroid_x": cx_full,
            })

            u_y1_min = min(u_y1_min, y1_full)
            u_x1_min = min(u_x1_min, x1_full)
            u_y2_max = max(u_y2_max, y2_full)
            u_x2_max = max(u_x2_max, x2_full)

        if u_y1_min == float("inf"):
            print(f"  WARNING: Tissue {tissue_id} mask is empty across all z!")
            continue

        union_h = u_y2_max - u_y1_min
        union_w = u_x2_max - u_x1_min

        if centered:
            half_h = 0
            half_w = 0
            for info in per_slice_info:
                if info["y1"] is None:
                    continue
                cy = info["centroid_y"]
                cx = info["centroid_x"]
                hh = max(cy - info["y1"], info["y2"] - cy)
                hw = max(cx - info["x1"], info["x2"] - cx)
                half_h = max(half_h, int(np.ceil(hh)))
                half_w = max(half_w, int(np.ceil(hw)))
            canvas_h = 2 * half_h
            canvas_w = 2 * half_w
            target_cy = half_h
            target_cx = half_w

            for info in per_slice_info:
                if info["y1"] is None:
                    info["paste_oy"] = None
                    info["paste_ox"] = None
                    continue
                info["paste_oy"] = int(round(target_cy - info["centroid_y"] + info["y1"]))
                info["paste_ox"] = int(round(target_cx - info["centroid_x"] + info["x1"]))

            tissue_bboxes[tissue_id] = {
                "y1": u_y1_min, "y2": u_y2_max,
                "x1": u_x1_min, "x2": u_x2_max,
                "height": union_h, "width": union_w,
                "canvas_h": canvas_h, "canvas_w": canvas_w,
                "target_center_y": target_cy,
                "target_center_x": target_cx,
                "per_slice": per_slice_info,
                "centered": True,
            }

            size_gpx = canvas_h * canvas_w / 1e9
            union_gpx = union_h * union_w / 1e9
            valid_z = sum(1 for i in per_slice_info if i["y1"] is not None)
            print(
                f"  Tissue {tissue_id}: canvas={canvas_w}x{canvas_h} "
                f"({size_gpx:.3f} Gpx) <- {valid_z}/{len(per_slice_info)} z "
                f"valid; union extent {union_w}x{union_h} ({union_gpx:.3f} Gpx)"
            )
        else:
            tissue_bboxes[tissue_id] = {
                "y1": u_y1_min, "y2": u_y2_max,
                "x1": u_x1_min, "x2": u_x2_max,
                "height": union_h, "width": union_w,
                "centered": False,
            }
            size_gpx = union_h * union_w / 1e9
            print(
                f"  Tissue {tissue_id}: union bbox=({u_x1_min},{u_y1_min})-"
                f"({u_x2_max},{u_y2_max}) size={union_w}x{union_h} "
                f"({size_gpx:.3f} Gpx)"
            )

    return tissue_bboxes


# ---------------------------------------------------------------------------
# Upscale a single downsampled mask to full resolution
# ---------------------------------------------------------------------------

def _upscale_mask(
    mask_ds: np.ndarray,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """Upscale a uint8 binary mask via nearest-neighbor."""
    return cv2.resize(
        mask_ds,
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


# ---------------------------------------------------------------------------
# Main crop-and-save
# ---------------------------------------------------------------------------

def crop_and_save_tissues(
    qptiff_paths: List[Path],
    matched_segmentations: List[np.ndarray],
    dilated_masks: Dict[int, List[np.ndarray]],
    tissue_bboxes: Dict[int, Dict],
    num_channels: int,
    scale_factor_x: int,
    scale_factor_y: int,
    num_tissues: int,
    output_dir: str,
    channel_names: List[str],
    compression: Optional[str] = "zlib",
    compression_level: int = 1,
) -> Dict[int, str]:
    """
    Crop full-resolution multichannel data for each tissue using per-slice
    dilated masks and write directly to OME-TIFF (single ZCYX series).

    If `tissue_bboxes[tid]` carries the new "per_slice" + "canvas_h/w" +
    "target_center_y/x" fields (i.e. produced by compute_bounding_boxes
    with centered=True), this function does CENTROID-ALIGNED placement:
    each z is cropped from its OWN tight bbox (no off-slide clipping)
    and pasted at an offset that lands the per-slice tissue centroid
    at the canvas center.

    Otherwise it falls back to the legacy union-bbox + clip-to-slide
    behavior that produces position artifacts when slides have different
    dimensions.

    Memory-efficient: yields one YX plane at a time via a generator so the
    entire volume never resides in memory.

    `compression`: tifffile compression name (default 'zlib' / deflate),
    or None to write uncompressed.  Compression matters: at 60 channels x
    17 z this turns a ~470 GB uncompressed write into ~40 GB on disk.
    """
    num_slices = len(qptiff_paths)
    output_paths = {}

    for tissue_id, bbox in tissue_bboxes.items():
        centered_mode = bool(bbox.get("centered", False)) and "per_slice" in bbox

        if centered_mode:
            h = bbox["canvas_h"]
            w = bbox["canvas_w"]
            target_cy = bbox["target_center_y"]
            target_cx = bbox["target_center_x"]
            per_slice = bbox["per_slice"]
        else:
            h, w = bbox["height"], bbox["width"]
            target_cy = target_cx = None
            per_slice = None

        est_gb = num_slices * num_channels * h * w / 1e9
        mode_label = "CENTERED" if centered_mode else "legacy union-bbox"
        print(f"\nProcessing tissue {tissue_id}/{num_tissues} "
              f"({w}x{h}, {num_slices} slices, {num_channels} channels, "
              f"~{est_gb:.1f} GB raw) [{mode_label}]...")

        output_path = os.path.join(
            output_dir, f"tissue_{tissue_id}_stacked.ome.tif"
        )
        output_paths[tissue_id] = output_path

        per_slice_masks_ds = dilated_masks[tissue_id]

        def _plane_generator(
            _bbox=bbox, _h=h, _w=w,
            _masks_ds=per_slice_masks_ds, _tid=tissue_id,
            _centered=centered_mode, _per_slice=per_slice,
        ):
            for z_idx, qptiff_path in enumerate(qptiff_paths):
                print(f"  Slice {z_idx + 1}/{num_slices}: "
                      f"{qptiff_path.name}")

                with tifffile.TiffFile(str(qptiff_path)) as t:
                    full_h = t.pages[0].shape[0]
                    full_w = t.pages[0].shape[1]

                if _centered:
                    info = _per_slice[z_idx]
                    if info["y1"] is None:
                        print("    Skipping (empty mask for this z)")
                        blank = np.zeros((_h, _w), dtype=np.uint8)
                        for _ in range(num_channels):
                            yield blank
                        continue

                    sy1 = max(0, min(info["y1"], full_h))
                    sy2 = max(0, min(info["y2"], full_h))
                    sx1 = max(0, min(info["x1"], full_w))
                    sx2 = max(0, min(info["x2"], full_w))
                    crop_h = sy2 - sy1
                    crop_w = sx2 - sx1
                    if crop_h <= 0 or crop_w <= 0:
                        print(
                            "    Skipping (per-slice tight bbox is empty "
                            "after clip - this should not happen)"
                        )
                        blank = np.zeros((_h, _w), dtype=np.uint8)
                        for _ in range(num_channels):
                            yield blank
                        continue

                    oy = info["paste_oy"] + (sy1 - info["y1"])
                    ox = info["paste_ox"] + (sx1 - info["x1"])

                    if oy < 0 or ox < 0 or \
                            oy + crop_h > _h or ox + crop_w > _w:
                        ty1 = max(0, oy)
                        ty2 = min(_h, oy + crop_h)
                        tx1 = max(0, ox)
                        tx2 = min(_w, ox + crop_w)
                        sub_oy = ty1 - oy
                        sub_ox = tx1 - ox
                        sub_sy1 = sy1 + sub_oy
                        sub_sx1 = sx1 + sub_ox
                        sy1, sx1 = sub_sy1, sub_sx1
                        sy2 = sy1 + (ty2 - ty1)
                        sx2 = sx1 + (tx2 - tx1)
                        crop_h = ty2 - ty1
                        crop_w = tx2 - tx1
                        oy, ox = ty1, tx1
                        if crop_h <= 0 or crop_w <= 0:
                            print(
                                "    WARN: paste fully outside canvas "
                                "after clip; emitting blanks"
                            )
                            blank = np.zeros((_h, _w), dtype=np.uint8)
                            for _ in range(num_channels):
                                yield blank
                            continue
                else:
                    _y1 = _bbox["y1"]; _x1 = _bbox["x1"]
                    _y2 = _bbox["y2"]; _x2 = _bbox["x2"]
                    sy1 = max(0, min(_y1, full_h))
                    sx1 = max(0, min(_x1, full_w))
                    sy2 = min(_y2, full_h)
                    sx2 = min(_x2, full_w)
                    crop_h = sy2 - sy1
                    crop_w = sx2 - sx1
                    if crop_h <= 0 or crop_w <= 0:
                        print("    Skipping (bbox outside this slice)")
                        blank = np.zeros((_h, _w), dtype=np.uint8)
                        for _ in range(num_channels):
                            yield blank
                        continue
                    oy = sy1 - _y1
                    ox = sx1 - _x1

                mask_full = _upscale_mask(
                    _masks_ds[z_idx], (full_h, full_w)
                )
                mask_crop = mask_full[sy1:sy2, sx1:sx2]
                del mask_full

                with tifffile.TiffFile(str(qptiff_path)) as t:
                    for ch in range(num_channels):
                        if ch % 10 == 0:
                            print(
                                f"    Channel {ch + 1}/{num_channels}..."
                            )

                        page_data = t.pages[ch].asarray()
                        cropped = page_data[sy1:sy2, sx1:sx2]
                        cropped = cropped * mask_crop

                        plane = np.zeros((_h, _w), dtype=np.uint8)
                        plane[oy:oy + crop_h, ox:ox + crop_w] = cropped
                        yield plane

                        del page_data, cropped, plane

                del mask_crop
                gc.collect()

        writer_kwargs = {}
        if compression is not None:
            writer_kwargs["compression"] = compression
            if compression_level is not None:
                writer_kwargs["compressionargs"] = {"level": compression_level}

        with tifffile.TiffWriter(
            output_path, bigtiff=True, ome=True
        ) as tw:
            tw.write(
                data=_plane_generator(),
                shape=(num_slices, num_channels, h, w),
                dtype=np.uint8,
                metadata={
                    "axes": "ZCYX",
                    "Channel": {"Name": channel_names},
                    "PhysicalSizeX": 0.5073519424785282,
                    "PhysicalSizeY": 0.5073519424785282,
                    "PhysicalSizeZ": 1.0,
                },
                **writer_kwargs,
            )

        size_gb = os.path.getsize(output_path) / 1e9
        print(f"  Tissue {tissue_id}: saved {output_path} "
              f"({size_gb:.2f} GB)")

    return output_paths
