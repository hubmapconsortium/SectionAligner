"""
QC visualization: generate color-coded tissue overlays and cross-slice
matching diagrams for visual verification.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import imageio.v3 as iio
import numpy as np


# Distinct colors for up to 12 tissues (BGR format for cv2)
TISSUE_COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (128, 255, 0),    # Spring green
    (0, 128, 255),    # Orange
    (255, 128, 0),    # Sky blue
    (128, 0, 255),    # Violet
    (0, 255, 128),    # Mint
    (255, 128, 128),  # Light blue
]


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize an image to 0-255 uint8 range using 1st-99th percentile."""
    p1, p99 = np.percentile(image, [1, 99])
    result = np.clip((image - p1) / (p99 - p1 + 1e-8) * 255, 0, 255)
    return result.astype(np.uint8)


def generate_segmentation_overlay(
    images: List[np.ndarray],
    label_images: List[np.ndarray],
    output_dir: str,
    matched: bool = True,
    alpha: float = 0.45,
) -> List[str]:
    """
    Generate color-coded overlay images showing tissue segmentation results.

    Args:
        images: List of 2D downsampled sum-projection images
        label_images: List of label images (0=bg, 1..N=tissue)
        output_dir: Directory to save overlay images
        matched: Whether labels are already matched across slices
        alpha: Blending factor for overlay

    Returns:
        List of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    for i, (img, labels) in enumerate(zip(images, label_images)):
        # Normalize background image
        bg = normalize_to_uint8(img)
        overlay = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        for lbl in unique_labels:
            mask = labels == lbl
            color = TISSUE_COLORS[(lbl - 1) % len(TISSUE_COLORS)]

            # Blend color onto tissue region
            overlay[mask] = (
                overlay[mask].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

            # Draw tissue label at centroid
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                text = f"T{lbl}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                thickness = 4

                # Text with outline for readability
                cv2.putText(
                    overlay, text, (cx - 30, cy + 15),
                    font, font_scale, (0, 0, 0), thickness + 2,
                )
                cv2.putText(
                    overlay, text, (cx - 30, cy + 15),
                    font, font_scale, (255, 255, 255), thickness,
                )

        # Draw tissue contours
        for lbl in unique_labels:
            mask = (labels == lbl).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            color = TISSUE_COLORS[(lbl - 1) % len(TISSUE_COLORS)]
            cv2.drawContours(overlay, contours, -1, color, 2)

        suffix = "matched" if matched else "raw"
        filename = f"slice_{i}_{suffix}_segmentation.png"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, overlay)
        saved_paths.append(path)
        print(f"  Saved segmentation overlay: {filename}")

    return saved_paths


def generate_matching_comparison(
    images: List[np.ndarray],
    label_images: List[np.ndarray],
    output_dir: str,
    alpha: float = 0.45,
) -> str:
    """
    Generate a side-by-side comparison image showing tissue matching
    across all z-slices, with consistent colors for matched tissues.

    Args:
        images: List of downsampled sum-projection images
        label_images: List of matched label images
        output_dir: Directory to save

    Returns:
        Path to saved comparison image
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create individual overlays
    overlays = []
    for img, labels in zip(images, label_images):
        bg = normalize_to_uint8(img)
        overlay = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]

        for lbl in unique_labels:
            mask = labels == lbl
            color = TISSUE_COLORS[(lbl - 1) % len(TISSUE_COLORS)]
            overlay[mask] = (
                overlay[mask].astype(np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

            # Label text
            ys, xs = np.where(mask)
            if len(ys) > 0:
                cy, cx = int(ys.mean()), int(xs.mean())
                text = f"T{lbl}"
                cv2.putText(
                    overlay, text, (cx - 30, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6,
                )
                cv2.putText(
                    overlay, text, (cx - 30, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4,
                )

        overlays.append(overlay)

    # Add slice labels at top
    for idx, overlay in enumerate(overlays):
        cv2.putText(
            overlay, f"Slice {idx}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4,
        )

    # Resize all to same height and stack horizontally
    target_h = max(o.shape[0] for o in overlays)
    resized = []
    for o in overlays:
        if o.shape[0] != target_h:
            scale = target_h / o.shape[0]
            new_w = int(o.shape[1] * scale)
            o = cv2.resize(o, (new_w, target_h))
        resized.append(o)

    # Add gap between slices
    gap = np.zeros((target_h, 10, 3), dtype=np.uint8)
    parts = []
    for j, r in enumerate(resized):
        parts.append(r)
        if j < len(resized) - 1:
            parts.append(gap)

    comparison = np.hstack(parts)
    path = os.path.join(output_dir, "tissue_matching_comparison.png")
    cv2.imwrite(path, comparison)
    print(f"  Saved matching comparison: tissue_matching_comparison.png")
    return path


def generate_bbox_overlay(
    images: List[np.ndarray],
    label_images: List[np.ndarray],
    tissue_bboxes: Dict[int, Dict],
    scale_factor_x: int,
    scale_factor_y: int,
    output_dir: str,
) -> str:
    """
    Generate overlay showing bounding boxes for each tissue at downsampled
    resolution for QC verification.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, (img, labels) in enumerate(zip(images, label_images)):
        bg = normalize_to_uint8(img)
        overlay = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        for tissue_id, bbox in tissue_bboxes.items():
            color = TISSUE_COLORS[(tissue_id - 1) % len(TISSUE_COLORS)]

            # Prefer the per-slice tight bbox (centered mode); fall back to
            # the legacy union bbox if no per-slice info is available.
            per_slice = bbox.get("per_slice")
            if per_slice is not None and i < len(per_slice) \
                    and per_slice[i]["y1"] is not None:
                info = per_slice[i]
                y1 = info["y1"] // scale_factor_y
                x1 = info["x1"] // scale_factor_x
                y2 = info["y2"] // scale_factor_y
                x2 = info["x2"] // scale_factor_x
                # Mark the per-slice centroid too (small filled circle)
                cy = int(info["centroid_y"] // scale_factor_y)
                cx = int(info["centroid_x"] // scale_factor_x)
                cv2.circle(overlay, (cx, cy), 6, color, -1)
            else:
                y1 = bbox["y1"] // scale_factor_y
                x1 = bbox["x1"] // scale_factor_x
                y2 = bbox["y2"] // scale_factor_y
                x2 = bbox["x2"] // scale_factor_x

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                overlay, f"T{tissue_id}", (x1 + 5, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3,
            )

        path = os.path.join(output_dir, f"slice_{i}_bboxes.png")
        cv2.imwrite(path, overlay)
        print(f"  Saved bbox overlay: slice_{i}_bboxes.png")

    return output_dir


def _crop_tissue_frames(
    images: List[np.ndarray],
    dilated_masks: Dict[int, List[np.ndarray]],
    tissue_bboxes: Dict[int, Dict],
    scale_factor_x: int,
    scale_factor_y: int,
    tissue_id: int,
) -> List[np.ndarray]:
    """
    Produce a list of RGB uint8 frames (one per z-slice) showing only the
    masked tissue on a black background.

    Uses the same bounding box as the real full-res crop (scaled to
    downsampled coordinates) so the preview faithfully represents the
    actual output canvas dimensions.  Everything outside the dilated mask
    is zeroed out.
    """
    if tissue_id not in tissue_bboxes:
        return []

    bbox = tissue_bboxes[tissue_id]
    centered_mode = bool(bbox.get("centered", False)) and "per_slice" in bbox

    if centered_mode:
        canvas_h = bbox["canvas_h"] // scale_factor_y
        canvas_w = bbox["canvas_w"] // scale_factor_x
        target_cy = bbox["target_center_y"] // scale_factor_y
        target_cx = bbox["target_center_x"] // scale_factor_x
        per_slice_info = bbox["per_slice"]
    else:
        y1 = bbox["y1"] // scale_factor_y
        x1 = bbox["x1"] // scale_factor_x
        y2 = bbox["y2"] // scale_factor_y
        x2 = bbox["x2"] // scale_factor_x
        canvas_h = y2 - y1
        canvas_w = x2 - x1

    if canvas_h <= 0 or canvas_w <= 0:
        return []

    per_slice_masks = dilated_masks[tissue_id]
    color = TISSUE_COLORS[(tissue_id - 1) % len(TISSUE_COLORS)]

    frames = []
    for z_idx, img in enumerate(images):
        h, w = img.shape[:2]

        if centered_mode:
            info = per_slice_info[z_idx]
            if info["y1"] is None:
                # Empty mask for this z; emit a blank-frame label
                frames.append(np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8))
                continue
            sy1 = max(0, min(info["y1"] // scale_factor_y, h))
            sx1 = max(0, min(info["x1"] // scale_factor_x, w))
            sy2 = max(0, min(info["y2"] // scale_factor_y, h))
            sx2 = max(0, min(info["x2"] // scale_factor_x, w))
            ds_paste_oy = info["paste_oy"] // scale_factor_y \
                + (sy1 - info["y1"] // scale_factor_y)
            ds_paste_ox = info["paste_ox"] // scale_factor_x \
                + (sx1 - info["x1"] // scale_factor_x)
            cy1, cx1, cy2, cx2 = sy1, sx1, sy2, sx2
            oy, ox = ds_paste_oy, ds_paste_ox
        else:
            cy1 = max(0, min(y1, h))
            cx1 = max(0, min(x1, w))
            cy2 = min(y2, h)
            cx2 = min(x2, w)
            oy = cy1 - y1
            ox = cx1 - x1

        frame = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        crop = img[cy1:cy2, cx1:cx2]
        bg = normalize_to_uint8(crop)
        bg_bgr = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        mask_crop = per_slice_masks[z_idx][cy1:cy2, cx1:cx2]
        bg_bgr[mask_crop == 0] = 0

        ch, cw = bg_bgr.shape[:2]
        # Clip paste to canvas bounds (defensive)
        ty1 = max(0, oy); ty2 = min(canvas_h, oy + ch)
        tx1 = max(0, ox); tx2 = min(canvas_w, ox + cw)
        sub_dy = ty1 - oy; sub_dx = tx1 - ox
        sub_h = ty2 - ty1; sub_w = tx2 - tx1
        if sub_h > 0 and sub_w > 0:
            frame[ty1:ty2, tx1:tx2] = bg_bgr[
                sub_dy:sub_dy + sub_h, sub_dx:sub_dx + sub_w
            ]

        mask_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if sub_h > 0 and sub_w > 0:
            mask_canvas[ty1:ty2, tx1:tx2] = mask_crop[
                sub_dy:sub_dy + sub_h, sub_dx:sub_dx + sub_w
            ]
        contours, _ = cv2.findContours(
            mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, color, 1)

        label = f"T{tissue_id} z={z_idx}"
        font_scale = max(0.5, min(canvas_w, canvas_h) / 800)
        thickness = max(1, int(font_scale * 2))
        cv2.putText(
            frame, label, (6, int(25 * font_scale + 8)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2,
        )
        cv2.putText(
            frame, label, (6, int(25 * font_scale + 8)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return frames


def generate_tissue_crop_gifs(
    images: List[np.ndarray],
    dilated_masks: Dict[int, List[np.ndarray]],
    tissue_bboxes: Dict[int, Dict],
    scale_factor_x: int,
    scale_factor_y: int,
    output_dir: str,
    fps: int = 3,
) -> Dict[int, str]:
    """
    For each tissue, create an animated GIF that cycles through every z-slice
    showing the cropped, masked region — a preview of what the final output
    stack looks like.

    Returns:
        dict  tissue_id -> path to saved GIF
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    for tissue_id in sorted(tissue_bboxes.keys()):
        frames = _crop_tissue_frames(
            images, dilated_masks, tissue_bboxes,
            scale_factor_x, scale_factor_y, tissue_id,
        )

        if not frames:
            continue

        path = os.path.join(output_dir, f"tissue_{tissue_id}_stack.gif")
        duration = 1000 // fps
        iio.imwrite(
            path,
            frames,
            extension=".gif",
            duration=duration,
            loop=0,
        )
        saved[tissue_id] = path
        print(f"  Saved tissue stack GIF: tissue_{tissue_id}_stack.gif "
              f"({len(frames)} frames)")

    return saved


def generate_tissue_crop_previews(
    images: List[np.ndarray],
    dilated_masks: Dict[int, List[np.ndarray]],
    tissue_bboxes: Dict[int, Dict],
    scale_factor_x: int,
    scale_factor_y: int,
    output_dir: str,
    max_cols: int = 6,
    thumb_max_side: int = 512,
) -> Dict[int, str]:
    """
    For each tissue, create a single PNG montage that tiles all z-slice
    crops into a grid — a compact overview of the whole stack.

    Returns:
        dict  tissue_id -> path to saved PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    for tissue_id in sorted(tissue_bboxes.keys()):
        frames = _crop_tissue_frames(
            images, dilated_masks, tissue_bboxes,
            scale_factor_x, scale_factor_y, tissue_id,
        )

        if not frames:
            continue

        fh, fw = frames[0].shape[:2]
        scale = min(1.0, thumb_max_side / max(fh, fw))
        th = max(1, int(fh * scale))
        tw = max(1, int(fw * scale))

        thumbs = []
        for f in frames:
            t = cv2.resize(
                cv2.cvtColor(f, cv2.COLOR_RGB2BGR), (tw, th),
                interpolation=cv2.INTER_AREA,
            )
            thumbs.append(t)

        n = len(thumbs)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols

        gap = 4
        canvas_h = rows * th + (rows - 1) * gap
        canvas_w = cols * tw + (cols - 1) * gap
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for idx, thumb in enumerate(thumbs):
            r, c = divmod(idx, cols)
            y = r * (th + gap)
            x = c * (tw + gap)
            canvas[y:y + th, x:x + tw] = thumb

        path = os.path.join(output_dir, f"tissue_{tissue_id}_preview.png")
        cv2.imwrite(path, canvas)
        saved[tissue_id] = path
        print(f"  Saved tissue preview: tissue_{tissue_id}_preview.png "
              f"({cols}x{rows} grid, {n} slices)")

    return saved
