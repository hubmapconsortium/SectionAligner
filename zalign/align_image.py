#!/usr/bin/env python
"""
Image-only z-slice alignment -- no segmentation required.

For each consecutive pair, three levels of registration on raw image channels:
  1. Coarse -- phase correlation on downscaled DAPI for global (dy, dx)
  2. Refined -- multi-channel consensus phase correlation (DAPI, CYTOKERATIN, CD45)
  3. Fine -- dense optical flow on DAPI for local deformable correction (optional)

Modes
-----
  Single pair (default):
      Align --z_mov to --z_ref and save one aligned TIF.

  --all_consecutive:
      Estimate each pairwise transform on RAW slices (z_k -> z_{k-1}) and
      compose them cumulatively so every slice is warped directly into
      z0's frame with a SINGLE interpolation. Output is one aligned stack
      (ZCYX) whose slices share a common reference (z0).

Usage
-----
    python align_image.py --input inputs/tissue_1_stacked.ome.tif
    python align_image.py --input inputs/tissue_1_stacked.ome.tif --skip_optical_flow
    python align_image.py --input inputs/tissue_stack.ome.tif --all_consecutive
"""

import argparse
import logging
import os
import sys
import time

import cv2
import numpy as np
import tifffile
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from utils import read_ome_tiff_metadata, read_zslice_all_channels, read_zslice_channels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Channel indices
DAPI_CH = 0
CYTOKERATIN_CH = 37
CD45_CH = 49
CONSENSUS_CHANNELS = [DAPI_CH, CYTOKERATIN_CH, CD45_CH]

# Downscale factor for registration (saves memory on large images)
DOWNSCALE = 4

# OpenCV remap() requires src and dst dimensions < SHRT_MAX (32768).
CV_REMAP_TILE = 28000


# ===================================================================
# Registration
# ===================================================================

def coarse_phase_correlation(ch0, ch1, downscale=8):
    """
    Coarse global shift via phase correlation on heavily downscaled images.
    Returns (dy, dx) in full-resolution pixels.
    """
    logger.info("Level 1: Coarse phase correlation (ds=%d) ...", downscale)
    r0 = ch0[::downscale, ::downscale].astype(np.float32)
    r1 = ch1[::downscale, ::downscale].astype(np.float32)
    shift_ds, _, _ = phase_cross_correlation(r0, r1, upsample_factor=10)
    shift = shift_ds * downscale
    logger.info("  Coarse shift (dy, dx): (%.2f, %.2f)", shift[0], shift[1])
    return shift[0], shift[1]


def refined_phase_correlation(img0, img1_shifted, channel_indices, downscale=4):
    """
    Multi-channel consensus phase correlation for sub-pixel refinement.
    img0, img1_shifted: (H, W, C) -- img1 already coarsely shifted.
    Returns residual (dy, dx).
    """
    logger.info("Level 2: Refined phase correlation (ds=%d, %d channels) ...",
                downscale, len(channel_indices))
    shifts = []
    for ci in channel_indices:
        c0 = img0[::downscale, ::downscale, ci].astype(np.float32)
        c1 = img1_shifted[::downscale, ::downscale, ci].astype(np.float32)
        shift_ds, _, _ = phase_cross_correlation(c0, c1, upsample_factor=20)
        shift_full = shift_ds * downscale
        logger.info("  Channel %d: residual (%.3f, %.3f)", ci, shift_full[0], shift_full[1])
        shifts.append(shift_full)

    avg = np.mean(shifts, axis=0)
    logger.info("  Consensus residual (dy, dx): (%.3f, %.3f)", avg[0], avg[1])
    return avg[0], avg[1]


def compute_optical_flow(ref, moving, downscale=4):
    """
    Dense optical flow (Farneback) between two single-channel images.
    Computed on downscaled images, then upscaled.
    Returns flow field (H, W, 2).
    """
    logger.info("Level 3: Dense optical flow (ds=%d) ...", downscale)
    ref_ds = cv2.resize(ref, None, fx=1.0 / downscale, fy=1.0 / downscale,
                        interpolation=cv2.INTER_AREA)
    mov_ds = cv2.resize(moving, None, fx=1.0 / downscale, fy=1.0 / downscale,
                        interpolation=cv2.INTER_AREA)

    ref_ds = np.clip(ref_ds, 0, 255).astype(np.uint8)
    mov_ds = np.clip(mov_ds, 0, 255).astype(np.uint8)

    flow_ds = cv2.calcOpticalFlowFarneback(
        ref_ds, mov_ds, flow=None,
        pyr_scale=0.5, levels=5, winsize=21,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0,
    )

    h, w = ref.shape[:2]
    flow = cv2.resize(flow_ds, (w, h), interpolation=cv2.INTER_LINEAR) * downscale
    mag = np.sqrt((flow ** 2).sum(axis=-1)).mean()
    logger.info("  Flow shape: %s, mean magnitude: %.2f px", flow.shape, mag)
    return flow


# ===================================================================
# Transform helpers
# ===================================================================

def cv2_remap_tiled(src, map_x, map_y, interpolation, border_mode,
                    border_value=None):
    """
    cv2.remap in spatial tiles so output dimensions can exceed OpenCV's
    SHRT_MAX (~32k) limit. `map_x` / `map_y` must match the desired output
    shape; `src` is the source image (2D or 3D with channel-last layout).
    """
    out_h, out_w = map_x.shape
    src_h, src_w = src.shape[:2]
    if (out_h < 32767 and out_w < 32767 and
            src_h < 32767 and src_w < 32767):
        kw = dict(
            interpolation=interpolation, borderMode=border_mode,
        )
        if border_value is not None:
            kw["borderValue"] = border_value
        return cv2.remap(src, map_x, map_y, **kw)

    if src.ndim == 2:
        out = np.empty((out_h, out_w), dtype=src.dtype)
    else:
        out = np.empty((out_h, out_w, src.shape[-1]), dtype=src.dtype)

    tile = CV_REMAP_TILE
    for rs in range(0, out_h, tile):
        re = min(rs + tile, out_h)
        for cs in range(0, out_w, tile):
            ce = min(cs + tile, out_w)
            mx = map_x[rs:re, cs:ce]
            my = map_y[rs:re, cs:ce]
            kw = dict(
                interpolation=interpolation, borderMode=border_mode,
            )
            if border_value is not None:
                kw["borderValue"] = border_value
            patch = cv2.remap(src, mx, my, **kw)
            if src.ndim == 2:
                out[rs:re, cs:ce] = patch
            else:
                out[rs:re, cs:ce, :] = patch
    return out


def estimate_pair_transform(img_ref, img_mov, skip_optical_flow=False):
    """
    Estimate rigid shift (+ optional dense flow) to align img_mov to img_ref.

    Parameters
    ----------
    img_ref, img_mov : (H, W, C) uint8
        Channel stacks of CONSENSUS_CHANNELS (channel 0 = DAPI).
    skip_optical_flow : bool
        If True, only estimate the rigid shift.

    Returns
    -------
    dict with keys:
        coarse_dy, coarse_dx, residual_dy, residual_dx,
        total_dy, total_dx, flow (or None),
        dapi_ref, dapi_mov_before, dapi_mov_rigid  -- DAPI snapshots for QC.
    """
    coarse_dy, coarse_dx = coarse_phase_correlation(
        img_ref[..., 0], img_mov[..., 0], downscale=8,
    )
    img_mov_coarse = shift_image(img_mov, coarse_dy, coarse_dx)
    residual_dy, residual_dx = refined_phase_correlation(
        img_ref, img_mov_coarse,
        channel_indices=list(range(img_ref.shape[-1])),
        downscale=DOWNSCALE,
    )
    total_dy = coarse_dy + residual_dy
    total_dx = coarse_dx + residual_dx
    img_mov_rigid = shift_image(img_mov, total_dy, total_dx)

    flow = None
    if not skip_optical_flow:
        flow = compute_optical_flow(
            img_ref[..., 0], img_mov_rigid[..., 0], downscale=DOWNSCALE,
        )

    return {
        "coarse_dy": coarse_dy, "coarse_dx": coarse_dx,
        "residual_dy": residual_dy, "residual_dx": residual_dx,
        "total_dy": total_dy, "total_dx": total_dx,
        "flow": flow,
        "dapi_ref": img_ref[..., 0],
        "dapi_mov_before": img_mov[..., 0],
        "dapi_mov_rigid": img_mov_rigid[..., 0],
    }


def compose_cumulative_step(cur_x, cur_y, pair_transform):
    """
    Compose a new pair transform into the running cumulative sampling map.

    `cur_x`, `cur_y` hold, for each pixel p in the z0 frame, the position in
    raw z_{k-1} to sample. After composing pair transform T_k (raw z_k -> raw
    z_{k-1}), the updated map gives positions in raw z_k to sample for the
    same z0 pixel p.

    The per-pair forward mapping (from the existing rigid + flow pipeline) is:
        sample z_k at (y - dy + flow_y(y, x), x - dx + flow_x(y, x))
    for pixel (y, x) in z_{k-1}'s frame. Composition recursion:
        M_k(p) = M_{k-1}(p) - shift_k + flow_k(M_{k-1}(p)).
    """
    flow = pair_transform["flow"]
    dy = np.float32(pair_transform["total_dy"])
    dx = np.float32(pair_transform["total_dx"])

    if flow is not None:
        flow_f = flow.astype(np.float32)
        flow_x_at = cv2_remap_tiled(
            flow_f[..., 0], cur_x, cur_y,
            cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE,
        )
        flow_y_at = cv2_remap_tiled(
            flow_f[..., 1], cur_x, cur_y,
            cv2.INTER_LINEAR, border_mode=cv2.BORDER_REPLICATE,
        )
        new_x = cur_x - dx + flow_x_at
        new_y = cur_y - dy + flow_y_at
    else:
        new_x = cur_x - dx
        new_y = cur_y - dy

    return new_x.astype(np.float32), new_y.astype(np.float32)


def apply_sampling_map(image, map_x, map_y):
    """Warp (H, W) or (H, W, C) image via a per-pixel sampling map.

    The output shape matches `map_x` (not the input image). Positions that
    fall outside the input image are filled with 0.
    """
    if image.ndim == 2:
        return cv2_remap_tiled(
            image, map_x, map_y, cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT, border_value=0,
        )
    out_h, out_w = map_x.shape
    warped = np.empty((out_h, out_w, image.shape[-1]), dtype=image.dtype)
    for c in range(image.shape[-1]):
        warped[..., c] = cv2_remap_tiled(
            image[..., c], map_x, map_y, cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT, border_value=0,
        )
    return warped


def compute_padded_canvas(pair_transforms, height, width, flow_safety=8.0):
    """
    Compute the output canvas size and top-left offset needed so that every
    raw slice fits completely after its cumulative warp into z0's frame --
    no clipping, no lost signal.

    A raw z_k pixel (y', x') lands in z0 frame at approximately
    (y' + cum_dy_k, x' + cum_dx_k), plus a per-pixel perturbation bounded by
    the composed optical flow magnitude. The union of those boxes across all
    k (including k=0) defines the padded canvas.

    Returns
    -------
    new_H, new_W : int
        Padded canvas size.
    top_pad, left_pad : int
        Offset from padded-canvas origin to the z0-frame origin.
    info : dict
        Per-axis min/max cumulative shift and flow margin used.
    """
    cum_dys = [0.0]
    cum_dxs = [0.0]
    cum_dy = 0.0
    cum_dx = 0.0
    flow_margin = 0.0
    for t in pair_transforms:
        cum_dy += t["total_dy"]
        cum_dx += t["total_dx"]
        cum_dys.append(cum_dy)
        cum_dxs.append(cum_dx)
        if t["flow"] is not None:
            mag = float(np.sqrt((t["flow"] ** 2).sum(axis=-1)).max())
            flow_margin += mag

    flow_margin += flow_safety

    min_dy = min(cum_dys) - flow_margin
    max_dy = max(cum_dys) + flow_margin
    min_dx = min(cum_dxs) - flow_margin
    max_dx = max(cum_dxs) + flow_margin

    top_pad = int(np.ceil(max(0.0, -min_dy)))
    bot_pad = int(np.ceil(max(0.0, max_dy)))
    left_pad = int(np.ceil(max(0.0, -min_dx)))
    right_pad = int(np.ceil(max(0.0, max_dx)))

    new_H = height + top_pad + bot_pad
    new_W = width + left_pad + right_pad

    info = {
        "cum_dys": cum_dys, "cum_dxs": cum_dxs,
        "min_dy": min_dy, "max_dy": max_dy,
        "min_dx": min_dx, "max_dx": max_dx,
        "flow_margin": flow_margin,
        "top_pad": top_pad, "bot_pad": bot_pad,
        "left_pad": left_pad, "right_pad": right_pad,
    }
    return new_H, new_W, top_pad, left_pad, info


def shift_image(image, dy, dx):
    """Translate image by (dy, dx) with sub-pixel interpolation."""
    if image.ndim == 2:
        return ndimage.shift(image, (dy, dx), order=1, mode="constant", cval=0)
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = ndimage.shift(
            image[..., c], (dy, dx), order=1, mode="constant", cval=0,
        )
    return out


def apply_flow_warp(image, flow):
    """Apply dense flow warp to single or multi-channel image."""
    h, w = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    if image.ndim == 2:
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    warped = np.empty_like(image)
    for c in range(image.shape[2]):
        warped[..., c] = cv2.remap(
            image[..., c], map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    return warped


# ===================================================================
# QC Visualization
# ===================================================================

def _agreement_overlay(ref, moving, ds=8):
    """Red/green overlay: ref=green, moving=red, aligned=yellow."""
    r = ref[::ds, ::ds].astype(np.float32)
    m = moving[::ds, ::ds].astype(np.float32)

    for arr in [r, m]:
        mx = np.percentile(arr, 99.9)
        if mx > 0:
            arr /= mx
        np.clip(arr, 0, 1, out=arr)

    rgb = np.stack([m, r, np.zeros_like(r)], axis=-1)
    return rgb


def save_qc_overlay(dapi0, dapi1_before, dapi1_after, output_path):
    """Before/after overlay: green = aligned, red = misaligned."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ds = 8
    before_rgb = _agreement_overlay(dapi0, dapi1_before, ds)
    after_rgb = _agreement_overlay(dapi0, dapi1_after, ds)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    axes[0].imshow(before_rgb)
    axes[0].set_title("Before alignment", fontsize=15, fontweight="bold")
    axes[0].axis("off")
    axes[1].imshow(after_rgb)
    axes[1].set_title("After alignment", fontsize=15, fontweight="bold")
    axes[1].axis("off")

    legend_elements = [
        Patch(facecolor="green", label="z0 only"),
        Patch(facecolor="red", label="z1 only"),
        Patch(facecolor="yellow", label="Aligned (both slices)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=13, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("QC overlay saved to %s", output_path)


def _save_qc_montage(output_dir, num_z):
    """Combine per-pair QC overlay PNGs into a single montage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.image as mpimg

    pairs = []
    for z in range(num_z - 1):
        p = os.path.join(output_dir, "qc_alignment_z%d_to_z%d.png" % (z + 1, z))
        if os.path.exists(p):
            pairs.append((z, z + 1, p))

    if not pairs:
        return

    ncols = 4
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (zr, zm, path) in enumerate(pairs):
        r, c = divmod(idx, ncols)
        img = mpimg.imread(path)
        axes[r, c].imshow(img)
        axes[r, c].set_title("z%d → z%d" % (zm, zr), fontsize=11, fontweight="bold")
        axes[r, c].axis("off")

    for idx in range(len(pairs), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    legend_elements = [
        Patch(facecolor="green", label="ref only"),
        Patch(facecolor="red", label="moving only"),
        Patch(facecolor="yellow", label="Aligned"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=12, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    montage_path = os.path.join(output_dir, "qc_alignment_montage.png")
    plt.savefig(montage_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("QC montage saved to %s", montage_path)


# ===================================================================
# Main
# ===================================================================

def align_pair(input_path, z_ref, z_mov, output_dir,
               skip_optical_flow=False, save_flow=False,
               return_aligned=False):
    """
    Align z-slice *z_mov* to z-slice *z_ref*.
    Saves QC overlay and parameters to *output_dir*.

    If return_aligned=True, returns the aligned z-slice as (C, Y, X) uint8
    instead of saving a per-slice TIF (used by --all_consecutive to build
    the combined stack).
    """
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    meta = read_ome_tiff_metadata(input_path)
    logger.info("Image: %s, shape=%s", input_path, meta["shape"])
    logger.info("Aligning z%d -> z%d (ref=z%d)", z_mov, z_ref, z_ref)

    logger.info("Loading registration channels (DAPI, CYTOKERATIN, CD45) ...")
    img0_ref = read_zslice_channels(input_path, z_ref, CONSENSUS_CHANNELS)
    img1_ref = read_zslice_channels(input_path, z_mov, CONSENSUS_CHANNELS)
    logger.info("  Loaded: %s", img0_ref.shape)

    coarse_dy, coarse_dx = coarse_phase_correlation(
        img0_ref[..., 0], img1_ref[..., 0], downscale=8,
    )
    img1_coarse = shift_image(img1_ref, coarse_dy, coarse_dx)

    residual_dy, residual_dx = refined_phase_correlation(
        img0_ref, img1_coarse,
        channel_indices=list(range(len(CONSENSUS_CHANNELS))),
        downscale=DOWNSCALE,
    )

    total_dy = coarse_dy + residual_dy
    total_dx = coarse_dx + residual_dx
    logger.info("Total rigid shift (dy, dx): (%.3f, %.3f)", total_dy, total_dx)
    logger.info("  = (%.1f um, %.1f um)", total_dy * 0.507, total_dx * 0.507)

    img1_rigid = shift_image(img1_ref, total_dy, total_dx)

    flow = None
    if not skip_optical_flow:
        flow = compute_optical_flow(
            img0_ref[..., 0], img1_rigid[..., 0], downscale=DOWNSCALE,
        )
        if save_flow:
            flow_path = os.path.join(output_dir, "flow_field_z%d.npy" % z_mov)
            np.save(flow_path, flow)
            logger.info("  Saved flow field to %s", flow_path)

    dapi1_before = img1_ref[..., 0]
    if flow is not None:
        dapi1_after = apply_flow_warp(img1_rigid[..., 0].astype(np.float32), flow)
    else:
        dapi1_after = img1_rigid[..., 0]

    save_qc_overlay(
        img0_ref[..., 0], dapi1_before, dapi1_after,
        os.path.join(output_dir, "qc_alignment_z%d_to_z%d.png" % (z_mov, z_ref)),
    )

    del img0_ref, img1_ref, img1_coarse, img1_rigid

    logger.info("Loading all channels of z-slice %d ...", z_mov)
    img1_all = read_zslice_all_channels(input_path, z_mov)
    logger.info("  Shape: %s", img1_all.shape)

    logger.info("Applying rigid shift to all channels ...")
    img1_all_shifted = shift_image(img1_all, total_dy, total_dx)
    del img1_all

    if flow is not None:
        logger.info("Applying optical flow warp to all channels ...")
        img1_all_aligned = apply_flow_warp(img1_all_shifted, flow)
        del img1_all_shifted
    else:
        img1_all_aligned = img1_all_shifted

    # (H, W, C) -> (C, Y, X) uint8
    aligned_cyx = np.transpose(img1_all_aligned.astype(np.uint8), (2, 0, 1))
    del img1_all_aligned

    if not return_aligned:
        out_path = os.path.join(output_dir, "z%d_aligned_all_channels.tif" % z_mov)
        logger.info("Saving aligned z%d (%s) ...", z_mov, aligned_cyx.shape)
        tifffile.imwrite(out_path, aligned_cyx, bigtiff=True, compression="zlib")
        logger.info("  Saved to %s", out_path)

    params_path = os.path.join(output_dir, "alignment_params_z%d.txt" % z_mov)
    with open(params_path, "w") as f:
        f.write("input: %s\n" % input_path)
        f.write("z_ref: %d\n" % z_ref)
        f.write("z_mov: %d\n" % z_mov)
        f.write("coarse_dy: %.4f\n" % coarse_dy)
        f.write("coarse_dx: %.4f\n" % coarse_dx)
        f.write("residual_dy: %.4f\n" % residual_dy)
        f.write("residual_dx: %.4f\n" % residual_dx)
        f.write("total_dy: %.4f\n" % total_dy)
        f.write("total_dx: %.4f\n" % total_dx)
        f.write("total_dy_um: %.2f\n" % (total_dy * 0.507))
        f.write("total_dx_um: %.2f\n" % (total_dx * 0.507))
        f.write("optical_flow: %s\n" % ("yes" if flow is not None else "no"))
        if flow is not None:
            mag = np.sqrt((flow ** 2).sum(axis=-1)).mean()
            f.write("flow_mean_magnitude_px: %.2f\n" % mag)
    logger.info("  Saved parameters to %s", params_path)

    elapsed = time.time() - t0
    logger.info("=== Pair z%d->z%d complete in %.1fs ===", z_mov, z_ref, elapsed)

    result = {
        "z_ref": z_ref, "z_mov": z_mov,
        "total_dy": total_dy, "total_dx": total_dx,
        "flow": flow is not None,
    }
    if return_aligned:
        result["aligned_cyx"] = aligned_cyx
    else:
        del aligned_cyx
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Align z-slices using image data only (no segmentation needed)",
    )
    parser.add_argument("--input", "-i", required=True, help="OME-TIFF path")
    parser.add_argument("--output_dir", "-o", default="outputs/aligned")
    parser.add_argument("--z_ref", type=int, default=0, help="Reference z-slice index")
    parser.add_argument("--z_mov", type=int, default=1, help="Moving z-slice index")
    parser.add_argument(
        "--all_consecutive", action="store_true",
        help="Align every slice to z0 by composing consecutive pairwise "
             "transforms (ignores --z_ref/--z_mov)",
    )
    parser.add_argument(
        "--save_flow", action="store_true",
        help="Save the per-pair flow field (single-pair mode) or the "
             "cumulative z_k->z_0 sampling map (--all_consecutive)",
    )
    parser.add_argument(
        "--skip_optical_flow", action="store_true",
        help="Skip optical flow (rigid-only, faster)",
    )
    args = parser.parse_args()

    meta = read_ome_tiff_metadata(args.input)
    num_z = meta["shape"][0]
    logger.info("Image: %s, %d z-slices", args.input, num_z)

    if args.all_consecutive:
        num_c = meta["shape"][1]
        height = meta["shape"][2]
        width = meta["shape"][3]
        dtype = np.dtype(meta["dtype"])

        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("=== Cumulative alignment to z0 (%d pairs) ===", num_z - 1)

        # -----------------------------------------------------------------
        # Step 1: estimate pairwise transforms on RAW slices.
        #   Save per-pair QC immediately (z_k aligned to raw z_{k-1}).
        #   Drop the heavy DAPI snapshots after QC so only shift + flow are
        #   kept for the composition step.
        # -----------------------------------------------------------------
        pair_transforms = []
        logger.info("Reading registration channels for z0 ...")
        prev_ch = read_zslice_channels(args.input, 0, CONSENSUS_CHANNELS)

        for k in range(1, num_z):
            logger.info("Reading registration channels for z%d ...", k)
            cur_ch = read_zslice_channels(args.input, k, CONSENSUS_CHANNELS)

            logger.info("--- Pair z%d -> z%d ---", k, k - 1)
            t = estimate_pair_transform(
                prev_ch, cur_ch, skip_optical_flow=args.skip_optical_flow,
            )

            if t["flow"] is not None:
                dapi_after = apply_flow_warp(
                    t["dapi_mov_rigid"].astype(np.float32), t["flow"],
                )
            else:
                dapi_after = t["dapi_mov_rigid"]
            save_qc_overlay(
                t["dapi_ref"], t["dapi_mov_before"], dapi_after,
                os.path.join(args.output_dir,
                             "qc_alignment_z%d_to_z%d.png" % (k, k - 1)),
            )

            t["has_flow"] = t["flow"] is not None
            t.pop("dapi_ref")
            t.pop("dapi_mov_before")
            t.pop("dapi_mov_rigid")
            pair_transforms.append(t)

            prev_ch = cur_ch
        del prev_ch

        # -----------------------------------------------------------------
        # Step 2: size the output canvas so every slice fits fully after its
        #   cumulative warp -- no clipping.
        # -----------------------------------------------------------------
        new_H, new_W, top_pad, left_pad, pad_info = compute_padded_canvas(
            pair_transforms, height, width,
        )
        logger.info(
            "Canvas: raw (%d, %d) -> padded (%d, %d)  [top=%d bot=%d left=%d right=%d]",
            height, width, new_H, new_W,
            pad_info["top_pad"], pad_info["bot_pad"],
            pad_info["left_pad"], pad_info["right_pad"],
        )
        logger.info(
            "  cumulative dy range [%.2f, %.2f], dx range [%.2f, %.2f], flow margin %.2f px",
            pad_info["min_dy"] + pad_info["flow_margin"],
            pad_info["max_dy"] - pad_info["flow_margin"],
            pad_info["min_dx"] + pad_info["flow_margin"],
            pad_info["max_dx"] - pad_info["flow_margin"],
            pad_info["flow_margin"],
        )
        logger.info("Output stack will be (%d, %d, %d, %d) ZCYX",
                     num_z, num_c, new_H, new_W)

        # -----------------------------------------------------------------
        # Step 3: stream aligned slices (per channel) into the output stack.
        #   The cumulative sampling map is built over the PADDED canvas so
        #   the identity position inside raw z_k maps to (y+top_pad, x+left_pad)
        #   of the output canvas before any per-pair composition.
        # -----------------------------------------------------------------
        stack_path = os.path.join(args.output_dir, "aligned_stack.ome.tif")
        cumulative_records = []  # (k, cum_dy, cum_dx, flow_bool)
        qc_cache = {}

        def aligned_plane_iterator():
            # --- z0: placed unchanged into the padded canvas ---
            logger.info("Writing z0 into padded canvas at offset (%d, %d) ...",
                         top_pad, left_pad)
            z0 = read_zslice_all_channels(args.input, 0)  # (H, W, C)
            qc_cache["dapi0_raw"] = z0[..., DAPI_CH].copy()
            for c in range(num_c):
                plane = np.zeros((new_H, new_W), dtype=dtype)
                plane[top_pad:top_pad + height,
                      left_pad:left_pad + width] = z0[..., c]
                yield plane
                del plane
            del z0

            # --- Running cumulative map defined over padded canvas ---
            # For padded pixel (yn, xn), its z0-frame coord is (yn - top_pad,
            # xn - left_pad). Start from that shifted identity, then compose
            # each pair's transform on top.
            grid_x, grid_y = np.meshgrid(
                np.arange(new_W, dtype=np.float32) - left_pad,
                np.arange(new_H, dtype=np.float32) - top_pad,
            )
            cur_x = grid_x.astype(np.float32, copy=True)
            cur_y = grid_y.astype(np.float32, copy=True)
            del grid_x, grid_y

            cum_dy = 0.0
            cum_dx = 0.0
            for k, t in enumerate(pair_transforms, start=1):
                logger.info("Composing cumulative map through pair z%d->z%d ...",
                             k, k - 1)
                cur_x, cur_y = compose_cumulative_step(cur_x, cur_y, t)
                cum_dy += t["total_dy"]
                cum_dx += t["total_dx"]
                cumulative_records.append(
                    (k, cum_dy, cum_dx, t["has_flow"]),
                )

                if args.save_flow:
                    map_path = os.path.join(
                        args.output_dir, "cumulative_map_z%d.npy" % k,
                    )
                    np.save(map_path, np.stack([cur_x, cur_y], axis=-1))
                    logger.info("  Saved cumulative map to %s", map_path)

                t["flow"] = None  # free per-pair flow

                logger.info("Loading raw z%d (all channels) ...", k)
                raw_k = read_zslice_all_channels(args.input, k)  # (H, W, C)

                if k == 1 or k == num_z - 1:
                    qc_cache["raw_z%d" % k] = raw_k[..., DAPI_CH].copy()

                logger.info("Warping z%d into padded z0 frame (per channel) ...", k)
                for c in range(num_c):
                    plane = cv2_remap_tiled(
                        raw_k[..., c], cur_x, cur_y,
                        cv2.INTER_LINEAR,
                        border_mode=cv2.BORDER_CONSTANT, border_value=0,
                    )
                    if c == DAPI_CH and (k == 1 or k == num_z - 1):
                        qc_cache["aligned_z%d" % k] = plane.copy()
                    yield plane
                    del plane
                del raw_k

        with tifffile.TiffWriter(stack_path, bigtiff=True) as writer:
            writer.write(
                data=aligned_plane_iterator(),
                shape=(num_z, num_c, new_H, new_W),
                dtype=dtype,
                metadata={"axes": "ZCYX"},
            )

        stack_size = os.path.getsize(stack_path)
        logger.info("Aligned stack saved: %s (%.1f GB)", stack_path, stack_size / 1e9)

        # -----------------------------------------------------------------
        # Step 4: per-slice param files (pairwise + cumulative) + summary.
        # -----------------------------------------------------------------
        cum_dy = 0.0
        cum_dx = 0.0
        for k, t in enumerate(pair_transforms, start=1):
            cum_dy += t["total_dy"]
            cum_dx += t["total_dx"]
            params_path = os.path.join(
                args.output_dir, "alignment_params_z%d.txt" % k,
            )
            with open(params_path, "w") as f:
                f.write("input: %s\n" % args.input)
                f.write("z_ref: 0 (cumulative composition via z%d)\n" % (k - 1))
                f.write("z_mov: %d\n" % k)
                f.write("pair_coarse_dy: %.4f\n" % t["coarse_dy"])
                f.write("pair_coarse_dx: %.4f\n" % t["coarse_dx"])
                f.write("pair_residual_dy: %.4f\n" % t["residual_dy"])
                f.write("pair_residual_dx: %.4f\n" % t["residual_dx"])
                f.write("pair_total_dy: %.4f\n" % t["total_dy"])
                f.write("pair_total_dx: %.4f\n" % t["total_dx"])
                f.write("cumulative_dy: %.4f\n" % cum_dy)
                f.write("cumulative_dx: %.4f\n" % cum_dx)
                f.write("cumulative_dy_um: %.2f\n" % (cum_dy * 0.507))
                f.write("cumulative_dx_um: %.2f\n" % (cum_dx * 0.507))
                f.write("optical_flow: %s\n"
                        % ("yes" if t["has_flow"] else "no"))

        summary_path = os.path.join(args.output_dir, "alignment_summary.txt")
        with open(summary_path, "w") as f:
            f.write("input: %s\n" % args.input)
            f.write("num_z: %d\n" % num_z)
            f.write("pairs_aligned: %d\n" % len(pair_transforms))
            f.write("reference: z0 (cumulative composition)\n")
            f.write("raw_shape: (%d, %d)\n" % (height, width))
            f.write("padded_shape: (%d, %d)\n" % (new_H, new_W))
            f.write("pad_top: %d\n" % pad_info["top_pad"])
            f.write("pad_bottom: %d\n" % pad_info["bot_pad"])
            f.write("pad_left: %d\n" % pad_info["left_pad"])
            f.write("pad_right: %d\n" % pad_info["right_pad"])
            f.write("z0_origin_in_canvas: (%d, %d)\n" % (top_pad, left_pad))
            f.write("flow_safety_margin_px: %.2f\n" % pad_info["flow_margin"])
            f.write("output_stack: %s\n\n" % stack_path)
            f.write("pair and cumulative-to-z0 shifts (px):\n")
            for (k, cdy, cdx, has_flow), t in zip(
                cumulative_records, pair_transforms,
            ):
                f.write(
                    "  z%d->z%d: pair dy=%+.3f dx=%+.3f | "
                    "cumulative(z%d->z0) dy=%+.3f dx=%+.3f (flow=%s)\n"
                    % (k, k - 1, t["total_dy"], t["total_dx"],
                       k, cdy, cdx, "yes" if has_flow else "no")
                )
        logger.info("Summary saved to %s", summary_path)

        # Extra QC: first and last slice overlaid on z0 in the padded canvas.
        if "dapi0_raw" in qc_cache:
            dapi0_padded = np.zeros((new_H, new_W),
                                    dtype=qc_cache["dapi0_raw"].dtype)
            dapi0_padded[top_pad:top_pad + height,
                         left_pad:left_pad + width] = qc_cache["dapi0_raw"]
            for k in sorted(set([1, num_z - 1])):
                if "aligned_z%d" % k not in qc_cache:
                    continue
                raw_padded = np.zeros((new_H, new_W),
                                      dtype=qc_cache["raw_z%d" % k].dtype)
                raw_padded[top_pad:top_pad + height,
                           left_pad:left_pad + width] = qc_cache["raw_z%d" % k]
                save_qc_overlay(
                    dapi0_padded, raw_padded, qc_cache["aligned_z%d" % k],
                    os.path.join(
                        args.output_dir,
                        "qc_alignment_z%d_to_z0_cumulative.png" % k,
                    ),
                )

        _save_qc_montage(args.output_dir, num_z)
    else:
        align_pair(
            args.input, z_ref=args.z_ref, z_mov=args.z_mov,
            output_dir=args.output_dir,
            skip_optical_flow=args.skip_optical_flow,
            save_flow=args.save_flow,
        )


if __name__ == "__main__":
    main()
