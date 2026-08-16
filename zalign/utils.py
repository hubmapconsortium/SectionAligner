"""
Shared utilities for DeepCell segmentation & z-slice alignment pipeline.

Provides:
  - Lazy OME-TIFF reading (one z-slice / channel at a time)
  - Tiling large images into overlapping patches
  - Stitching tiled label masks back together (resolving overlaps)
  - Relabeling helpers for consistent cell IDs
"""

import logging
import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _detect_layout(path: str):
    """
    Detect whether an OME-TIFF is stored as:
      - "zcyx": single series with shape (Z, C, Y, X)
      - "multi_series": multiple series, each (C, Y, X) — one per z-slice

    Returns (layout, num_z, num_c, height, width, dtype).
    """
    with tifffile.TiffFile(path) as tif:
        s0 = tif.series[0]
        if len(tif.series) > 1 and s0.axes.upper() == "CYX":
            num_z = len(tif.series)
            num_c, height, width = s0.shape
            return "multi_series", num_z, num_c, height, width, s0.dtype
        elif "Z" in s0.axes.upper():
            ax = s0.axes.upper()
            zi = ax.index("Z")
            ci = ax.index("C")
            yi = ax.index("Y")
            xi = ax.index("X")
            return ("zcyx", s0.shape[zi], s0.shape[ci],
                    s0.shape[yi], s0.shape[xi], s0.dtype)
        else:
            return "zcyx", 1, s0.shape[0], s0.shape[1], s0.shape[2], s0.dtype


def read_ome_tiff_metadata(path: str) -> dict:
    """Return shape, axes, dtype, and channel names from an OME-TIFF."""
    import xml.etree.ElementTree as ET

    layout, num_z, num_c, height, width, dtype = _detect_layout(path)

    channel_names = []
    with tifffile.TiffFile(path) as tif:
        if tif.ome_metadata:
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            root = ET.fromstring(tif.ome_metadata)
            for ch in root.findall(".//ome:Channel", ns):
                channel_names.append(ch.get("Name", ""))

    shape = (num_z, num_c, height, width)
    return {
        "shape": shape,
        "axes": "ZCYX",
        "dtype": str(dtype),
        "channel_names": channel_names,
        "_layout": layout,
    }


def read_zslice_channels(path: str, z_index: int, channel_indices: list) -> np.ndarray:
    """
    Read specific channels from one z-slice of an OME-TIFF.
    Handles both single-series ZCYX and multi-series CYX layouts.

    Returns
    -------
    np.ndarray of shape (H, W, len(channel_indices)), dtype matching the file.
    """
    import zarr
    layout, *_ = _detect_layout(path)

    with tifffile.TiffFile(path) as tif:
        if layout == "multi_series":
            series = tif.series[z_index]
            store = series.aszarr()
            za = zarr.open(store, mode="r")
            planes = []
            for c in channel_indices:
                logger.info("  Reading z=%d (series), channel=%d ...", z_index, c)
                planes.append(np.array(za[c, :, :]))
        else:
            series = tif.series[0]
            store = series.aszarr()
            za = zarr.open(store, mode="r")
            planes = []
            for c in channel_indices:
                logger.info("  Reading z=%d, channel=%d ...", z_index, c)
                planes.append(np.array(za[z_index, c, :, :]))

    return np.stack(planes, axis=-1)


def read_zslice_all_channels(path: str, z_index: int) -> np.ndarray:
    """
    Read ALL channels of one z-slice. Returns (H, W, C) array.
    Handles both single-series ZCYX and multi-series CYX layouts.
    """
    import zarr
    layout, *_ = _detect_layout(path)

    with tifffile.TiffFile(path) as tif:
        if layout == "multi_series":
            series = tif.series[z_index]
            store = series.aszarr()
            za = zarr.open(store, mode="r")
            data = np.array(za)  # (C, Y, X)
        else:
            series = tif.series[0]
            store = series.aszarr()
            za = zarr.open(store, mode="r")
            data = np.array(za[z_index])  # (C, Y, X)

    return np.transpose(data, (1, 2, 0))


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def compute_tile_coords(
    img_height: int,
    img_width: int,
    tile_size: int = 2048,
    overlap: int = 128,
) -> list:
    """
    Compute (row_start, row_end, col_start, col_end) for overlapping tiles
    that cover the full image.

    Returns a list of (rs, re, cs, ce) tuples.
    """
    step = tile_size - overlap
    coords = []
    row = 0
    while row < img_height:
        re = min(row + tile_size, img_height)
        rs = max(re - tile_size, 0)
        col = 0
        while col < img_width:
            ce = min(col + tile_size, img_width)
            cs = max(ce - tile_size, 0)
            coords.append((rs, re, cs, ce))
            if ce >= img_width:
                break
            col += step
        if re >= img_height:
            break
        row += step
    return coords


def extract_tiles(image: np.ndarray, coords: list) -> np.ndarray:
    """
    Extract tiles from image given coordinate list.

    Parameters
    ----------
    image : (H, W, C)
    coords : list of (rs, re, cs, ce)

    Returns
    -------
    tiles : (N, tile_h, tile_w, C)
    """
    tiles = []
    for rs, re, cs, ce in coords:
        tiles.append(image[rs:re, cs:ce, :])
    return np.array(tiles)


# ---------------------------------------------------------------------------
# Stitching label masks
# ---------------------------------------------------------------------------

def stitch_masks(
    tile_masks: np.ndarray,
    coords: list,
    img_height: int,
    img_width: int,
    overlap: int = 128,
) -> np.ndarray:
    """
    Stitch tiled label masks into a single full-size mask, relabeling to
    ensure globally unique cell IDs and resolving overlaps by keeping the
    inner (non-overlap) region of each tile.

    Parameters
    ----------
    tile_masks : (N, tile_h, tile_w)  — integer label masks per tile
    coords : list of (rs, re, cs, ce)
    img_height, img_width : full image dimensions
    overlap : overlap used during tiling

    Returns
    -------
    full_mask : (img_height, img_width), int32 label mask
    """
    full_mask = np.zeros((img_height, img_width), dtype=np.int32)
    max_label = 0
    half_ov = overlap // 2

    for idx, (rs, re, cs, ce) in enumerate(coords):
        tile = tile_masks[idx].copy()
        tile_h, tile_w = tile.shape

        # Relabel so IDs don't collide across tiles
        mask_nonzero = tile > 0
        tile[mask_nonzero] += max_label

        # Determine the inner region of this tile (trim overlap margins)
        # Only trim a margin if the tile is NOT at the image boundary
        inner_top = half_ov if rs > 0 else 0
        inner_bot = tile_h - half_ov if re < img_height else tile_h
        inner_left = half_ov if cs > 0 else 0
        inner_right = tile_w - half_ov if ce < img_width else tile_w

        # Write inner region into full mask
        full_mask[
            rs + inner_top : rs + inner_bot,
            cs + inner_left : cs + inner_right,
        ] = tile[inner_top:inner_bot, inner_left:inner_right]

        # Update max label
        if tile.max() > max_label:
            max_label = int(tile.max())

    return full_mask


# ---------------------------------------------------------------------------
# Relabeling helpers
# ---------------------------------------------------------------------------

def relabel_sequential(mask: np.ndarray) -> np.ndarray:
    """Relabel a mask so that cell IDs are sequential starting from 1."""
    from skimage.segmentation import relabel_sequential as _relabel
    relabeled, _, _ = _relabel(mask)
    return relabeled
