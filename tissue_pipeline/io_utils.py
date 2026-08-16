"""
I/O utilities for loading qptiff images, creating downsampled projections,
reading channel names, and saving OME-TIFF outputs.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tifffile


def find_qptiff_files(input_dir: str) -> List[Path]:
    """Find and sort qptiff files in the input directory."""
    input_path = Path(input_dir)
    qptiff_files = sorted(input_path.glob("*.qptiff"), key=lambda x: x.name)
    if not qptiff_files:
        raise FileNotFoundError(f"No .qptiff files found in {input_dir}")
    print(f"Found {len(qptiff_files)} qptiff files:")
    for f in qptiff_files:
        print(f"  {f.name}")
    return qptiff_files


def load_channel_names(input_dir: str) -> List[str]:
    """Read channel names from channelnames.txt or MarkerList.txt."""
    input_path = Path(input_dir)
    for fname in ("channelnames.txt", "MarkerList.txt"):
        path = input_path / fname
        if path.exists():
            with open(path, "r") as f:
                names = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(names)} channel names from {fname}")
            return names
    raise FileNotFoundError(
        f"No channel names file found in {input_dir} "
        f"(looked for channelnames.txt, MarkerList.txt)"
    )


def get_image_metadata(qptiff_path: Path) -> dict:
    """Extract metadata from a qptiff file without loading image data."""
    with tifffile.TiffFile(str(qptiff_path)) as t:
        series0 = t.series[0]
        shape = series0.shape
        dtype = series0.dtype
        axes = series0.axes

        # Try to get physical pixel size from first page
        page = t.pages[0]
        pixel_size = None
        if hasattr(page, "tags"):
            # qptiff stores resolution in TIFF tags
            x_res_tag = page.tags.get("XResolution")
            y_res_tag = page.tags.get("YResolution")
            if x_res_tag and y_res_tag:
                x_res = x_res_tag.value
                y_res = y_res_tag.value
                if isinstance(x_res, tuple) and len(x_res) == 2:
                    pixel_size = {
                        "X": x_res[1] / x_res[0] if x_res[0] != 0 else None,
                        "Y": y_res[1] / y_res[0] if y_res[0] != 0 else None,
                    }

    return {
        "shape": shape,
        "dtype": dtype,
        "axes": axes,
        "pixel_size": pixel_size,
    }


def load_or_create_downsampled(
    input_dir: str, qptiff_paths: List[Path], force_recreate: bool = False
) -> Tuple[List[np.ndarray], int, int]:
    """
    Load cached downsampled sum-projections from NPZ, or create them from
    the qptiff files by summing all channels and block-reducing.

    Returns:
        (list of 2D sum-projected images, scale_factor_x, scale_factor_y)
    """
    npz_path = Path(input_dir) / "downsizedimgs_compressed.npz"

    # Default physical pixel size (matching the existing codebase)
    desired_physical_pixel_size = 4.058815539828226  # microns
    physical_pixel_size = 0.5073519424785282  # microns

    scale_factor_x = int(desired_physical_pixel_size / physical_pixel_size)
    scale_factor_y = int(desired_physical_pixel_size / physical_pixel_size)

    def _prepare_cached_images(data: np.lib.npyio.NpzFile) -> Optional[List[np.ndarray]]:
        """Load cached arrays only if they are known to match qptiff order.

        Older caches were written as arr_0, arr_1, ... without filename
        metadata. That is unsafe for this dataset because the cached array order
        can differ from the sorted qptiff order, which causes segmentation
        masks to be applied to the wrong raw slices. New caches include a
        ``filenames`` array and are reordered by filename when loaded.
        """

        expected = [p.name for p in qptiff_paths]

        if "filenames" in data.files:
            cache_method = str(data["cache_method"]) if "cache_method" in data.files else ""
            if cache_method != "exact_blocksum_v1":
                print(
                    "  Cached downsampled file is ordered but was not built "
                    "with exact 8x8 block sums. Rebuilding exact cache."
                )
                return None
            filenames = [str(x) for x in data["filenames"].tolist()]
            image_keys = [k for k in data.files if k.startswith("arr_")]
            by_name = {}
            for idx, fname in enumerate(filenames):
                key = f"arr_{idx}"
                if key in data.files:
                    by_name[fname] = data[key]
                elif idx < len(image_keys):
                    by_name[fname] = data[image_keys[idx]]
            missing = [fname for fname in expected if fname not in by_name]
            if missing:
                print(
                    "  Cached downsampled file has filename metadata but is "
                    f"missing arrays for: {missing}"
                )
                return None
            arrays = [by_name[fname] for fname in expected]
        else:
            image_keys = [k for k in data.files if k.startswith("arr_")]
            arrays = [data[k] for k in image_keys]
            if len(arrays) != len(qptiff_paths):
                print(
                    f"  Cached downsampled file has {len(arrays)} arrays, "
                    f"expected {len(qptiff_paths)}."
                )
                return None

            mismatches = []
            for idx, (arr, qptiff_path) in enumerate(zip(arrays, qptiff_paths)):
                shape = arr.squeeze().shape
                expected_full = (shape[0] * scale_factor_y, shape[1] * scale_factor_x)
                with tifffile.TiffFile(str(qptiff_path)) as tif:
                    full_shape = tif.pages[0].shape
                if tuple(expected_full) != tuple(full_shape):
                    mismatches.append((idx, qptiff_path.name, shape, full_shape))
            if mismatches:
                print(
                    "  WARNING: cached downsampled image order does not match "
                    "the qptiff order. Rebuilding cache with filename metadata."
                )
                for idx, fname, shape, full_shape in mismatches[:8]:
                    print(
                        f"    idx={idx}: cache shape {shape} -> "
                        f"{shape[0] * scale_factor_y}x{shape[1] * scale_factor_x}, "
                        f"but {fname} is {full_shape}"
                    )
                if len(mismatches) > 8:
                    print(f"    ... {len(mismatches) - 8} more mismatches")
                return None

        images = []
        for arr in arrays:
            if arr.ndim > 2:
                arr = arr.squeeze()
            if arr.dtype == np.uint64 or arr.dtype == np.int64:
                arr = arr.astype(np.float32)
            images.append(arr)
        return images

    def _downsample_exact_blocksum(qptiff_path: Path) -> np.ndarray:
        """Create the 8x downsampled sum projection for one qptiff.

        This reproduces the legacy semantics exactly:
          sum over channels AND sum over every 8x8 full-resolution block.

        It streams one channel page at a time, so memory is bounded by one
        full-resolution channel plus the downsampled accumulator instead of
        the full 60 x Y x X stack.
        """

        with tifffile.TiffFile(str(qptiff_path)) as tif:
            full_h, full_w = tif.pages[0].shape
            target_h = full_h // scale_factor_y
            target_w = full_w // scale_factor_x
            crop_h = target_h * scale_factor_y
            crop_w = target_w * scale_factor_x
            acc = np.zeros((target_h, target_w), dtype=np.float32)
            channel_pages = tif.series[0].shape[0]
            print(
                f"    Exact block-sum: {channel_pages} channels, "
                f"{full_h}x{full_w} -> {target_h}x{target_w}"
            )
            for ch in range(channel_pages):
                page = tif.pages[ch].asarray()
                page = page[:crop_h, :crop_w]
                block_sum = page.reshape(
                    target_h,
                    scale_factor_y,
                    target_w,
                    scale_factor_x,
                ).sum(axis=(1, 3), dtype=np.uint32)
                acc += block_sum.astype(np.float32)
                if (ch + 1) % 10 == 0 or ch == channel_pages - 1:
                    print(f"      channel {ch + 1}/{channel_pages}")
                del page, block_sum
        return acc

    if npz_path.exists() and not force_recreate:
        print(f"Loading cached downsampled images from {npz_path}")
        with np.load(str(npz_path), allow_pickle=False) as data:
            images = _prepare_cached_images(data)
        if images is not None:
            print(
                f"Loaded {len(images)} images, shapes: "
                f"{[img.shape for img in images]}, "
                f"dtypes: {[img.dtype for img in images]}"
            )
            return images, scale_factor_x, scale_factor_y

    print("Creating downsampled sum-projections from qptiff files...")

    images = []
    for i, qptiff_path in enumerate(qptiff_paths):
        print(f"  Processing {qptiff_path.name} ({i+1}/{len(qptiff_paths)})...")
        reduced = _downsample_exact_blocksum(qptiff_path)
        images.append(reduced)
        print(f"    Downsampled sum shape: {reduced.shape}")

    # Cache for future runs
    if npz_path.exists():
        backup = npz_path.with_suffix(npz_path.suffix + ".legacy_mismatched")
        if not backup.exists():
            shutil.copy2(npz_path, backup)
            print(f"Backed up previous cache to {backup}")
    np.savez_compressed(
        str(npz_path),
        *images,
        filenames=np.asarray([p.name for p in qptiff_paths]),
        scale_factor_x=np.asarray(scale_factor_x),
        scale_factor_y=np.asarray(scale_factor_y),
        cache_method=np.asarray("exact_blocksum_v1"),
    )
    print(f"Saved downsampled images to {npz_path}")

    return images, scale_factor_x, scale_factor_y


def load_channel_lazy(qptiff_path: Path, channel_idx: int) -> np.ndarray:
    """
    Lazily load a single channel from a qptiff file.
    Returns 2D array (Y, X) for the requested channel.
    """
    with tifffile.TiffFile(str(qptiff_path)) as t:
        page = t.pages[channel_idx]
        return page.asarray()


def load_channel_crop(
    qptiff_path: Path,
    channel_idx: int,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> np.ndarray:
    """
    Load a single channel from a qptiff file, cropped to a bounding box.
    Uses tifffile's built-in slicing for memory efficiency.
    Returns 2D array (H, W) for the cropped region.
    """
    with tifffile.TiffFile(str(qptiff_path)) as t:
        page = t.pages[channel_idx]
        # Read the full page and crop (tifffile doesn't support sub-region
        # reads for all compression types, but reading + slicing is still
        # more memory efficient than loading all channels)
        full = page.asarray()
        return full[y1:y2, x1:x2]


def save_tissue_ome_tiff(
    stack: np.ndarray,
    output_path: str,
    channel_names: List[str],
    physical_pixel_sizes: Optional[Tuple[float, float, float]] = None,
):
    """
    Save a 3D multichannel tissue stack as OME-TIFF.

    Args:
        stack: Array with shape (Z, C, Y, X)
        output_path: Path to save the OME-TIFF
        channel_names: List of channel name strings
        physical_pixel_sizes: Optional (Z, Y, X) physical sizes in microns
    """
    from aicsimageio.writers import OmeTiffWriter
    from aicsimageio import types

    if physical_pixel_sizes is not None:
        pps = types.PhysicalPixelSizes(
            Z=physical_pixel_sizes[0],
            Y=physical_pixel_sizes[1],
            X=physical_pixel_sizes[2],
        )
    else:
        # Default values from existing codebase
        pps = types.PhysicalPixelSizes(
            Z=1.0,
            Y=0.5073519424785282,
            X=0.5073519424785282,
        )

    print(f"  Saving OME-TIFF: {output_path}")
    print(f"    Shape: {stack.shape} (ZCYX), dtype: {stack.dtype}")
    print(f"    Physical pixel sizes: Z={pps.Z}, Y={pps.Y}, X={pps.X}")

    OmeTiffWriter.save(
        stack,
        output_path,
        dim_order="ZCYX",
        channel_names=channel_names,
        physical_pixel_sizes=pps,
    )
    print(f"    Saved successfully ({os.path.getsize(output_path) / 1e9:.2f} GB)")
