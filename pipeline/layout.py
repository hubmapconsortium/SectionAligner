"""Canonical on-disk layout for a pipeline run.

Everything a run produces lives under a single ``work_dir`` so the whole
pipeline is inspectable and restartable stage-by-stage::

    work_dir/
      01_stacked/                         # stage 1 (all tissues)
        tissue_{k}_stacked.ome.tif
        qc/
      02_aligned/tissue_{k}/              # stage 2 (per tissue)
        aligned_stack.ome.tif
      03_tiles/tissue_{k}/                # stage 3 (per tissue)
        tile_y####_x####.tif
        tile_metadata.json
      04_segmentation/tissue_{k}/         # stage 4 (per tissue, per tile)
        tiles/tile_y####_x####/3D_cell_mask.tif ...
        masks/tile_y####_x####.tif        # gathered masks for stitching
      05_stitched/tissue_{k}/             # stage 5 (per tissue)
        stitched_3D_cell_mask.tif
      logs/
      slurm/
"""

from __future__ import annotations

import glob
import os
import re


class Layout:
    """Resolve all paths for a run given its ``work_dir``."""

    def __init__(self, work_dir: str, tile_prefix: str = "tile") -> None:
        self.work = os.path.abspath(work_dir)
        self.tile_prefix = tile_prefix

    # -- top-level directories ------------------------------------------
    @property
    def stacked_dir(self) -> str:
        return os.path.join(self.work, "01_stacked")

    def aligned_dir(self, tissue: int) -> str:
        return os.path.join(self.work, "02_aligned", f"tissue_{tissue}")

    def tiles_dir(self, tissue: int) -> str:
        return os.path.join(self.work, "03_tiles", f"tissue_{tissue}")

    def seg_dir(self, tissue: int) -> str:
        return os.path.join(self.work, "04_segmentation", f"tissue_{tissue}")

    def seg_tiles_dir(self, tissue: int) -> str:
        return os.path.join(self.seg_dir(tissue), "tiles")

    def masks_dir(self, tissue: int, source_mask_name: str | None = None) -> str:
        base = os.path.join(self.seg_dir(tissue), "masks")
        if source_mask_name is None:
            return base
        # Keep e.g. cell and nuclear gathered masks in separate subdirectories
        # so a per-mask stitch reads only its own tiles.
        return os.path.join(base, _strip_tif(source_mask_name))

    def stitched_dir(self, tissue: int) -> str:
        return os.path.join(self.work, "05_stitched", f"tissue_{tissue}")

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.work, "logs")

    @property
    def slurm_dir(self) -> str:
        return os.path.join(self.work, "slurm")

    # -- stage output files ---------------------------------------------
    def stacked_tissue(self, tissue: int) -> str:
        return os.path.join(self.stacked_dir, f"tissue_{tissue}_stacked.ome.tif")

    def aligned_stack(self, tissue: int) -> str:
        return os.path.join(self.aligned_dir(tissue), "aligned_stack.ome.tif")

    def tile_metadata(self, tissue: int) -> str:
        return os.path.join(self.tiles_dir(tissue), f"{self.tile_prefix}_metadata.json")

    def seg_tile_result(self, tissue: int, tile_filename: str) -> str:
        """Per-tile 3DCellComposer results directory for a tile file name."""
        stem = _strip_tif(tile_filename)
        return os.path.join(self.seg_tiles_dir(tissue), stem)

    def stitched_mask(self, tissue: int, source_mask_name: str) -> str:
        return os.path.join(self.stitched_dir(tissue), f"stitched_{source_mask_name}")

    # -- discovery ------------------------------------------------------
    def discover_stacked_tissues(self) -> list[int]:
        """Tissue ids that already have a stacked OME-TIFF on disk."""
        pattern = os.path.join(self.stacked_dir, "tissue_*_stacked.ome.tif")
        ids = []
        for path in glob.glob(pattern):
            match = re.search(r"tissue_(\d+)_stacked\.ome\.tif$", os.path.basename(path))
            if match:
                ids.append(int(match.group(1)))
        return sorted(ids)

    def ensure_run_dirs(self) -> None:
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.slurm_dir, exist_ok=True)


def _strip_tif(name: str) -> str:
    for ext in (".tif", ".tiff"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name
