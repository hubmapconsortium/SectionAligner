"""Where each stage's tool lives inside this repository.

Every stage of the pipeline resolves inside the checkout, so a run needs no path
outside it: ``REPO_ROOT`` is derived from this file's own location, which means
the pipeline works from any clone and any working directory. Stage 4 arrives as
a git submodule rather than committed files, so it is the one tool that can be
absent from an otherwise complete checkout.

Each stage has a *script* (what gets run) and a *dir* (the working directory it
runs in).  The two differ for tools that import their own sibling modules --
stage 2 does ``from utils import ...`` and stage 4 does
``from segmentation_2D... import ...`` -- so those must run from their own
directory.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stage 1 - tissue matching & stacking
STACK_DIR = os.path.join(REPO_ROOT, "tissue_pipeline")
STACK_SCRIPT = os.path.join(STACK_DIR, "run_pipeline.py")

# Stage 2 - z-slice alignment
ALIGN_DIR = os.path.join(REPO_ROOT, "zalign")
ALIGN_SCRIPT = os.path.join(ALIGN_DIR, "align_image.py")

# Stage 3 - 3D tiling
TILE_DIR = REPO_ROOT
TILE_SCRIPT = os.path.join(REPO_ROOT, "3Dtiler.py")

# Stage 4 - 3D cell segmentation
SEGMENT_DIR = os.path.join(REPO_ROOT, "3DCellComposer")
SEGMENT_SCRIPT = os.path.join(SEGMENT_DIR, "run_3DCellComposer.py")

# Stage 5 - 3D stitching
STITCH_DIR = REPO_ROOT
STITCH_SCRIPT = os.path.join(REPO_ROOT, "3Dstitcher.py")

STAGE_SCRIPTS = {
    1: STACK_SCRIPT,
    2: ALIGN_SCRIPT,
    3: TILE_SCRIPT,
    4: SEGMENT_SCRIPT,
    5: STITCH_SCRIPT,
}

# Stages whose tool is a git submodule: missing files mean an uninitialised
# submodule, which needs a different fix than a broken checkout.
SUBMODULE_STAGES = {4: SEGMENT_DIR}

# Channel names are read from the raw input directory, in the same order stage 1
# looks for them, so stages 1 and 4 always agree on the marker list.
CHANNEL_NAME_FILES = ("channelnames.txt", "MarkerList.txt")
