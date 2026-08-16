"""Translate configuration into concrete :class:`~pipeline.commands.Step`s.

There is one builder per pipeline stage.  Stage 4 (segmentation) has two
flavours: a *local* step that processes a single named tile, and a *SLURM
array* step whose tile is chosen at run time from ``SLURM_ARRAY_TASK_ID``.
Both share :func:`segment_invocation` so the tool is always called the same
way.
"""

from __future__ import annotations

import json
import os
import shlex

from . import paths
from .commands import Step, py
from .layout import Layout, _strip_tif


# ---------------------------------------------------------------------------
# Stage 1 - tissue matching & stacking  (tissue_pipeline/run_pipeline.py)
# ---------------------------------------------------------------------------
def stack_step(cfg, layout: Layout) -> Step:
    argv = [
        "python", "-u", paths.STACK_SCRIPT,
        "--input", cfg.input_dir,
        "--output", layout.stacked_dir,
        "--num-tissues", str(cfg.stack.num_tissues),
        "--padding", str(cfg.stack.padding),
    ]
    if cfg.stack.skip_crop:
        argv.append("--skip-crop")
    return Step(
        name="stack",
        stage="stack",
        env=cfg.conda["stack"],
        cwd=paths.STACK_DIR,
        invocation=py(argv),
    )


# ---------------------------------------------------------------------------
# Stage 2 - z-slice alignment  (zalign/align_image.py)
# ---------------------------------------------------------------------------
def align_step(cfg, layout: Layout, tissue: int) -> Step:
    argv = [
        "python", "-u", paths.ALIGN_SCRIPT,
        "--input", layout.stacked_tissue(tissue),
        "--output_dir", layout.aligned_dir(tissue),
    ]
    if cfg.align.all_consecutive:
        argv.append("--all_consecutive")
    if cfg.align.save_flow:
        argv.append("--save_flow")
    if cfg.align.skip_optical_flow:
        argv.append("--skip_optical_flow")
    return Step(
        name=f"align_tissue{tissue}",
        stage="align",
        env=cfg.conda["align"],
        cwd=paths.ALIGN_DIR,
        pre_lines=[f"mkdir -p {shlex.quote(layout.aligned_dir(tissue))}"],
        invocation=py(argv),
    )


# ---------------------------------------------------------------------------
# Stage 3 - 3D tiling  (3Dtiler.py)
# ---------------------------------------------------------------------------
def tile_step(cfg, layout: Layout, tissue: int) -> Step:
    argv = [
        "python", "-u", paths.TILE_SCRIPT,
        "--input_file", layout.aligned_stack(tissue),
        "--output_dir", layout.tiles_dir(tissue),
        "--num-tiles", str(cfg.tile.num_tiles),
        "--overlap", str(cfg.tile.overlap),
        "--prefix", cfg.tile.prefix,
    ]
    return Step(
        name=f"tile_tissue{tissue}",
        stage="tile",
        env=cfg.conda["tile"],
        cwd=paths.TILE_DIR,
        pre_lines=[f"mkdir -p {shlex.quote(layout.tiles_dir(tissue))}"],
        invocation=py(argv),
    )


# ---------------------------------------------------------------------------
# Stage 4 - 3D cell segmentation  (3DCellComposer/run_3DCellComposer.py)
# ---------------------------------------------------------------------------
def segment_invocation(cfg, image_expr: str, results_expr: str) -> str:
    """Build the run_3DCellComposer.py command line.

    ``image_expr`` and ``results_expr`` are inserted verbatim so callers can
    pass either a shell-quoted literal path (local) or a shell variable such
    as ``"${TILE_DIR}/${TILE}"`` (SLURM array).
    """
    seg = cfg.segment
    parts = [
        "python", "-u", paths.SEGMENT_SCRIPT,
        image_expr,
        shlex.quote(seg.nucleus_markers),
        shlex.quote(seg.cytoplasm_markers),
        shlex.quote(seg.membrane_markers),
        "--segmentation_method", shlex.quote(seg.segmentation_method),
        "--results_path", results_expr,
        "--channel_names", shlex.quote(cfg.resolved_channel_names_file()),
    ]
    # These flags use argparse `type=bool`, where ANY value is truthy; only
    # pass them (with "True") when we actually want them enabled.
    if seg.skip_blender:
        parts += ["--skip_blender", "True"]
    if seg.skip_eval:
        parts += ["--skip_eval", "True"]
    if seg.skip_yz:
        parts += ["--skipYZ", "True"]
    if seg.clear_cache:
        parts += ["--clear_cache", "True"]
    for extra in seg.extra_args:
        parts.append(shlex.quote(str(extra)))
    return " ".join(parts)


def _segment_exports(cfg) -> list[str]:
    exports = ["LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"]
    token = cfg.resolved_deepcell_token()
    if token:
        exports.append(f"DEEPCELL_ACCESS_TOKEN={shlex.quote(token)}")
    else:
        # Fall back to whatever the parent/submitting environment provides.
        src = cfg.segment.deepcell_token_env
        exports.append(f'DEEPCELL_ACCESS_TOKEN="${{{src}:-}}"')
    return exports


def segment_step_local(cfg, layout: Layout, tissue: int, tile_filename: str) -> Step:
    tile_path = os.path.join(layout.tiles_dir(tissue), tile_filename)
    results_dir = layout.seg_tile_result(tissue, tile_filename)
    invocation = segment_invocation(
        cfg, shlex.quote(tile_path), shlex.quote(results_dir)
    )
    stem = _strip_tif(tile_filename)
    return Step(
        name=f"segment_tissue{tissue}_{stem}",
        stage="segment",
        env=cfg.conda["segment"],
        cwd=paths.SEGMENT_DIR,
        exports=_segment_exports(cfg),
        pre_lines=[f"mkdir -p {shlex.quote(results_dir)}"],
        invocation=invocation,
    )


def segment_step_slurm_array(cfg, layout: Layout, tissue: int) -> Step:
    """A single array step; the tile is selected from SLURM_ARRAY_TASK_ID."""
    pick_tile = (
        "TILE=\"$(python3 -c 'import json,sys; "
        "print(json.load(open(sys.argv[1]))[\"tiles\"][int(sys.argv[2])][\"filename\"])' "
        "\"$META\" \"$SLURM_ARRAY_TASK_ID\")\""
    )
    pre_lines = [
        f"META={shlex.quote(layout.tile_metadata(tissue))}",
        f"TILE_DIR={shlex.quote(layout.tiles_dir(tissue))}",
        f"RESULTS_ROOT={shlex.quote(layout.seg_tiles_dir(tissue))}",
        pick_tile,
        'TILE_STEM="${TILE%.tif}"',
        'RESULTS_DIR="${RESULTS_ROOT}/${TILE_STEM}"',
        'mkdir -p "$RESULTS_DIR"',
        'echo "[segment] array task ${SLURM_ARRAY_TASK_ID}: ${TILE} -> ${RESULTS_DIR}"',
    ]
    invocation = segment_invocation(cfg, '"${TILE_DIR}/${TILE}"', '"${RESULTS_DIR}"')
    return Step(
        name=f"segment_tissue{tissue}",
        stage="segment",
        env=cfg.conda["segment"],
        cwd=paths.SEGMENT_DIR,
        exports=_segment_exports(cfg),
        pre_lines=pre_lines,
        invocation=invocation,
    )


# ---------------------------------------------------------------------------
# Stage 4 -> 5 bridge: gather per-tile masks into a stitch-ready directory
# ---------------------------------------------------------------------------
def read_tile_filenames(layout: Layout, tissue: int) -> list[str]:
    with open(layout.tile_metadata(tissue), "r") as handle:
        meta = json.load(handle)
    return [t["filename"] for t in meta["tiles"]]


def gather_masks(layout: Layout, tissue: int, source_mask_name: str, log=print) -> tuple[int, int]:
    """Symlink each per-tile mask to ``masks/<stem>/<tile_filename>``.

    3Dstitcher expects a directory of mask tiles whose filenames match the
    tiler metadata; 3DCellComposer instead writes ``3D_cell_mask.tif`` inside a
    per-tile results directory. This bridges the two.
    """
    tiles = read_tile_filenames(layout, tissue)
    masks_dir = layout.masks_dir(tissue, source_mask_name)
    os.makedirs(masks_dir, exist_ok=True)
    ok = 0
    for tile_filename in tiles:
        stem = _strip_tif(tile_filename)
        src = os.path.join(layout.seg_tiles_dir(tissue), stem, source_mask_name)
        dst = os.path.join(masks_dir, tile_filename)
        if os.path.exists(src):
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            ok += 1
        else:
            log(f"[gather] MISSING {src}")
    log(f"[gather] tissue {tissue}: linked {ok}/{len(tiles)} '{source_mask_name}' masks")
    return ok, len(tiles)


def _gather_shell_snippet() -> str:
    """A stdlib-only python3 snippet used inside SLURM stitch jobs."""
    return (
        "python3 -c '\n"
        "import json, os, sys\n"
        "meta = json.load(open(sys.argv[1]))\n"
        "masks_dir, src_root, name = sys.argv[2], sys.argv[3], sys.argv[4]\n"
        "os.makedirs(masks_dir, exist_ok=True)\n"
        "ok = 0\n"
        "for t in meta[\"tiles\"]:\n"
        "    fn = t[\"filename\"]\n"
        "    stem = fn[:-4] if fn.endswith(\".tif\") else fn\n"
        "    src = os.path.join(src_root, stem, name)\n"
        "    dst = os.path.join(masks_dir, fn)\n"
        "    if os.path.exists(src):\n"
        "        if os.path.lexists(dst): os.remove(dst)\n"
        "        os.symlink(os.path.abspath(src), dst); ok += 1\n"
        "    else: print(\"[gather] MISSING\", src)\n"
        "print(\"[gather] linked %d/%d masks\" % (ok, len(meta[\"tiles\"])))\n"
        "' \"$META\" \"$MASKS_DIR\" \"$RESULTS_ROOT\" \"$MASK_NAME\""
    )


# ---------------------------------------------------------------------------
# Stage 5 - 3D stitching  (3Dstitcher.py)
# ---------------------------------------------------------------------------
def stitch_step_local(cfg, layout: Layout, tissue: int, source_mask_name: str) -> Step:
    """Assumes masks were already gathered (call :func:`gather_masks` first)."""
    masks_dir = layout.masks_dir(tissue, source_mask_name)
    out_path = layout.stitched_mask(tissue, source_mask_name)
    argv = [
        "python", "-u", paths.STITCH_SCRIPT,
        masks_dir,
        layout.tile_metadata(tissue),
        out_path,
        "--relabel", cfg.stitch.relabel,
    ]
    stem = _strip_tif(source_mask_name)
    return Step(
        name=f"stitch_tissue{tissue}_{stem}",
        stage="stitch",
        env=cfg.conda["stitch"],
        cwd=paths.STITCH_DIR,
        pre_lines=[f"mkdir -p {shlex.quote(layout.stitched_dir(tissue))}"],
        invocation=py(argv),
    )


def stitch_step_slurm(cfg, layout: Layout, tissue: int, source_mask_name: str) -> Step:
    """A single SLURM job that gathers masks then runs 3Dstitcher."""
    masks_dir = layout.masks_dir(tissue, source_mask_name)
    out_path = layout.stitched_mask(tissue, source_mask_name)
    pre_lines = [
        f"META={shlex.quote(layout.tile_metadata(tissue))}",
        f"RESULTS_ROOT={shlex.quote(layout.seg_tiles_dir(tissue))}",
        f"MASKS_DIR={shlex.quote(masks_dir)}",
        f"MASK_NAME={shlex.quote(source_mask_name)}",
        f"STITCH_OUT={shlex.quote(out_path)}",
        'mkdir -p "$(dirname "$STITCH_OUT")"',
        "echo '[stitch] gathering per-tile masks ...'",
        _gather_shell_snippet(),
    ]
    invocation = (
        f"python -u {shlex.quote(paths.STITCH_SCRIPT)} "
        '"$MASKS_DIR" "$META" "$STITCH_OUT" '
        f"--relabel {shlex.quote(cfg.stitch.relabel)}"
    )
    stem = _strip_tif(source_mask_name)
    return Step(
        name=f"stitch_tissue{tissue}_{stem}",
        stage="stitch",
        env=cfg.conda["stitch"],
        cwd=paths.STITCH_DIR,
        pre_lines=pre_lines,
        invocation=invocation,
    )
