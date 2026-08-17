# SectionAligner

SectionAligner takes raw multiplexed tissue sections all the way to a stitched
3D cell-segmentation volume. The five stages of that workflow all live in this
repository and are driven by a single orchestrator, so a run is described
entirely by one config file: `python run_pipeline.py --config config.yaml`.

```
 raw *.qptiff              tissue_k_stacked      aligned_stack        tiles/             per-tile masks        stitched volume
 (N tissues / slice)  ─▶   .ome.tif (ZCYX)  ─▶   .ome.tif (ZCYX)  ─▶  tile_y*_x*.tif ─▶  3D_cell_mask.tif ─▶   stitched_3D_cell_mask.tif
        │                        │                     │                    │                   │                      │
   ┌────┴─────┐            ┌─────┴─────┐         ┌──────┴─────┐       ┌──────┴─────┐      ┌───────┴──────┐        ┌──────┴──────┐
   │ Stage 1  │            │  Stage 2  │         │  Stage 3   │       │  Stage 4   │      │              │        │  Stage 5    │
   │  STACK   │            │  ALIGN    │         │   TILE     │       │  SEGMENT   │      │  (per tile)  │        │  STITCH     │
   └──────────┘            └───────────┘         └────────────┘       └────────────┘      └──────────────┘        └─────────────┘
```

| # | Stage | Tool in this repo | In → Out |
|---|-------|-------------------|----------|
| 1 | **Stack** | `tissue_pipeline/run_pipeline.py` | directory of `*.qptiff` (N tissues per slice) → `tissue_{k}_stacked.ome.tif` per tissue (ZCYX) |
| 2 | **Align** | `zalign/align_image.py` | stacked volume → `aligned_stack.ome.tif` (ZCYX, z-slices registered) |
| 3 | **Tile** | `3Dtiler.py` | aligned volume → `tile_y####_x####.tif` + `tile_metadata.json` |
| 4 | **Segment** | `3DCellComposer/run_3DCellComposer.py` | each tile → `3D_cell_mask.tif`, `3D_nuclear_mask.tif`, `metrics.json`, … |
| 5 | **Stitch** | `3Dstitcher.py` | per-tile masks + metadata → `stitched_3D_cell_mask.tif` (ZYX label volume) |

Each stage runs as a subprocess in its own conda environment (named in
`config.yaml`). The orchestrator builds the commands, manages the output
directory layout, gathers per-tile masks before stitching, and chains SLURM jobs
with `afterok` dependencies when requested. Stage code paths are resolved from
the location of `pipeline/paths.py`, so any clone works from any directory
without configuration.

## Repository layout

```
SectionAligner/
  run_pipeline.py       pipeline CLI entry point
  config.example.yaml   template configuration (copy to config.yaml)
  pipeline/
    paths.py              where each stage's tool lives in this repo
    config.py             load and validate YAML config
    layout.py             fixed work_dir paths and naming conventions
    stages.py             per-stage command construction
    executor.py           local subprocess vs SLURM submission
    orchestrator.py       stage DAG and tissue/tile fan-out
  tissue_pipeline/      stage 1: tissue matching, stacking and cropping
  zalign/               stage 2: z-slice registration (rigid + optical flow)
  3Dtiler.py            stage 3: split a volume into overlapping 3D tiles
  3DCellComposer/       stage 4: 3D cell segmentation per tile (git submodule)
  3Dstitcher.py         stage 5: stitch per-tile masks into one volume
  qc/                   debug and QC tools, run by hand outside a pipeline run
  main.py               standalone single-image tissue detection + alignment
```

Stages 2 and 4 were previously separate checkouts. `zalign/` holds the
DeepCell-era alignment scripts and is committed here directly. `3DCellComposer/`
is a submodule pinned to the `sectionaligner-pipeline` branch of
`hubmapconsortium/3DCellComposer`, which carries the changes this pipeline
depends on — most importantly `--channel_names`, which upstream does not have.

To change stage 4, commit inside the submodule, push that branch, then commit
the moved pointer in this repo:

```bash
cd 3DCellComposer
git commit -am "..." && git push
cd .. && git add 3DCellComposer && git commit -m "bump 3DCellComposer"
```

`git submodule update --remote 3DCellComposer` pulls the latest tip of that
branch when someone else has moved it.

## Quick start

Stage 4 is a submodule, so clone with `--recurse-submodules` (or run
`git submodule update --init --recursive` in an existing clone):

```bash
git clone --recurse-submodules git@github.com:murphygroup/SectionAligner.git
```

```bash
conda activate <env with PyYAML>      # the orchestrator needs nothing else
cp config.example.yaml config.yaml    # then edit input_dir, work_dir, markers

# 1) Dry run — prints every command and writes every sbatch script, runs nothing.
python run_pipeline.py --config config.yaml --dry-run

# 2) Run the whole pipeline locally (sequential subprocesses on this node).
python run_pipeline.py --config config.yaml --executor local

# 3) Submit the whole pipeline to SLURM (chained with afterok dependencies).
python run_pipeline.py --config config.yaml --executor slurm
```

### Running only some stages / tissues

`--stages` accepts ranges and lists; `--tissues` filters which tissues to process:

```bash
# Re-tile, re-segment and re-stitch only tissues 2 and 5:
python run_pipeline.py -c config.yaml --stages 3-5 --tissues 2,5

# Just the alignment step for all tissues:
python run_pipeline.py -c config.yaml --stages 2
```

Stages can be resumed independently as long as the previous stage's outputs
exist in `work_dir` (the layout is fixed and discoverable).

## Output layout

Everything for a run lives under `work_dir`:

```
work_dir/
  01_stacked/         tissue_{k}_stacked.ome.tif      + qc/
  02_aligned/tissue_{k}/   aligned_stack.ome.tif      + params/QC
  03_tiles/tissue_{k}/     tile_y####_x####.tif       + tile_metadata.json
  04_segmentation/tissue_{k}/
      tiles/tile_y####_x####/  3D_cell_mask.tif, 3D_nuclear_mask.tif, metrics.json, ...
      masks/3D_cell_mask/      tile_y####_x####.tif   (symlinks, gathered for stitching)
  05_stitched/tissue_{k}/  stitched_3D_cell_mask.tif  (+ _contributions.tif)
  logs/               <step>.log (local) / <step>-<jobid>.out (slurm)
  slurm/              generated .sbatch scripts
```

## Environments

`run_pipeline.py` itself needs only **Python 3.8+ and PyYAML**. The heavy
scientific dependencies belong to the per-stage conda environments named under
`conda:` in `config.yaml` — that mapping is the only machine-specific setting
left, since stage code is no longer configurable.

| Stage | requirements |
|-------|--------------|
| 1 stack | `tissue_pipeline/requirements.txt` |
| 2 align | `zalign/requirements.txt` |
| 3 tile, 5 stitch | `requirements.txt` (repo root) |
| 4 segment | `3DCellComposer/requirements.txt` |

Stage 4 needs a DeepCell/Mesmer token: `export DEEPCELL_ACCESS_TOKEN=...` or set
`segment.deepcell_token` in the config.

Marker names for stage 4 must match the channel list. By default the pipeline
reads the same list stage 1 does — `channelnames.txt` or `MarkerList.txt` in
`input_dir` — so the two cannot drift apart; override with
`segment.channel_names_file`.

## How execution works

- **local** — stages run one after another in this process. Output streams to
  the console and to `work_dir/logs/<step>.log`. Good for small/test data or a
  single big allocation. Stage 4 runs every tile sequentially, so for a full
  100-tile tissue on a GPU, SLURM is usually a better fit.
- **slurm** — one job (or array) per stage per tissue, chained with
  `--dependency=afterok`. Stage 4 is submitted as an array (`0..num_tiles-1`);
  each task reads `tile_metadata.json` at run time and segments its own tile.
  Stage 5 gathers masks and stitches in a single dependent job.

## Notes & caveats

- **`num_tiles`** should be a perfect square (100 → 10×10). The tiler builds
  an approximately-square grid and asserts full coverage; a perfect square
  guarantees the requested count is produced and the SLURM array size matches.
- **Segment all tiles.** By default every tile is segmented so stage 5 can
  reconstruct the whole tissue.
- **Memory.** Stage 3 loads the whole aligned volume into RAM (large-memory
  node); tune SLURM memory/time in `config.yaml` for your cluster.
- **Partitions/envs** in `config.example.yaml` are example values for one
  cluster; change them to match yours.

## QC and debugging

These are run by hand against a finished run, not as pipeline stages.

**Stitch seams.** The hardest thing to judge in stage 5 is whether two labels on
either side of a tile overlap are one cell or two.
`qc/debug_stitch_edge_merges.py` reads the per-tile masks *before* stitching and,
for every overlapping tile pair, reports label-pair IoU, coverage and centroid
distance — the evidence a merge decision rests on. Pass it the gathered masks and
the tiling metadata, optionally with the stitched result and its
`_contributions.tif` sidecar to summarise what stitching actually did:

```bash
python qc/debug_stitch_edge_merges.py \
    work_dir/04_segmentation/tissue_1/masks/3D_cell_mask \
    work_dir/03_tiles/tissue_1/tile_metadata.json \
    --output-dir stitch_edge_debug \
    --stitched-mask work_dir/05_stitched/tissue_1/stitched_3D_cell_mask.tif

python qc/plot_stitch_edge_merge_report.py \
    stitch_edge_debug/edge_merge_report.json \
    --output stitch_edge_debug/edge_merge_qc.html
```

The second script turns that JSON into an HTML report with per-seam figures; the
first also writes red/green/yellow boundary overlays per seam unless you pass
`--no-boundary-pngs`.

**Stage 1 cropping.** `tissue_pipeline/validate_centered_tissue2.py` re-runs the
centered-crop logic for a single tissue and writes to its own directory so the
existing stack is untouched; `--plan-only` stops before writing the OME-TIFF. It
reads the dataset from `$SECTIONALIGNER_DATA_ROOT` (defaulting to the original
scratch directory).

**Segmentation scoring** is not here: it is `evaluation/single_method_eval_3D.py`
inside the 3DCellComposer submodule, which stage 4 runs unless you set
`segment.skip_eval`.

## Standalone tissue detection and alignment (`main.py`)

`main.py` predates the pipeline and still works on a single OME-TIFF, doing
tissue detection, cropping and alignment in one process:

```bash
python main.py --input_path "path/to/image.ome.tiff" --output_dir "path/to/output" \
  --num_tissue 8 --pixel_size [0.5073519424785282, 0.5073519424785282] --crop_only False
```

Key options: `--num_tissue` (tissues to detect, default 8), `--pixel_size`
(microns; determines downsampling, so it affects results), `--crop_only` (detect
and crop without aligning). Others: `--level`, `--thresh`, `--kernel_size`,
`--holes_thresh`, `--scale_factor`, `--padding`, `--connect`,
`--output_file_basename`, `--align_upsample_factor`, `--optimize` (Optuna
parameter search). For new work prefer stages 1–2 of the pipeline, which handle
multi-slice datasets and tissue matching across slices.

## Contact

Ted Zhang [tedz@andrew.cmu.edu] \
Bob Murphy [murphy@andrew.cmu.edu]
