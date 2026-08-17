"""Wire stages + executor into the full pipeline DAG.

Shape of the DAG::

    stage 1 (once)
        └── for each tissue k:
              stage 2 ─▶ stage 3 ─▶ stage 4 (fan-out over tiles) ─▶ stage 5 (fan-in)

Local runs everything sequentially on one node. SLURM submits one job (or
array) per stage per tissue and chains them with ``afterok`` dependencies.
"""

from __future__ import annotations

import json
import os

from . import stages
from .config import PipelineConfig
from .executor import LocalExecutor, SlurmExecutor, SlurmResources
from .layout import Layout


class Orchestrator:
    def __init__(self, cfg: PipelineConfig, stages_to_run: list[int],
                 tissues_filter: list[int] | None, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.stages_to_run = sorted(set(stages_to_run))
        self.tissues_filter = tissues_filter
        self.dry_run = dry_run
        self.layout = Layout(cfg.work_dir, tile_prefix=cfg.tile.prefix)

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.layout.ensure_run_dirs()
        if self.cfg.executor == "slurm":
            self._run_slurm()
        else:
            self._run_local()

    # ------------------------------------------------------------------
    # Local
    # ------------------------------------------------------------------
    def _run_local(self) -> None:
        cfg, layout = self.cfg, self.layout
        ex = LocalExecutor(layout.logs_dir, dry_run=self.dry_run)

        if 1 in self.stages_to_run:
            ex.run(stages.stack_step(cfg, layout))

        tissues = self._resolve_tissues_local()
        if not tissues:
            raise SystemExit(
                "No tissues to process. Run stage 1 first, or check "
                f"{layout.stacked_dir} / config 'stack.num_tissues'."
            )
        print(f"\n>>> Processing tissues: {tissues}", flush=True)

        for tissue in tissues:
            if 2 in self.stages_to_run:
                ex.run(stages.align_step(cfg, layout, tissue))
            if 3 in self.stages_to_run:
                ex.run(stages.tile_step(cfg, layout, tissue))
            if 4 in self.stages_to_run:
                meta = layout.tile_metadata(tissue)
                if self.dry_run and not os.path.isfile(meta):
                    print(f">>> tissue {tissue}: (dry-run) tiles not created yet; "
                          f"one segment step will run per tile from {meta}. "
                          f"Showing one representative command:", flush=True)
                    ex.run(stages.segment_step_local(cfg, layout, tissue, "tile_y0000_x0000.tif"))
                else:
                    tile_files = self._tile_files_or_die(tissue)
                    print(f">>> tissue {tissue}: segmenting {len(tile_files)} tiles", flush=True)
                    for tile_filename in tile_files:
                        ex.run(stages.segment_step_local(cfg, layout, tissue, tile_filename))
            if 5 in self.stages_to_run:
                for mask in cfg.stitch.source_masks:
                    if not self.dry_run:
                        stages.gather_masks(layout, tissue, mask)
                    ex.run(stages.stitch_step_local(cfg, layout, tissue, mask))

        print("\n>>> Local pipeline finished.", flush=True)
        self._print_outputs(tissues)

    # ------------------------------------------------------------------
    # SLURM
    # ------------------------------------------------------------------
    def _run_slurm(self) -> None:
        cfg, layout = self.cfg, self.layout
        ex = SlurmExecutor(layout.slurm_dir, layout.logs_dir, dry_run=self.dry_run)

        stack_job: str | None = None
        if 1 in self.stages_to_run:
            stack_job = ex.submit(stages.stack_step(cfg, layout), self._res("stack"))

        tissues = self._resolve_tissues_slurm()
        print(f"\n>>> Submitting jobs for tissues: {tissues}", flush=True)

        submitted: list[str] = []
        for tissue in tissues:
            prev = stack_job  # shared stage-1 dependency (may be None)

            if 2 in self.stages_to_run:
                job = ex.submit(stages.align_step(cfg, layout, tissue),
                                self._res("align"), _deps(prev))
                submitted.append(job)
                prev = job

            if 3 in self.stages_to_run:
                job = ex.submit(stages.tile_step(cfg, layout, tissue),
                                self._res("tile"), _deps(prev))
                submitted.append(job)
                prev = job

            if 4 in self.stages_to_run:
                array_size = self._array_size(tissue)
                job = ex.submit(stages.segment_step_slurm_array(cfg, layout, tissue),
                                self._res("segment", array=f"0-{array_size - 1}"),
                                _deps(prev))
                submitted.append(job)
                prev = job

            if 5 in self.stages_to_run:
                for mask in cfg.stitch.source_masks:
                    job = ex.submit(stages.stitch_step_slurm(cfg, layout, tissue, mask),
                                    self._res("stitch"), _deps(prev))
                    submitted.append(job)

        print(f"\n>>> Submitted {len(submitted)} jobs.", flush=True)
        if not self.dry_run:
            print("    Monitor with:  squeue -u $USER")
        print(f"    sbatch scripts: {layout.slurm_dir}")
        print(f"    logs:           {layout.logs_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _res(self, stage: str, array: str | None = None) -> SlurmResources:
        s = getattr(self.cfg.slurm, stage)
        return SlurmResources(
            partition=s.partition, cpus=s.cpus, mem=s.mem, time=s.time,
            gres=s.gres, account=self.cfg.slurm.account, array=array,
            extra_directives=tuple(self.cfg.slurm.extra_directives),
        )

    def _apply_filter(self, tissues: list[int]) -> list[int]:
        if self.tissues_filter is None:
            return tissues
        wanted = set(self.tissues_filter)
        return [t for t in tissues if t in wanted]

    def _resolve_tissues_local(self) -> list[int]:
        discovered = self.layout.discover_stacked_tissues()
        if not discovered:
            discovered = list(range(1, self.cfg.stack.num_tissues + 1))
        return self._apply_filter(discovered)

    def _resolve_tissues_slurm(self) -> list[int]:
        # Stage-1 outputs don't exist yet at submit time, so enumerate by count.
        discovered = self.layout.discover_stacked_tissues()
        if not discovered or 1 in self.stages_to_run:
            discovered = list(range(1, self.cfg.stack.num_tissues + 1))
        return self._apply_filter(discovered)

    def _tile_files_or_die(self, tissue: int) -> list[str]:
        meta = self.layout.tile_metadata(tissue)
        if not os.path.isfile(meta):
            raise SystemExit(
                f"tissue {tissue}: tile metadata not found ({meta}). "
                "Run stage 3 (tile) before stage 4 (segment)."
            )
        return stages.read_tile_filenames(self.layout, tissue)

    def _array_size(self, tissue: int) -> int:
        """Actual tile count if tiling already ran, else the configured count."""
        meta = self.layout.tile_metadata(tissue)
        if os.path.isfile(meta):
            with open(meta, "r") as handle:
                return len(json.load(handle)["tiles"])
        return self.cfg.tile.num_tiles

    def _print_outputs(self, tissues: list[int]) -> None:
        print("\nOutputs:")
        for tissue in tissues:
            for mask in self.cfg.stitch.source_masks:
                path = self.layout.stitched_mask(tissue, mask)
                exists = "ok" if os.path.exists(path) else "--"
                print(f"  [{exists}] tissue {tissue}: {path}")


def _deps(job: str | None) -> list[str]:
    return [job] if job else []
