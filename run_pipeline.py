#!/usr/bin/env python3
"""SectionAligner — orchestrate 3D tissue → cell segmentation.

Five stages: stack → align → tile → segment → stitch. Every stage runs a tool
from this repository, so a run is fully described by one config file.

Run the whole pipeline, or any contiguous/selected subset of stages, either
locally (sequential subprocesses) or on SLURM (chained jobs).

Examples
--------
    # Dry run: print every command / write every sbatch script, run nothing.
    python run_pipeline.py --config config.yaml --dry-run

    # Full pipeline locally.
    python run_pipeline.py --config config.yaml --executor local

    # Submit the full pipeline to SLURM.
    python run_pipeline.py --config config.yaml --executor slurm

    # Only re-run tiling + segmentation + stitching for tissues 2 and 5.
    python run_pipeline.py --config config.yaml --stages 3-5 --tissues 2,5
"""

from __future__ import annotations

import argparse
import sys

from pipeline import paths
from pipeline.config import load_config, validate_config
from pipeline.orchestrator import Orchestrator


ALL_STAGES = [1, 2, 3, 4, 5]
STAGE_NAMES = {
    1: "stack   (match tissues and stack z-slices)",
    2: "align   (register z-slices)",
    3: "tile    (split volume into tiles)",
    4: "segment (3D cell segmentation per tile)",
    5: "stitch  (reassemble tile masks)",
}


def parse_stages(spec: str) -> list[int]:
    """Parse "1-5", "2,3", "3", "1-3,5" -> sorted list of stage numbers."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            result.update(range(int(lo), int(hi) + 1))
        else:
            result.add(int(part))
    invalid = sorted(s for s in result if s not in ALL_STAGES)
    if invalid:
        raise SystemExit(f"invalid stage(s): {invalid} (valid: 1-5)")
    return sorted(result)


def parse_int_list(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    return [int(x) for x in spec.replace(",", " ").split()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Path to config YAML")
    parser.add_argument("--stages", "-s", default="1-5",
                        help="Stages to run: e.g. '1-5', '2,3', '3-5' (default: 1-5)")
    parser.add_argument("--tissues", "-t", default=None,
                        help="Comma/space list of tissue ids to process (default: all)")
    parser.add_argument("--executor", "-e", choices=["local", "slurm"], default=None,
                        help="Override the executor set in the config file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands / write sbatch scripts without executing")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.executor:
        cfg.executor = args.executor

    stages_to_run = parse_stages(args.stages)
    tissues_filter = parse_int_list(args.tissues)

    problems = validate_config(cfg, stages_to_run)
    if problems:
        print("Configuration problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 2

    print("=" * 72)
    print("SectionAligner pipeline")
    print("=" * 72)
    print(f"  repo     : {paths.REPO_ROOT}")
    print(f"  work_dir : {cfg.work_dir}")
    print(f"  executor : {cfg.executor}" + ("  (DRY RUN)" if args.dry_run else ""))
    print(f"  stages   : {stages_to_run}")
    for stage in stages_to_run:
        print(f"             {stage}. {STAGE_NAMES[stage]}")
    print(f"  tissues  : {tissues_filter if tissues_filter else 'all'}")
    print("=" * 72)

    orchestrator = Orchestrator(cfg, stages_to_run, tissues_filter, dry_run=args.dry_run)
    orchestrator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
