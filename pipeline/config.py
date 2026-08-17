"""Load, validate and hold pipeline configuration (from a YAML file)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any

from . import paths

try:
    import yaml
except ImportError as exc:  # pragma: no cover - surfaced with a clear message
    raise SystemExit(
        "PyYAML is required to run the pipeline orchestrator.\n"
        "Install it with:  pip install pyyaml"
    ) from exc


# ---------------------------------------------------------------------------
# Defaults derived from the stages' own production job scripts.  Stage *code*
# is not configurable: every tool lives in this repo (see paths.py).  Only the
# conda environment each stage runs in is, since that is machine-specific.
# ---------------------------------------------------------------------------
DEFAULT_CONDA = {
    "stack": "bigstream",         # tissue_pipeline/run_pipeline.py
    "align": "3DCellComposer",    # zalign/align_image.py
    "tile": "hubmap",             # 3Dtiler.py
    "segment": "3DCellComposer",  # 3DCellComposer/run_3DCellComposer.py
    "stitch": "hubmap",           # 3Dstitcher.py
}

# Per-stage SLURM resources copied from the real jobs in the repos.
DEFAULT_SLURM = {
    "stack":   {"partition": "WORKSPACES-CPU", "cpus": 8,  "mem": "500G",  "time": "24:00:00", "gres": None},
    "align":   {"partition": "batch",          "cpus": 16, "mem": "200G",  "time": "48:00:00", "gres": None},
    "tile":    {"partition": "WORKSPACES-CPU", "cpus": 8,  "mem": "3500G", "time": "12:00:00", "gres": None},
    "segment": {"partition": "GPU",            "cpus": 8,  "mem": "64G",   "time": "24:00:00", "gres": "gpu:1"},
    "stitch":  {"partition": "WORKSPACES-CPU", "cpus": 8,  "mem": "500G",  "time": "12:00:00", "gres": None},
}


@dataclass
class StackCfg:
    num_tissues: int = 8
    padding: int = 50
    skip_crop: bool = False


@dataclass
class AlignCfg:
    all_consecutive: bool = True
    save_flow: bool = True
    skip_optical_flow: bool = False


@dataclass
class TileCfg:
    num_tiles: int = 100
    overlap: int = 25
    prefix: str = "tile"


@dataclass
class SegmentCfg:
    nucleus_markers: str = "DAPI"
    cytoplasm_markers: str = "VIM"
    membrane_markers: str = "CD45,CD34,CD90,CD11C"
    segmentation_method: str = "deepcell"
    # Leave empty to use the marker list shipped with the raw input (the same
    # file stage 1 reads); set a path to override.
    channel_names_file: str = ""
    skip_blender: bool = True
    skip_eval: bool = False
    skip_yz: bool = False
    clear_cache: bool = False
    # Token used by DeepCell/Mesmer.  If empty, the value is inherited from the
    # submitting/parent environment variable named ``deepcell_token_env``.
    deepcell_token: str = ""
    deepcell_token_env: str = "DEEPCELL_ACCESS_TOKEN"
    # Extra raw args appended verbatim to run_3DCellComposer.py, e.g.
    # ["--downsample_vector", "1,2,2", "--min_slices", "3"].
    extra_args: list[str] = field(default_factory=list)


@dataclass
class StitchCfg:
    relabel: str = "global"                      # "global" or "local"
    # Which per-tile 3DCellComposer output(s) to stitch back into a full volume.
    source_masks: list[str] = field(default_factory=lambda: ["3D_cell_mask.tif"])


@dataclass
class SlurmStage:
    partition: str
    cpus: int
    mem: str
    time: str
    gres: str | None = None


@dataclass
class SlurmCfg:
    account: str | None = None
    extra_directives: list[str] = field(default_factory=list)
    stack: SlurmStage = None       # type: ignore[assignment]
    align: SlurmStage = None       # type: ignore[assignment]
    tile: SlurmStage = None        # type: ignore[assignment]
    segment: SlurmStage = None     # type: ignore[assignment]
    stitch: SlurmStage = None      # type: ignore[assignment]


@dataclass
class PipelineConfig:
    work_dir: str
    input_dir: str
    conda: dict[str, str]
    executor: str
    stack: StackCfg
    align: AlignCfg
    tile: TileCfg
    segment: SegmentCfg
    stitch: StitchCfg
    slurm: SlurmCfg

    # -- convenience accessors -----------------------------------------
    def resolved_channel_names_file(self) -> str:
        """The marker list for stage 4: config value, else the input's own."""
        if self.segment.channel_names_file:
            return _abspath(self.segment.channel_names_file)
        if not self.input_dir:
            return ""
        for name in paths.CHANNEL_NAME_FILES:
            candidate = os.path.join(self.input_dir, name)
            if os.path.isfile(candidate):
                return candidate
        return ""

    def resolved_deepcell_token(self) -> str:
        """Literal token from config, else the value of the configured env var."""
        if self.segment.deepcell_token:
            return self.segment.deepcell_token
        return os.environ.get(self.segment.deepcell_token_env, "")


def _abspath(path: str) -> str:
    """Absolute, ~-expanded path.

    Config paths are handed to stages that each run from their own directory, so
    a relative path has to be resolved once, here, against the directory the
    orchestrator was started from.
    """
    return os.path.abspath(os.path.expanduser(path))


def _merge(defaults: dict, override: dict | None) -> dict:
    out = dict(defaults)
    if override:
        out.update({k: v for k, v in override.items() if v is not None})
    return out


def load_config(path: str) -> PipelineConfig:
    with open(path, "r") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    if "work_dir" not in raw:
        raise SystemExit("config error: 'work_dir' is required")

    if "repos" in raw:
        print(
            "config warning: 'repos:' is ignored - all five stages now run from "
            f"this repository ({paths.REPO_ROOT}). Remove it from your config.",
            file=sys.stderr,
        )

    conda = _merge(DEFAULT_CONDA, raw.get("conda"))

    slurm_raw = raw.get("slurm") or {}
    slurm = SlurmCfg(
        account=slurm_raw.get("account"),
        extra_directives=list(slurm_raw.get("extra_directives", [])),
    )
    for stage in ("stack", "align", "tile", "segment", "stitch"):
        merged = _merge(DEFAULT_SLURM[stage], slurm_raw.get(stage))
        setattr(slurm, stage, SlurmStage(**merged))

    input_dir = raw.get("input_dir", "")
    cfg = PipelineConfig(
        work_dir=_abspath(raw["work_dir"]),
        input_dir=_abspath(input_dir) if input_dir else "",
        conda=conda,
        executor=raw.get("executor", "local"),
        stack=StackCfg(**(raw.get("stack") or {})),
        align=AlignCfg(**(raw.get("align") or {})),
        tile=TileCfg(**(raw.get("tile") or {})),
        segment=SegmentCfg(**(raw.get("segment") or {})),
        stitch=StitchCfg(**(raw.get("stitch") or {})),
        slurm=slurm,
    )
    return cfg


def validate_config(cfg: PipelineConfig, stages: list[int]) -> list[str]:
    """Return a list of human-readable problems (empty means OK)."""
    problems: list[str] = []

    if cfg.executor not in ("local", "slurm"):
        problems.append(f"executor must be 'local' or 'slurm', got '{cfg.executor}'")

    # Every stage's tool ships with this repo; a missing one means an incomplete
    # checkout rather than a misconfiguration.
    for stage in stages:
        script = paths.STAGE_SCRIPTS.get(stage)
        if script and not os.path.isfile(script):
            submodule = paths.SUBMODULE_STAGES.get(stage)
            if submodule:
                problems.append(
                    f"stage {stage}: the {os.path.basename(submodule)} submodule is "
                    "not checked out. Run: git submodule update --init --recursive"
                )
            else:
                problems.append(
                    f"stage {stage}: tool missing from this repository: {script}"
                )

    if 1 in stages:
        if not cfg.input_dir:
            problems.append("stage 1: 'input_dir' is required (raw qptiff directory)")
        elif not os.path.isdir(cfg.input_dir):
            problems.append(f"stage 1: input_dir does not exist: {cfg.input_dir}")

    if 4 in stages:
        channel_names = cfg.resolved_channel_names_file()
        if not channel_names:
            problems.append(
                "stage 4: no marker list found. Set 'segment.channel_names_file', "
                "or add one of "
                f"{'/'.join(paths.CHANNEL_NAME_FILES)} to input_dir"
                + (f" ({cfg.input_dir})" if cfg.input_dir else "")
            )
        elif not os.path.isfile(channel_names):
            problems.append(f"stage 4: channel_names_file not found: {channel_names}")
        if cfg.segment.segmentation_method not in ("deepcell", "cellpose", "custom"):
            problems.append(
                "stage 4: segmentation_method must be deepcell/cellpose/custom, "
                f"got '{cfg.segment.segmentation_method}'"
            )

    if 5 in stages and cfg.stitch.relabel not in ("global", "local"):
        problems.append(f"stage 5: stitch.relabel must be global/local, got '{cfg.stitch.relabel}'")

    return problems
