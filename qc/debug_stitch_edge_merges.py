#!/usr/bin/env python3
"""Audit cell-label continuity across stitching overlap edges.

This script inspects the original per-tile mask outputs before stitching. For
each overlapping tile pair in the tiling metadata it measures which labels touch
the same overlap voxels and reports label-pair IoU, coverage, and centroid
distance. Those metrics are the evidence used to decide whether two edge labels
probably represent the same cell.

Example:
    python debug_stitch_edge_merges.py \
        3DCODEX_3_TILES \
        3DCODEX_3_TILES/tile_metadata.json \
        --output-dir stitch_edge_debug
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image


Bounds = Tuple[int, int, int, int, int, int]
Overlap = Tuple[int, int, int, int, int, int]


def load_metadata(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        metadata = json.load(handle)
    if "tiles" not in metadata or not isinstance(metadata["tiles"], list):
        raise ValueError(f"{path} does not look like 3Dstitcher tile metadata")
    return metadata


def tile_bounds(tile: Dict[str, Any]) -> Bounds:
    pos = tile["position"]
    z0, z1 = pos["z"]
    y0, y1 = pos["y"]
    x0, x1 = pos["x"]
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def pair_overlap(a: Bounds, b: Bounds) -> Optional[Overlap]:
    z0 = max(a[0], b[0])
    z1 = min(a[1], b[1])
    y0 = max(a[2], b[2])
    y1 = min(a[3], b[3])
    x0 = max(a[4], b[4])
    x1 = min(a[5], b[5])
    if z0 >= z1 or y0 >= y1 or x0 >= x1:
        return None
    return z0, z1, y0, y1, x0, x1


def overlap_shape(overlap: Overlap) -> Tuple[int, int, int]:
    z0, z1, y0, y1, x0, x1 = overlap
    return z1 - z0, y1 - y0, x1 - x0


def classify_overlap(a: Bounds, b: Bounds, overlap: Overlap) -> str:
    _, _, y0, y1, x0, x1 = overlap
    overlap_y = y1 - y0
    overlap_x = x1 - x0
    min_tile_y = min(a[3] - a[2], b[3] - b[2])
    min_tile_x = min(a[5] - a[4], b[5] - b[4])
    thin_y = overlap_y < min_tile_y
    thin_x = overlap_x < min_tile_x
    if thin_x and thin_y:
        return "corner"
    if thin_x:
        return "vertical_x"
    if thin_y:
        return "horizontal_y"
    return "area"


def resolve_tile_path(tiles_dir: Path, filename: str) -> Path:
    path = tiles_dir / filename
    if path.exists():
        return path
    stem_path = tiles_dir / Path(filename).name
    if stem_path.exists():
        return stem_path
    return path


def mask_to_zyx(mask: np.ndarray, channel: int) -> np.ndarray:
    """Normalize a 2D, 3D, or zcyx mask array to zyx."""
    if mask.ndim == 2:
        return mask[np.newaxis, :, :]
    if mask.ndim == 3:
        return mask
    if mask.ndim == 4:
        if channel >= mask.shape[1]:
            raise ValueError(
                f"Requested channel {channel}, but mask only has {mask.shape[1]} channels"
            )
        return mask[:, channel, :, :]
    raise ValueError(f"Expected a 2D/3D/4D mask, got shape {mask.shape}")


class TileCache:
    def __init__(self, tiles_dir: Path, tiles: Sequence[Dict[str, Any]], channel: int, size: int):
        self.tiles_dir = tiles_dir
        self.tiles = tiles
        self.channel = channel
        self.size = max(1, size)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, tile_idx: int) -> np.ndarray:
        if tile_idx in self._cache:
            mask = self._cache.pop(tile_idx)
            self._cache[tile_idx] = mask
            return mask

        tile = self.tiles[tile_idx]
        path = resolve_tile_path(self.tiles_dir, tile["filename"])
        if not path.exists():
            raise FileNotFoundError(path)

        mask = mask_to_zyx(tifffile.imread(path), self.channel)
        self._cache[tile_idx] = mask
        while len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return mask


def local_slices(bounds: Bounds, overlap: Overlap) -> Tuple[slice, slice, slice]:
    z0, _, y0, _, x0, _ = bounds
    oz0, oz1, oy0, oy1, ox0, ox1 = overlap
    return (
        slice(oz0 - z0, oz1 - z0),
        slice(oy0 - y0, oy1 - y0),
        slice(ox0 - x0, ox1 - x0),
    )


def label_counts(mask: np.ndarray) -> Dict[int, int]:
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def centroid(mask: np.ndarray, label: int, cache: Dict[int, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if label in cache:
        return cache[label]
    coords = np.argwhere(mask == label)
    if coords.size == 0:
        result = (math.nan, math.nan, math.nan)
    else:
        z, y, x = coords.mean(axis=0)
        result = (float(z), float(y), float(x))
    cache[label] = result
    return result


def distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    if any(math.isnan(value) for value in (*a, *b)):
        return math.nan
    return float(math.sqrt(sum((av - bv) ** 2 for av, bv in zip(a, b))))


def rule_key(
    iou: float,
    max_coverage: float,
    centroid_dist: float,
    shared_voxels: int,
    args: argparse.Namespace,
) -> str:
    rules: List[str] = []
    if iou >= args.iou_threshold:
        rules.append("iou")
    if max_coverage >= args.coverage_threshold:
        rules.append("coverage")
    if (
        shared_voxels >= args.centroid_min_shared_pixels
        and not math.isnan(centroid_dist)
        and centroid_dist <= args.centroid_dist_threshold
    ):
        rules.append("centroid")
    return "+".join(rules)


def boundary_2d(labels: np.ndarray) -> np.ndarray:
    fg = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    if labels.size == 0:
        return boundary

    boundary[0, :] |= fg[0, :]
    boundary[-1, :] |= fg[-1, :]
    boundary[:, 0] |= fg[:, 0]
    boundary[:, -1] |= fg[:, -1]

    y_diff = labels[1:, :] != labels[:-1, :]
    boundary[1:, :] |= fg[1:, :] & y_diff
    boundary[:-1, :] |= fg[:-1, :] & y_diff

    x_diff = labels[:, 1:] != labels[:, :-1]
    boundary[:, 1:] |= fg[:, 1:] & x_diff
    boundary[:, :-1] |= fg[:, :-1] & x_diff
    return boundary


def boundary_projection(
    mask: np.ndarray,
    matched_labels: set[int],
    z_selection: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if z_selection == "max":
        slices = range(mask.shape[0])
    else:
        z_index = int(z_selection)
        if z_index < 0 or z_index >= mask.shape[0]:
            raise ValueError(f"Boundary z index {z_index} is outside overlap z range 0-{mask.shape[0] - 1}")
        slices = [z_index]

    any_boundary = np.zeros(mask.shape[1:], dtype=bool)
    matched_boundary = np.zeros(mask.shape[1:], dtype=bool)
    for z_index in slices:
        z_mask = mask[z_index]
        z_boundary = boundary_2d(z_mask)
        any_boundary |= z_boundary
        if matched_labels:
            matched_boundary |= z_boundary & np.isin(z_mask, list(matched_labels))
    return any_boundary, matched_boundary


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep).strip("_")


def write_boundary_overlay_png(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    matched_labels_a: set[int],
    matched_labels_b: set[int],
    pair_id: str,
    tile_a: str,
    tile_b: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_a, merged_boundary_a = boundary_projection(mask_a, matched_labels_a, args.boundary_z)
    boundary_b, merged_boundary_b = boundary_projection(mask_b, matched_labels_b, args.boundary_z)

    merged = merged_boundary_a | merged_boundary_b
    red = boundary_a & ~merged
    green = boundary_b & ~merged & ~red

    image = np.zeros((*boundary_a.shape, 3), dtype=np.uint8)
    image[green] = (0, 255, 0)
    image[red] = (255, 0, 0)
    image[merged] = (255, 255, 0)

    png = Image.fromarray(image, mode="RGB")
    if args.boundary_scale > 1:
        width, height = png.size
        nearest = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        png = png.resize(
            (width * args.boundary_scale, height * args.boundary_scale),
            resample=nearest,
        )

    name = "{pair}_{a}__{b}_boundary_overlay.png".format(
        pair=safe_name(pair_id),
        a=safe_name(Path(tile_a).stem),
        b=safe_name(Path(tile_b).stem),
    )
    path = output_dir / name
    png.save(path)
    return path


def iter_overlapping_pairs(
    tiles: Sequence[Dict[str, Any]], skip_corners: bool
) -> Iterable[Tuple[int, int, Bounds, Bounds, Overlap, str]]:
    bounds = [tile_bounds(tile) for tile in tiles]
    for i in range(len(tiles)):
        for j in range(i + 1, len(tiles)):
            overlap = pair_overlap(bounds[i], bounds[j])
            if overlap is None:
                continue
            seam_type = classify_overlap(bounds[i], bounds[j], overlap)
            if skip_corners and seam_type == "corner":
                continue
            yield i, j, bounds[i], bounds[j], overlap, seam_type


def analyze_pair(
    tile_i: int,
    tile_j: int,
    bounds_i: Bounds,
    bounds_j: Bounds,
    overlap: Overlap,
    seam_type: str,
    cache: TileCache,
    tiles: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    stitched_mask: Optional[np.ndarray],
    contributions: Optional[np.ndarray],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    mask_i = cache.get(tile_i)[local_slices(bounds_i, overlap)]
    mask_j = cache.get(tile_j)[local_slices(bounds_j, overlap)]

    counts_i = label_counts(mask_i)
    counts_j = label_counts(mask_j)
    nonzero_i = int(np.count_nonzero(mask_i))
    nonzero_j = int(np.count_nonzero(mask_j))
    both = (mask_i > 0) & (mask_j > 0)
    shared_nonzero = int(np.count_nonzero(both))

    pair_rows: List[Dict[str, Any]] = []
    matches_by_label_i: Dict[int, set[int]] = defaultdict(set)
    matches_by_label_j: Dict[int, set[int]] = defaultdict(set)
    rule_counts: Dict[str, int] = defaultdict(int)
    centroid_i: Dict[int, Tuple[float, float, float]] = {}
    centroid_j: Dict[int, Tuple[float, float, float]] = {}

    if shared_nonzero:
        label_pairs = np.stack([mask_i[both], mask_j[both]], axis=1)
        unique_pairs, shared_counts = np.unique(label_pairs, axis=0, return_counts=True)
        for (label_i, label_j), shared_count in zip(unique_pairs, shared_counts):
            shared = int(shared_count)
            if shared < args.min_shared_pixels:
                continue

            label_i_int = int(label_i)
            label_j_int = int(label_j)
            area_i = counts_i[label_i_int]
            area_j = counts_j[label_j_int]
            union = area_i + area_j - shared
            iou = float(shared / union) if union else 0.0
            coverage_i = float(shared / area_i) if area_i else 0.0
            coverage_j = float(shared / area_j) if area_j else 0.0
            max_coverage = max(coverage_i, coverage_j)
            centroid_dist = distance(
                centroid(mask_i, label_i_int, centroid_i),
                centroid(mask_j, label_j_int, centroid_j),
            )
            rules = rule_key(iou, max_coverage, centroid_dist, shared, args)
            matched = bool(rules)
            if matched:
                matches_by_label_i[label_i_int].add(label_j_int)
                matches_by_label_j[label_j_int].add(label_i_int)
                rule_counts[rules] += 1

            pair_rows.append(
                {
                    "pair_id": f"{tile_i}-{tile_j}",
                    "tile_a": tiles[tile_i]["filename"],
                    "tile_b": tiles[tile_j]["filename"],
                    "tile_a_index": tile_i,
                    "tile_b_index": tile_j,
                    "seam_type": seam_type,
                    "label_a": label_i_int,
                    "label_b": label_j_int,
                    "shared_voxels": shared,
                    "area_a_overlap": area_i,
                    "area_b_overlap": area_j,
                    "iou": iou,
                    "coverage_a": coverage_i,
                    "coverage_b": coverage_j,
                    "max_coverage": max_coverage,
                    "centroid_dist": centroid_dist,
                    "matched": matched,
                    "match_rule": rules,
                }
            )

    matched_rows = [row for row in pair_rows if row["matched"]]
    labels_i = set(counts_i)
    labels_j = set(counts_j)
    matched_i = set(matches_by_label_i)
    matched_j = set(matches_by_label_j)

    stitched_nonzero = None
    stitched_labels = None
    contribution_overlap_voxels = None
    if stitched_mask is not None:
        oz0, oz1, oy0, oy1, ox0, ox1 = overlap
        stitched_roi = mask_to_zyx(stitched_mask[oz0:oz1, oy0:oy1, ox0:ox1], 0)
        stitched_nonzero = int(np.count_nonzero(stitched_roi))
        stitched_labels = int(np.unique(stitched_roi[stitched_roi > 0]).size)
    if contributions is not None:
        oz0, oz1, oy0, oy1, ox0, ox1 = overlap
        contribution_roi = mask_to_zyx(contributions[oz0:oz1, oy0:oy1, ox0:ox1], 0)
        contribution_overlap_voxels = int(np.count_nonzero(contribution_roi > 1))

    shape_z, shape_y, shape_x = overlap_shape(overlap)
    summary = {
        "pair_id": f"{tile_i}-{tile_j}",
        "tile_a": tiles[tile_i]["filename"],
        "tile_b": tiles[tile_j]["filename"],
        "tile_a_index": tile_i,
        "tile_b_index": tile_j,
        "seam_type": seam_type,
        "overlap_global_z": [overlap[0], overlap[1]],
        "overlap_global_y": [overlap[2], overlap[3]],
        "overlap_global_x": [overlap[4], overlap[5]],
        "overlap_shape_zyx": [shape_z, shape_y, shape_x],
        "labels_a_in_overlap": len(labels_i),
        "labels_b_in_overlap": len(labels_j),
        "nonzero_voxels_a": nonzero_i,
        "nonzero_voxels_b": nonzero_j,
        "shared_nonzero_voxels": shared_nonzero,
        "shared_fraction_a": float(shared_nonzero / nonzero_i) if nonzero_i else 0.0,
        "shared_fraction_b": float(shared_nonzero / nonzero_j) if nonzero_j else 0.0,
        "candidate_pairs": len(pair_rows),
        "matched_pairs": len(matched_rows),
        "matched_labels_a": len(matched_i),
        "matched_labels_b": len(matched_j),
        "unmatched_labels_a": len(labels_i - matched_i),
        "unmatched_labels_b": len(labels_j - matched_j),
        "split_like_labels_a": sum(1 for values in matches_by_label_i.values() if len(values) > 1),
        "merge_like_labels_b": sum(1 for values in matches_by_label_j.values() if len(values) > 1),
        "match_rule_counts": dict(sorted(rule_counts.items())),
        "mean_matched_iou": mean(row["iou"] for row in matched_rows),
        "mean_matched_max_coverage": mean(row["max_coverage"] for row in matched_rows),
        "mean_matched_centroid_dist": mean(row["centroid_dist"] for row in matched_rows),
        "stitched_nonzero_voxels": stitched_nonzero,
        "stitched_labels": stitched_labels,
        "contribution_overlap_voxels": contribution_overlap_voxels,
    }

    if args.write_boundary_pngs:
        boundary_path = write_boundary_overlay_png(
            mask_i,
            mask_j,
            matched_i,
            matched_j,
            summary["pair_id"],
            summary["tile_a"],
            summary["tile_b"],
            args.output_dir / "boundary_pngs",
            args,
        )
        summary["boundary_png"] = str(boundary_path)

    return summary, pair_rows


def mean(values: Iterable[float]) -> Optional[float]:
    valid = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_optional_volume(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return tifffile.memmap(path)
    except ValueError:
        return tifffile.imread(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug cell-label merging evidence across tile overlap edges."
    )
    parser.add_argument("tiles_dir", type=Path, help="Directory containing mask tile TIFFs")
    parser.add_argument("metadata_file", type=Path, help="Tiling metadata JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stitch_edge_debug"),
        help="Directory for JSON/CSV outputs",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Channel to read if a mask tile is still zcyx",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=6,
        help="Number of tile masks to keep in memory",
    )
    parser.add_argument(
        "--skip-corners",
        action="store_true",
        help="Skip diagonal corner-only overlap pairs",
    )
    parser.add_argument(
        "--min-shared-pixels",
        type=int,
        default=5,
        help="Minimum shared voxels for a label pair to be scored",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.05)
    parser.add_argument("--coverage-threshold", type=float, default=0.40)
    parser.add_argument("--centroid-dist-threshold", type=float, default=6.0)
    parser.add_argument("--centroid-min-shared-pixels", type=int, default=5)
    parser.add_argument(
        "--no-boundary-pngs",
        action="store_false",
        dest="write_boundary_pngs",
        help="Do not write per-seam red/green/yellow pre-merge boundary overlays",
    )
    parser.set_defaults(write_boundary_pngs=True)
    parser.add_argument(
        "--boundary-z",
        default="max",
        help="Z slice to render for boundary PNGs, or 'max' for a max projection",
    )
    parser.add_argument(
        "--boundary-scale",
        type=int,
        default=4,
        help="Nearest-neighbor scale factor for narrow seam PNGs",
    )
    parser.add_argument(
        "--stitched-mask",
        type=Path,
        help="Optional stitched mask TIFF to summarize in each overlap region",
    )
    parser.add_argument(
        "--contributions",
        type=Path,
        help="Optional 3Dstitcher *_contributions.tif sidecar",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = load_metadata(args.metadata_file)
    tiles = metadata["tiles"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache = TileCache(args.tiles_dir, tiles, args.channel, args.cache_size)
    stitched_mask = load_optional_volume(args.stitched_mask)
    contributions = load_optional_volume(args.contributions)

    summaries: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    pairs = list(iter_overlapping_pairs(tiles, args.skip_corners))
    print(f"Found {len(pairs)} overlapping tile pairs")

    for n, (i, j, bounds_i, bounds_j, overlap, seam_type) in enumerate(pairs, start=1):
        print(f"[{n}/{len(pairs)}] {tiles[i]['filename']} <-> {tiles[j]['filename']}")
        try:
            summary, rows = analyze_pair(
                i,
                j,
                bounds_i,
                bounds_j,
                overlap,
                seam_type,
                cache,
                tiles,
                args,
                stitched_mask,
                contributions,
            )
        except FileNotFoundError as exc:
            print(f"  missing tile, skipped: {exc}")
            continue
        summaries.append(summary)
        candidates.extend(rows)

    summary_csv = args.output_dir / "edge_pair_summary.csv"
    candidate_csv = args.output_dir / "edge_label_candidates.csv"
    report_json = args.output_dir / "edge_merge_report.json"
    write_csv(summary_csv, summaries)
    write_csv(candidate_csv, candidates)

    report = {
        "metadata_file": str(args.metadata_file),
        "tiles_dir": str(args.tiles_dir),
        "thresholds": {
            "iou": args.iou_threshold,
            "coverage": args.coverage_threshold,
            "centroid_dist": args.centroid_dist_threshold,
            "centroid_min_shared_pixels": args.centroid_min_shared_pixels,
            "min_shared_pixels": args.min_shared_pixels,
        },
        "n_tiles": len(tiles),
        "n_overlapping_pairs": len(pairs),
        "n_analyzed_pairs": len(summaries),
        "n_candidate_label_pairs": len(candidates),
        "n_matched_label_pairs": sum(1 for row in candidates if row["matched"]),
        "boundary_pngs_enabled": args.write_boundary_pngs,
        "boundary_png_color_legend": {
            "red": "pre-merge boundary from tile_a labels not selected for merging",
            "green": "pre-merge boundary from tile_b labels not selected for merging",
            "yellow": "pre-merge boundary from labels selected for merge resolution",
        },
        "boundary_z": args.boundary_z,
        "boundary_scale": args.boundary_scale,
        "summary_csv": str(summary_csv),
        "candidate_csv": str(candidate_csv),
        "pair_summaries": summaries,
    }
    report_json.write_text(json.dumps(report, indent=2))

    worst = sorted(
        summaries,
        key=lambda row: (
            row["unmatched_labels_a"] + row["unmatched_labels_b"],
            row["split_like_labels_a"] + row["merge_like_labels_b"],
        ),
        reverse=True,
    )[:10]
    print("\nWrote:")
    print(f"  {report_json}")
    print(f"  {summary_csv}")
    print(f"  {candidate_csv}")
    print("\nWorst seams by unmatched overlap labels:")
    for row in worst:
        print(
            "  {pair_id} {seam_type}: unmatched={unmatched} matched_pairs={matched}".format(
                pair_id=row["pair_id"],
                seam_type=row["seam_type"],
                unmatched=row["unmatched_labels_a"] + row["unmatched_labels_b"],
                matched=row["matched_pairs"],
            )
        )


if __name__ == "__main__":
    main()
