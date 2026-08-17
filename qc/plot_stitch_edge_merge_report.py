#!/usr/bin/env python3
"""Render QC figures from debug_stitch_edge_merges.py outputs.

Example:
    python plot_stitch_edge_merge_report.py \
        stitch_edge_debug/edge_merge_report.json \
        --output stitch_edge_debug/edge_merge_qc.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

np: Any = None
plt: Any = None


def load_report(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def load_candidate_rows(report: Dict[str, Any], report_path: Path) -> List[Dict[str, Any]]:
    candidate_path = Path(report["candidate_csv"])
    if not candidate_path.is_absolute():
        cwd_path = Path.cwd() / candidate_path
        report_relative_path = report_path.parent / candidate_path.name
        if cwd_path.exists():
            candidate_path = cwd_path
        else:
            candidate_path = report_relative_path
    if not candidate_path.exists():
        return []
    with candidate_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return math.nan
    return float(value)


def is_true(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def seam_label(summary: Dict[str, Any]) -> str:
    return "{pair_id}\n{kind}".format(
        pair_id=summary["pair_id"],
        kind=summary["seam_type"].replace("_", " "),
    )


def plot_pair_bars(ax: Any, summaries: List[Dict[str, Any]]) -> None:
    labels = [seam_label(row) for row in summaries]
    matched = np.array([row["matched_pairs"] for row in summaries], dtype=float)
    unmatched = np.array(
        [row["unmatched_labels_a"] + row["unmatched_labels_b"] for row in summaries],
        dtype=float,
    )
    x = np.arange(len(summaries))
    ax.bar(x, matched, label="matched label pairs", color="#3b82f6")
    ax.bar(x, unmatched, bottom=matched, label="unmatched overlap labels", color="#f97316")
    ax.set_title("Seam Match Counts")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8)


def plot_quality_scatter(ax: Any, candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        ax.text(0.5, 0.5, "No label candidates", ha="center", va="center")
        ax.set_axis_off()
        return
    valid = [
        row
        for row in candidates
        if not math.isnan(as_float(row, "iou"))
        and not math.isnan(as_float(row, "max_coverage"))
    ]
    iou = np.array([as_float(row, "iou") for row in valid], dtype=float)
    coverage = np.array([as_float(row, "max_coverage") for row in valid], dtype=float)
    matched = np.array([is_true(row.get("matched")) for row in valid], dtype=bool)

    ax.scatter(iou[~matched], coverage[~matched], s=8, alpha=0.35, label="not matched")
    ax.scatter(iou[matched], coverage[matched], s=10, alpha=0.6, label="matched")
    ax.set_title("Label-Pair Evidence")
    ax.set_xlabel("IoU in overlap")
    ax.set_ylabel("max coverage")
    ax.legend(fontsize=8)


def plot_rule_counts(ax: Any, candidates: List[Dict[str, Any]]) -> None:
    counts = Counter(
        row.get("match_rule") or "not matched"
        for row in candidates
        if row.get("match_rule") or not is_true(row.get("matched"))
    )
    if not counts:
        ax.text(0.5, 0.5, "No rule counts", ha="center", va="center")
        ax.set_axis_off()
        return
    labels, values = zip(*counts.most_common())
    x = np.arange(len(labels))
    ax.bar(x, values, color="#10b981")
    ax.set_title("Decision Rule Counts")
    ax.set_ylabel("label pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)


def plot_unmatched_heat(ax: Any, summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        ax.text(0.5, 0.5, "No seam summaries", ha="center", va="center")
        ax.set_axis_off()
        return
    values = np.array(
        [
            [
                row["unmatched_labels_a"],
                row["unmatched_labels_b"],
                row["split_like_labels_a"],
                row["merge_like_labels_b"],
            ]
            for row in summaries
        ],
        dtype=float,
    )
    image = ax.imshow(values.T, aspect="auto", cmap="magma")
    ax.set_title("Potential Edge Problems")
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["unmatched A", "unmatched B", "split-like A", "merge-like B"])
    ax.set_xticks(np.arange(len(summaries)))
    ax.set_xticklabels([row["pair_id"] for row in summaries], rotation=45, ha="right", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.4g}"
    return str(value)


def problem_score(summary: Dict[str, Any]) -> int:
    return (
        summary["unmatched_labels_a"]
        + summary["unmatched_labels_b"]
        + summary["split_like_labels_a"]
        + summary["merge_like_labels_b"]
    )


def boundary_href(summary: Dict[str, Any], output: Path) -> str:
    path_value = summary.get("boundary_png")
    if not path_value:
        return "NA"
    path = Path(path_value)
    try:
        href = path.relative_to(output.parent)
    except ValueError:
        href = path
    return '<a href="{href}">overlay</a>'.format(href=html.escape(str(href)))


def html_bar(label: str, matched: int, unmatched: int, max_total: int) -> str:
    total = matched + unmatched
    matched_width = 0 if max_total == 0 else 100 * matched / max_total
    unmatched_width = 0 if max_total == 0 else 100 * unmatched / max_total
    return f"""
      <div class="bar-row">
        <div class="bar-label">{html.escape(label)}</div>
        <div class="bar-track">
          <span class="bar matched" style="width:{matched_width:.2f}%"></span>
          <span class="bar unmatched" style="width:{unmatched_width:.2f}%"></span>
        </div>
        <div class="bar-value">{matched} matched / {unmatched} unmatched</div>
      </div>
    """


def write_html_report(
    report: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    output: Path,
    top: int,
) -> None:
    thresholds = report.get("thresholds", {})
    matched_candidates = sum(1 for row in candidates if is_true(row.get("matched")))
    rule_counts = Counter(row.get("match_rule") or "not matched" for row in candidates)
    ranked = sorted(summaries, key=problem_score, reverse=True)
    max_total = max(
        [row["matched_pairs"] + row["unmatched_labels_a"] + row["unmatched_labels_b"] for row in summaries]
        or [1]
    )

    bars = "\n".join(
        html_bar(
            seam_label(row).replace("\n", " "),
            row["matched_pairs"],
            row["unmatched_labels_a"] + row["unmatched_labels_b"],
            max_total,
        )
        for row in ranked
    )
    rule_items = "\n".join(
        f"<li><strong>{html.escape(rule)}</strong>: {count}</li>"
        for rule, count in rule_counts.most_common()
    )
    top_rows = "\n".join(
        """
        <tr>
          <td>{pair}</td><td>{seam}</td><td>{matched}</td><td>{unmatched}</td>
          <td>{split}</td><td>{merge}</td><td>{iou}</td><td>{coverage}</td>
          <td>{centroid}</td><td>{boundary}</td>
        </tr>
        """.format(
            pair=html.escape(row["pair_id"]),
            seam=html.escape(row["seam_type"]),
            matched=row["matched_pairs"],
            unmatched=row["unmatched_labels_a"] + row["unmatched_labels_b"],
            split=row["split_like_labels_a"],
            merge=row["merge_like_labels_b"],
            iou=fmt(row["mean_matched_iou"]),
            coverage=fmt(row["mean_matched_max_coverage"]),
            centroid=fmt(row["mean_matched_centroid_dist"]),
            boundary=boundary_href(row, output),
        )
        for row in ranked[:top]
    )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Stitch Edge Merge QC</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; color: #111827; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 1rem; min-width: 11rem; }}
    .card .value {{ font-size: 1.8rem; font-weight: 700; }}
    .bar-row {{ display: grid; grid-template-columns: 12rem 1fr 14rem; gap: .75rem; align-items: center; margin: .35rem 0; }}
    .bar-track {{ height: 1rem; background: #e5e7eb; border-radius: 999px; overflow: hidden; white-space: nowrap; }}
    .bar {{ display: inline-block; height: 100%; }}
    .matched {{ background: #3b82f6; }}
    .unmatched {{ background: #f97316; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
    th, td {{ border: 1px solid #d1d5db; padding: .45rem .55rem; text-align: left; }}
    th {{ background: #f3f4f6; }}
    code {{ background: #f3f4f6; padding: .1rem .25rem; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Stitch Edge Merge QC</h1>
  <p>Report source: <code>{html.escape(str(report.get("metadata_file", "")))}</code></p>
  <div class="cards">
    <div class="card"><div class="value">{len(summaries)}</div><div>analyzed seams</div></div>
    <div class="card"><div class="value">{len(candidates)}</div><div>label candidates</div></div>
    <div class="card"><div class="value">{matched_candidates}</div><div>matched label pairs</div></div>
    <div class="card"><div class="value">{sum(problem_score(row) for row in summaries)}</div><div>problem indicators</div></div>
  </div>

  <h2>Thresholds</h2>
  <p>
    IoU >= <code>{fmt(thresholds.get("iou"))}</code>,
    coverage >= <code>{fmt(thresholds.get("coverage"))}</code>,
    centroid distance <= <code>{fmt(thresholds.get("centroid_dist"))}</code>,
    min shared pixels = <code>{fmt(thresholds.get("min_shared_pixels"))}</code>.
  </p>

  <h2>Seam Match Counts</h2>
  <p><span style="color:#3b82f6">blue</span> is matched label pairs; <span style="color:#f97316">orange</span> is unmatched overlap labels.</p>
  {bars}

  <h2>Boundary Overlay Legend</h2>
  <p>
    Overlay PNGs are rendered from the pre-resolution tile masks on the cell boundary layer:
    <strong style="color:#b91c1c">red</strong> is tile A, 
    <strong style="color:#15803d">green</strong> is tile B, and
    <strong style="color:#a16207">yellow</strong> marks labels selected for merge resolution.
  </p>

  <h2>Decision Rules</h2>
  <ul>{rule_items}</ul>

  <h2>Top Seams To Inspect</h2>
  <table>
    <thead>
      <tr>
        <th>pair</th><th>seam</th><th>matched pairs</th><th>unmatched labels</th>
        <th>split-like A</th><th>merge-like B</th><th>mean IoU</th>
        <th>mean coverage</th><th>mean centroid dist</th><th>boundary PNG</th>
      </tr>
    </thead>
    <tbody>{top_rows}</tbody>
  </table>
</body>
</html>
"""
    output.write_text(document)


def write_png_report(
    summaries: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    output: Path,
) -> None:
    global np, plt
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Stitch Edge Merge QC", fontsize=16)
    plot_pair_bars(axes[0, 0], summaries)
    plot_quality_scatter(axes[0, 1], candidates)
    plot_rule_counts(axes[1, 0], candidates)
    plot_unmatched_heat(axes[1, 1], summaries)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def print_top_seams(summaries: List[Dict[str, Any]], limit: int) -> None:
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["unmatched_labels_a"] + row["unmatched_labels_b"],
            row["split_like_labels_a"] + row["merge_like_labels_b"],
        ),
        reverse=True,
    )
    print("Top seams by unmatched/split/merge indicators:")
    for row in ranked[:limit]:
        print(
            "  {pair_id} {seam}: matched={matched} unmatched={unmatched} "
            "split_like={split_like} merge_like={merge_like}".format(
                pair_id=row["pair_id"],
                seam=row["seam_type"],
                matched=row["matched_pairs"],
                unmatched=row["unmatched_labels_a"] + row["unmatched_labels_b"],
                split_like=row["split_like_labels_a"],
                merge_like=row["merge_like_labels_b"],
            )
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render stitch edge merge QC report.")
    parser.add_argument("report_json", type=Path, help="edge_merge_report.json")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output .html or .png path. Defaults to edge_merge_qc.html beside the report.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of worst seams to print",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = load_report(args.report_json)
    summaries = report.get("pair_summaries", [])
    candidates = load_candidate_rows(report, args.report_json)

    output = args.output or args.report_json.with_name("edge_merge_qc.html")
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".png":
        write_png_report(summaries, candidates, output)
    else:
        write_html_report(report, summaries, candidates, output, args.top)

    print(f"Wrote {output}")
    print_top_seams(summaries, args.top)


if __name__ == "__main__":
    main()
