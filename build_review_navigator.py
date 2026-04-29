#!/usr/bin/env python3
"""Build a human-friendly review navigator from grouping outputs.

This is intentionally separate from the matcher so results can be reorganized
without rerunning the expensive image matching pipeline.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class GroupInfo:
    angle_id: str
    status: str
    size: int
    representative_filename: str
    min_confidence: float
    max_confidence: float
    filenames: List[str]
    shoot_ids: Tuple[str, ...]
    primary_shoot: str
    folder: Path


@dataclass
class NeighborInfo:
    other_group: str
    max_confidence: float
    count: int
    best_pair_a: str
    best_pair_b: str
    top_reason: str
    same_shoot: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a useful browser/review view from matcher outputs.")
    parser.add_argument("--groups", type=Path, required=True, help="Path to angle groups CSV.")
    parser.add_argument("--pair-scores", type=Path, required=True, help="Path to pair scores CSV.")
    parser.add_argument("--grouped-dir", type=Path, required=True, help="Path to grouped folder view.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for the navigator.")
    return parser.parse_args()


def shoot_id_for_filename(filename: str) -> str:
    match = re.match(r"^(g\d+)", filename)
    if match:
        return match.group(1)
    stem = Path(filename).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return stem


def load_groups(groups_csv: Path, grouped_dir: Path) -> Tuple[Dict[str, GroupInfo], Dict[str, str]]:
    rows = list(csv.DictReader(groups_csv.open(encoding="utf-8", newline="")))
    by_group: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_group[row["angle_id"]].append(row)

    folder_lookup = {
        path.name.split("__", 1)[0]: path
        for path in grouped_dir.iterdir()
        if path.is_dir() and "__" in path.name
    }

    groups: Dict[str, GroupInfo] = {}
    filename_to_group: Dict[str, str] = {}
    for angle_id, group_rows in sorted(by_group.items()):
        group_rows.sort(key=lambda row: row["filename"])
        filenames = [row["filename"] for row in group_rows]
        for filename in filenames:
            filename_to_group[filename] = angle_id
        shoot_ids = tuple(sorted({shoot_id_for_filename(filename) for filename in filenames}))
        primary_shoot = shoot_ids[0] if len(shoot_ids) == 1 else "mixed"
        folder = folder_lookup.get(angle_id)
        if folder is None:
            raise FileNotFoundError(f"Could not find grouped folder for {angle_id} in {grouped_dir}")
        confidences = [float(row["confidence"]) for row in group_rows]
        groups[angle_id] = GroupInfo(
            angle_id=angle_id,
            status=group_rows[0]["status"],
            size=len(group_rows),
            representative_filename=group_rows[0]["representative_filename"],
            min_confidence=min(confidences),
            max_confidence=max(confidences),
            filenames=filenames,
            shoot_ids=shoot_ids,
            primary_shoot=primary_shoot,
            folder=folder,
        )
    return groups, filename_to_group


def load_neighbors(pair_scores_csv: Path, groups: Dict[str, GroupInfo], filename_to_group: Dict[str, str]) -> Dict[str, List[NeighborInfo]]:
    raw_neighbors: Dict[Tuple[str, str], dict] = {}
    with pair_scores_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            group_a = filename_to_group.get(row["filename_a"])
            group_b = filename_to_group.get(row["filename_b"])
            if not group_a or not group_b or group_a == group_b or row["decision"] != "reject":
                continue
            confidence = float(row["confidence"])
            key = tuple(sorted((group_a, group_b)))
            entry = raw_neighbors.setdefault(
                key,
                {
                    "max_confidence": 0.0,
                    "count": 0,
                    "best_pair_a": "",
                    "best_pair_b": "",
                    "reasons": Counter(),
                },
            )
            entry["count"] += 1
            entry["reasons"][row["reason"]] += 1
            if confidence >= entry["max_confidence"]:
                entry["max_confidence"] = confidence
                entry["best_pair_a"] = row["filename_a"]
                entry["best_pair_b"] = row["filename_b"]

    neighbors_by_group: Dict[str, List[NeighborInfo]] = defaultdict(list)
    for (group_a, group_b), entry in raw_neighbors.items():
        top_reason = entry["reasons"].most_common(1)[0][0]
        same_shoot = bool(set(groups[group_a].shoot_ids) & set(groups[group_b].shoot_ids))
        neighbor_a = NeighborInfo(
            other_group=group_b,
            max_confidence=entry["max_confidence"],
            count=entry["count"],
            best_pair_a=entry["best_pair_a"],
            best_pair_b=entry["best_pair_b"],
            top_reason=top_reason,
            same_shoot=same_shoot,
        )
        neighbor_b = NeighborInfo(
            other_group=group_a,
            max_confidence=entry["max_confidence"],
            count=entry["count"],
            best_pair_a=entry["best_pair_a"],
            best_pair_b=entry["best_pair_b"],
            top_reason=top_reason,
            same_shoot=same_shoot,
        )
        neighbors_by_group[group_a].append(neighbor_a)
        neighbors_by_group[group_b].append(neighbor_b)

    for group_id, items in neighbors_by_group.items():
        items.sort(
            key=lambda item: (
                0 if item.same_shoot else 1,
                -item.max_confidence,
                -item.count,
                item.other_group,
            )
        )
    return neighbors_by_group


def link_relative(target: Path, link_path: Path) -> None:
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def write_by_shoot(output_dir: Path, groups: Dict[str, GroupInfo]) -> Dict[str, Path]:
    by_shoot_dir = output_dir / "by_shoot"
    by_shoot_dir.mkdir(parents=True, exist_ok=True)
    shoot_map: Dict[str, List[GroupInfo]] = defaultdict(list)
    for group in groups.values():
        shoot_key = group.primary_shoot if group.primary_shoot != "mixed" else "mixed_" + "_".join(group.shoot_ids)
        shoot_map[shoot_key].append(group)

    shoot_folders: Dict[str, Path] = {}
    for shoot_key, shoot_groups in sorted(shoot_map.items()):
        shoot_groups.sort(key=lambda group: group.angle_id)
        folder = by_shoot_dir / f"{shoot_key}__{len(shoot_groups)}_groups"
        folder.mkdir()
        shoot_folders[shoot_key] = folder
        summary_lines = [
            f"shoot={shoot_key}",
            f"group_count={len(shoot_groups)}",
            "",
            "angle_id,status,size,min_confidence,representative_filename",
        ]
        for group in shoot_groups:
            link_relative(group.folder, folder / group.folder.name)
            summary_lines.append(
                f"{group.angle_id},{group.status},{group.size},{group.min_confidence:.4f},{group.representative_filename}"
            )
        (folder / "SHOOT_SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return shoot_folders


def score_group(
    group: GroupInfo, neighbors: List[NeighborInfo], same_shoot_other_groups: List[GroupInfo]
) -> Tuple[float, List[str], List[NeighborInfo]]:
    score = 0.0
    reasons: List[str] = []

    same_shoot_neighbor_map = {
        neighbor.other_group: neighbor for neighbor in neighbors if neighbor.same_shoot
    }
    compare_neighbors: List[NeighborInfo] = []
    for other_group in sorted(same_shoot_other_groups, key=lambda item: item.angle_id):
        compare_neighbors.append(
            same_shoot_neighbor_map.get(
                other_group.angle_id,
                NeighborInfo(
                    other_group=other_group.angle_id,
                    max_confidence=0.0,
                    count=0,
                    best_pair_a="",
                    best_pair_b="",
                    top_reason="same_shoot_context",
                    same_shoot=True,
                ),
            )
        )
    compare_neighbors.sort(key=lambda item: (-item.max_confidence, item.other_group))
    compare_neighbors = compare_neighbors[:3]

    if group.status == "singleton":
        score += 100.0
        reasons.append("singleton: matcher found no confident partner")
        if same_shoot_other_groups:
            score += 20.0
            reasons.append(f"same shoot has {len(same_shoot_other_groups) + 1} separate groups")

    if len(group.shoot_ids) > 1:
        score += 90.0
        reasons.append("group mixes multiple shoot prefixes")

    if group.min_confidence < 0.97:
        penalty = (0.97 - group.min_confidence) * 200.0
        score += 35.0 + penalty
        reasons.append(f"low member confidence: min={group.min_confidence:.4f}")

    if group.size >= 10:
        score += 15.0 + min(group.size, 25)
        reasons.append(f"large group: n={group.size}")

    if compare_neighbors:
        best_neighbor = compare_neighbors[0]
        if best_neighbor.max_confidence > 0.0:
            score += 25.0 + best_neighbor.max_confidence * 20.0 + min(best_neighbor.count, 10)
            reasons.append(
                f"same-shoot near miss: compare {group.angle_id} with {best_neighbor.other_group} "
                f"(reject conf {best_neighbor.max_confidence:.4f}, {best_neighbor.count} pair links)"
            )

    return score, reasons, compare_neighbors


def build_review_queue(
    output_dir: Path,
    groups: Dict[str, GroupInfo],
    neighbors_by_group: Dict[str, List[NeighborInfo]],
    shoot_folders: Dict[str, Path],
) -> None:
    queue_dir = output_dir / "review_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)

    queue_rows = []
    for group in groups.values():
        shoot_key = group.primary_shoot if group.primary_shoot != "mixed" else "mixed_" + "_".join(group.shoot_ids)
        same_shoot_other_groups = [
            candidate
            for candidate in groups.values()
            if candidate.primary_shoot == group.primary_shoot and candidate.angle_id != group.angle_id
        ]
        score, reasons, compare_neighbors = score_group(
            group,
            neighbors_by_group.get(group.angle_id, []),
            same_shoot_other_groups,
        )
        if not reasons:
            continue
        queue_rows.append(
            {
                "angle_id": group.angle_id,
                "shoot_key": shoot_key,
                "status": group.status,
                "group_size": group.size,
                "min_confidence": group.min_confidence,
                "score": score,
                "reasons": reasons,
                "compare_neighbors": compare_neighbors,
                "group": group,
            }
        )

    queue_rows.sort(
        key=lambda row: (
            -row["score"],
            row["status"] != "singleton",
            row["min_confidence"],
            row["angle_id"],
        )
    )

    summary_lines = [
        "rank,angle_id,shoot_key,status,group_size,min_confidence,review_score,why,compare_groups"
    ]
    for rank, row in enumerate(queue_rows, start=1):
        group = row["group"]
        compare_groups = [neighbor.other_group for neighbor in row["compare_neighbors"]]
        why = " | ".join(row["reasons"])
        summary_lines.append(
            f"{rank},{group.angle_id},{row['shoot_key']},{group.status},{group.size},"
            f"{group.min_confidence:.4f},{row['score']:.2f},\"{why}\",\"{' '.join(compare_groups)}\""
        )

        label = (
            f"{rank:02d}__{group.angle_id}__{group.status}__n{group.size}__"
            f"min{group.min_confidence:.4f}"
        )
        item_dir = queue_dir / label
        item_dir.mkdir()
        link_relative(group.folder, item_dir / f"primary__{group.folder.name}")

        shoot_folder = shoot_folders.get(row["shoot_key"])
        if shoot_folder is not None:
            link_relative(shoot_folder, item_dir / f"shoot_view__{shoot_folder.name}")

        for neighbor in row["compare_neighbors"]:
            other = groups[neighbor.other_group]
            link_relative(other.folder, item_dir / f"compare__{other.folder.name}")

        issue_lines = [
            f"rank={rank}",
            f"angle_id={group.angle_id}",
            f"shoot_key={row['shoot_key']}",
            f"status={group.status}",
            f"group_size={group.size}",
            f"min_confidence={group.min_confidence:.4f}",
            f"max_confidence={group.max_confidence:.4f}",
            f"representative={group.representative_filename}",
            f"review_score={row['score']:.2f}",
            "",
            "Why this is in the queue:",
        ]
        issue_lines.extend(f"- {reason}" for reason in row["reasons"])
        if row["compare_neighbors"]:
            issue_lines.append("")
            issue_lines.append("Suggested comparisons:")
            for neighbor in row["compare_neighbors"]:
                issue_lines.append(
                    f"- {neighbor.other_group}: conf={neighbor.max_confidence:.4f}, "
                    f"count={neighbor.count}, same_shoot={str(neighbor.same_shoot).lower()}, "
                    f"reason={neighbor.top_reason}, best_pair={neighbor.best_pair_a} <-> {neighbor.best_pair_b}"
                )
        (item_dir / "WHY_THIS_IS_HERE.txt").write_text("\n".join(issue_lines) + "\n", encoding="utf-8")

    (output_dir / "review_queue.csv").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def write_root_readme(output_dir: Path) -> None:
    text = """Start here:

- `review_queue/` is the ranked list of groups worth checking first.
- `review_queue.csv` explains why each queued item matters.
- `by_shoot/` is the sane browser view when you want to inspect one property at a time.

How to use this:

1. Open `review_queue/01__...`
2. Look at `WHY_THIS_IS_HERE.txt`
3. Open `primary__...`
4. If compare folders are present, compare them before moving on
5. If you want the whole property context, open the `shoot_view__...` link
"""
    (output_dir / "README.txt").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    groups, filename_to_group = load_groups(args.groups, args.grouped_dir)
    neighbors_by_group = load_neighbors(args.pair_scores, groups, filename_to_group)
    shoot_folders = write_by_shoot(args.output_dir, groups)
    build_review_queue(args.output_dir, groups, neighbors_by_group, shoot_folders)
    write_root_readme(args.output_dir)
    print(f"Wrote review navigator to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
