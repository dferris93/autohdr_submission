#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score challenge predictions.csv against a manifest CSV using exact group-set matching."
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to predictions.csv with columns filename,group_id",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to manifest CSV with columns group_id,filename",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the score summary text file.",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["group_id"]].add(row["filename"])
    return groups


def load_manifest(path: Path) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["group_id"]].add(row["filename"])
    return groups


def main() -> None:
    args = parse_args()

    predicted = load_predictions(args.predictions)
    truth = load_manifest(args.manifest)

    predicted_sets = {frozenset(items) for items in predicted.values()}
    truth_sets = {frozenset(items) for items in truth.values()}
    exact_matches = len(predicted_sets & truth_sets)
    score = exact_matches / float(len(truth_sets)) if truth_sets else 0.0

    text = "\n".join(
        [
            f"predictions={args.predictions}",
            f"manifest={args.manifest}",
            f"predicted_groups={len(predicted_sets)}",
            f"truth_groups={len(truth_sets)}",
            f"exact_matches={exact_matches}",
            f"score={score:.6f}",
        ]
    ) + "\n"

    print(text, end="")
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
