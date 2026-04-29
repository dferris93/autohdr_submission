"""AutoHDR challenge entrypoint backed by the conservative geometry-first matcher.

This wrapper keeps the base matcher strict, then applies a narrow
challenge-only post-pass to images that are still unmatched singletons:
  * only original singleton groups are considered as merge sources;
  * each singleton can be merged at most once;
  * tiny verified tail fragments may attach to a larger group when one member
    has direct relaxed support and the rest are strongly verified internally;
  * it does not change the reusable core matcher used elsewhere in the repo.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from group_camera_angles import (
    ImageFeatures,
    PairScore,
    coarse_descriptor,
    compare_pair,
    default_thresholds,
    edge_image,
    extract_features,
    gradient_image,
    groups_from_assignments,
    run_grouping,
)

INPUT_DIR = Path(os.environ.get("AUTOHDR_INPUT_DIR", "/input/images"))
OUTPUT_DIR = Path(os.environ.get("AUTOHDR_OUTPUT_DIR", "/output"))
PAIR_CACHE_PATH = os.environ.get("AUTOHDR_PAIR_CACHE_PATH")
MAX_CANDIDATES = int(os.environ.get("AUTOHDR_MAX_CANDIDATES", "12"))
MAX_SIZE = int(os.environ.get("AUTOHDR_MAX_SIZE", "1024"))
FEATURES_PER_VARIANT = int(os.environ.get("AUTOHDR_FEATURES_PER_VARIANT", "2500"))
LARGE_DATASET_THRESHOLD = int(os.environ.get("AUTOHDR_LARGE_DATASET_THRESHOLD", "5000"))
LARGE_MAX_SIZE = int(os.environ.get("AUTOHDR_LARGE_MAX_SIZE", "256"))
LARGE_NEIGHBORS = int(os.environ.get("AUTOHDR_LARGE_NEIGHBORS", "192"))
LARGE_MIN_SIMILARITY = float(os.environ.get("AUTOHDR_LARGE_MIN_SIMILARITY", "0.90"))
LARGE_MAX_ASPECT_DELTA = float(os.environ.get("AUTOHDR_LARGE_MAX_ASPECT_DELTA", "0.03"))
LARGE_HYBRID = os.environ.get("AUTOHDR_LARGE_HYBRID", "1") != "0"
LARGE_SUPER_MIN_SIMILARITY = float(os.environ.get("AUTOHDR_LARGE_SUPER_MIN_SIMILARITY", "0.84"))
LARGE_MAX_COMPONENT_SIZE = int(os.environ.get("AUTOHDR_LARGE_MAX_COMPONENT_SIZE", "64"))
LARGE_COMPONENT_MAX_SIZE = int(os.environ.get("AUTOHDR_LARGE_COMPONENT_MAX_SIZE", "768"))
LARGE_COMPONENT_FEATURES = int(os.environ.get("AUTOHDR_LARGE_COMPONENT_FEATURES", "900"))
LARGE_COMPONENT_CANDIDATES = int(os.environ.get("AUTOHDR_LARGE_COMPONENT_CANDIDATES", "16"))
LARGE_COMPONENT_POSTPASS = os.environ.get("AUTOHDR_LARGE_COMPONENT_POSTPASS", "0") == "1"
LARGE_HYBRID_TIME_BUDGET = float(os.environ.get("AUTOHDR_LARGE_HYBRID_TIME_BUDGET", "2400"))
LARGE_BLOCKWISE_CANDIDATES = os.environ.get("AUTOHDR_LARGE_BLOCKWISE_CANDIDATES", "1") != "0"
LARGE_BLOCK_SIZE = int(os.environ.get("AUTOHDR_LARGE_BLOCK_SIZE", "512"))
LARGE_CANDIDATE_MIN_SIMILARITY = float(os.environ.get("AUTOHDR_LARGE_CANDIDATE_MIN_SIMILARITY", "0.45"))
LARGE_GEOMETRY_BRIDGE = os.environ.get("AUTOHDR_LARGE_GEOMETRY_BRIDGE", "1") != "0"
LARGE_BRIDGE_MIN_SIMILARITY = float(os.environ.get("AUTOHDR_LARGE_BRIDGE_MIN_SIMILARITY", "0.84"))
LARGE_BRIDGE_MAX_PAIRS = int(os.environ.get("AUTOHDR_LARGE_BRIDGE_MAX_PAIRS", "20000"))
LARGE_BRIDGE_TIME_BUDGET = float(os.environ.get("AUTOHDR_LARGE_BRIDGE_TIME_BUDGET", "900"))
LARGE_BRIDGE_BATCH_SIZE = int(os.environ.get("AUTOHDR_LARGE_BRIDGE_BATCH_SIZE", "512"))
LARGE_BRIDGE_EDGES_PER_COMPONENT_PAIR = int(os.environ.get("AUTOHDR_LARGE_BRIDGE_EDGES_PER_COMPONENT_PAIR", "1"))
LARGE_BRIDGE_ACCEPT_STRONG_REJECT = os.environ.get("AUTOHDR_LARGE_BRIDGE_ACCEPT_STRONG_REJECT", "0") != "0"
SUPPORTED = {".jpg", ".jpeg", ".png"}


@dataclass
class RelaxedGroupMatch:
    source_group_id: int
    target_group_id: int
    average_support: float
    source_supports: List[Tuple[int, float]]


@dataclass
class CompactImageFeature:
    index: int
    path: Path
    filename: str
    width: int
    height: int
    descriptor: "object"


@dataclass(frozen=True)
class CompactNeighborPair:
    similarity: float
    left: int
    right: int


class DSU:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def build_feature_groups(
    features: Sequence[ImageFeatures], assignments: Dict[int, object]
) -> Dict[int, List[ImageFeatures]]:
    groups: Dict[int, List[ImageFeatures]] = {}
    for feature in features:
        group_id = assignments[feature.index].group_id
        groups.setdefault(group_id, []).append(feature)
    return {
        group_id: sorted(group_features, key=lambda item: item.filename)
        for group_id, group_features in groups.items()
    }


def normalized_average_descriptor(members: Sequence[ImageFeatures]) -> "object":
    import numpy as np

    matrix = np.vstack([member.coarse_descriptor for member in members]).astype(np.float32, copy=False)
    vector = np.mean(matrix, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        vector /= norm
    return vector.astype(np.float32, copy=False)


def group_affinity(
    source_members: Sequence[ImageFeatures],
    target_members: Sequence[ImageFeatures],
    source_descriptor: "object",
    target_descriptor: "object",
) -> float:
    descriptor_similarity = float(source_descriptor @ target_descriptor)
    member_similarity = max_member_similarity(source_members, target_members)
    return 0.55 * member_similarity + 0.45 * descriptor_similarity


def max_member_similarity(source: Sequence[ImageFeatures], target: Sequence[ImageFeatures]) -> float:
    best = -1.0
    for source_member in source:
        for target_member in target:
            similarity = float(source_member.coarse_descriptor @ target_member.coarse_descriptor)
            if similarity > best:
                best = similarity
    return best


def relaxed_support_score(score: PairScore) -> Optional[float]:
    if score.method == "none":
        return None
    if math.isinf(score.median_reprojection_error) or math.isinf(score.avg_corner_shift_ratio):
        return None
    if score.good_matches < 40 or score.ransac_inliers < 20:
        return None
    if score.inlier_ratio < 0.28:
        return None
    if score.median_reprojection_error > 3.0:
        return None
    if score.avg_corner_shift_ratio > 1.00 or score.max_corner_shift_ratio > 1.80:
        return None
    if score.edge_score < 0.14:
        return None
    if score.confidence < 0.45:
        return None

    inliers = min(1.0, score.ransac_inliers / 60.0)
    matches = min(1.0, score.good_matches / 100.0)
    ratio = min(1.0, score.inlier_ratio / 0.60)
    error = max(0.0, 1.0 - score.median_reprojection_error / 3.0)
    shift = max(0.0, 1.0 - score.avg_corner_shift_ratio / 1.00)
    structure = min(1.0, score.edge_score / 0.30)
    return (
        0.28 * inliers
        + 0.18 * matches
        + 0.18 * ratio
        + 0.16 * error
        + 0.10 * shift
        + 0.10 * structure
    )


def singleton_pair_support(score: PairScore) -> Optional[float]:
    if score.method == "none":
        return None
    if math.isinf(score.median_reprojection_error) or math.isinf(score.avg_corner_shift_ratio):
        return None
    if score.good_matches < 14 or score.ransac_inliers < 10:
        return None
    if score.inlier_ratio < 0.75:
        return None
    if score.median_reprojection_error > 0.35:
        return None
    if score.avg_corner_shift_ratio > 0.60 or score.max_corner_shift_ratio > 0.90:
        return None
    if score.edge_score < 0.30:
        return None
    if score.confidence < 0.52:
        return None
    return 0.45 * score.confidence + 0.30 * score.edge_score + 0.25 * score.inlier_ratio


def low_feature_tiny_motion_support(score: PairScore) -> Optional[float]:
    if score.method == "none":
        return None
    if math.isinf(score.median_reprojection_error) or math.isinf(score.avg_corner_shift_ratio):
        return None
    if score.good_matches < 7 or score.ransac_inliers < 7:
        return None
    if score.inlier_ratio < 0.90:
        return None
    if score.median_reprojection_error > 0.45:
        return None
    if score.avg_corner_shift_ratio > 0.12 or score.max_corner_shift_ratio > 0.22:
        return None
    if score.edge_score < 0.30:
        return None
    if score.confidence < 0.60:
        return None
    return 0.45 * score.confidence + 0.35 * score.inlier_ratio + 0.20 * score.edge_score


def cached_pair_score(
    cache: Dict[Tuple[int, int], PairScore],
    left: ImageFeatures,
    right: ImageFeatures,
) -> PairScore:
    key = (left.index, right.index) if left.index < right.index else (right.index, left.index)
    cached = cache.get(key)
    if cached is None:
        cached = compare_pair(left, right, default_thresholds())
        cache[key] = cached
    return cached


def add_pair_request(
    pair_requests: Dict[Tuple[int, int], Tuple[ImageFeatures, ImageFeatures]],
    pair_cache: Dict[Tuple[int, int], PairScore],
    left: ImageFeatures,
    right: ImageFeatures,
) -> None:
    key = (left.index, right.index) if left.index < right.index else (right.index, left.index)
    if key not in pair_cache:
        pair_requests[key] = (left, right)


def prefetch_pair_scores(
    pair_cache: Dict[Tuple[int, int], PairScore],
    pair_requests: Dict[Tuple[int, int], Tuple[ImageFeatures, ImageFeatures]],
    *,
    jobs: int,
) -> None:
    if not pair_requests:
        return

    items = list(pair_requests.items())
    print(
        f"Challenge singleton post-pass: scoring {len(items)} uncached pair(s) with {jobs} worker(s).",
        file=sys.stdout,
        flush=True,
    )
    thresholds = default_thresholds()
    if jobs <= 1 or len(items) <= 1:
        for key, (left, right) in items:
            pair_cache[key] = compare_pair(left, right, thresholds)
        return

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(compare_pair, left, right, thresholds): key
            for key, (left, right) in items
        }
        for future in as_completed(futures):
            pair_cache[futures[future]] = future.result()


def candidate_target_supports(
    source_member: ImageFeatures,
    target_members: Sequence[ImageFeatures],
    pair_cache: Dict[Tuple[int, int], PairScore],
    *,
    max_target_candidates: int,
) -> List[Tuple[int, float]]:
    ranked_targets = sorted(
        target_members,
        key=lambda target: float(source_member.coarse_descriptor @ target.coarse_descriptor),
        reverse=True,
    )
    supports: List[Tuple[int, float]] = []
    for target_member in ranked_targets[:max_target_candidates]:
        score = cached_pair_score(pair_cache, source_member, target_member)
        support = relaxed_support_score(score)
        if support is None:
            support = low_feature_tiny_motion_support(score)
        if support is None:
            continue
        supports.append((target_member.index, support))
    supports.sort(key=lambda item: item[1], reverse=True)
    return supports


def request_candidate_target_pairs(
    pair_requests: Dict[Tuple[int, int], Tuple[ImageFeatures, ImageFeatures]],
    pair_cache: Dict[Tuple[int, int], PairScore],
    source_member: ImageFeatures,
    target_members: Sequence[ImageFeatures],
    *,
    max_target_candidates: int,
) -> None:
    ranked_targets = sorted(
        target_members,
        key=lambda target: float(source_member.coarse_descriptor @ target.coarse_descriptor),
        reverse=True,
    )
    for target_member in ranked_targets[:max_target_candidates]:
        add_pair_request(pair_requests, pair_cache, source_member, target_member)


def collect_challenge_pair_requests(
    groups: Dict[int, List[ImageFeatures]],
    group_ids: Sequence[int],
    descriptors: Dict[int, object],
    singleton_group_ids: Sequence[int],
    pair_cache: Dict[Tuple[int, int], PairScore],
) -> Dict[Tuple[int, int], Tuple[ImageFeatures, ImageFeatures]]:
    pair_requests: Dict[Tuple[int, int], Tuple[ImageFeatures, ImageFeatures]] = {}

    for source_group_id in singleton_group_ids:
        source_members = groups[source_group_id]
        ranked_targets = []
        for target_group_id in group_ids:
            if target_group_id == source_group_id:
                continue
            affinity = group_affinity(
                source_members,
                groups[target_group_id],
                descriptors[source_group_id],
                descriptors[target_group_id],
            )
            ranked_targets.append((affinity, target_group_id))
        ranked_targets.sort(reverse=True)

        for _affinity, target_group_id in ranked_targets[:24]:
            target_members = groups[target_group_id]
            if len(target_members) == 1:
                add_pair_request(pair_requests, pair_cache, source_members[0], target_members[0])
            else:
                request_candidate_target_pairs(
                    pair_requests,
                    pair_cache,
                    source_members[0],
                    target_members,
                    max_target_candidates=8,
                )

    for offset, left_group_id in enumerate(singleton_group_ids):
        left_members = groups[left_group_id]
        for right_group_id in singleton_group_ids[offset + 1 :]:
            right_members = groups[right_group_id]
            affinity = group_affinity(
                left_members,
                right_members,
                descriptors[left_group_id],
                descriptors[right_group_id],
            )
            if affinity >= 0.60:
                add_pair_request(pair_requests, pair_cache, left_members[0], right_members[0])

    for source_group_id in [group_id for group_id in group_ids if len(groups[group_id]) == 2]:
        source_members = groups[source_group_id]
        ranked_targets = []
        for target_group_id in group_ids:
            if target_group_id == source_group_id or len(groups[target_group_id]) < 4:
                continue
            affinity = group_affinity(
                source_members,
                groups[target_group_id],
                descriptors[source_group_id],
                descriptors[target_group_id],
            )
            if affinity >= 0.52:
                ranked_targets.append((affinity, target_group_id))
        ranked_targets.sort(reverse=True)

        add_pair_request(pair_requests, pair_cache, source_members[0], source_members[1])
        for _affinity, target_group_id in ranked_targets[:16]:
            target_members = groups[target_group_id]
            for source_member in source_members:
                request_candidate_target_pairs(
                    pair_requests,
                    pair_cache,
                    source_member,
                    target_members,
                    max_target_candidates=8,
                )

    for source_group_id in [group_id for group_id in group_ids if 3 <= len(groups[group_id]) <= 12]:
        source_members = groups[source_group_id]
        ranked_targets = []
        for target_group_id in group_ids:
            if target_group_id == source_group_id:
                continue
            target_members = groups[target_group_id]
            if len(target_members) < 3 or len(target_members) > 16:
                continue
            size_ratio = max(len(source_members), len(target_members)) / float(min(len(source_members), len(target_members)))
            if size_ratio > 4.0:
                continue
            affinity = group_affinity(
                source_members,
                target_members,
                descriptors[source_group_id],
                descriptors[target_group_id],
            )
            if affinity >= 0.60:
                ranked_targets.append((affinity, target_group_id))
        ranked_targets.sort(reverse=True)

        for _affinity, target_group_id in ranked_targets[:8]:
            target_members = groups[target_group_id]
            for source_member in source_members:
                request_candidate_target_pairs(
                    pair_requests,
                    pair_cache,
                    source_member,
                    target_members,
                    max_target_candidates=8,
                )
            for target_member in target_members:
                request_candidate_target_pairs(
                    pair_requests,
                    pair_cache,
                    target_member,
                    source_members,
                    max_target_candidates=8,
                )

    return pair_requests



def evaluate_relaxed_group_match(
    source_group_id: int,
    source_members: Sequence[ImageFeatures],
    target_group_id: int,
    target_members: Sequence[ImageFeatures],
    pair_cache: Dict[Tuple[int, int], PairScore],
    *,
    max_target_candidates: int,
) -> Optional[RelaxedGroupMatch]:
    supports: List[Tuple[int, float]] = []
    target_supports_by_source: List[List[Tuple[int, float]]] = []
    for source_member in source_members:
        candidate_supports = candidate_target_supports(
            source_member,
            target_members,
            pair_cache,
            max_target_candidates=max_target_candidates,
        )
        if not candidate_supports:
            return None
        supports.append((source_member.index, candidate_supports[0][1]))
        target_supports_by_source.append(candidate_supports)

    average_support = sum(support for _member, support in supports) / float(len(supports))
    minimum_support = min(support for _member, support in supports)
    if average_support < 0.60 or minimum_support < 0.52:
        return None

    # Singleton-to-group merges are the easiest way to create false positives.
    # Require corroboration from at least two distinct target members.
    if len(source_members) == 1 and len(target_members) > 1:
        corroborating_targets = {
            target_member
            for target_member, support in target_supports_by_source[0]
            if support >= 0.58
        }
        if len(corroborating_targets) < 2:
            best_target, best_support = target_supports_by_source[0][0]
            best_score = cached_pair_score(
                pair_cache,
                source_members[0],
                next(member for member in target_members if member.index == best_target),
            )
            very_strong_singleton = (
                best_support >= 0.80
                and best_score.confidence >= 0.80
                and best_score.avg_corner_shift_ratio <= 0.05
                and best_score.edge_score >= 0.25
            )
            tiny_low_feature_singleton = (
                best_support >= 0.72
                and low_feature_tiny_motion_support(best_score) is not None
            )
            if not very_strong_singleton and not tiny_low_feature_singleton:
                return None

    return RelaxedGroupMatch(
        source_group_id=source_group_id,
        target_group_id=target_group_id,
        average_support=average_support,
        source_supports=supports,
    )


def evaluate_verified_tail_match(
    source_group_id: int,
    source_members: Sequence[ImageFeatures],
    target_group_id: int,
    target_members: Sequence[ImageFeatures],
    pair_cache: Dict[Tuple[int, int], PairScore],
    *,
    max_target_candidates: int,
) -> Optional[RelaxedGroupMatch]:
    if len(source_members) != 2 or len(target_members) < 4:
        return None

    direct_supports: List[Tuple[int, float]] = []
    unsupported_members: List[ImageFeatures] = []
    supported_members: List[ImageFeatures] = []
    for source_member in source_members:
        candidate_supports = candidate_target_supports(
            source_member,
            target_members,
            pair_cache,
            max_target_candidates=max_target_candidates,
        )
        if candidate_supports and candidate_supports[0][1] >= 0.70:
            direct_supports.append((source_member.index, candidate_supports[0][1]))
            supported_members.append(source_member)
        else:
            unsupported_members.append(source_member)

    if len(supported_members) != 1 or len(unsupported_members) != 1:
        return None

    bridge_score = cached_pair_score(pair_cache, unsupported_members[0], supported_members[0])
    if not bridge_score.accepted:
        return None
    if bridge_score.confidence < 0.90 or bridge_score.ransac_inliers < 50:
        return None
    if bridge_score.avg_corner_shift_ratio > 0.05 or bridge_score.max_corner_shift_ratio > 0.10:
        return None

    # Penalize the indirect member so this cannot outrank a fully direct match.
    direct_support = direct_supports[0][1]
    indirect_support = min(0.70, 0.85 * bridge_score.confidence)
    average_support = (direct_support + indirect_support) / 2.0
    if average_support < 0.72:
        return None

    return RelaxedGroupMatch(
        source_group_id=source_group_id,
        target_group_id=target_group_id,
        average_support=average_support,
        source_supports=[
            direct_supports[0],
            (unsupported_members[0].index, indirect_support),
        ],
    )


def evaluate_medium_fragment_match(
    source_group_id: int,
    source_members: Sequence[ImageFeatures],
    target_group_id: int,
    target_members: Sequence[ImageFeatures],
    pair_cache: Dict[Tuple[int, int], PairScore],
    *,
    max_target_candidates: int,
) -> Optional[RelaxedGroupMatch]:
    if len(source_members) < 3 or len(target_members) < 3:
        return None
    if len(source_members) > 12 or len(target_members) > 16:
        return None
    size_ratio = max(len(source_members), len(target_members)) / float(min(len(source_members), len(target_members)))
    if size_ratio > 4.0:
        return None

    supports: List[Tuple[int, float]] = []
    supported_targets = set()
    for source_member in source_members:
        candidate_supports = candidate_target_supports(
            source_member,
            target_members,
            pair_cache,
            max_target_candidates=max_target_candidates,
        )
        if not candidate_supports:
            return None
        target_index, support = candidate_supports[0]
        supports.append((source_member.index, support))
        if support >= 0.60:
            supported_targets.add(target_index)

    average_support = sum(support for _member, support in supports) / float(len(supports))
    minimum_support = min(support for _member, support in supports)
    source_coverage = sum(1 for _member, support in supports if support >= 0.60)
    target_coverage = len(supported_targets)

    if average_support < 0.66 or minimum_support < 0.58:
        return None
    if source_coverage < max(3, math.ceil(0.75 * len(source_members))):
        return None
    if target_coverage < min(3, len(target_members)):
        return None

    return RelaxedGroupMatch(
        source_group_id=source_group_id,
        target_group_id=target_group_id,
        average_support=average_support,
        source_supports=supports,
    )


def challenge_merge_groups(
    features: Sequence[ImageFeatures],
    assignments: Dict[int, object],
    initial_scores: Optional[Sequence[PairScore]] = None,
) -> List[List[str]]:
    groups = build_feature_groups(features, assignments)
    group_ids = sorted(groups)
    if len(group_ids) <= 1:
        return [[member.filename for member in groups[group_ids[0]]]] if group_ids else []

    descriptors = {
        group_id: normalized_average_descriptor(group_members)
        for group_id, group_members in groups.items()
    }
    pair_cache: Dict[Tuple[int, int], PairScore] = {}
    if initial_scores:
        for score in initial_scores:
            pair_cache[tuple(sorted((score.i, score.j)))] = score
        print(
            f"Challenge singleton post-pass: seeded {len(pair_cache)} pair score(s).",
            file=sys.stdout,
            flush=True,
        )

    singleton_group_ids = [group_id for group_id in group_ids if len(groups[group_id]) == 1]
    singleton_group_id_set = set(singleton_group_ids)
    print(
        f"Challenge singleton post-pass: evaluating {len(singleton_group_ids)} unmatched singleton(s).",
        file=sys.stdout,
        flush=True,
    )
    prefetch_pair_scores(
        pair_cache,
        collect_challenge_pair_requests(
            groups,
            group_ids,
            descriptors,
            singleton_group_ids,
            pair_cache,
        ),
        jobs=max(1, min(16, os.cpu_count() or 1)),
    )

    proposals: List[Tuple[float, RelaxedGroupMatch]] = []
    for source_group_id in singleton_group_ids:
        source_members = groups[source_group_id]
        ranked_targets = []
        for target_group_id in group_ids:
            if target_group_id == source_group_id:
                continue
            affinity = group_affinity(
                source_members,
                groups[target_group_id],
                descriptors[source_group_id],
                descriptors[target_group_id],
            )
            ranked_targets.append((affinity, target_group_id))
        ranked_targets.sort(reverse=True)

        matches: List[RelaxedGroupMatch] = []
        for _affinity, target_group_id in ranked_targets[:24]:
            target_members = groups[target_group_id]
            if len(target_members) == 1:
                score = cached_pair_score(pair_cache, source_members[0], target_members[0])
                support = singleton_pair_support(score)
                if support is None:
                    continue
                matches.append(
                    RelaxedGroupMatch(
                        source_group_id=source_group_id,
                        target_group_id=target_group_id,
                        average_support=support,
                        source_supports=[(source_members[0].index, support)],
                    )
                )
                continue

            match = evaluate_relaxed_group_match(
                source_group_id,
                source_members,
                target_group_id,
                target_members,
                pair_cache,
                max_target_candidates=8,
            )
            if match is not None:
                matches.append(match)

        if not matches:
            continue

        matches.sort(key=lambda item: item.average_support, reverse=True)
        best = matches[0]
        second = matches[1].average_support if len(matches) > 1 else 0.0
        threshold = 0.52 if best.target_group_id in singleton_group_id_set else 0.62
        if best.average_support < threshold:
            continue
        if best.average_support < second + 0.05:
            continue
        proposals.append((best.average_support, best))

    singleton_best_matches: Dict[int, Tuple[float, RelaxedGroupMatch]] = {}
    singleton_second_scores: Dict[int, float] = {}
    singleton_pair_checks = 0
    for offset, left_group_id in enumerate(singleton_group_ids):
        left_members = groups[left_group_id]
        for right_group_id in singleton_group_ids[offset + 1 :]:
            right_members = groups[right_group_id]
            affinity = group_affinity(
                left_members,
                right_members,
                descriptors[left_group_id],
                descriptors[right_group_id],
            )
            if affinity < 0.60:
                continue

            singleton_pair_checks += 1
            score = cached_pair_score(pair_cache, left_members[0], right_members[0])
            support = singleton_pair_support(score)
            if support is None or support < 0.52:
                continue

            left_match = RelaxedGroupMatch(
                source_group_id=left_group_id,
                target_group_id=right_group_id,
                average_support=support,
                source_supports=[(left_members[0].index, support)],
            )
            right_match = RelaxedGroupMatch(
                source_group_id=right_group_id,
                target_group_id=left_group_id,
                average_support=support,
                source_supports=[(right_members[0].index, support)],
            )
            for source_group_id, match in (
                (left_group_id, left_match),
                (right_group_id, right_match),
            ):
                current = singleton_best_matches.get(source_group_id)
                if current is None or support > current[0]:
                    if current is not None:
                        singleton_second_scores[source_group_id] = current[0]
                    singleton_best_matches[source_group_id] = (support, match)
                else:
                    singleton_second_scores[source_group_id] = max(
                        singleton_second_scores.get(source_group_id, 0.0),
                        support,
                    )

    for source_group_id, (support, match) in singleton_best_matches.items():
        reverse = singleton_best_matches.get(match.target_group_id)
        if reverse is None or reverse[1].target_group_id != source_group_id:
            continue
        if support < singleton_second_scores.get(source_group_id, 0.0) + 0.05:
            continue
        if support < singleton_second_scores.get(match.target_group_id, 0.0) + 0.05:
            continue
        proposals.append((support, match))
    print(
        f"Challenge singleton post-pass: checked {singleton_pair_checks} singleton pair(s).",
        file=sys.stdout,
        flush=True,
    )

    tail_source_group_ids = [group_id for group_id in group_ids if len(groups[group_id]) == 2]
    print(
        f"Challenge singleton post-pass: evaluating {len(tail_source_group_ids)} two-image tail fragment(s).",
        file=sys.stdout,
        flush=True,
    )
    for source_group_id in tail_source_group_ids:
        source_members = groups[source_group_id]
        ranked_targets = []
        for target_group_id in group_ids:
            if target_group_id == source_group_id or len(groups[target_group_id]) < 4:
                continue
            affinity = group_affinity(
                source_members,
                groups[target_group_id],
                descriptors[source_group_id],
                descriptors[target_group_id],
            )
            if affinity < 0.52:
                continue
            ranked_targets.append((affinity, target_group_id))
        ranked_targets.sort(reverse=True)

        matches: List[RelaxedGroupMatch] = []
        for _affinity, target_group_id in ranked_targets[:16]:
            match = evaluate_verified_tail_match(
                source_group_id,
                source_members,
                target_group_id,
                groups[target_group_id],
                pair_cache,
                max_target_candidates=8,
            )
            if match is not None:
                matches.append(match)

        if not matches:
            continue

        matches.sort(key=lambda item: item.average_support, reverse=True)
        best = matches[0]
        second = matches[1].average_support if len(matches) > 1 else 0.0
        if best.average_support < 0.72:
            continue
        if best.average_support < second + 0.06:
            continue
        proposals.append((best.average_support, best))

    medium_best_matches: Dict[int, RelaxedGroupMatch] = {}
    medium_source_group_ids = [group_id for group_id in group_ids if 3 <= len(groups[group_id]) <= 12]
    print(
        f"Challenge singleton post-pass: evaluating {len(medium_source_group_ids)} medium fragment group(s).",
        file=sys.stdout,
        flush=True,
    )
    for source_group_id in medium_source_group_ids:
        source_members = groups[source_group_id]
        ranked_targets = []
        for target_group_id in group_ids:
            if target_group_id == source_group_id:
                continue
            target_members = groups[target_group_id]
            if len(target_members) < 3 or len(target_members) > 16:
                continue
            size_ratio = max(len(source_members), len(target_members)) / float(min(len(source_members), len(target_members)))
            if size_ratio > 4.0:
                continue
            affinity = group_affinity(
                source_members,
                target_members,
                descriptors[source_group_id],
                descriptors[target_group_id],
            )
            if affinity < 0.60:
                continue
            ranked_targets.append((affinity, target_group_id))
        ranked_targets.sort(reverse=True)

        matches: List[RelaxedGroupMatch] = []
        for _affinity, target_group_id in ranked_targets[:8]:
            match = evaluate_medium_fragment_match(
                source_group_id,
                source_members,
                target_group_id,
                groups[target_group_id],
                pair_cache,
                max_target_candidates=8,
            )
            if match is not None:
                matches.append(match)

        if not matches:
            continue

        matches.sort(key=lambda item: item.average_support, reverse=True)
        best = matches[0]
        second = matches[1].average_support if len(matches) > 1 else 0.0
        if best.average_support < 0.66:
            continue
        if best.average_support < second + 0.04:
            continue
        medium_best_matches[source_group_id] = best
        for match in matches:
            if match.average_support >= 0.72:
                proposals.append((match.average_support - 0.03, match))

    for source_group_id, best in medium_best_matches.items():
        reverse = medium_best_matches.get(best.target_group_id)
        if reverse is None or reverse.target_group_id != source_group_id:
            continue
        if source_group_id > best.target_group_id:
            continue
        proposals.append((min(best.average_support, reverse.average_support), best))
    for source_group_id, best in medium_best_matches.items():
        reverse = medium_best_matches.get(best.target_group_id)
        if reverse is not None and reverse.target_group_id == source_group_id:
            continue
        if best.average_support >= 0.72:
            proposals.append((best.average_support - 0.02, best))

    proposals.sort(key=lambda item: item[0], reverse=True)
    dsu = DSU(len(group_ids))
    group_index = {group_id: offset for offset, group_id in enumerate(group_ids)}
    consumed_singletons = set()
    consumed_tail_fragments = set()
    applied = 0
    for _score, proposal in proposals:
        source_group_id = proposal.source_group_id
        target_group_id = proposal.target_group_id
        if source_group_id in singleton_group_id_set:
            if source_group_id in consumed_singletons:
                continue
            if target_group_id in singleton_group_id_set and target_group_id in consumed_singletons:
                continue
        elif len(groups[source_group_id]) == 2:
            if source_group_id in consumed_tail_fragments:
                continue
        elif 3 <= len(groups[source_group_id]) <= 12:
            pass
        else:
            continue

        left = group_index[source_group_id]
        right = group_index[target_group_id]
        if dsu.find(left) == dsu.find(right):
            continue

        dsu.union(left, right)
        if source_group_id in singleton_group_id_set:
            consumed_singletons.add(source_group_id)
            if target_group_id in singleton_group_id_set:
                consumed_singletons.add(target_group_id)
        else:
            if len(groups[source_group_id]) == 2:
                consumed_tail_fragments.add(source_group_id)
        applied += 1

    merged_feature_groups: Dict[int, List[ImageFeatures]] = {}
    for group_id, members in groups.items():
        root = dsu.find(group_index[group_id])
        merged_feature_groups.setdefault(root, []).extend(members)
    merged_feature_groups = {
        root: sorted(members, key=lambda member: member.filename)
        for root, members in merged_feature_groups.items()
    }
    merged_descriptors = {
        root: normalized_average_descriptor(members)
        for root, members in merged_feature_groups.items()
    }
    merged_roots = sorted(merged_feature_groups)
    followup_proposals: List[Tuple[float, int, int]] = []
    for source_root in merged_roots:
        source_members = merged_feature_groups[source_root]
        if len(source_members) < 3 or len(source_members) > 16:
            continue
        ranked_targets = []
        for target_root in merged_roots:
            if target_root == source_root:
                continue
            target_members = merged_feature_groups[target_root]
            if len(target_members) < 3 or len(target_members) > 16:
                continue
            size_ratio = max(len(source_members), len(target_members)) / float(min(len(source_members), len(target_members)))
            if size_ratio > 4.0:
                continue
            affinity = group_affinity(
                source_members,
                target_members,
                merged_descriptors[source_root],
                merged_descriptors[target_root],
            )
            if affinity < 0.60:
                continue
            ranked_targets.append((affinity, target_root))
        ranked_targets.sort(reverse=True)
        for _affinity, target_root in ranked_targets[:6]:
            match = evaluate_medium_fragment_match(
                source_root,
                source_members,
                target_root,
                merged_feature_groups[target_root],
                pair_cache,
                max_target_candidates=8,
            )
            if match is not None and match.average_support >= 0.70:
                followup_proposals.append((match.average_support, source_root, target_root))

    followup_proposals.sort(reverse=True)
    followup_applied = 0
    for _score, source_root, target_root in followup_proposals:
        if dsu.find(source_root) == dsu.find(target_root):
            continue
        dsu.union(source_root, target_root)
        applied += 1
        followup_applied += 1
    if followup_proposals:
        print(
            f"Challenge singleton post-pass: applied {followup_applied}/{len(followup_proposals)} follow-up medium merge(s).",
            file=sys.stdout,
            flush=True,
        )

    merged: Dict[int, List[str]] = {}
    for group_id, members in groups.items():
        root = dsu.find(group_index[group_id])
        merged.setdefault(root, []).extend(member.filename for member in members)

    merged_groups = [
        sorted(filenames)
        for _root, filenames in sorted(merged.items(), key=lambda item: min(item[1]))
    ]
    print(
        f"Challenge singleton post-pass: applied {applied} merge(s); final groups={len(merged_groups)}.",
        file=sys.stdout,
        flush=True,
    )
    return merged_groups


def _large_progress(label: str, completed: int, total: int) -> None:
    if total <= 0:
        return
    stride = max(1, total // 100)
    if completed not in {0, total} and completed % stride != 0:
        return
    percent = int(round(100.0 * completed / float(total)))
    print(f"{label}: {percent:3d}% ({completed}/{total})", file=sys.stdout, flush=True)


def extract_compact_feature(index: int, path: Path, max_size: int) -> CompactImageFeature:
    import cv2
    import numpy as np

    data = np.fromfile(str(path), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"Could not decode image: {path}")

    height, width = gray.shape[:2]
    largest = max(height, width)
    if largest > max_size:
        scale = max_size / float(largest)
        gray = cv2.resize(
            gray,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = gray.shape[:2]

    normalized = cv2.equalizeHist(gray)
    edges = edge_image(normalized)
    gradient = gradient_image(normalized)
    descriptor = coarse_descriptor(normalized, edges, gradient)
    return CompactImageFeature(
        index=index,
        path=path,
        filename=path.name,
        width=width,
        height=height,
        descriptor=descriptor,
    )


def extract_compact_features(
    paths: Sequence[Path],
    *,
    max_size: int,
    jobs: int,
) -> List[CompactImageFeature]:
    if not paths:
        return []

    _large_progress("Compact feature extraction", 0, len(paths))
    features: List[Optional[CompactImageFeature]] = [None] * len(paths)
    if jobs <= 1 or len(paths) <= 1:
        for index, path in enumerate(paths):
            features[index] = extract_compact_feature(index, path, max_size)
            _large_progress("Compact feature extraction", index + 1, len(paths))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(extract_compact_feature, index, path, max_size): index
                for index, path in enumerate(paths)
            }
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                features[index] = future.result()
                completed += 1
                _large_progress("Compact feature extraction", completed, len(paths))

    return [feature for feature in features if feature is not None]


def compact_groups_from_features(
    features: Sequence[CompactImageFeature],
    *,
    neighbors: int,
    min_similarity: float,
    max_aspect_delta: float,
) -> List[List[str]]:
    pairs = compact_neighbor_pairs(features, neighbors=neighbors)
    components = compact_components_from_pairs(
        features,
        pairs,
        min_similarity=min_similarity,
        max_aspect_delta=max_aspect_delta,
    )
    return filenames_from_components(features, components)


def compact_neighbor_pairs(
    features: Sequence[CompactImageFeature],
    *,
    neighbors: int,
) -> List[CompactNeighborPair]:
    import numpy as np

    if not features:
        return []
    if len(features) == 1:
        return []

    matrix = np.vstack([feature.descriptor for feature in features]).astype(np.float32, copy=False)
    for feature in features:
        feature.descriptor = None
    if LARGE_BLOCKWISE_CANDIDATES:
        return compact_neighbor_pairs_blockwise(
            matrix,
            neighbors=neighbors,
            block_size=LARGE_BLOCK_SIZE,
            min_similarity=LARGE_CANDIDATE_MIN_SIMILARITY,
        )

    return compact_neighbor_pairs_flann(matrix, neighbors=neighbors)


def compact_neighbor_pairs_flann(
    matrix: "object",
    *,
    neighbors: int,
) -> List[CompactNeighborPair]:
    import cv2

    if hasattr(cv2, "setNumThreads"):
        cv2.setNumThreads(1)

    k = max(2, min(len(matrix), neighbors + 1))
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=4),
        dict(checks=128),
    )
    print(
        f"Large dataset mode: querying {k - 1} compact neighbor(s) per image.",
        file=sys.stdout,
        flush=True,
    )
    matches = matcher.knnMatch(matrix, matrix, k=k)

    pairs: List[CompactNeighborPair] = []
    seen_pairs = set()
    for candidate_matches in matches:
        for match in candidate_matches:
            left = int(match.queryIdx)
            right = int(match.trainIdx)
            if left == right:
                continue
            pair = (left, right) if left < right else (right, left)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            similarity = 1.0 - (float(match.distance) * float(match.distance)) / 2.0
            pairs.append(CompactNeighborPair(similarity, pair[0], pair[1]))

    pairs.sort(key=lambda item: item.similarity, reverse=True)
    return pairs


def compact_neighbor_pairs_blockwise(
    matrix: "object",
    *,
    neighbors: int,
    block_size: int,
    min_similarity: float,
) -> List[CompactNeighborPair]:
    import gc
    import numpy as np

    count = int(matrix.shape[0])
    topk = max(1, min(neighbors, count - 1))
    block_size = max(1, block_size)
    print(
        "Large dataset mode: blockwise exact compact search "
        f"top_k={topk} block_size={block_size} min_similarity={min_similarity:0.3f}.",
        file=sys.stdout,
        flush=True,
    )
    pairs: Dict[Tuple[int, int], float] = {}
    started_at = time.time()
    for start in range(0, count, block_size):
        stop = min(count, start + block_size)
        scores = matrix[start:stop] @ matrix.T
        rows = stop - start
        row_indexes = np.arange(rows)
        scores[row_indexes, start + row_indexes] = -np.inf

        if topk >= count - 1:
            top_indexes = np.argsort(scores, axis=1)[:, ::-1]
        else:
            partition = np.argpartition(scores, -topk, axis=1)[:, -topk:]
            partition_scores = np.take_along_axis(scores, partition, axis=1)
            order = np.argsort(partition_scores, axis=1)[:, ::-1]
            top_indexes = np.take_along_axis(partition, order, axis=1)

        for row_offset in range(rows):
            left = start + row_offset
            for right_value in top_indexes[row_offset]:
                right = int(right_value)
                similarity = float(scores[row_offset, right])
                if not np.isfinite(similarity) or similarity < min_similarity:
                    continue
                pair = (left, right) if left < right else (right, left)
                current = pairs.get(pair)
                if current is None or similarity > current:
                    pairs[pair] = similarity

        completed = stop
        if completed == count or completed % max(block_size, count // 20) < block_size:
            elapsed = time.time() - started_at
            print(
                "Large dataset mode: compact search "
                f"{completed}/{count} image(s), pairs={len(pairs)}, elapsed={elapsed:0.1f}s.",
                file=sys.stdout,
                flush=True,
            )

    gc.collect()
    result = [
        CompactNeighborPair(similarity, left, right)
        for (left, right), similarity in pairs.items()
    ]
    result.sort(key=lambda item: item.similarity, reverse=True)
    print(
        f"Large dataset mode: retained {len(result)} compact candidate pair(s).",
        file=sys.stdout,
        flush=True,
    )
    return result


def compact_components_from_pairs(
    features: Sequence[CompactImageFeature],
    pairs: Sequence[CompactNeighborPair],
    *,
    min_similarity: float,
    max_aspect_delta: float,
) -> List[List[int]]:
    dsu = DSU(len(features))
    accepted = 0
    considered = 0
    for pair in pairs:
        if pair.similarity < min_similarity:
            continue
        considered += 1

        left_feature = features[pair.left]
        right_feature = features[pair.right]
        left_aspect = left_feature.width / max(1.0, float(left_feature.height))
        right_aspect = right_feature.width / max(1.0, float(right_feature.height))
        if abs(left_aspect - right_aspect) > max_aspect_delta:
            continue

        dsu.union(pair.left, pair.right)
        accepted += 1

    groups: Dict[int, List[int]] = {}
    for feature in features:
        groups.setdefault(dsu.find(feature.index), []).append(feature.index)

    print(
        "Large dataset mode: "
        f"accepted {accepted}/{considered} compact pair(s) at similarity>={min_similarity:0.3f}; "
        f"groups={len(groups)}.",
        file=sys.stdout,
        flush=True,
    )
    return [
        sorted(indexes, key=lambda index: features[index].filename)
        for _root, indexes in sorted(groups.items(), key=lambda item: min(features[index].filename for index in item[1]))
    ]


def filenames_from_components(
    features: Sequence[CompactImageFeature], components: Sequence[Sequence[int]]
) -> List[List[str]]:
    return [
        sorted(features[index].filename for index in component)
        for component in components
    ]


def split_oversized_component(
    component: Sequence[int],
    features: Sequence[CompactImageFeature],
    pairs: Sequence[CompactNeighborPair],
    *,
    max_size: int,
    max_aspect_delta: float,
) -> List[List[int]]:
    if len(component) <= max_size:
        return [list(component)]

    component_set = set(component)
    component_pairs = [
        pair for pair in pairs if pair.left in component_set and pair.right in component_set
    ]
    threshold = LARGE_SUPER_MIN_SIMILARITY + 0.04
    while threshold <= 0.98:
        parts = compact_components_from_pairs(
            features,
            component_pairs,
            min_similarity=threshold,
            max_aspect_delta=max_aspect_delta,
        )
        parts = [part for part in parts if part and part[0] in component_set]
        if parts and max(len(part) for part in parts) <= max_size:
            print(
                "Large hybrid mode: split oversized component "
                f"n={len(component)} into {len(parts)} part(s) at similarity>={threshold:0.3f}.",
                file=sys.stdout,
                flush=True,
            )
            return parts
        threshold += 0.04

    print(
        "Large hybrid mode: oversized component "
        f"n={len(component)} remains too large; using compact fallback for it.",
        file=sys.stdout,
        flush=True,
    )
    return [list(component)]


def refine_large_component(component: Sequence[int], paths: Sequence[Path]) -> List[List[str]]:
    if len(component) == 1:
        return [[paths[component[0]].name]]

    component_paths = [paths[index] for index in component]
    jobs = max(1, min(8, os.cpu_count() or 1, len(component_paths)))
    features, assignments, _representatives, scores = run_grouping(
        component_paths,
        max_size=LARGE_COMPONENT_MAX_SIZE,
        features_per_variant=LARGE_COMPONENT_FEATURES,
        thresholds=default_thresholds(),
        max_candidates=LARGE_COMPONENT_CANDIDATES,
        jobs=jobs,
        pair_cache_path=None,
    )
    if LARGE_COMPONENT_POSTPASS:
        return challenge_merge_groups(features, assignments, scores)
    return groups_from_assignments(features, assignments)


def group_images_large_hybrid(
    paths: Sequence[Path],
    compact_features: Sequence[CompactImageFeature],
) -> List[List[str]]:
    started_at = time.time()
    pairs = compact_neighbor_pairs(compact_features, neighbors=LARGE_NEIGHBORS)
    super_components = compact_components_from_pairs(
        compact_features,
        pairs,
        min_similarity=LARGE_SUPER_MIN_SIMILARITY,
        max_aspect_delta=LARGE_MAX_ASPECT_DELTA,
    )

    bounded_components: List[List[int]] = []
    for component in super_components:
        bounded_components.extend(
            split_oversized_component(
                component,
                compact_features,
                pairs,
                max_size=LARGE_MAX_COMPONENT_SIZE,
                max_aspect_delta=LARGE_MAX_ASPECT_DELTA,
            )
        )

    bounded_components.sort(
        key=lambda component: (-len(component), min(compact_features[index].filename for index in component))
    )
    refined_groups: List[List[str]] = []
    compact_fallback_components = 0
    budget_fallback_components = 0
    refined_components = 0
    refined_images = 0
    total_components = len(bounded_components)
    for offset, component in enumerate(bounded_components, start=1):
        if len(component) == 1:
            refined_groups.append([compact_features[component[0]].filename])
            continue
        if time.time() - started_at >= LARGE_HYBRID_TIME_BUDGET:
            budget_fallback_components += 1
            refined_groups.append(sorted(compact_features[index].filename for index in component))
            continue
        if len(component) > LARGE_MAX_COMPONENT_SIZE:
            compact_fallback_components += 1
            refined_groups.append(sorted(compact_features[index].filename for index in component))
            continue

        print(
            "Large hybrid mode: refining component "
            f"{offset}/{total_components} with {len(component)} image(s).",
            file=sys.stdout,
            flush=True,
        )
        refined_groups.extend(refine_large_component(component, paths))
        refined_components += 1
        refined_images += len(component)

    print(
        "Large hybrid mode: "
        f"refined {refined_components} component(s), {refined_images} image(s); "
        f"compact fallback components={compact_fallback_components}; "
        f"budget fallback components={budget_fallback_components}; "
        f"final groups={len(refined_groups)}.",
        file=sys.stdout,
        flush=True,
    )
    return [sorted(group) for group in sorted(refined_groups, key=lambda group: min(group))]


def _large_aspect_delta(
    left_feature: CompactImageFeature, right_feature: CompactImageFeature
) -> float:
    left_aspect = left_feature.width / max(1.0, float(left_feature.height))
    right_aspect = right_feature.width / max(1.0, float(right_feature.height))
    return abs(left_aspect - right_aspect)


def large_bridge_accepts(score: PairScore) -> bool:
    if score.decision == "verified":
        return True
    if not LARGE_BRIDGE_ACCEPT_STRONG_REJECT:
        return False
    return (
        score.ransac_inliers >= 18
        and score.inlier_ratio >= 0.45
        and score.median_reprojection_error <= 3.0
        and score.avg_corner_shift_ratio <= 0.12
    )


def extract_large_bridge_features(
    indexes: Sequence[int],
    paths: Sequence[Path],
    jobs: int,
) -> Dict[int, ImageFeatures]:
    features: Dict[int, ImageFeatures] = {}
    if not indexes:
        return features
    with ThreadPoolExecutor(max_workers=min(jobs, len(indexes))) as executor:
        futures = {
            executor.submit(
                extract_features,
                index,
                paths[index],
                LARGE_COMPONENT_MAX_SIZE,
                LARGE_COMPONENT_FEATURES,
            ): index
            for index in indexes
        }
        for future in as_completed(futures):
            features[futures[future]] = future.result()
    return features


def score_large_bridge_pair(
    pair: CompactNeighborPair,
    features: Dict[int, ImageFeatures],
) -> Tuple[CompactNeighborPair, PairScore]:
    return pair, compare_pair(features[pair.left], features[pair.right], default_thresholds())


def group_images_large_bridge(
    paths: Sequence[Path],
    compact_features: Sequence[CompactImageFeature],
) -> List[List[str]]:
    """Compact grouping plus bounded geometric checks on borderline edges."""
    started_at = time.time()
    pairs = compact_neighbor_pairs(compact_features, neighbors=LARGE_NEIGHBORS)
    base_components = compact_components_from_pairs(
        compact_features,
        pairs,
        min_similarity=LARGE_MIN_SIMILARITY,
        max_aspect_delta=LARGE_MAX_ASPECT_DELTA,
    )

    component_for_index: Dict[int, int] = {}
    for component_id, component in enumerate(base_components):
        for index in component:
            component_for_index[index] = component_id

    candidates: List[CompactNeighborPair] = []
    component_pair_counts: Dict[Tuple[int, int], int] = {}
    for pair in pairs:
        if pair.similarity >= LARGE_MIN_SIMILARITY:
            continue
        if pair.similarity < LARGE_BRIDGE_MIN_SIMILARITY:
            break
        left_component = component_for_index[pair.left]
        right_component = component_for_index[pair.right]
        if left_component == right_component:
            continue
        if _large_aspect_delta(compact_features[pair.left], compact_features[pair.right]) > LARGE_MAX_ASPECT_DELTA:
            continue
        component_pair = (
            (left_component, right_component)
            if left_component < right_component
            else (right_component, left_component)
        )
        if (
            LARGE_BRIDGE_EDGES_PER_COMPONENT_PAIR > 0
            and component_pair_counts.get(component_pair, 0) >= LARGE_BRIDGE_EDGES_PER_COMPONENT_PAIR
        ):
            continue
        component_pair_counts[component_pair] = component_pair_counts.get(component_pair, 0) + 1
        candidates.append(pair)
        if len(candidates) >= LARGE_BRIDGE_MAX_PAIRS:
            break

    print(
        "Large bridge mode: "
        f"checking {len(candidates)} borderline compact edge(s) "
        f"at {LARGE_BRIDGE_MIN_SIMILARITY:0.3f}<sim<{LARGE_MIN_SIMILARITY:0.3f}.",
        file=sys.stdout,
        flush=True,
    )

    dsu = DSU(len(base_components))
    accepted = 0
    checked = 0
    jobs = max(1, min(16, os.cpu_count() or 1))
    batch_size = max(1, LARGE_BRIDGE_BATCH_SIZE)
    for start in range(0, len(candidates), batch_size):
        if time.time() - started_at >= LARGE_BRIDGE_TIME_BUDGET:
            break
        batch = candidates[start : start + batch_size]
        batch_feature_indexes = sorted({pair.left for pair in batch} | {pair.right for pair in batch})
        batch_features = extract_large_bridge_features(batch_feature_indexes, paths, jobs)
        with ThreadPoolExecutor(max_workers=min(jobs, len(batch))) as executor:
            futures = [executor.submit(score_large_bridge_pair, pair, batch_features) for pair in batch]
            for future in as_completed(futures):
                pair, score = future.result()
                checked += 1
                if not large_bridge_accepts(score):
                    continue
                left_component = component_for_index[pair.left]
                right_component = component_for_index[pair.right]
                if dsu.find(left_component) == dsu.find(right_component):
                    continue
                dsu.union(left_component, right_component)
                accepted += 1
        print(
            "Large bridge mode: "
            f"checked {checked}/{len(candidates)} edge(s), accepted={accepted}, "
            f"elapsed={time.time() - started_at:0.1f}s.",
            file=sys.stdout,
            flush=True,
        )

    merged: Dict[int, List[str]] = {}
    for component_id, component in enumerate(base_components):
        root = dsu.find(component_id)
        merged.setdefault(root, [])
        merged[root].extend(compact_features[index].filename for index in component)

    groups = [sorted(filenames) for filenames in merged.values()]
    print(
        "Large bridge mode: "
        f"base groups={len(base_components)}, final groups={len(groups)}, "
        f"accepted={accepted}/{checked}.",
        file=sys.stdout,
        flush=True,
    )
    return [sorted(group) for group in sorted(groups, key=lambda group: min(group))]


def group_images_large(paths: Sequence[Path]) -> List[List[str]]:
    """Memory-bounded fallback for challenge-scale private inputs."""
    jobs = max(1, min(16, os.cpu_count() or 1))
    print(
        "Large dataset mode: "
        f"images={len(paths)} max_size={LARGE_MAX_SIZE} neighbors={LARGE_NEIGHBORS} "
        f"min_similarity={LARGE_MIN_SIMILARITY:0.3f}",
        file=sys.stdout,
        flush=True,
    )
    features = extract_compact_features(paths, max_size=LARGE_MAX_SIZE, jobs=jobs)
    if LARGE_GEOMETRY_BRIDGE:
        return group_images_large_bridge(paths, features)
    if LARGE_HYBRID:
        return group_images_large_hybrid(paths, features)

    return compact_groups_from_features(
        features,
        neighbors=LARGE_NEIGHBORS,
        min_similarity=LARGE_MIN_SIMILARITY,
        max_aspect_delta=LARGE_MAX_ASPECT_DELTA,
    )


def group_images(image_paths: List[str]) -> List[List[str]]:
    """Group challenge images and return basename-only groups."""
    paths = [Path(path) for path in image_paths]
    if len(paths) >= LARGE_DATASET_THRESHOLD:
        return group_images_large(paths)

    jobs = max(1, min(16, os.cpu_count() or 1))
    features, assignments, _representatives, _scores = run_grouping(
        paths,
        max_size=MAX_SIZE,
        features_per_variant=FEATURES_PER_VARIANT,
        thresholds=default_thresholds(),
        max_candidates=MAX_CANDIDATES,
        jobs=jobs,
        pair_cache_path=Path(PAIR_CACHE_PATH) if PAIR_CACHE_PATH else None,
    )
    return challenge_merge_groups(features, assignments, _scores)


def main() -> None:
    images = sorted([
        str(path) for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in SUPPORTED
    ])
    print(f"Loaded {len(images)} images from {INPUT_DIR}")

    groups = group_images(images)
    print(f"Predicted {len(groups)} groups")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictions.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "group_id"])
        for group_id, group in enumerate(groups):
            for filename in sorted(group):
                writer.writerow([os.path.basename(filename), group_id])

    print(f"Wrote {sum(len(group) for group in groups)} predictions to {out_path}")


if __name__ == "__main__":
    main()
