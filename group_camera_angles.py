#!/usr/bin/env python3
"""Group real estate shoot JPEGs by camera angle.

The matcher is intentionally conservative:
  * filenames and EXIF are not used for grouping;
  * strong SIFT + homography evidence creates verified groups;
  * very light/dark brackets may join as tentative only when edge/gradient
    structure aligns under tiny camera motion;
  * otherwise images remain singleton for manual review.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency. Install with:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install numpy opencv-python"
    ) from exc


IMAGE_EXTENSIONS = {".jpg", ".jpeg"}


@dataclass(frozen=True)
class Thresholds:
    ratio: float
    min_good_matches: int
    min_inliers: int
    min_inlier_ratio: float
    max_median_reprojection_error: float
    max_avg_corner_shift: float
    max_corner_shift: float
    min_edge_score: float
    tentative_edge_score: float
    max_tentative_corner_shift: float


@dataclass
class ImageFeatures:
    index: int
    path: Path
    filename: str
    original_width: int
    original_height: int
    width: int
    height: int
    gray: "np.ndarray"
    normalized: "np.ndarray"
    edges: "np.ndarray"
    gradient: "np.ndarray"
    coarse_descriptor: "np.ndarray"
    keypoints: List["cv2.KeyPoint"]
    descriptors: Optional["np.ndarray"]
    clipped_ratio: float
    load_warning: str = ""

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def original_diagonal(self) -> float:
        return math.hypot(self.original_width, self.original_height)

    @property
    def resize_scale_x(self) -> float:
        return self.width / max(1.0, float(self.original_width))

    @property
    def resize_scale_y(self) -> float:
        return self.height / max(1.0, float(self.original_height))


@dataclass(frozen=True)
class PairScore:
    i: int
    j: int
    filename_a: str
    filename_b: str
    decision: str
    confidence: float
    good_matches: int
    ransac_inliers: int
    inlier_ratio: float
    median_reprojection_error: float
    avg_corner_shift_ratio: float
    max_corner_shift_ratio: float
    avg_corner_shift_pixels: float
    max_corner_shift_pixels: float
    edge_score: float
    gradient_score: float
    method: str
    reason: str

    @property
    def accepted(self) -> bool:
        return self.decision in {"verified", "tentative"}


@dataclass
class MemberAssignment:
    group_id: int
    confidence: float
    status: str


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


def default_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(12, cpu_count))


def default_thresholds() -> Thresholds:
    return Thresholds(
        ratio=0.75,
        min_good_matches=45,
        min_inliers=25,
        min_inlier_ratio=0.30,
        max_median_reprojection_error=5.0,
        max_avg_corner_shift=0.08,
        max_corner_shift=0.16,
        min_edge_score=0.50,
        tentative_edge_score=0.62,
        max_tentative_corner_shift=0.10,
    )


_LAST_PROGRESS_PERCENT: Dict[Tuple[str, int], int] = {}


def render_progress_bar(label: str, completed: int, total: int, *, started_at: Optional[float] = None) -> None:
    total = max(1, total)
    completed = max(0, min(completed, total))
    percent = int(round(100.0 * completed / float(total)))
    key = (label, total)
    last_percent = _LAST_PROGRESS_PERCENT.get(key)
    if last_percent is not None and percent == last_percent and completed < total:
        return
    _LAST_PROGRESS_PERCENT[key] = percent
    suffix = ""
    if started_at is not None and completed > 0:
        elapsed = max(0.0, time.time() - started_at)
        rate = elapsed / float(completed)
        remaining = max(0.0, rate * (total - completed))
        suffix = f" elapsed={elapsed:0.1f}s eta={remaining:0.1f}s"
    print(
        f"{label}: {percent:3d}% ({completed}/{total}){suffix}",
        end="\n",
        file=sys.stdout,
        flush=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group JPEG images by camera angle using conservative geometric matching."
    )
    parser.add_argument("input_folder", type=Path, help="Folder containing randomized JPEGs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("angle_groups.csv"),
        help="Output CSV path. Default: angle_groups.csv",
    )
    parser.add_argument(
        "--pair-scores",
        type=Path,
        default=Path("pair_scores.csv"),
        help="Pairwise debug CSV path. Default: pair_scores.csv",
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("angle_group_review"),
        help="Directory for contact sheet review artifacts. Default: angle_group_review",
    )
    parser.add_argument("--recursive", action="store_true", help="Scan input folder recursively.")
    parser.add_argument("--skip-contact-sheets", action="store_true", help="Do not write PNG contact sheets.")
    parser.add_argument("--max-size", type=int, default=1024, help="Maximum working image side. Default: 1024")
    parser.add_argument(
        "--features-per-variant",
        type=int,
        default=2500,
        help="SIFT features per normalized variant. Default: 2500",
    )
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio threshold. Default: 0.75")
    parser.add_argument("--min-good-matches", type=int, default=45, help="Verified match threshold. Default: 45")
    parser.add_argument("--min-inliers", type=int, default=25, help="Verified RANSAC inlier threshold. Default: 25")
    parser.add_argument("--min-inlier-ratio", type=float, default=0.30, help="Verified inlier ratio. Default: 0.30")
    parser.add_argument(
        "--max-median-error",
        type=float,
        default=5.0,
        help="Maximum median reprojection error in pixels. Default: 5",
    )
    parser.add_argument(
        "--max-avg-corner-shift",
        type=float,
        default=0.08,
        help="Maximum average warped-corner movement as image diagonal ratio. Default: 0.08",
    )
    parser.add_argument(
        "--max-corner-shift",
        type=float,
        default=0.16,
        help="Maximum single warped-corner movement as image diagonal ratio. Default: 0.16",
    )
    parser.add_argument(
        "--min-edge-score",
        type=float,
        default=0.50,
        help="Structural score required for borderline verified matches. Default: 0.50",
    )
    parser.add_argument(
        "--tentative-edge-score",
        type=float,
        default=0.62,
        help="Structural score required for tentative extreme-bracket joins. Default: 0.62",
    )
    parser.add_argument(
        "--max-tentative-corner-shift",
        type=float,
        default=0.10,
        help="Maximum average corner shift for tentative joins. Default: 0.10",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=40,
        help="Maximum coarse candidates per image before geometric verification. Default: 40",
    )
    parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Compare every image pair instead of candidate pruning.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs(),
        help="Worker threads for feature extraction and pair scoring. Default: auto",
    )
    return parser.parse_args(argv)


def list_images(folder: Path, recursive: bool) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {folder}")

    iterator: Iterable[Path]
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    images = sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No JPEG images found in {folder}")
    return images


def imread_bgr(path: Path) -> "np.ndarray":
    # imdecode handles non-ASCII paths more reliably than cv2.imread.
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def resize_max_side(image: "np.ndarray", max_size: int) -> "np.ndarray":
    height, width = image.shape[:2]
    largest = max(height, width)
    if largest <= max_size:
        return image
    scale = max_size / float(largest)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def gamma_lut(gamma: float) -> "np.ndarray":
    table = [((value / 255.0) ** gamma) * 255.0 for value in range(256)]
    return np.array(table, dtype=np.uint8)


def normalize_variants(gray: "np.ndarray") -> List[Tuple[str, "np.ndarray"]]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants: List[Tuple[str, "np.ndarray"]] = [("clahe", clahe.apply(gray))]

    equalized = cv2.equalizeHist(gray)
    variants.append(("equalized", equalized))

    brightened = cv2.LUT(gray, gamma_lut(0.55))
    variants.append(("bright_clahe", clahe.apply(brightened)))

    darkened = cv2.LUT(gray, gamma_lut(1.8))
    variants.append(("dark_clahe", clahe.apply(darkened)))

    log_image = np.log1p(gray.astype(np.float32))
    log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    variants.append(("log_clahe", clahe.apply(log_image)))

    return variants


def build_feature_mask(gray: "np.ndarray") -> Tuple["np.ndarray", float]:
    valid = ((gray > 8) & (gray < 247)).astype(np.uint8) * 255
    clipped_ratio = 1.0 - float(np.count_nonzero(valid)) / float(valid.size)

    kernel = np.ones((5, 5), dtype=np.uint8)
    valid = cv2.morphologyEx(valid, cv2.MORPH_OPEN, kernel)
    valid = cv2.dilate(valid, kernel, iterations=1)

    if np.count_nonzero(valid) < valid.size * 0.05:
        # Extremely clipped frame: let SIFT try the whole image instead of using
        # a nearly empty mask.
        valid[:, :] = 255
    return valid, clipped_ratio


def gradient_image(gray: "np.ndarray") -> "np.ndarray":
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)


def edge_image(gray: "np.ndarray") -> "np.ndarray":
    median = float(np.median(gray))
    lower = int(max(20, 0.66 * median))
    upper = int(min(240, 1.33 * median + 30))
    if upper <= lower:
        lower, upper = 50, 150
    edges = cv2.Canny(gray, lower, upper)
    if np.count_nonzero(edges) < edges.size * 0.005:
        edges = cv2.Canny(gray, 30, 100)
    return edges


def coarse_descriptor(normalized: "np.ndarray", edges: "np.ndarray", gradient: "np.ndarray") -> "np.ndarray":
    small_gradient = cv2.resize(gradient, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    edge_float = (edges > 0).astype(np.float32)
    edge_blur = cv2.GaussianBlur(edge_float, (5, 5), 0)
    small_edges = cv2.resize(edge_blur, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_gray = cv2.resize(normalized, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    vector = np.concatenate([small_gradient.ravel(), small_edges.ravel(), small_gray.ravel()])
    vector -= np.mean(vector)
    norm = np.linalg.norm(vector)
    if norm > 1e-8:
        vector /= norm
    return vector.astype(np.float32, copy=False)


def create_sift(features_per_variant: int) -> "cv2.SIFT":
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This OpenCV build does not expose SIFT. Install a recent opencv-python "
            "or opencv-contrib-python package."
        )
    return cv2.SIFT_create(nfeatures=features_per_variant, contrastThreshold=0.025)


def rootsift(descriptors: Optional["np.ndarray"]) -> Optional["np.ndarray"]:
    if descriptors is None or len(descriptors) == 0:
        return None
    descriptors = descriptors.astype(np.float32, copy=False)
    eps = 1e-7
    descriptors /= descriptors.sum(axis=1, keepdims=True) + eps
    descriptors = np.sqrt(descriptors)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors /= norms + eps
    return descriptors.astype(np.float32, copy=False)


def extract_features(index: int, path: Path, max_size: int, features_per_variant: int) -> ImageFeatures:
    original_bgr = imread_bgr(path)
    original_height, original_width = original_bgr.shape[:2]
    bgr = resize_max_side(original_bgr, max_size)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    mask, clipped_ratio = build_feature_mask(gray)
    variants = normalize_variants(gray)
    normalized = variants[0][1]
    edges = edge_image(normalized)
    gradient = gradient_image(normalized)
    coarse = coarse_descriptor(normalized, edges, gradient)
    sift = create_sift(features_per_variant)

    all_keypoints: List["cv2.KeyPoint"] = []
    all_descriptors: List["np.ndarray"] = []
    for _name, variant in variants:
        keypoints, descriptors = sift.detectAndCompute(variant, mask)
        descriptors = rootsift(descriptors)
        if descriptors is None or not keypoints:
            continue
        all_keypoints.extend(keypoints)
        all_descriptors.append(descriptors)

    merged_descriptors: Optional["np.ndarray"]
    if all_descriptors:
        merged_descriptors = np.vstack(all_descriptors).astype(np.float32, copy=False)
    else:
        merged_descriptors = None

    return ImageFeatures(
        index=index,
        path=path,
        filename=path.name,
        original_width=original_width,
        original_height=original_height,
        width=width,
        height=height,
        gray=gray,
        normalized=normalized,
        edges=edges,
        gradient=gradient,
        coarse_descriptor=coarse,
        keypoints=all_keypoints,
        descriptors=merged_descriptors,
        clipped_ratio=clipped_ratio,
    )


def extract_all_features(
    paths: Sequence[Path], max_size: int, features_per_variant: int, jobs: int
) -> List[ImageFeatures]:
    started_at = time.time()
    render_progress_bar("Feature extraction", 0, len(paths), started_at=started_at)
    if jobs <= 1 or len(paths) <= 1:
        features: List[ImageFeatures] = []
        for index, path in enumerate(paths):
            feature = extract_features(index, path, max_size, features_per_variant)
            features.append(feature)
            render_progress_bar("Feature extraction", index + 1, len(paths), started_at=started_at)
        return features

    features: List[Optional[ImageFeatures]] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {
            executor.submit(extract_features, index, path, max_size, features_per_variant): (index, path)
            for index, path in enumerate(paths)
        }
        completed = 0
        for future in as_completed(future_map):
            index, path = future_map[future]
            feature = future.result()
            features[index] = feature
            completed += 1
            render_progress_bar("Feature extraction", completed, len(paths), started_at=started_at)
    return [feature for feature in features if feature is not None]


def match_descriptors(
    a: ImageFeatures, b: ImageFeatures, ratio: float
) -> List["cv2.DMatch"]:
    if a.descriptors is None or b.descriptors is None:
        return []
    if len(a.descriptors) < 4 or len(b.descriptors) < 4:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = matcher.knnMatch(a.descriptors, b.descriptors, k=2)
    good: List["cv2.DMatch"] = []
    for candidates in knn:
        if len(candidates) != 2:
            continue
        first, second = candidates
        if first.distance <= ratio * second.distance:
            good.append(first)
    return good


def estimate_homography(
    a: ImageFeatures, b: ImageFeatures, matches: Sequence["cv2.DMatch"]
) -> Tuple[Optional["np.ndarray"], "np.ndarray", float]:
    if len(matches) < 4:
        return None, np.zeros((0,), dtype=bool), float("inf")

    src = np.float32([a.keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([b.keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return None, np.zeros((len(matches),), dtype=bool), float("inf")

    inlier_mask = mask.ravel().astype(bool)
    if np.count_nonzero(inlier_mask) == 0:
        return homography, inlier_mask, float("inf")

    projected = cv2.perspectiveTransform(src[inlier_mask], homography)
    errors = np.linalg.norm(projected - dst[inlier_mask], axis=2).ravel()
    median_error = float(np.median(errors)) if len(errors) else float("inf")
    return homography, inlier_mask, median_error


def homography_to_original_scale(a: ImageFeatures, b: ImageFeatures, homography: "np.ndarray") -> "np.ndarray":
    source_scale = np.array(
        [[a.resize_scale_x, 0.0, 0.0], [0.0, a.resize_scale_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    target_scale_inv = np.array(
        [
            [1.0 / max(b.resize_scale_x, 1e-8), 0.0, 0.0],
            [0.0, 1.0 / max(b.resize_scale_y, 1e-8), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return target_scale_inv @ homography @ source_scale


def corner_shift_metrics(
    a: ImageFeatures, b: ImageFeatures, homography: "np.ndarray"
) -> Tuple[float, float, float, float]:
    corners = np.float32(
        [[0, 0], [a.width - 1, 0], [a.width - 1, a.height - 1], [0, a.height - 1]]
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    original = corners.reshape(-1, 2)

    if a.width == b.width and a.height == b.height:
        reference = original
    else:
        scale_x = b.width / max(1.0, float(a.width))
        scale_y = b.height / max(1.0, float(a.height))
        reference = original * np.array([scale_x, scale_y], dtype=np.float32)

    distances = np.linalg.norm(warped - reference, axis=1)
    diagonal = max(a.diagonal, b.diagonal, 1.0)
    avg_ratio = float(np.mean(distances) / diagonal)
    max_ratio = float(np.max(distances) / diagonal)

    original_homography = homography_to_original_scale(a, b, homography)
    original_corners = np.float32(
        [
            [0, 0],
            [a.original_width - 1, 0],
            [a.original_width - 1, a.original_height - 1],
            [0, a.original_height - 1],
        ]
    ).reshape(-1, 1, 2)
    warped_original = cv2.perspectiveTransform(original_corners, original_homography).reshape(-1, 2)
    original_reference = original_corners.reshape(-1, 2)
    if a.original_width != b.original_width or a.original_height != b.original_height:
        original_reference = original_reference * np.array(
            [
                b.original_width / max(1.0, float(a.original_width)),
                b.original_height / max(1.0, float(a.original_height)),
            ],
            dtype=np.float32,
        )

    original_distances = np.linalg.norm(warped_original - original_reference, axis=1)
    return (
        avg_ratio,
        max_ratio,
        float(np.mean(original_distances)),
        float(np.max(original_distances)),
    )


def structural_alignment(
    a: ImageFeatures, b: ImageFeatures, homography: "np.ndarray"
) -> Tuple[float, float]:
    size = (b.width, b.height)
    warped_edges = cv2.warpPerspective(a.edges, homography, size, flags=cv2.INTER_NEAREST)
    warped_gradient = cv2.warpPerspective(a.gradient, homography, size, flags=cv2.INTER_LINEAR)

    kernel = np.ones((3, 3), dtype=np.uint8)
    edge_a = cv2.dilate((warped_edges > 0).astype(np.uint8), kernel)
    edge_b = cv2.dilate((b.edges > 0).astype(np.uint8), kernel)

    edge_sum = int(np.count_nonzero(edge_a) + np.count_nonzero(edge_b))
    if edge_sum == 0:
        edge_score = 0.0
    else:
        intersection = int(np.count_nonzero(edge_a & edge_b))
        edge_score = (2.0 * intersection) / float(edge_sum)

    valid = (warped_gradient > 0) | (b.gradient > 0)
    if np.count_nonzero(valid) < 50:
        gradient_score = 0.0
    else:
        left = warped_gradient[valid].astype(np.float32)
        right = b.gradient[valid].astype(np.float32)
        left -= float(np.mean(left))
        right -= float(np.mean(right))
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        gradient_score = 0.0 if denominator <= 1e-8 else float(np.dot(left, right) / denominator)
        gradient_score = max(0.0, min(1.0, gradient_score))

    combined = 0.65 * edge_score + 0.35 * gradient_score
    return float(combined), gradient_score


def ecc_small_motion(a: ImageFeatures, b: ImageFeatures) -> Tuple[Optional["np.ndarray"], str]:
    source = a.gradient.astype(np.float32)
    target = b.gradient.astype(np.float32)
    if source.shape != target.shape:
        source = cv2.resize(source, (b.width, b.height), interpolation=cv2.INTER_AREA)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        cv2.findTransformECC(target, source, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)
    except cv2.error as exc:
        return None, f"ecc_failed:{exc.code}"

    homography = np.eye(3, dtype=np.float32)
    homography[:2, :] = warp
    return homography, "ecc_euclidean"


def confidence_from_scores(
    inliers: int,
    inlier_ratio: float,
    median_error: float,
    avg_corner_shift: float,
    edge_score: float,
    thresholds: Thresholds,
) -> float:
    inlier_component = min(1.0, inliers / max(1.0, thresholds.min_inliers * 2.5))
    ratio_component = min(1.0, inlier_ratio / max(0.01, thresholds.min_inlier_ratio * 2.0))
    error_component = 0.0 if math.isinf(median_error) else max(
        0.0, 1.0 - median_error / max(1.0, thresholds.max_median_reprojection_error * 2.0)
    )
    motion_component = max(0.0, 1.0 - avg_corner_shift / max(0.01, thresholds.max_corner_shift))
    edge_component = min(1.0, edge_score / max(0.01, thresholds.min_edge_score))
    confidence = (
        0.25 * inlier_component
        + 0.20 * ratio_component
        + 0.20 * error_component
        + 0.20 * motion_component
        + 0.15 * edge_component
    )
    return round(float(max(0.0, min(1.0, confidence))), 4)


def evaluate_homography_pair(
    a: ImageFeatures,
    b: ImageFeatures,
    matches: Sequence["cv2.DMatch"],
    homography: "np.ndarray",
    inlier_mask: "np.ndarray",
    median_error: float,
    thresholds: Thresholds,
    method: str,
) -> PairScore:
    good_matches = len(matches)
    inliers = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inliers / max(1.0, float(good_matches))
    avg_shift, max_shift, avg_shift_pixels, max_shift_pixels = corner_shift_metrics(a, b, homography)
    edge_score, gradient_score = structural_alignment(a, b, homography)

    enough_features = (
        good_matches >= thresholds.min_good_matches
        and inliers >= thresholds.min_inliers
        and inlier_ratio >= thresholds.min_inlier_ratio
    )
    motion_ok = avg_shift <= thresholds.max_avg_corner_shift and max_shift <= thresholds.max_corner_shift
    reprojection_ok = median_error <= thresholds.max_median_reprojection_error
    structural_ok = edge_score >= thresholds.min_edge_score
    very_strong_features = (
        inliers >= int(thresholds.min_inliers * 1.8)
        and inlier_ratio >= thresholds.min_inlier_ratio * 1.25
        and median_error <= thresholds.max_median_reprojection_error * 0.75
    )

    if enough_features and motion_ok and reprojection_ok and (structural_ok or very_strong_features):
        decision = "verified"
        reason = "feature_homography_verified"
    elif (
        motion_ok
        and avg_shift <= thresholds.max_tentative_corner_shift
        and edge_score >= thresholds.tentative_edge_score
        and (inliers >= 8 or method.startswith("ecc"))
    ):
        decision = "tentative"
        reason = "structural_alignment_tentative"
    else:
        decision = "reject"
        failed = []
        if not enough_features:
            failed.append("weak_features")
        if not motion_ok:
            failed.append("motion_too_large")
        if not reprojection_ok:
            failed.append("reprojection_error")
        if not structural_ok and not very_strong_features:
            failed.append("weak_structure")
        reason = ",".join(failed) or "thresholds_not_met"

    confidence = confidence_from_scores(
        inliers, inlier_ratio, median_error, avg_shift, edge_score, thresholds
    )

    return PairScore(
        i=a.index,
        j=b.index,
        filename_a=a.filename,
        filename_b=b.filename,
        decision=decision,
        confidence=confidence,
        good_matches=good_matches,
        ransac_inliers=inliers,
        inlier_ratio=round(float(inlier_ratio), 4),
        median_reprojection_error=round(float(median_error), 4)
        if not math.isinf(median_error)
        else float("inf"),
        avg_corner_shift_ratio=round(float(avg_shift), 4),
        max_corner_shift_ratio=round(float(max_shift), 4),
        avg_corner_shift_pixels=round(float(avg_shift_pixels), 4),
        max_corner_shift_pixels=round(float(max_shift_pixels), 4),
        edge_score=round(float(edge_score), 4),
        gradient_score=round(float(gradient_score), 4),
        method=method,
        reason=reason,
    )


def reject_pair(a: ImageFeatures, b: ImageFeatures, reason: str, good_matches: int = 0) -> PairScore:
    return PairScore(
        i=a.index,
        j=b.index,
        filename_a=a.filename,
        filename_b=b.filename,
        decision="reject",
        confidence=0.0,
        good_matches=good_matches,
        ransac_inliers=0,
        inlier_ratio=0.0,
        median_reprojection_error=float("inf"),
        avg_corner_shift_ratio=float("inf"),
        max_corner_shift_ratio=float("inf"),
        avg_corner_shift_pixels=float("inf"),
        max_corner_shift_pixels=float("inf"),
        edge_score=0.0,
        gradient_score=0.0,
        method="none",
        reason=reason,
    )


def _compare_pair_directional(a: ImageFeatures, b: ImageFeatures, thresholds: Thresholds) -> PairScore:
    matches = match_descriptors(a, b, thresholds.ratio)
    homography, inlier_mask, median_error = estimate_homography(a, b, matches)
    if homography is not None:
        score = evaluate_homography_pair(
            a, b, matches, homography, inlier_mask, median_error, thresholds, "sift_homography"
        )
        if score.accepted:
            return score

    # Fallback for very dark/bright bracket frames: try only tiny Euclidean motion
    # on exposure-normalized gradient maps. This cannot create verified matches.
    ecc_homography, method = ecc_small_motion(a, b)
    if ecc_homography is not None:
        empty_mask = np.zeros((len(matches),), dtype=bool)
        fallback_score = evaluate_homography_pair(
            a,
            b,
            matches,
            ecc_homography,
            empty_mask,
            float("inf"),
            thresholds,
            method,
        )
        if fallback_score.decision == "tentative":
            return fallback_score

    if homography is not None:
        return score
    return reject_pair(a, b, f"no_homography,{method}", len(matches))


def compare_pair(a: ImageFeatures, b: ImageFeatures, thresholds: Thresholds) -> PairScore:
    forward = _compare_pair_directional(a, b, thresholds)
    reverse = _compare_pair_directional(b, a, thresholds)

    def rank(score: PairScore) -> Tuple[int, int, float, int, float, float]:
        return (
            1 if score.accepted else 0,
            1 if score.decision == "verified" else 0,
            score.confidence,
            score.ransac_inliers,
            score.inlier_ratio,
            -score.median_reprojection_error,
        )

    return forward if rank(forward) >= rank(reverse) else reverse


PAIR_SCORE_CHECKPOINT_FIELDS = [
    "i",
    "j",
    "filename_a",
    "filename_b",
    "decision",
    "confidence",
    "good_matches",
    "ransac_inliers",
    "inlier_ratio",
    "median_reprojection_error",
    "avg_corner_shift_ratio",
    "max_corner_shift_ratio",
    "avg_corner_shift_pixels",
    "max_corner_shift_pixels",
    "edge_score",
    "gradient_score",
    "method",
    "reason",
]


def pair_score_to_row(score: PairScore) -> Dict[str, object]:
    return {
        "i": score.i,
        "j": score.j,
        "filename_a": score.filename_a,
        "filename_b": score.filename_b,
        "decision": score.decision,
        "confidence": score.confidence,
        "good_matches": score.good_matches,
        "ransac_inliers": score.ransac_inliers,
        "inlier_ratio": score.inlier_ratio,
        "median_reprojection_error": score.median_reprojection_error,
        "avg_corner_shift_ratio": score.avg_corner_shift_ratio,
        "max_corner_shift_ratio": score.max_corner_shift_ratio,
        "avg_corner_shift_pixels": score.avg_corner_shift_pixels,
        "max_corner_shift_pixels": score.max_corner_shift_pixels,
        "edge_score": score.edge_score,
        "gradient_score": score.gradient_score,
        "method": score.method,
        "reason": score.reason,
    }


def pair_score_from_row(row: Dict[str, str]) -> PairScore:
    return PairScore(
        i=int(row["i"]),
        j=int(row["j"]),
        filename_a=row["filename_a"],
        filename_b=row["filename_b"],
        decision=row["decision"],
        confidence=float(row["confidence"]),
        good_matches=int(row["good_matches"]),
        ransac_inliers=int(row["ransac_inliers"]),
        inlier_ratio=float(row["inlier_ratio"]),
        median_reprojection_error=float(row["median_reprojection_error"]),
        avg_corner_shift_ratio=float(row["avg_corner_shift_ratio"]),
        max_corner_shift_ratio=float(row["max_corner_shift_ratio"]),
        avg_corner_shift_pixels=float(row["avg_corner_shift_pixels"]),
        max_corner_shift_pixels=float(row["max_corner_shift_pixels"]),
        edge_score=float(row["edge_score"]),
        gradient_score=float(row["gradient_score"]),
        method=row["method"],
        reason=row["reason"],
    )


def load_pair_score_checkpoint(
    path: Optional[Path], pair_indexes: Sequence[Tuple[int, int]], features: Sequence[ImageFeatures]
) -> Dict[Tuple[int, int], PairScore]:
    if path is None or not path.exists():
        return {}

    expected_pairs = set(pair_indexes)
    scores: Dict[Tuple[int, int], PairScore] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                score = pair_score_from_row(row)
            except (KeyError, TypeError, ValueError):
                continue
            key = tuple(sorted((score.i, score.j)))
            if key not in expected_pairs:
                continue
            if score.filename_a != features[score.i].filename or score.filename_b != features[score.j].filename:
                continue
            scores[key] = score
    return scores


def append_pair_score_checkpoint(path: Optional[Path], scores: Sequence[PairScore]) -> None:
    if path is None or not scores:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_SCORE_CHECKPOINT_FIELDS)
        if write_header:
            writer.writeheader()
        for score in scores:
            writer.writerow(pair_score_to_row(score))
        handle.flush()


def score_pairs(
    features: Sequence[ImageFeatures],
    pair_indexes: Sequence[Tuple[int, int]],
    thresholds: Thresholds,
    jobs: int,
    pair_cache_path: Optional[Path] = None,
) -> List[PairScore]:
    total_pairs = len(pair_indexes)
    if total_pairs == 0:
        return []

    cached_scores = load_pair_score_checkpoint(pair_cache_path, pair_indexes, features)
    missing_pair_indexes = [
        pair_index for pair_index in pair_indexes if pair_index not in cached_scores
    ]
    if cached_scores:
        print(
            f"Pair scoring: loaded {len(cached_scores)} cached score(s); {len(missing_pair_indexes)} remaining.",
            file=sys.stdout,
            flush=True,
        )
    if not missing_pair_indexes:
        return [cached_scores[pair_index] for pair_index in pair_indexes]

    started_at = time.time()
    completed = len(cached_scores)
    render_progress_bar("Pair scoring", completed, total_pairs, started_at=started_at)
    if jobs <= 1 or len(missing_pair_indexes) <= 1:
        for i, j in missing_pair_indexes:
            score = compare_pair(features[i], features[j], thresholds)
            cached_scores[(i, j)] = score
            append_pair_score_checkpoint(pair_cache_path, [score])
            completed += 1
            render_progress_bar("Pair scoring", completed, total_pairs, started_at=started_at)
        return [cached_scores[pair_index] for pair_index in pair_indexes]

    scores: Dict[Tuple[int, int], PairScore] = dict(cached_scores)
    checkpoint_buffer: List[PairScore] = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        future_map = {
            executor.submit(compare_pair, features[i], features[j], thresholds): pair_index
            for pair_index in missing_pair_indexes
            for i, j in [pair_index]
        }
        for future in as_completed(future_map):
            pair_index = future_map[future]
            score = future.result()
            scores[pair_index] = score
            checkpoint_buffer.append(score)
            if len(checkpoint_buffer) >= 100:
                append_pair_score_checkpoint(pair_cache_path, checkpoint_buffer)
                checkpoint_buffer = []
            completed += 1
            render_progress_bar("Pair scoring", completed, total_pairs, started_at=started_at)
    append_pair_score_checkpoint(pair_cache_path, checkpoint_buffer)
    return [scores[pair_index] for pair_index in pair_indexes]


def all_pair_scores(features: Sequence[ImageFeatures], thresholds: Thresholds, jobs: int) -> List[PairScore]:
    pair_indexes = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    return score_pairs(features, pair_indexes, thresholds, jobs)


def candidate_pairs(features: Sequence[ImageFeatures], max_candidates: int) -> List[Tuple[int, int]]:
    if len(features) <= 1:
        return []
    if max_candidates <= 0 or max_candidates >= len(features) - 1:
        return [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]

    matrix = np.vstack([feature.coarse_descriptor for feature in features]).astype(np.float32, copy=False)
    similarity = matrix @ matrix.T
    np.fill_diagonal(similarity, -np.inf)

    neighbors: List[set[int]] = []
    topk = min(max_candidates, len(features) - 1)
    for index in range(len(features)):
        row = similarity[index]
        if topk == len(features) - 1:
            chosen = np.argsort(row)[::-1]
        else:
            partition = np.argpartition(row, -topk)[-topk:]
            chosen = partition[np.argsort(row[partition])[::-1]]
        neighbors.append({int(candidate) for candidate in chosen if np.isfinite(row[candidate])})

    pairs = set()
    for i in range(len(features)):
        for j in neighbors[i]:
            if i == j:
                continue
            pair = (i, j) if i < j else (j, i)
            pairs.add(pair)
        # Also keep a few mutual-near candidates for borderline cases.
        for j in range(i + 1, len(features)):
            if i in neighbors[j]:
                pairs.add((i, j))
    return sorted(pairs)


def selected_pair_scores(
    features: Sequence[ImageFeatures],
    thresholds: Thresholds,
    max_candidates: int,
    disable_pruning: bool,
    jobs: int,
    pair_cache_path: Optional[Path] = None,
) -> List[PairScore]:
    if disable_pruning:
        pair_indexes = [(i, j) for i in range(len(features)) for j in range(i + 1, len(features))]
    else:
        pair_indexes = candidate_pairs(features, max_candidates)
    return score_pairs(features, pair_indexes, thresholds, jobs, pair_cache_path)


def choose_representative(members: Sequence[int], pair_lookup: Dict[Tuple[int, int], PairScore]) -> int:
    if len(members) == 1:
        return members[0]
    best_member = members[0]
    best_score = -1.0
    for member in members:
        confidences = []
        for other in members:
            if member == other:
                continue
            score = pair_lookup.get(tuple(sorted((member, other))))
            if score is not None and score.accepted:
                confidences.append(score.confidence)
        average = sum(confidences) / len(confidences) if confidences else 0.0
        if average > best_score:
            best_score = average
            best_member = member
    return best_member


def member_confidence(
    member: int,
    members: Sequence[int],
    representative: int,
    pair_lookup: Dict[Tuple[int, int], PairScore],
    fallback: float,
) -> float:
    if member == representative:
        return 1.0

    representative_pair = pair_lookup.get(tuple(sorted((member, representative))))
    if representative_pair is not None and representative_pair.accepted:
        return representative_pair.confidence

    confidences = []
    for other in members:
        if member == other:
            continue
        score = pair_lookup.get(tuple(sorted((member, other))))
        if score is not None and score.accepted:
            confidences.append(score.confidence)
    if confidences:
        return max(confidences)
    return fallback


def cluster_images(
    features: Sequence[ImageFeatures], scores: Sequence[PairScore]
) -> Tuple[Dict[int, MemberAssignment], Dict[int, int]]:
    pair_lookup = {tuple(sorted((score.i, score.j))): score for score in scores}
    dsu = DSU(len(features))

    verified = sorted(
        (score for score in scores if score.decision == "verified"),
        key=lambda score: score.confidence,
        reverse=True,
    )
    for score in verified:
        dsu.union(score.i, score.j)

    components: Dict[int, List[int]] = {}
    for idx in range(len(features)):
        components.setdefault(dsu.find(idx), []).append(idx)

    assignments: Dict[int, MemberAssignment] = {}
    for provisional_group, members in enumerate(components.values()):
        for member in members:
            status = "verified" if len(members) > 1 else "singleton"
            assignments[member] = MemberAssignment(provisional_group, 1.0, status)

    representatives = {
        group_id: choose_representative(
            [idx for idx, assignment in assignments.items() if assignment.group_id == group_id], pair_lookup
        )
        for group_id in sorted({assignment.group_id for assignment in assignments.values()})
    }

    # Attach singleton extreme brackets to existing verified groups only when
    # there is a clear best tentative match.
    tentative = sorted(
        (score for score in scores if score.decision == "tentative"),
        key=lambda score: score.confidence,
        reverse=True,
    )
    for score in tentative:
        assign_a = assignments[score.i]
        assign_b = assignments[score.j]
        if assign_a.group_id == assign_b.group_id:
            continue

        group_a_members = [idx for idx, item in assignments.items() if item.group_id == assign_a.group_id]
        group_b_members = [idx for idx, item in assignments.items() if item.group_id == assign_b.group_id]
        a_singleton = len(group_a_members) == 1
        b_singleton = len(group_b_members) == 1
        if a_singleton == b_singleton:
            continue

        singleton = score.i if a_singleton else score.j
        target_group = assign_b.group_id if a_singleton else assign_a.group_id
        target_members = [idx for idx, item in assignments.items() if item.group_id == target_group]

        competing = [
            candidate
            for candidate in tentative
            if singleton in (candidate.i, candidate.j)
            and assignments[candidate.i if candidate.j == singleton else candidate.j].group_id != target_group
        ]
        if competing and competing[0].confidence > score.confidence - 0.05:
            continue

        support = 0
        for member in target_members:
            pair = pair_lookup.get(tuple(sorted((singleton, member))))
            if pair is not None and pair.accepted:
                support += 1
        if support < 1:
            continue

        assignments[singleton] = MemberAssignment(target_group, score.confidence, "tentative")

    # Renumber groups in stable representative-filename order.
    groups: Dict[int, List[int]] = {}
    for idx, assignment in assignments.items():
        groups.setdefault(assignment.group_id, []).append(idx)

    final_representatives = {
        group_id: choose_representative(members, pair_lookup) for group_id, members in groups.items()
    }
    ordered_groups = sorted(
        groups.keys(), key=lambda group_id: features[final_representatives[group_id]].filename
    )
    group_remap = {old_group: new_group + 1 for new_group, old_group in enumerate(ordered_groups)}

    final_assignments: Dict[int, MemberAssignment] = {}
    for group_id, members in groups.items():
        representative = final_representatives[group_id]
        for idx in members:
            assignment = assignments[idx]
            final_assignments[idx] = MemberAssignment(
                group_remap[assignment.group_id],
                member_confidence(idx, members, representative, pair_lookup, assignment.confidence),
                assignment.status,
            )

    final_rep_by_group = {
        group_remap[group_id]: final_representatives[group_id] for group_id in ordered_groups
    }
    return final_assignments, final_rep_by_group


def write_pair_scores(path: Path, scores: Sequence[PairScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "filename_a",
                "filename_b",
                "decision",
                "confidence",
                "good_matches",
                "ransac_inliers",
                "inlier_ratio",
                "median_reprojection_error",
                "avg_corner_shift_ratio",
                "max_corner_shift_ratio",
                "avg_corner_shift_pixels",
                "max_corner_shift_pixels",
                "edge_score",
                "gradient_score",
                "method",
                "reason",
            ],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "filename_a": score.filename_a,
                    "filename_b": score.filename_b,
                    "decision": score.decision,
                    "confidence": f"{score.confidence:.4f}",
                    "good_matches": score.good_matches,
                    "ransac_inliers": score.ransac_inliers,
                    "inlier_ratio": f"{score.inlier_ratio:.4f}",
                    "median_reprojection_error": score.median_reprojection_error,
                    "avg_corner_shift_ratio": score.avg_corner_shift_ratio,
                    "max_corner_shift_ratio": score.max_corner_shift_ratio,
                    "avg_corner_shift_pixels": score.avg_corner_shift_pixels,
                    "max_corner_shift_pixels": score.max_corner_shift_pixels,
                    "edge_score": f"{score.edge_score:.4f}",
                    "gradient_score": f"{score.gradient_score:.4f}",
                    "method": score.method,
                    "reason": score.reason,
                }
            )


def write_groups_csv(
    path: Path,
    features: Sequence[ImageFeatures],
    assignments: Dict[int, MemberAssignment],
    representatives: Dict[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "angle_id",
                "filename",
                "representative_filename",
                "group_size",
                "confidence",
                "status",
            ],
        )
        writer.writeheader()
        group_sizes: Dict[int, int] = {}
        for assignment in assignments.values():
            group_sizes[assignment.group_id] = group_sizes.get(assignment.group_id, 0) + 1

        for feature in sorted(features, key=lambda item: (assignments[item.index].group_id, item.filename)):
            assignment = assignments[feature.index]
            representative = features[representatives[assignment.group_id]]
            confidence = 1.0 if feature.index == representative.index else assignment.confidence
            writer.writerow(
                {
                    "angle_id": f"angle_{assignment.group_id:03d}",
                    "filename": feature.filename,
                    "representative_filename": representative.filename,
                    "group_size": group_sizes[assignment.group_id],
                    "confidence": f"{confidence:.4f}",
                    "status": assignment.status,
                }
            )


def make_thumbnail(path: Path, cell_width: int, cell_height: int) -> "np.ndarray":
    image = resize_max_side(imread_bgr(path), max(cell_width, cell_height))
    height, width = image.shape[:2]
    scale = min(cell_width / width, (cell_height - 28) / height)
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((cell_height, cell_width, 3), 245, dtype=np.uint8)
    y = max(0, (cell_height - 28 - resized.shape[0]) // 2)
    x = max(0, (cell_width - resized.shape[1]) // 2)
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def write_contact_sheets(
    review_dir: Path,
    features: Sequence[ImageFeatures],
    assignments: Dict[int, MemberAssignment],
    representatives: Dict[int, int],
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    groups: Dict[int, List[ImageFeatures]] = {}
    for feature in features:
        groups.setdefault(assignments[feature.index].group_id, []).append(feature)

    status_colors = {
        "verified": (30, 140, 30),
        "tentative": (0, 190, 220),
        "singleton": (130, 130, 130),
    }
    cell_width = 260
    cell_height = 220
    font = cv2.FONT_HERSHEY_SIMPLEX

    for group_id, members in sorted(groups.items()):
        members = sorted(members, key=lambda item: item.filename)
        columns = min(4, max(1, len(members)))
        rows = int(math.ceil(len(members) / columns))
        sheet = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)
        representative = representatives[group_id]

        for offset, feature in enumerate(members):
            row = offset // columns
            col = offset % columns
            thumb = make_thumbnail(feature.path, cell_width, cell_height)
            assignment = assignments[feature.index]
            color = status_colors.get(assignment.status, (80, 80, 80))
            if feature.index == representative:
                color = (200, 80, 20)
            cv2.rectangle(thumb, (0, 0), (cell_width - 1, cell_height - 1), color, 4)
            label = feature.filename[:34]
            if feature.index == representative:
                label = f"* {label}"
            cv2.putText(thumb, label, (8, cell_height - 10), font, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            y0 = row * cell_height
            x0 = col * cell_width
            sheet[y0 : y0 + cell_height, x0 : x0 + cell_width] = thumb

        output = review_dir / f"angle_{group_id:03d}.png"
        cv2.imwrite(str(output), sheet)


def configure_opencv_runtime(jobs: int) -> None:
    if hasattr(cv2, "setNumThreads") and jobs > 1:
        cv2.setNumThreads(1)
    if hasattr(cv2, "ocl"):
        cv2.ocl.setUseOpenCL(False)


def run_grouping(
    paths: Sequence[Path],
    *,
    max_size: int = 1024,
    features_per_variant: int = 2500,
    thresholds: Optional[Thresholds] = None,
    max_candidates: int = 40,
    disable_pruning: bool = False,
    jobs: Optional[int] = None,
    pair_cache_path: Optional[Path] = None,
) -> Tuple[List[ImageFeatures], Dict[int, MemberAssignment], Dict[int, int], List[PairScore]]:
    if not paths:
        raise ValueError("No input images provided.")

    thresholds = thresholds or default_thresholds()
    jobs = default_jobs() if jobs is None else max(1, jobs)
    extract_jobs = min(4, jobs)
    pair_jobs = jobs

    configure_opencv_runtime(jobs)
    print(f"Found {len(paths)} images.", file=sys.stderr)
    print(
        f"Using {extract_jobs} extraction worker(s) and {pair_jobs} pair worker(s).",
        file=sys.stderr,
    )

    try:
        features = extract_all_features(paths, max_size, features_per_variant, extract_jobs)
    except Exception as exc:
        raise RuntimeError(f"Failed while extracting image features: {exc}") from exc

    if len(features) == 1:
        assignments = {0: MemberAssignment(1, 1.0, "singleton")}
        representatives = {1: 0}
        scores: List[PairScore] = []
    else:
        scores = selected_pair_scores(
            features,
            thresholds,
            max_candidates,
            disable_pruning,
            pair_jobs,
            pair_cache_path,
        )
        assignments, representatives = cluster_images(features, scores)

    return features, assignments, representatives, scores


def groups_from_assignments(
    features: Sequence[ImageFeatures], assignments: Dict[int, MemberAssignment]
) -> List[List[str]]:
    groups: Dict[int, List[str]] = {}
    for feature in features:
        groups.setdefault(assignments[feature.index].group_id, []).append(feature.filename)
    return [sorted(groups[group_id]) for group_id in sorted(groups)]


def build_thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        ratio=args.ratio,
        min_good_matches=args.min_good_matches,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_median_reprojection_error=args.max_median_error,
        max_avg_corner_shift=args.max_avg_corner_shift,
        max_corner_shift=args.max_corner_shift,
        min_edge_score=args.min_edge_score,
        tentative_edge_score=args.tentative_edge_score,
        max_tentative_corner_shift=args.max_tentative_corner_shift,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    thresholds = build_thresholds(args)
    paths = list_images(args.input_folder, args.recursive)
    features, assignments, representatives, scores = run_grouping(
        paths,
        max_size=args.max_size,
        features_per_variant=args.features_per_variant,
        thresholds=thresholds,
        max_candidates=args.max_candidates,
        disable_pruning=args.disable_pruning,
        jobs=args.jobs,
    )

    write_groups_csv(args.output, features, assignments, representatives)
    write_pair_scores(args.pair_scores, scores)

    if not args.skip_contact_sheets:
        write_contact_sheets(args.review_dir, features, assignments, representatives)

    group_count = len({assignment.group_id for assignment in assignments.values()})
    verified_pairs = sum(1 for score in scores if score.decision == "verified")
    tentative_pairs = sum(1 for score in scores if score.decision == "tentative")
    print(
        f"Wrote {args.output} with {group_count} groups. "
        f"Verified pairs={verified_pairs}, tentative pairs={tentative_pairs}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
