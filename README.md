# Camera Angle Grouping

This repository groups real estate photos by camera viewpoint. It is built for
AutoHDR-style image sets where filenames, ordering, and metadata are not useful,
and where the same camera angle may appear as multiple exposure brackets.

The matcher is deliberately conservative: it prefers leaving uncertain photos as
singletons over merging two different viewpoints. That matters because the
challenge score only gives credit for exact filename sets.

## How It Works

There are two entrypoints:

- `group_camera_angles.py` is the reusable geometry-first matcher.
- `autohdr-challenge-starter/solution.py` is the AutoHDR competition wrapper. It
  calls the core matcher for small and medium inputs, applies targeted repair
  passes for common split cases, and switches to a memory-bounded large-dataset
  path for challenge-scale runs.

The core pipeline is:

1. Load JPEG images and resize them to a working max side, defaulting to `1024`.
2. Normalize exposure and contrast so HDR brackets compare more by structure
   than brightness.
3. Extract grayscale images, normalized images, edge maps, gradient maps, coarse
   descriptors, and SIFT/RootSIFT features.
4. Use coarse descriptor similarity to select candidate neighbors instead of
   scoring every possible image pair.
5. Score candidate pairs with bidirectional feature matching. Each pair is
   checked both ways because one-way KNN matching can be order-sensitive.
6. Estimate a homography with RANSAC and gate matches by good match count,
   inlier count, inlier ratio, reprojection error, warped-corner shift, and
   edge/gradient agreement.
7. Cluster verified and safe tentative pairs with union-find.
8. Leave unresolved images as singleton groups.

For large inputs, `solution.py` avoids building an all-pairs similarity matrix.
It extracts compact descriptors, finds neighbors blockwise, forms high-confidence
compact groups, and optionally checks borderline compact edges with bounded SIFT
geometry batches.

## Install Without Docker

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The local requirements use OpenCV and NumPy. The Docker image installs
`opencv-python-headless`, which is better suited for containers.

## Run The Core Matcher

Use this when you want diagnostic output and contact sheets for a folder of
JPEGs:

```bash
. .venv/bin/activate
python group_camera_angles.py /path/to/images --output angle_groups.csv
```

Useful options:

```bash
python group_camera_angles.py /path/to/images \
  --output angle_groups.csv \
  --pair-scores pair_scores.csv \
  --review-dir angle_group_review \
  --max-size 1024 \
  --features-per-variant 2500 \
  --max-candidates 40 \
  --jobs 12
```

Outputs:

- `angle_groups.csv`: filename-to-group assignments for local review.
- `pair_scores.csv`: pairwise geometry diagnostics.
- `angle_group_review/`: contact-sheet PNGs, unless `--skip-contact-sheets` is
  passed.

## Run The Challenge Wrapper Without Docker

The challenge wrapper writes the competition-format CSV:

```csv
filename,group_id
```

Run it locally by setting the same paths the container would receive:

```bash
. .venv/bin/activate
mkdir -p results/local_output

AUTOHDR_INPUT_DIR=/path/to/images \
AUTOHDR_OUTPUT_DIR=results/local_output \
python autohdr-challenge-starter/solution.py
```

The result is:

```bash
results/local_output/predictions.csv
```

If you have a public manifest, score the predictions locally:

```bash
python score_challenge_predictions.py \
  results/local_output/predictions.csv \
  /path/to/public_manifest.csv
```

The included medium-data helper script expects images at `data/images` and the
manifest at `data/public_manifest.csv`:

```bash
./run_autohdr_medium_dataset.sh
```

## Run With Docker

Build the competition container from the challenge starter directory:

```bash
cd autohdr-challenge-starter
docker build --platform linux/amd64 -t autohdr-camera-angle-grouping:latest .
```

The `--platform linux/amd64` flag matches the competition runtime and is
especially important when building on non-amd64 machines.

Run the container against local images:

```bash
mkdir -p /tmp/autohdr-output

docker run --rm \
  --memory=32g \
  --cpus=16 \
  -v /path/to/images:/input/images:ro \
  -v /tmp/autohdr-output:/output \
  autohdr-camera-angle-grouping:latest
```

The container reads:

```text
/input/images
```

and writes:

```text
/output/predictions.csv
```

Score Docker output locally, if a manifest is available:

```bash
python score_challenge_predictions.py \
  /tmp/autohdr-output/predictions.csv \
  /path/to/public_manifest.csv
```

## Configuration

The challenge wrapper is configured with environment variables.

Common small/medium settings:

- `AUTOHDR_INPUT_DIR`: image directory, default `/input/images`.
- `AUTOHDR_OUTPUT_DIR`: output directory, default `/output`.
- `AUTOHDR_PAIR_CACHE_PATH`: optional pair-score checkpoint CSV. For competition
  containers, keep this under `/output` if enabled.
- `AUTOHDR_MAX_CANDIDATES`: coarse neighbors per image, default `12`.
- `AUTOHDR_MAX_SIZE`: working max side for the core matcher, default `1024`.
- `AUTOHDR_FEATURES_PER_VARIANT`: SIFT features per normalized variant, default
  `2500`.
- `AUTOHDR_LARGE_DATASET_THRESHOLD`: image count at which large mode is used,
  default `5000`.

Large-dataset settings:

- `AUTOHDR_LARGE_MAX_SIZE`: compact descriptor max side, default `256`.
- `AUTOHDR_LARGE_NEIGHBORS`: compact neighbors per image, default `192`.
- `AUTOHDR_LARGE_MIN_SIMILARITY`: compact-only merge threshold, default `0.90`.
- `AUTOHDR_LARGE_GEOMETRY_BRIDGE`: enable bounded SIFT bridge checks, default
  `1`.
- `AUTOHDR_LARGE_BRIDGE_MIN_SIMILARITY`: lower similarity bound for bridge
  checks, default `0.84`.
- `AUTOHDR_LARGE_BRIDGE_MAX_PAIRS`: max bridge pairs to check, default `20000`.
- `AUTOHDR_LARGE_BRIDGE_TIME_BUDGET`: bridge time budget in seconds, default
  `900`.

Example with a resumable pair-score cache:

```bash
AUTOHDR_INPUT_DIR=data/images \
AUTOHDR_OUTPUT_DIR=results/medium_output \
AUTOHDR_PAIR_CACHE_PATH=results/medium_pair_scores_checkpoint.csv \
AUTOHDR_MAX_CANDIDATES=12 \
python autohdr-challenge-starter/solution.py
```

## Repository Layout

- `group_camera_angles.py`: core image matcher and local CLI.
- `autohdr-challenge-starter/solution.py`: competition entrypoint.
- `autohdr-challenge-starter/Dockerfile`: Docker build used for submissions.
- `autohdr-challenge-starter/group_camera_angles.py`: packaged copy of the core
  matcher for Docker builds.
- `score_challenge_predictions.py`: exact-set scorer for public manifests.
- `run_autohdr_medium_dataset.sh`: local helper for the medium public dataset.
- `requirements.txt`: local Python dependencies.

When changing `group_camera_angles.py`, copy it into the challenge starter before
building Docker:

```bash
cp group_camera_angles.py autohdr-challenge-starter/group_camera_angles.py
```

## Competition Notes

The AutoHDR container contract is:

- Read images from `/input/images`.
- Write `/output/predictions.csv`.
- Use CSV header `filename,group_id`.
- Group IDs may be arbitrary.
- The score is `exact_matches / total_reference_groups`, where a predicted group
  only counts if its filename set exactly matches a truth group.

The container has no internet access during competition execution, so all code
and dependencies must be present in the built image.
