# AGENTS.md

## Algorithm Overview

This project groups real estate photos by visual geometry rather than filenames,
metadata, or ordering. The practical question for each candidate pair is whether
image A can be mapped onto image B as the same camera viewpoint, allowing for
exposure differences, small shifts, crops, and bracketed HDR captures.

There are two main layers:
- `group_camera_angles.py` is the conservative core matcher.
- `autohdr-challenge-starter/solution.py` is the competition wrapper that runs
  either the high-quality core matcher plus targeted repairs, or the large
  memory-bounded path for the private 66K-image run.

Core pipeline:
1. Load and resize images, currently with a default max side of `1024`.
2. Normalize exposure/contrast so HDR bracket differences are less dominant.
3. Extract grayscale/normalized images, edges, gradient maps, coarse visual
   descriptors, and SIFT/RootSIFT keypoints/descriptors.
4. Pick candidate pairs from coarse descriptor similarity instead of scoring all
   possible pairs.
5. Score candidate pairs geometrically with bidirectional feature matching.
6. Cluster verified pairs with DSU/union-find.
7. Attach only safe tentative singleton brackets.
8. Return base groups, leaving uncertain images as singletons.

The core matcher is intentionally conservative: a false split is usually easier
to recover than a false merge, and the competition score requires exact filename
sets. Pair scoring matches RootSIFT descriptors with Lowe ratio filtering,
estimates a homography with RANSAC, and checks good match count, inlier count,
inlier ratio, reprojection error, corner shift, and edge/gradient agreement. It
also has an ECC small-motion fallback for difficult exposure brackets. Pair
decisions are effectively `verified`, `tentative`, or reject.

Pair scoring is bidirectional. The matcher evaluates both `a -> b` and `b -> a`
and keeps the stronger result because one-way KNN descriptor matching is
order-sensitive.

Candidate pruning is the main speed/accuracy tradeoff. The small/medium
competition path currently uses `AUTOHDR_MAX_CANDIDATES=12`, which keeps
runtime manageable by scoring each image against its nearest coarse visual
neighbors instead of all pairs.

On small/medium datasets, the challenge post-pass runs after base grouping and
only targets known under-grouping shapes. It tries singleton-to-group recovery
with relaxed but corroborated geometry, singleton-to-singleton recovery with
stricter gates, low-feature tiny-motion support, two-image tail recovery, and
medium-fragment recovery for balanced split groups. The post-pass now batches
missing pair-score requests through `prefetch_pair_scores()` so expensive
recovery checks can use parallel workers instead of serial `cached_pair_score()`
calls.

On large datasets, `solution.py` switches to a memory-bounded hybrid path. It
extracts compact grayscale descriptors, finds candidate neighbors blockwise
instead of materializing an `N x N` matrix, creates safe compact groups at a
high threshold, then SIFT-checks borderline compact edges before merging groups.
The older bounded-component refinement path remains available behind env vars
but is no longer the default because public proxy runs showed it could over-split
or damage compact groups.

Final output is `/output/predictions.csv` with header `filename,group_id`. Group
IDs are arbitrary; exact filename sets are what matter for scoring.

## Current Handoff

Repo:
- `/home/dferris/camera_angle_grouping`

Challenge starter entrypoint:
- `autohdr-challenge-starter/solution.py`

Core matcher:
- `group_camera_angles.py`

Packaged matcher copy for Docker:
- `autohdr-challenge-starter/group_camera_angles.py`

## Competition Contract

Official local docs:
- `autohdr-challenge-starter/README.md`
- `autohdr-challenge-starter/SCORING.md`
- `autohdr-challenge-starter/SUBMISSION_GUIDE.md`
- `autohdr-challenge-starter/submission.yaml`

Container contract:
- read images from `/input/images`
- write `/output/predictions.csv`
- CSV header must be `filename,group_id`
- group IDs can be arbitrary
- score is exact set matches only: `exact_matches / total_reference_groups`
- no internet during execution
- only `/output` is writable by default
- Docker image must be Linux/AMD64

Current Docker packaging status:
- `autohdr-challenge-starter/Dockerfile` now copies both `solution.py` and `group_camera_angles.py`
- `autohdr-challenge-starter/group_camera_angles.py` was copied from root `group_camera_angles.py`; re-copy it after editing root matcher
- Docker image `autohdr-camera-angle-grouping:latest` was built and smoke-tested; see Current Build State.

## Dataset State

Medium/public dataset:
- source zip: `data/autohdr_medium_5000.zip`
- images: `data/images`
- manifest: `data/public_manifest.csv`
- image count: `2126`
- truth groups: `538`

Sample dataset:
- images: `data/autohdr_sample_500/images`
- manifest: `data/autohdr_sample_500/public_manifest.csv`
- image count in this unpack: `366`
- truth groups: `69`

## Important Runtime/Checkpoint Notes

Pair scoring checkpoint support was added to `group_camera_angles.py`.

Relevant functions:
- `load_pair_score_checkpoint()`
- `append_pair_score_checkpoint()`
- `score_pairs(..., pair_cache_path=...)`
- `run_grouping(..., pair_cache_path=...)`

Challenge wrapper env vars:
- `AUTOHDR_INPUT_DIR`, default `/input/images`
- `AUTOHDR_OUTPUT_DIR`, default `/output`
- `AUTOHDR_PAIR_CACHE_PATH`, default unset; do not set for competition unless it points under `/output`
- `AUTOHDR_MAX_CANDIDATES`, default currently `12`
- `AUTOHDR_MAX_SIZE`, default `1024`
- `AUTOHDR_FEATURES_PER_VARIANT`, default currently `2500`
- `AUTOHDR_LARGE_DATASET_THRESHOLD`, default `5000`

Current small/medium competition defaults in `autohdr-challenge-starter/solution.py`:
- `MAX_CANDIDATES=12`
- `FEATURES_PER_VARIANT=2500`
- jobs: `min(16, os.cpu_count())`

Why candidate default is 12:
- old 40-candidate path was too slow
- 366-image sample with 12 candidates produced `3678` base pair scores instead of `12958`
- sample fresh runtime around 21 minutes including threaded post-pass

## Large Dataset Path

Why this exists:
- first private run had `66462` images and exhausted a 16 GB container
- current submission targets `cpu-xlarge`: 16 vCPU, 32 GB RAM, 45 min timeout
- the large path avoids full pairwise matrices and bounds SIFT refinement work

Switching behavior:
- if image count is at least `AUTOHDR_LARGE_DATASET_THRESHOLD=5000`,
  `solution.py` uses `group_images_large()`
- below that threshold, it uses the original `run_grouping()` plus
  `challenge_merge_groups()` path

Large path defaults:
- compact descriptor max side: `AUTOHDR_LARGE_MAX_SIZE=256`
- compact neighbor count: `AUTOHDR_LARGE_NEIGHBORS=192`
- compact-only merge threshold: `AUTOHDR_LARGE_MIN_SIMILARITY=0.90`
- aspect-ratio gate: `AUTOHDR_LARGE_MAX_ASPECT_DELTA=0.03`
- geometry bridge enabled: `AUTOHDR_LARGE_GEOMETRY_BRIDGE=1`
- bridge lower similarity bound: `AUTOHDR_LARGE_BRIDGE_MIN_SIMILARITY=0.84`
- bridge max checked pairs: `AUTOHDR_LARGE_BRIDGE_MAX_PAIRS=20000`
- bridge time budget: `AUTOHDR_LARGE_BRIDGE_TIME_BUDGET=900`
- bridge batch size: `AUTOHDR_LARGE_BRIDGE_BATCH_SIZE=512`
- bridge edges per compact-group pair: `AUTOHDR_LARGE_BRIDGE_EDGES_PER_COMPONENT_PAIR=1`
- bridge accepts strict high-inlier rejected pairs: `AUTOHDR_LARGE_BRIDGE_ACCEPT_STRONG_REJECT=0`
- hybrid mode enabled: `AUTOHDR_LARGE_HYBRID=1`
- super-component threshold: `AUTOHDR_LARGE_SUPER_MIN_SIMILARITY=0.84`
- max compact component size before splitting/fallback: `AUTOHDR_LARGE_MAX_COMPONENT_SIZE=64`
- reduced geometry component max side: `AUTOHDR_LARGE_COMPONENT_MAX_SIZE=768`
- reduced geometry features per variant: `AUTOHDR_LARGE_COMPONENT_FEATURES=900`
- reduced geometry candidates: `AUTOHDR_LARGE_COMPONENT_CANDIDATES=16`
- component post-pass disabled: `AUTOHDR_LARGE_COMPONENT_POSTPASS=0`
- hybrid time budget: `AUTOHDR_LARGE_HYBRID_TIME_BUDGET=2400`
- blockwise candidate search enabled: `AUTOHDR_LARGE_BLOCKWISE_CANDIDATES=1`
- block size: `AUTOHDR_LARGE_BLOCK_SIZE=512`
- blockwise candidate min similarity: `AUTOHDR_LARGE_CANDIDATE_MIN_SIMILARITY=0.45`

Large path behavior:
- extracts compact features for all images with 16 workers
- finds exact top compact neighbors block by block, not by allocating a full
  all-pairs similarity matrix
- unions high-similarity pairs into compact groups at `0.90`
- SIFT-checks one representative edge for each borderline compact group pair
  between `0.84` and `0.90`
- extracts SIFT features only for the current bridge batch, keeping bridge
  memory bounded under the 32 GB target
- merges compact groups only for geometrically verified bridge pairs
- if `AUTOHDR_LARGE_GEOMETRY_BRIDGE=0`, the older bounded-component hybrid path
  can still run

## Core Code Changes

### Bidirectional Pair Scoring

File:
- `group_camera_angles.py`

What changed:
- added `_compare_pair_directional()`
- changed `compare_pair()` to score both `a -> b` and `b -> a`
- `compare_pair()` keeps the stronger directional result

Why:
- one-way KNN ratio matching made pair scoring direction-sensitive

### Pair-Score Checkpointing

File:
- `group_camera_angles.py`

What changed:
- pair scores can be loaded/appended as CSV checkpoints
- `run_grouping()` accepts `pair_cache_path`

Why:
- full/medium runs are expensive and must be resumable locally

### Challenge Post-Pass Current Shape

File:
- `autohdr-challenge-starter/solution.py`

Current post-pass behavior:
- base matcher remains conservative
- singleton recovery:
  - singleton-to-group via `evaluate_relaxed_group_match()`
  - singleton-to-singleton all-pair scan filtered by coarse affinity
  - `singleton_pair_support()`
  - `low_feature_tiny_motion_support()`
- two-image tail recovery:
  - `evaluate_verified_tail_match()`
  - one member has direct relaxed support into larger group
  - other member is strongly verified internally to that member
- medium fragment recovery:
  - `evaluate_medium_fragment_match()`
  - intended for split groups like `11393` and `84850`
  - currently allows strong non-reciprocal medium proposals and a follow-up medium pass after initial DSU merges
- post-pass missing pair scores are collected by `collect_challenge_pair_requests()` and scored in parallel by `prefetch_pair_scores()`

Important: post-pass used to be effectively single-threaded through `cached_pair_score()`. It is now batched and threaded.

## Sample 500 Validation

Best current sample result with candidate-12 defaults:
- output: `results/autohdr_sample_500_candidate12_output/predictions.csv`
- score: `results/autohdr_sample_500_candidate12_score.txt`
- predicted groups: `69`
- truth groups: `69`
- exact matches: `69`
- score: `1.000000`

This run used:
- `AUTOHDR_MAX_CANDIDATES=12` default
- `AUTOHDR_FEATURES_PER_VARIANT=2500` default
- threaded post-pass
- fresh checkpoint: `results/autohdr_sample_500_candidate12_pair_scores_checkpoint.csv`

Do not use the `conform_threaded` run as a quality reference:
- it used `FEATURES_PER_VARIANT=1800`
- score was only `66/69 = 0.956522`
- it falsely merged truth groups `34011` and `56036`

Large-path sample validation:
- compact-only forced sample:
  - output: `results/autohdr_sample_500_large_fallback_output2/predictions.csv`
  - score: `62/69 = 0.898551`
- hybrid forced sample before blockwise:
  - output: `results/autohdr_sample_500_large_hybrid_output/predictions.csv`
  - score: `66/69 = 0.956522`
- blockwise hybrid forced sample with 120s budget:
  - output: `results/autohdr_sample_500_blockwise_120s_output/predictions.csv`
  - predicted groups: `72`
  - exact matches: `66/69`
  - score: `0.956522`
- geometry bridge forced sample:
  - output: `results/autohdr_sample_500_large_bridge_output/predictions.csv`
  - predicted groups: `74`
  - exact matches: `65/69`
  - score: `0.942029`

## Medium Known-Case Targeted Validation

Known problem groups included in targeted subset:
- bad merge probes: `16919`, `29773`, `747`, `92151`, `88910`, `91771`
- split probes: `11393`, `84850`, `40615`, `29734`, `28778`, `58773`

Subset files:
- images symlink dir: `results/medium_known_subset/images`
- manifest: `results/medium_known_subset/public_manifest.csv`
- total: `74` images, `12` truth groups

Current best completed targeted run before last patch:
- output: `results/medium_known_subset/output_v4/predictions.csv`
- score: `results/medium_known_subset/score_v4.txt`
- predicted groups: `13`
- truth groups: `12`
- exact matches: `11`
- score: `0.916667`
- remaining miss: truth group `11393` split `15 -> 9 + 6`
- no bad merges reported in this subset

Earlier targeted progression:
- initial current code: `9/12`, missed `11393`, `84850`, `29734`
- after medium pass: `10/12`, missed `11393`, `29734`
- after relaxing low-feature median and medium proposals: `11/12`, missed only `11393`

## Current Build State

The current upload target was reverted to the safer v3 bridge after the v4
expanded-bridge/384-size full-corpus test appeared too merge-heavy.

Build command used from `autohdr-challenge-starter`:
- `docker build --platform linux/amd64 -t autohdr-camera-angle-grouping:latest .`

Built image:
- tags: `autohdr-camera-angle-grouping:latest`, `dferris93/autohdr-solution:v3`
- image id / manifest list sha: `sha256:7f27771b334d20ff692da137b08f0ff6a2b3a93f79af7c0e36019278e27dc5c7`
- platform from manifest inspect: `linux/amd64`
- linux/amd64 image manifest: `sha256:2c1e20b955c9509eaf538ea6487b73d7204d83130d67299c011ffbabf5e8e3c4`

Pre-build checks completed:
- `. .venv/bin/activate && python -m py_compile autohdr-challenge-starter/solution.py group_camera_angles.py autohdr-challenge-starter/group_camera_angles.py`
- `cp group_camera_angles.py autohdr-challenge-starter/group_camera_angles.py`

Docker smoke test:
- created real copied input files in `results/docker_smoke/images`
- ran:
  - `docker run --rm --memory=32g --cpus=16 -v /home/dferris/camera_angle_grouping/results/docker_smoke/images:/input/images:ro -v /home/dferris/camera_angle_grouping/results/docker_smoke/output_v4:/output autohdr-camera-angle-grouping:latest`
- result: passed
- wrote `results/docker_smoke/output_v4/predictions.csv`
- smoke output had correct `filename,group_id` header and 6 image rows

Important caveat:
- The v3 image includes the safer large-dataset geometry bridge path.
- It was compiled and Docker-smoke-tested.
- It was NOT run on the full private 66K-image dataset locally.
- Last scored normal small-path state was:
  - sample 500 candidate-12: `69/69 = 1.000000`
  - medium known subset: `11/12 = 0.916667`, only `11393` split `15 -> 9 + 6`
- Large path validation:
  - forced public medium compact-only: `502/538 = 0.933086`
  - forced public medium bridge: `510/538 = 0.947955`
  - rejected v4 32 GB Docker public medium bridge: `515/538 = 0.957249`
  - rejected v4 384-size full corpus, scored by inferred filename prefix: `0.798138`
  - private v2 score before bridge: `88.54%`

## Current Submission State

The submission is live on Codabench.

Docker Hub image pushed:
- `dferris93/autohdr-solution:v3`

Docker Hub digest reported by push:
- `sha256:7f27771b334d20ff692da137b08f0ff6a2b3a93f79af7c0e36019278e27dc5c7`

Public-read verification:
- unauthenticated manifest inspect succeeded with an empty temporary Docker config
- manifest includes `linux/amd64`

Submission files:
- `autohdr-challenge-starter/submission.yaml`
- `autohdr-challenge-starter/submission.zip`
- root copy: `submission.zip`
- both submission zips were regenerated after updating `submission.yaml`

Final `submission.yaml` values:
- `docker_image: dferris93/autohdr-solution:v3`
- `machine_type: cpu-xlarge`
- `email: dan@usrsbin.com`

Prediction/scoring behavior:
- the container prediction step writes `/output/predictions.csv`
- stdout includes:
  - `Loaded N images from /input/images`
  - `Predicted N groups`
  - `Wrote N predictions to /output/predictions.csv`
- the container does not print the private score because it has no private answer key
- Codabench's separate scoring step is expected to compute and display the actual score from `predictions.csv`

Security note:
- Docker login was completed through the Docker web flow.
- Docker credentials were stored unencrypted in `/home/dferris/.docker/config.json`.
- Rotate/remove the Docker Hub credential after the submission is accepted.

## Full Medium Run Status

No complete full public medium run exists with the current v3 bridge code.

Artifacts that do NOT indicate a complete full run:
- `results/autohdr_medium_5000_challenge_output_v2/` is empty/incomplete
- `results/autohdr_medium_5000_run_v2.log` is from a killed stale run

Old baseline full public score:
- predicted groups: `544`
- exact matches: `523`
- score: `523 / 538 = 0.972119`

Forced large-path public medium experiments:
- compact-only fallback:
  - output: `results/autohdr_medium_5000_large_fallback_output/predictions.csv`
  - score: `502/538 = 0.933086`
- 300s component-hybrid budget:
  - output: `results/autohdr_medium_5000_large_budget300_output/predictions.csv`
  - score: `502/538 = 0.933086`
- geometry bridge:
  - output: `results/autohdr_medium_5000_large_bridge_output/predictions.csv`
  - score: `510/538 = 0.947955`
- 32 GB Docker geometry bridge, v4 defaults except forced large threshold:
  - output: `results/docker_medium_32gb_bridge082_output/predictions.csv`
  - score: `515/538 = 0.957249`
  - rejected for upload after full-corpus inferred-prefix scoring showed over-merging
- 32 GB Docker geometry bridge at `AUTOHDR_LARGE_BRIDGE_MIN_SIMILARITY=0.80`:
  - output: `results/docker_medium_32gb_bridge080_output/predictions.csv`
  - score: `515/538 = 0.957249`
  - slower than `0.82`; it hit the old 900s bridge budget after `6144/9523` checked edges
- 32 GB Docker full corpus with v4 plus `AUTOHDR_LARGE_MAX_SIZE=384`:
  - output: `results/docker_all_65k_v4_size384_output/predictions.csv`
  - inferred-prefix score: `14835/18587 = 0.798138`
  - predicted groups: `15942`; inferred truth groups: `18587`
  - diagnosis: too many false merges, do not upload this variant
- reduced SIFT prototype:
  - score: `513/538 = 0.9535315985`
  - informative but too slow as a direct full approach

Current private-run target:
- fit within 32 GB RAM and the `cpu-xlarge` timeout
- exact match rate around the current top-10 threshold is the practical goal

## Process/Tooling Notes

Some long `exec_command` runs were started without a TTY, so Ctrl-C via `write_stdin` may fail.

If a stale process is stuck:
- use a narrow `pkill -f` with escalated permissions
- avoid killing unrelated Python processes

For htop:
- base pair scoring uses `ThreadPoolExecutor`
- post-pass now has threaded prefetch, but some later evaluation is CPU-light and may look single-threaded
- enable thread display in htop with `H` if needed
