# Scoring — AutoHDR Image Grouping Challenge

## Metric

```
score = exact_matches / total_groups
```

Your score is the fraction of reference groups you predicted exactly right.

## How It Works

### 1. Your Output

Your container writes `/output/predictions.csv`:

```csv
filename,group_id
living_room_dark.jpg,0
living_room_mid.jpg,0
living_room_bright.jpg,0
kitchen_wide_dark.jpg,1
kitchen_wide_bright.jpg,1
bathroom_dark.jpg,2
bathroom_bright.jpg,2
backyard.jpg,3
```

### 2. What We Compare Against

We have a private answer key with the correct groupings. Each group is a set of filenames that share the same camera angle.

### 3. Exact Match Comparison

Each group (yours and ours) is converted to a **set of filenames**. Order doesn't matter. Group IDs don't matter. Only which filenames are grouped together.

**Your group 0:** `{living_room_dark.jpg, living_room_mid.jpg, living_room_bright.jpg}`
**Reference group 0:** `{living_room_dark.jpg, living_room_mid.jpg, living_room_bright.jpg}`
**→ Exact match**

A group counts as an exact match **only if the sets are identical**. Partial overlaps, missing one file, or having one extra file all count as a miss.

### 4. Examples

**Reference groups (8 total):**
| Group | Files |
|-------|-------|
| 0 | living_room_dark, living_room_mid, living_room_bright |
| 1 | kitchen_wide_dark, kitchen_wide_bright |
| 2 | kitchen_close |
| 3 | master_bed_dark, master_bed_mid, master_bed_bright |
| 4 | master_bed_window_dark, master_bed_window_bright |
| 5 | bathroom_dark, bathroom_bright |
| 6 | backyard |
| 7 | exterior_front_day, exterior_front_dusk |

**Scenario A — Perfect score (1.0):**
You predict exactly these 8 groups. Score = 8/8 = 1.0

**Scenario B — Baseline (each image alone):**
You put every image in its own group (16 groups). The 2 single-image reference groups (kitchen_close, backyard) are matched exactly. Score = 2/8 = 0.25

**Scenario C — Almost right:**
You get 7 groups correct but merge `kitchen_close` into the `kitchen_wide` group. That's wrong for both groups (kitchen_wide now has 3 files instead of 2, and kitchen_close is missing). Score = 6/8 = 0.75

**Scenario D — Over-splitting:**
You split `living_room` into two groups: `{dark, mid}` and `{bright}`. Neither matches the reference group `{dark, mid, bright}`. Score drops by 1.

### 5. What Doesn't Matter

- **Row order** in the CSV — rows can appear in any order
- **Group ID values** — you can use `0, 1, 2` or `a, b, c` or `cat, dog, fish`
- **Group ID order** — group 0 doesn't need to correspond to reference group 0
- **Column order** — as long as `filename` and `group_id` headers are present

### 6. What Does Matter

- **Exact filename match** — `IMG_001.jpg` must match exactly (case-sensitive)
- **Every file must appear** — missing files won't be in any group
- **No partial credit** — a group with 2 out of 3 correct files scores 0 for that group
- **Duplicates** — if a file appears in multiple groups, only the last occurrence counts

### 7. Code

The scoring logic is ~15 lines:

```python
from collections import defaultdict

def load_groups(path):
    """Load CSV into a list of frozensets (unordered groups)."""
    buckets = defaultdict(set)
    for row in csv.DictReader(open(path)):
        buckets[row["group_id"]].add(row["filename"])
    return [frozenset(v) for v in buckets.values()]

reference = set(load_groups("answer_key.csv"))
predicted = set(load_groups("predictions.csv"))

exact_matches = reference & predicted  # set intersection
score = len(exact_matches) / len(reference)
```

The `frozenset` comparison means order is irrelevant — only the membership of each group matters.
