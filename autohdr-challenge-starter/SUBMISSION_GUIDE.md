# How to Submit — AutoHDR Image Grouping Challenge

## Overview

You build a Docker container that groups images. We run it in the cloud, score the output, and update the leaderboard.

Your container reads images from `/input/images/` and writes a CSV to `/output/predictions.csv`.

## Step-by-Step

### 1. Write Your Solution

Create a `solution.py` that:
- Reads images from `/input/images/`
- Groups them by camera angle
- Writes `predictions.csv` to `/output/`

```python
# predictions.csv format:
# filename,group_id
# IMG_001.jpg,0
# IMG_002.jpg,0
# IMG_003.jpg,1
```

`group_id` can be any value — it just needs to be the same for images in the same group. Order doesn't matter.

### 2. Create a Dockerfile

```dockerfile
FROM python:3.11-slim

# Install your dependencies
RUN pip install --no-cache-dir numpy opencv-python-headless Pillow scipy scikit-learn

WORKDIR /app
COPY solution.py .

CMD ["python", "solution.py"]
```

### 3. Build for Linux/AMD64

**This is critical.** Our cloud runs Linux x86_64. If you're on a Mac (Apple Silicon), you MUST build with the platform flag or your container will crash.

```bash
# Mac (Apple Silicon) — REQUIRED flag:
docker build --platform linux/amd64 -t yourusername/your-solution:v1 .

# Linux or Windows (Intel/AMD) — no flag needed:
docker build -t yourusername/your-solution:v1 .
```

**Common error if you skip `--platform`:**
```
exec python failed: Exec format error
```

### 4. Test Locally

```bash
# Run against sample data
docker run --rm \
  -v /path/to/sample/images:/input/images:ro \
  -v /tmp/output:/output \
  yourusername/your-solution:v1

# Check the output
cat /tmp/output/predictions.csv
```

### 5. Push to Docker Hub

```bash
# Log in (first time only)
docker login

# Push your image
docker push yourusername/your-solution:v1
```

**Make sure the image is PUBLIC** on Docker Hub. Our system pulls it — if it's private, the submission will fail.

To check: go to hub.docker.com → your repository → Settings → Visibility → **Public**.

### 6. Submit on Codabench

Create a `submission.yaml` file:

```yaml
docker_image: yourusername/your-solution:v1
machine_type: cpu-large
email: your-registered-email@example.com
```

ZIP it and upload:

```bash
zip submission.zip submission.yaml
```

Upload `submission.zip` on the Codabench competition page under "My Submissions".

### Machine Types

| Type | vCPU | RAM | Timeout | Best For |
|------|------|-----|---------|----------|
| `cpu-large` | 8 | 16 GB | 30 min | Most solutions |
| `cpu-xlarge` | 16 | 32 GB | 45 min | Heavy compute |

### Container Contract

| | Path | Access |
|---|---|---|
| **Input** | `/input/images/` | Read-only — JPEG/PNG images |
| **Output** | `/output/predictions.csv` | Write your predictions here |
| **Network** | None | No internet access during execution |
| **Filesystem** | Read-only | Only `/output` is writable |

### Quick Checklist

- [ ] `solution.py` reads from `/input/images/` and writes to `/output/predictions.csv`
- [ ] `predictions.csv` has headers: `filename,group_id`
- [ ] Docker image built with `--platform linux/amd64` (if on Mac)
- [ ] Image pushed to Docker Hub and set to **Public**
- [ ] `submission.yaml` has correct `docker_image`, `machine_type`, and `email`
- [ ] Email matches the one you registered with at bounty.autohdr.com
- [ ] Phone number verified on bounty.autohdr.com

### Common Issues

**"Exec format error"** — You built on Mac without `--platform linux/amd64`. Rebuild with the flag.

**"Failed to pull Docker image"** — Image is private or doesn't exist. Check Docker Hub visibility.

**"Email not registered"** — Register at bounty.autohdr.com first with the same email.

**"Phone number not verified"** — Complete the phone verification step after registering.

**"Container timed out"** — Your solution took too long. Optimize or use a larger machine type.

**Score is 0** — Your container didn't write `predictions.csv`, or the CSV format is wrong. Test locally first.
