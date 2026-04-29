#!/usr/bin/env python3
"""Build a tiny static HTML/JS browser for review results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from build_review_navigator import GroupInfo, NeighborInfo, load_groups, load_neighbors, score_group


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static browser frontend for review results.")
    parser.add_argument("--groups", type=Path, required=True, help="Path to angle groups CSV.")
    parser.add_argument("--pair-scores", type=Path, required=True, help="Path to pair scores CSV.")
    parser.add_argument("--grouped-dir", type=Path, required=True, help="Path to grouped folder view.")
    parser.add_argument("--review-dir", type=Path, required=True, help="Path to contact-sheet directory.")
    parser.add_argument("--navigator-dir", type=Path, required=True, help="Path to review navigator output.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for static site assets.")
    return parser.parse_args()


def relative(from_dir: Path, target: Path) -> str:
    return os.path.relpath(target, from_dir).replace(os.sep, "/")


def shoot_folder_map(navigator_dir: Path) -> Dict[str, Path]:
    by_shoot = navigator_dir / "by_shoot"
    mapping: Dict[str, Path] = {}
    for path in by_shoot.iterdir():
        if path.is_dir() and "__" in path.name:
            key = path.name.split("__", 1)[0]
            mapping[key] = path
    return mapping


def build_data(
    groups: Dict[str, GroupInfo],
    neighbors_by_group: Dict[str, List[NeighborInfo]],
    site_dir: Path,
    review_dir: Path,
    navigator_dir: Path,
) -> dict:
    shoot_groups: Dict[str, List[GroupInfo]] = {}
    for group in groups.values():
        shoot_groups.setdefault(group.primary_shoot, []).append(group)
    for values in shoot_groups.values():
        values.sort(key=lambda item: item.angle_id)

    shoot_folders = shoot_folder_map(navigator_dir)

    group_cards = []
    for group in groups.values():
        same_shoot_other_groups = [
            candidate
            for candidate in shoot_groups.get(group.primary_shoot, [])
            if candidate.angle_id != group.angle_id
        ]
        review_score, reasons, compare_neighbors = score_group(
            group,
            neighbors_by_group.get(group.angle_id, []),
            same_shoot_other_groups,
        )
        tags = []
        if group.status == "singleton":
            tags.append("singleton")
        if group.min_confidence < 0.97:
            tags.append("low_confidence")
        if group.size >= 10:
            tags.append("large_group")
        if same_shoot_other_groups:
            tags.append("same_shoot_multi_group")

        compare_cards = []
        for neighbor in compare_neighbors:
            other = groups[neighbor.other_group]
            compare_cards.append(
                {
                    "angleId": other.angle_id,
                    "status": other.status,
                    "size": other.size,
                    "minConfidence": round(other.min_confidence, 4),
                    "reviewImage": relative(site_dir, review_dir / f"{other.angle_id}.png"),
                    "groupFolder": relative(site_dir, other.folder),
                    "maxRejectConfidence": round(neighbor.max_confidence, 4),
                    "pairCount": neighbor.count,
                    "reason": neighbor.top_reason,
                }
            )

        shoot_folder = shoot_folders.get(group.primary_shoot)
        group_cards.append(
            {
                "angleId": group.angle_id,
                "shootKey": group.primary_shoot,
                "status": group.status,
                "size": group.size,
                "minConfidence": round(group.min_confidence, 4),
                "maxConfidence": round(group.max_confidence, 4),
                "reviewScore": round(review_score, 2),
                "representativeFilename": group.representative_filename,
                "reasons": reasons,
                "tags": tags,
                "reviewImage": relative(site_dir, review_dir / f"{group.angle_id}.png"),
                "groupFolder": relative(site_dir, group.folder),
                "shootFolder": relative(site_dir, shoot_folder) if shoot_folder else "",
                "compareGroups": compare_cards,
                "filenames": group.filenames,
            }
        )

    group_cards.sort(key=lambda item: (-item["reviewScore"], item["angleId"]))

    shoots = []
    for shoot_key, values in sorted(shoot_groups.items()):
        shoots.append(
            {
                "shootKey": shoot_key,
                "groupCount": len(values),
                "singletonCount": sum(1 for group in values if group.status == "singleton"),
                "angleIds": [group.angle_id for group in values],
                "folder": relative(site_dir, shoot_folders[shoot_key]) if shoot_key in shoot_folders else "",
            }
        )

    return {
        "groups": group_cards,
        "shoots": shoots,
        "generatedFrom": {
            "groupsCsv": str(args.groups),
            "pairScoresCsv": str(args.pair_scores),
        },
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Angle Review</title>
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div>
        <h1>Camera Angle Review</h1>
        <p>Ranked review queue, shoot browser, and quick compare links.</p>
      </div>
      <div class="stats" id="stats"></div>
    </header>

    <section class="controls">
      <label>
        Search
        <input id="searchInput" type="search" placeholder="angle_024, g22994, DSC_9060">
      </label>
      <label>
        Filter
        <select id="filterSelect">
          <option value="all">All groups</option>
          <option value="queue">Review queue only</option>
          <option value="singleton">Singletons</option>
          <option value="low_confidence">Low confidence</option>
          <option value="large_group">Large groups</option>
          <option value="same_shoot_multi_group">Same-shoot multi-group</option>
        </select>
      </label>
      <label>
        Sort
        <select id="sortSelect">
          <option value="review">Review score</option>
          <option value="confidence">Lowest confidence first</option>
          <option value="size">Largest groups first</option>
          <option value="angle">Angle ID</option>
          <option value="shoot">Shoot key</option>
        </select>
      </label>
      <label>
        Shoot
        <select id="shootSelect">
          <option value="">All shoots</option>
        </select>
      </label>
    </section>

    <main class="layout">
      <aside class="listPanel">
        <div class="listHeader">
          <strong id="resultCount"></strong>
        </div>
        <div id="groupList" class="groupList"></div>
      </aside>
      <section class="detailPanel" id="detailPanel">
        <div class="emptyState">Pick a group and I’ll stop making you guess.</div>
      </section>
    </main>
  </div>

  <script src="./data.js"></script>
  <script src="./app.js"></script>
</body>
</html>
"""


APP_JS = """const state = {
  search: '',
  filter: 'all',
  sort: 'review',
  shoot: '',
  selectedAngleId: null,
};

const elements = {
  searchInput: document.getElementById('searchInput'),
  filterSelect: document.getElementById('filterSelect'),
  sortSelect: document.getElementById('sortSelect'),
  shootSelect: document.getElementById('shootSelect'),
  groupList: document.getElementById('groupList'),
  detailPanel: document.getElementById('detailPanel'),
  resultCount: document.getElementById('resultCount'),
  stats: document.getElementById('stats'),
};

const groups = [...window.REVIEW_DATA.groups];
const shootMap = new Map(window.REVIEW_DATA.shoots.map((shoot) => [shoot.shootKey, shoot]));

function scoreLabel(group) {
  return group.reviewScore > 0 ? `score ${group.reviewScore.toFixed(2)}` : 'not queued';
}

function populateShoots() {
  for (const shoot of window.REVIEW_DATA.shoots) {
    const option = document.createElement('option');
    option.value = shoot.shootKey;
    option.textContent = `${shoot.shootKey} (${shoot.groupCount})`;
    elements.shootSelect.appendChild(option);
  }
}

function updateStats() {
  const singletons = groups.filter((group) => group.status === 'singleton').length;
  const lowConfidence = groups.filter((group) => group.tags.includes('low_confidence')).length;
  const queued = groups.filter((group) => group.reviewScore > 0).length;
  elements.stats.textContent = `${groups.length} groups | ${queued} queued | ${singletons} singletons | ${lowConfidence} low confidence`;
}

function filteredGroups() {
  const search = state.search.trim().toLowerCase();
  let rows = groups.filter((group) => {
    if (state.filter === 'queue' && !(group.reviewScore > 0)) return false;
    if (state.filter !== 'all' && state.filter !== 'queue' && !group.tags.includes(state.filter)) return false;
    if (state.shoot && group.shootKey !== state.shoot) return false;
    if (!search) return true;
    const haystack = [
      group.angleId,
      group.shootKey,
      group.representativeFilename,
      ...group.filenames,
      ...group.reasons,
    ].join(' ').toLowerCase();
    return haystack.includes(search);
  });

  rows.sort((a, b) => {
    if (state.sort === 'confidence') return a.minConfidence - b.minConfidence || a.angleId.localeCompare(b.angleId);
    if (state.sort === 'size') return b.size - a.size || a.angleId.localeCompare(b.angleId);
    if (state.sort === 'angle') return a.angleId.localeCompare(b.angleId);
    if (state.sort === 'shoot') return a.shootKey.localeCompare(b.shootKey) || a.angleId.localeCompare(b.angleId);
    return b.reviewScore - a.reviewScore || a.minConfidence - b.minConfidence || a.angleId.localeCompare(b.angleId);
  });
  return rows;
}

function renderList() {
  const rows = filteredGroups();
  elements.resultCount.textContent = `${rows.length} groups`;
  elements.groupList.innerHTML = '';
  for (const group of rows) {
    const button = document.createElement('button');
    button.className = `groupCard${group.angleId === state.selectedAngleId ? ' selected' : ''}`;
    button.type = 'button';
    button.innerHTML = `
      <img src="${group.reviewImage}" alt="${group.angleId}">
      <div class="groupMeta">
        <div class="groupTitleRow">
          <strong>${group.angleId}</strong>
          <span class="pill">${group.status}</span>
        </div>
        <div class="muted">${group.shootKey} | n=${group.size} | min=${group.minConfidence.toFixed(4)}</div>
        <div class="muted">${scoreLabel(group)}</div>
        <div class="reasonPreview">${group.reasons.slice(0, 2).join(' | ') || 'No special review flags'}</div>
      </div>
    `;
    button.addEventListener('click', () => {
      state.selectedAngleId = group.angleId;
      renderList();
      renderDetail(group);
    });
    elements.groupList.appendChild(button);
  }

  if (!rows.length) {
    elements.detailPanel.innerHTML = '<div class="emptyState">No groups match the current filters.</div>';
    return;
  }

  if (!state.selectedAngleId || !rows.some((group) => group.angleId === state.selectedAngleId)) {
    state.selectedAngleId = rows[0].angleId;
  }
  const selected = rows.find((group) => group.angleId === state.selectedAngleId);
  if (selected) renderDetail(selected);
}

function compareCards(group) {
  if (!group.compareGroups.length) return '<p class="muted">No same-shoot compare groups for this one.</p>';
  return group.compareGroups.map((compare) => `
    <a class="compareCard" href="${compare.groupFolder}">
      <img src="${compare.reviewImage}" alt="${compare.angleId}">
      <div>
        <strong>${compare.angleId}</strong>
        <div class="muted">${compare.status} | n=${compare.size} | min=${compare.minConfidence.toFixed(4)}</div>
        <div class="muted">reject conf ${compare.maxRejectConfidence.toFixed(4)} | links ${compare.pairCount}</div>
      </div>
    </a>
  `).join('');
}

function renderDetail(group) {
  const shoot = shootMap.get(group.shootKey);
  const reasons = group.reasons.length
    ? `<ul class="reasonList">${group.reasons.map((reason) => `<li>${reason}</li>`).join('')}</ul>`
    : '<p class="muted">No special review reasons.</p>';

  const shootLink = group.shootFolder
    ? `<a href="${group.shootFolder}" target="_blank" rel="noreferrer">Open shoot folder</a>`
    : '';

  elements.detailPanel.innerHTML = `
    <div class="detailHeader">
      <div>
        <h2>${group.angleId}</h2>
        <div class="muted">${group.shootKey} | ${group.status} | n=${group.size} | min=${group.minConfidence.toFixed(4)} | ${scoreLabel(group)}</div>
      </div>
      <div class="detailLinks">
        <a href="${group.groupFolder}" target="_blank" rel="noreferrer">Open group folder</a>
        ${shootLink}
      </div>
    </div>

    <div class="hero">
      <img class="heroImage" src="${group.reviewImage}" alt="${group.angleId}">
      <div class="heroMeta">
        <p><strong>Representative:</strong> ${group.representativeFilename}</p>
        <p><strong>Shoot:</strong> ${group.shootKey}${shoot ? ` (${shoot.groupCount} groups in shoot)` : ''}</p>
        <p><strong>Tags:</strong> ${group.tags.join(', ') || 'none'}</p>
        <p><strong>Images:</strong> ${group.filenames.length}</p>
      </div>
    </div>

    <section class="panel">
      <h3>Why This Is Here</h3>
      ${reasons}
    </section>

    <section class="panel">
      <h3>Same-Shoot Compare Groups</h3>
      <div class="compareGrid">${compareCards(group)}</div>
    </section>

    <section class="panel">
      <h3>Member Filenames</h3>
      <div class="filenameList">${group.filenames.map((name) => `<code>${name}</code>`).join('')}</div>
    </section>
  `;
}

function wireEvents() {
  elements.searchInput.addEventListener('input', (event) => {
    state.search = event.target.value;
    renderList();
  });
  elements.filterSelect.addEventListener('change', (event) => {
    state.filter = event.target.value;
    renderList();
  });
  elements.sortSelect.addEventListener('change', (event) => {
    state.sort = event.target.value;
    renderList();
  });
  elements.shootSelect.addEventListener('change', (event) => {
    state.shoot = event.target.value;
    renderList();
  });
}

populateShoots();
updateStats();
wireEvents();
renderList();
"""


STYLES_CSS = """:root {
  color-scheme: light;
  --bg: #f3f0e8;
  --panel: #fffdf8;
  --ink: #1e1d1a;
  --muted: #6b655c;
  --line: #ddd4c5;
  --accent: #a84b2f;
  --accent-soft: #f4d8cd;
  --selected: #2d5d7b;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: linear-gradient(180deg, #f7f2e7 0%, #eee5d6 100%);
  color: var(--ink);
  font: 15px/1.45 Georgia, "Times New Roman", serif;
}
.app {
  max-width: 1600px;
  margin: 0 auto;
  padding: 24px;
}
.topbar, .controls, .panel, .listPanel, .detailPanel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(60, 40, 10, 0.06);
}
.topbar {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  padding: 24px;
  margin-bottom: 18px;
}
.topbar h1 {
  margin: 0 0 6px;
  font-size: 34px;
}
.topbar p, .muted {
  color: var(--muted);
}
.controls {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  padding: 16px;
  margin-bottom: 18px;
}
.controls label {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 13px;
  color: var(--muted);
}
input, select {
  width: 100%;
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 12px;
  background: white;
  font: inherit;
}
.layout {
  display: grid;
  grid-template-columns: 420px minmax(0, 1fr);
  gap: 18px;
  min-height: 70vh;
}
.listPanel {
  overflow: hidden;
}
.listHeader {
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
}
.groupList {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 12px;
  max-height: calc(100vh - 260px);
  overflow: auto;
}
.groupCard {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 12px;
  width: 100%;
  text-align: left;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #fff;
  padding: 10px;
  cursor: pointer;
}
.groupCard:hover, .groupCard.selected {
  border-color: var(--selected);
  box-shadow: 0 0 0 3px rgba(45, 93, 123, 0.12);
}
.groupCard img, .compareCard img, .heroImage {
  width: 100%;
  display: block;
  border-radius: 10px;
  border: 1px solid var(--line);
}
.groupMeta {
  min-width: 0;
}
.groupTitleRow {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 2px 10px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 12px;
}
.reasonPreview {
  margin-top: 6px;
  font-size: 13px;
}
.detailPanel {
  padding: 20px;
}
.detailHeader {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
}
.detailHeader h2 {
  margin: 0 0 6px;
  font-size: 28px;
}
.detailLinks {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}
a {
  color: var(--selected);
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
.hero {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr);
  gap: 18px;
  margin: 18px 0;
}
.heroMeta {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #fff;
}
.panel {
  padding: 16px;
  margin-bottom: 16px;
}
.panel h3 {
  margin: 0 0 12px;
}
.reasonList {
  margin: 0;
  padding-left: 20px;
}
.compareGrid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 12px;
}
.compareCard {
  display: grid;
  grid-template-columns: 110px minmax(0, 1fr);
  gap: 12px;
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 10px;
  background: white;
}
.filenameList {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
code {
  background: #f6f0e4;
  border: 1px solid #e7dccc;
  border-radius: 999px;
  padding: 4px 10px;
  font: 12px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace;
}
.emptyState {
  color: var(--muted);
  min-height: 240px;
  display: grid;
  place-items: center;
  font-size: 18px;
}
@media (max-width: 1100px) {
  .controls, .layout, .hero {
    grid-template-columns: 1fr;
  }
  .groupList {
    max-height: none;
  }
}
"""


def write_site(output_dir: Path, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    (output_dir / "app.js").write_text(APP_JS, encoding="utf-8")
    (output_dir / "styles.css").write_text(STYLES_CSS, encoding="utf-8")
    data_js = "window.REVIEW_DATA = " + json.dumps(data, indent=2) + ";\n"
    (output_dir / "data.js").write_text(data_js, encoding="utf-8")


def main() -> int:
    global args
    args = parse_args()
    groups, filename_to_group = load_groups(args.groups, args.grouped_dir)
    neighbors_by_group = load_neighbors(args.pair_scores, groups, filename_to_group)
    data = build_data(groups, neighbors_by_group, args.output_dir, args.review_dir, args.navigator_dir)
    write_site(args.output_dir, data)
    print(f"Wrote static review frontend to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
