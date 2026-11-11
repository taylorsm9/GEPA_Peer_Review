"""Static candidate-tree figure for the README: each GEPA candidate's val score at the iteration it was found,
linked to its parent, with the path from the seed to the best candidate highlighted.

Reads runs/gepa/main/iterations/*/meta.json.
Writes figures/candidate_tree_{light,dark}.svg; the README picks one with <picture> by color scheme.

    .venv/bin/python scripts/plot_candidate_tree.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

RUN = ROOT / "runs" / "gepa" / "main"
OUT = ROOT / "figures"
SEED_X = -3  # the seed has no iteration; draw it just left of iteration 0
LABELS = {1: (-1.0, 0.003, "right", "1"), 5: (1.0, -0.004, "left", "5"),  # (dx, dy, ha, text)
          8: (1.0, 0.0015, "left", "8 (second best)"), 28: (0, 0.004, "center", "28 (best)")}
THEMES = {  # transparent background; ink and series steps for GitHub's light and dark pages
    "light": {"text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "path": "#2a78d6", "other": "#a3a29c",
              "ring": "#ffffff"},
    "dark": {"text": "#f0efec", "muted": "#c3c2b7", "grid": "#33332f", "path": "#3987e5", "other": "#6e6d68",
             "ring": "#0d1117"},
}


def candidates():
    """{candidate_idx: (x, val score, parent candidate_idx or None)} for accepted candidates and the seed."""
    metas = [json.load(p.open()) for p in RUN.glob("iterations/*/meta.json")]
    metas = [m for m in metas if m.get("accepted", True) and m.get("candidate_idx") is not None]
    by_id = {m["iteration_id"]: m["candidate_idx"] for m in metas}
    out = {}
    for m in metas:
        parents = [by_id[p] for p in m.get("parent_iteration_ids") or [] if p in by_id]
        c = m["candidate_idx"]
        out[c] = (SEED_X if c == 0 else m["trace_i"], m["avg_val_score"], parents[0] if parents else None)
    return out


def lineage(cands, best):
    path, c = [best], best
    while cands[c][2] is not None:
        c = cands[c][2]
        path.append(c)
    return set(path)


def draw(cands, best, theme):
    t = THEMES[theme]
    on_path = lineage(cands, best)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=100)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    for c, (x, y, p) in cands.items():
        if p is None:
            continue
        px, py, _ = cands[p]
        hot = c in on_path and p in on_path
        ax.plot([px, x], [py, y], color=t["path"] if hot else t["other"], lw=2 if hot else 1,
                alpha=1 if hot else 0.6, zorder=3 if hot else 1, solid_capstyle="round")
    for hot in (False, True):
        pts = [(x, y) for c, (x, y, _) in cands.items() if (c in on_path) == hot]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=64 if hot else 40, color=t["path"] if hot else t["other"], edgecolor=t["ring"],
                   linewidth=2, zorder=4 if hot else 2,
                   label="Path from the seed to the best candidate" if hot else "Other candidates")

    for c, (dx, dy, ha, label) in LABELS.items():
        x, y, _ = cands[c]
        ax.text(x + dx, y + dy, label, color=t["text"], fontsize=9, ha=ha, va="center", zorder=5)

    ax.set_xlim(SEED_X - 2, 67)
    ax.set_ylim(0.795, 0.922)
    ax.set_xticks([SEED_X, 0, 10, 20, 30, 40, 50, 60], ["seed", "0", "10", "20", "30", "40", "50", "60"])
    ax.set_xlabel("GEPA iteration the candidate was found", color=t["muted"], fontsize=10)
    ax.set_ylabel("Val score (40 papers)", color=t["muted"], fontsize=10)
    ax.set_title("Each GEPA candidate's val score, linked to the candidate it was edited from",
                 color=t["text"], fontsize=11, loc="left")
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"candidate_tree_{theme}.svg", transparent=True)
    plt.close(fig)


def main():
    cands = candidates()
    best = max(cands, key=lambda c: cands[c][1])
    for theme in THEMES:
        draw(cands, best, theme)
    print(f"{len(cands)} candidates, best {best} ({cands[best][1]:.4f}) -> {OUT}")


if __name__ == "__main__":
    main()
