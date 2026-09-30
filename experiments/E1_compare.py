"""E1: spec v0.2 vs v0.3 comparison, on the seeds both runs share.

    uv run python -m experiments.E1_compare --old <path to v0.2 E1.json> --new outputs/E1_v03/E1.json

Writes results/E1_v03_compare.png (mean level and fraction at level >= 2,
v0.2 dashed vs v0.3 solid, per condition) and prints a markdown table of
matched-seed epoch-400 numbers plus paired v0.3 - v0.2 differences.
Uncertainty is mean ± standard error across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.E1 import CONDITIONS  # noqa: E402
from experiments.E1_plot import COLORS, fmt, style_axes  # noqa: E402


def load(path: str) -> dict[int, dict]:
    return {r["seed"]: r for r in json.loads(Path(path).read_text())["results"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", default="outputs/E1_v03/E1.json")
    ap.add_argument("--out", default="results/E1_v03_compare.png")
    args = ap.parse_args()
    old, new = load(args.old), load(args.new)
    seeds = sorted(set(old) & set(new))

    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(15, 6), sharex=True, sharey="row")
    for col, name in enumerate(CONDITIONS):
        for row, key in enumerate(("mean_level", "frac_ge2")):
            ax = axes[row, col]
            for data, ls, lab in ((old, "--", "v0.2"), (new, "-", "v0.3")):
                c = np.array([data[s]["conditions"][name]["curves"][key] for s in seeds])
                t = np.arange(1, c.shape[1] + 1)
                m, se = c.mean(0), c.std(0, ddof=1) / np.sqrt(len(c))
                ax.plot(t, m, color=COLORS[name], ls=ls, lw=2, label=lab)
                ax.fill_between(t, m - se, m + se, color=COLORS[name], alpha=0.15, lw=0)
            style_axes(ax)
            if row == 0:
                ax.set_title(name, fontsize=10)
            if row == 1:
                ax.set_xlabel("Epoch")
                ax.set_ylim(0, 1)
        axes[0, col].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel("Mean level")
    axes[1, 0].set_ylabel("Fraction at level ≥ 2")
    fig.suptitle(f"E1 under spec v0.2 (dashed) vs v0.3 (solid): 19 non-elder minds, "
                 f"mean ± SE over matched seeds {seeds[0]}–{seeds[-1]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    L = [f"Matched seeds: {seeds}\n",
         "| Condition | mean level @400 v0.2 | v0.3 | paired Δ | frac ≥2 @400 v0.2 | v0.3 | paired Δ | frac ≥1 @400 v0.2 | v0.3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name in CONDITIONS:
        row = [name]
        for key in ("mean_level", "frac_ge2", "frac_ge1"):
            o = [old[s]["conditions"][name]["curves"][key][-1] for s in seeds]
            n = [new[s]["conditions"][name]["curves"][key][-1] for s in seeds]
            row += [fmt(o), fmt(n)]
            if key != "frac_ge1":
                row.append(fmt(np.array(n) - np.array(o)))
        L.append("| " + " | ".join(row) + " |")
    print("\n".join(L))


if __name__ == "__main__":
    main()
