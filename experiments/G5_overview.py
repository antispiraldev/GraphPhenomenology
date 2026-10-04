"""G5 synthesis figure: one panel per experiment, spec v0.3 results.

    uv run python -m experiments.G5_overview

Reads the saved v0.3 outputs (outputs/ is not committed; the G4c worktree outputs were
copied to outputs/archive/ at commit 42bda23) and writes results/G5_overview.png.
Every number plotted is also reported in the corresponding results/E*_v03.md.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARCHIVE = "outputs/archive/"
SOURCES = {
    "E0": "outputs/E0_v03/sweep.json",
    "E1": ARCHIVE + "*G4c_E1*/E1_v03/E1.json",
    "E2": ARCHIVE + "*G4c_E2*/E2v03_summary.json",
    "E3": "results/E3_v03_summary.json",
    "E4": ARCHIVE + "*G4c_E4*/E4_v03_compare.json",
    "E5": ARCHIVE + "*G4c_E5*/E5/summary.json",
}
RECORDING_LIMIT = 2e-4  # D8: recordings < 1 in 5,000 windows
INK, MUTED = "#222222", "#777777"
C_V03, GREY = "#2a78d6", "#b0b0b0"


def load(key: str) -> dict:
    paths = glob.glob(SOURCES[key])
    if len(paths) != 1:
        raise FileNotFoundError(f"{key}: expected one match for {SOURCES[key]}, got {paths}")
    return json.loads(Path(paths[0]).read_text())


def style(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)


def panel_e0(ax) -> None:
    runs = load("E0")["results"]
    kinds = ("object", "mirror", "recording")
    k = [sum(r["totals"][f"one_sided_vs_{c}"] for r in runs) for c in kinds]
    n = [sum(r["totals"][f"windows_vs_{c}"] for r in runs) for c in kinds]
    rate = [a / b for a, b in zip(k, n)]
    ax.bar(kinds, rate, color=[GREY, GREY, C_V03])
    ax.axhline(RECORDING_LIMIT, color="#d6452a", ls="--", lw=1)
    ax.text(2.45, RECORDING_LIMIT * 1.3, "limit for recordings", ha="right", fontsize=8, color="#d6452a")
    for i, (a, b) in enumerate(zip(k, n)):
        y = rate[i] * 1.6 if a else 3e-7
        ax.text(i, y, f"{a:,} / {b:,}" + ("" if a else "\n(never)"), ha="center", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(5e-8, 1e-2)
    ax.set_ylabel("false recognitions per window", fontsize=9)
    style(ax, "E0  Impostors are not recognized")


def panel_e1(ax) -> None:
    res = load("E1")["results"]
    colors = {"bootstrap (no elder)": "#52514e", "elder+elder": "#eb6834", "mixed (no elder)": "#e87ba4"}
    names = {"bootstrap (no elder)": "peers only", "elder+elder": "learn from those ahead",
             "mixed (no elder)": "both"}
    for cond, col in colors.items():
        y = np.mean([r["conditions"][cond]["curves"]["mean_level"] for r in res], axis=0)
        ax.plot(np.arange(1, len(y) + 1), y, color=col, lw=2, label=names[cond])
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("mean level", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "E1  Learning from those ahead helps")


def panel_e2(ax) -> None:
    t = load("E2")["table"]
    th = np.array([r["theta"] for r in t])
    rate = np.array([r["recording"]["rate"] for r in t])
    hi = np.array([r["recording"]["hi"] for r in t])
    pos = rate > 0
    ax.plot(th[pos], rate[pos], "o-", color="#d6452a", ms=4, label="recordings fooled")
    ax.plot(th[~pos], hi[~pos], "v", color="#d6452a", ms=5, mfc="none", label="(none seen: upper bound)")
    ax.axhline(RECORDING_LIMIT, color="#d6452a", ls="--", lw=1)
    ax.axvline(0.4, color=MUTED, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("strictness θ", fontsize=9)
    ax.set_ylabel("recording false-recognition rate", fontsize=9, color="#d6452a")
    ax2 = ax.twinx()
    ax2.plot(th, [r["frac_ge2"][0] for r in t], "s-", color=C_V03, ms=4, label="minds at level ≥ 2")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("fraction at level ≥ 2", fontsize=9, color=C_V03)
    ax2.spines[["top"]].set_visible(False)
    ax2.tick_params(labelsize=9)
    ax.text(0.41, 3e-2, "default 0.4", fontsize=8, color=MUTED)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="lower left")
    style(ax, "E2  Strictness: θ = 0.4 is the sweet spot")


def panel_e3(ax) -> None:
    d = load("E3")
    for key, col, lab in (("rho_N10", "#9bc2ea", "10 minds"), ("rho_N20", C_V03, "20 minds"),
                          ("rho_N40", "#173f75", "40 minds")):
        pts = sorted(d[key], key=lambda r: r["x"])
        x = [r["x"] for r in pts]
        ax.errorbar(x, [r["phi"] for r in pts], yerr=[r["se"] for r in pts], color=col, lw=2,
                    marker="o", ms=4, capsize=2, label=lab)
    ax.set_xlabel("reactivity ρ", fontsize=9)
    ax.set_ylabel("fraction at level ≥ 2", fontsize=9)
    ax.set_xlim(0.1, 0.65)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax, "E3  A sharp threshold, same at every size")


def panel_e4(ax) -> None:
    coh = load("E4")["cohort"]
    k = np.arange(1, 5)
    c = np.array(coh["v0.3_K5"]["counts"], dtype=float)
    h = 100 * c[1:5, 1] / c[1:5, 0]
    ax.bar(k, h, width=0.6, color=C_V03)
    for i, v in zip(k, h):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
    ax.text(4.4, h[0] * 0.95, f"the {coh['v0.3_K5']['minds']} minds that\nreached level 5,\nfollowed step by step",
            ha="right", va="top", fontsize=8, color=MUTED)
    ax.set_xticks(k, [f"{i}→{i + 1}" for i in k])
    ax.set_xlabel("step (same minds at every step)", fontsize=9)
    ax.set_ylabel("chance of climbing per meeting (%)", fontsize=9)
    style(ax, "E4  Each step is harder: no snowball")


def panel_e5(ax) -> None:
    by = load("E5")["key_test"]["shuffled"]["by_class_level"]
    lv = [1, 2, 3, 4]
    for cls, col, off, lab in (("cobuilt_top", "#1baf7a", -0.18, "with the partner it was built with"),
                               ("stranger", GREY, 0.18, "with an equal-level stranger")):
        y = [100 * by[f"{cls}|{L}"]["probe_successes"] / by[f"{cls}|{L}"]["probe_starts"] for L in lv]
        ax.bar(np.array(lv) + off, y, width=0.36, color=col, label=lab)
        for L, v in zip(lv, y):
            if v == 0:
                ax.text(L + off, 0.1, "0", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(lv, [f"L{L}" for L in lv])
    ax.set_xlabel("level of the shared routine", fontsize=9)
    ax.set_ylabel("routine replay succeeds (%)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "E5  Private languages")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/G5_overview.png")
    a = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for ax, f in zip(axes.flat, (panel_e0, panel_e1, panel_e2, panel_e3, panel_e4, panel_e5)):
        f(ax)
    fig.suptitle("Mutual recognition model, spec v0.3: one result per experiment", fontsize=14, x=0.01,
                 ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
