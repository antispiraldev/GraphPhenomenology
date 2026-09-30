"""E3 figures from the sweep outputs (run after the sweeps in results/E3.md).

    uv run python -m experiments.E3_plot [--tag-prefix E3] [--out-prefix results/E3] [--note TEXT]

Writes <out-prefix>_sweeps.png, <out-prefix>_size.png and <out-prefix>_summary.json.
Sweep tags are looked up as <tag-prefix>_rho_coarse etc.; missing tags are skipped
(so a sweep run without its refinement still plots).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

from experiments.E3_analyze import summarize  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SWEEPS = {
    "rho": (["rho_coarse", "rho_fine"], "reactivity ρ", False),
    "p_contact": (["p_coarse"], "contact probability p_contact", [0.02, 0.05, 0.1, 0.2, 0.4]),
    "q_edge": (["q_coarse", "q_fine"], "initial edge density q_edge", [0.03, 0.05, 0.08, 0.12, 0.2]),
}
SIZES = [(10, "rho_N10", ORANGE), (20, None, BLUE), (40, "rho_N40", AQUA)]


def existing(prefix: str, names: list[str]) -> list[str]:
    tags = [f"{prefix}_{n}" for n in names]
    return [t for t in tags if (Path("outputs") / t / "sweep.json").exists()]


def style(ax, xlabel: str, ylabel: str, log: bool, ticks: list[float] | None = None) -> None:
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="E3")
    ap.add_argument("--out-prefix", default="results/E3")
    ap.add_argument("--note", default="20 minds, 200 epochs, 8 seeds/point")
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    summary = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex="col")
    for col, (axis, (tags, label, log)) in enumerate(SWEEPS.items()):
        tags = existing(args.tag_prefix, tags)
        rows = summarize(tags, axis, with_isolation=(axis == "p_contact"))
        summary[axis] = rows
        x = [r["x"] for r in rows]
        ax = axes[0, col]
        ax.errorbar(x, [r["phi"] for r in rows], yerr=[r["se"] for r in rows], color=BLUE, lw=2,
                    marker="o", ms=5, capsize=3, label="level ≥ 2 (order parameter)")
        ax.errorbar(x, [r["ge1"] for r in rows], yerr=[r["ge1_se"] for r in rows], color=ORANGE, lw=2,
                    marker="s", ms=5, capsize=3, label="level ≥ 1")
        if axis == "p_contact":
            ax.plot(x, [1 - r["iso"] for r in rows], color=MUTED, ls="--", lw=1.5, label="minds with ≥1 contact")
        if axis == "q_edge":
            ax.plot(x, [r["comp"] for r in rows], color=MUTED, ls="--", lw=1.5, label="windows with computable C")
        if axis == "rho":
            ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
        else:
            ax.axvline({"p_contact": 0.2, "q_edge": 0.08}[axis], color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(-0.02, 1.02)
        style(ax, "", "fraction of minds (final epoch)" if col == 0 else "", bool(log), log or None)
        ax.set_title(f"Sweep of {label}", color=INK)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
        ax = axes[1, col]
        ax.plot(x, [r["var"] for r in rows], color=BLUE, lw=2, marker="o", ms=5, label="level ≥ 2")
        ax.plot(x, [r["ge1_var"] for r in rows], color=ORANGE, lw=2, marker="s", ms=5, label="level ≥ 1")
        style(ax, label, "variance across seeds" if col == 0 else "", bool(log), log or None)
        ax.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"E3: development vs. coupling parameters ({args.note}; dotted = default)", color=INK)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_sweeps.png", dpi=130)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for n, tag, color in SIZES:
        tags = existing(args.tag_prefix, [tag] if tag else ["rho_coarse", "rho_fine"])
        rows = [r for r in summarize(tags, "rho") if r["x"] <= 0.6]
        summary[f"rho_N{n}"] = rows
        x = [r["x"] for r in rows]
        axes[0].errorbar(x, [r["ge1"] for r in rows], yerr=[r["ge1_se"] for r in rows], color=color, lw=2,
                         marker="o", ms=5, capsize=3, label=f"N = {n}")
        axes[1].errorbar(x, [r["phi"] for r in rows], yerr=[r["se"] for r in rows], color=color, lw=2,
                         marker="o", ms=5, capsize=3, label=f"N = {n}")
        axes[2].plot(x, [n * r["ge1_var"] for r in rows], color=color, lw=2, marker="o", ms=5, label=f"N = {n}")
    for ax, yl, t in zip(axes, ["fraction at level ≥ 1", "fraction at level ≥ 2", "N × variance of level-≥1 fraction"],
                         ["Onset: level ≥ 1", "Order parameter: level ≥ 2", "Fluctuations (susceptibility)"]):
        style(ax, "reactivity ρ", yl, False)
        ax.set_title(t, color=INK)
        ax.legend(frameon=False, fontsize=8.5)
    axes[0].set_ylim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    fig.suptitle("E3 size check on ρ (mean contact degree held ≈ 3.8: p_contact = 0.42 / 0.2 / 0.1)", color=INK)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_size.png", dpi=130)
    Path(f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
