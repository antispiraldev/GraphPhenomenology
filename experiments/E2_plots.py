"""Figures for E2 (credulity sweep). Called from experiments/E2.py analyze."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _style(ax) -> None:
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _rate_points(table, kind="recording"):
    """(theta, rate, lo, hi, observed_zero) per row; zero-count rows are drawn at their 95% upper bound."""
    out = []
    for r in table:
        c = r[kind]
        out.append((r["theta"], c["rate"], c["lo"], c["hi"], c["k"] == 0))
    return out


def _tradeoff_panel(ax, table, key, ylabel) -> None:
    pts = _rate_points(table)
    dev = [r[key] for r in table]
    xs = [hi if zero else rate for _, rate, _, hi, zero in pts]
    ax.plot(xs, [d[0] for d in dev], color=BLUE, lw=1.5, alpha=0.5, zorder=1)
    zero_labels: dict[float, list[str]] = {}
    for (th, rate, lo, hi, zero), (m, h) in zip(pts, dev):
        if zero:
            ax.errorbar(hi, m, yerr=h, fmt="none", ecolor=BLUE, elinewidth=1, alpha=0.6)
            ax.annotate("", xy=(hi / 2.2, m), xytext=(hi, m),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
            ax.plot(hi, m, "o", ms=8, mfc="white", mec=BLUE, mew=1.8, zorder=3)
            zero_labels.setdefault(round(m, 3), []).append(f"{th:g}")
            continue
        ax.errorbar(rate, m, xerr=[[rate - lo], [hi - rate]], yerr=h, fmt="none",
                    ecolor=BLUE, elinewidth=1, alpha=0.6)
        ax.plot(rate, m, "o", ms=8, color=BLUE, mec="white", mew=1.5, zorder=3)
        off = (-4, -15) if th == 0 else (7, 6)
        ax.annotate(f"θ={th:g}", (rate, m), xytext=off, textcoords="offset points", fontsize=9, color=INK)
    hi0 = next(hi for _, _, _, hi, zero in pts if zero) if zero_labels else None
    lowest = min(zero_labels) if zero_labels else None
    for m, labs in zero_labels.items():
        ax.annotate("θ=" + ", ".join(labs), (hi0, m), xytext=(9, -13 if m == lowest else 5),
                    textcoords="offset points", fontsize=9, color=INK)
    ax.axvline(CRIT, color=ORANGE, lw=1.5, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Recording false-positive rate\n(one-sided recognitions per mind–recording window, log)")
    ax.set_ylabel(ylabel)
    _style(ax)


CRIT = 1 / 5000


def plot_tradeoff(table, criterion, path: Path) -> None:
    global CRIT
    CRIT = criterion
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
    _tradeoff_panel(axes[0], table, "frac_ge2", "Minds at level ≥ 2 at epoch 300 (fraction)")
    _tradeoff_panel(axes[1], table, "mean_level_e25", "Mean mind level at epoch 25 (early speed)")
    axes[0].set_title("Final development", color=INK, fontsize=11, loc="left")
    axes[1].set_title("Speed of development", color=INK, fontsize=11, loc="left")
    for ax in axes:
        ax.text(CRIT * 1.1, 0.98, "criterion\n1 in 5,000", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=8, va="top")
    fig.suptitle("E2: credulity trade-off — each point is one θ (20 minds + 12 controls, 300 epochs)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.text(0.01, -0.03, "Filled: observed rate with exact 95% CI. Hollow with arrow: none observed, drawn at the "
             "95% upper bound. Vertical bars: 95% CI over seeds (10 per θ; 32 for θ = 0.3–0.45).",
             fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_by_theta(table, criterion, path: Path) -> None:
    th = np.array([r["theta"] for r in table])
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.5), dpi=150, sharex=True)

    ax = axes[0]
    for t, rate, lo, hi, zero in _rate_points(table):
        if zero:
            ax.plot([t, t], [1e-6, hi], color=BLUE, lw=1, alpha=0.6)
            ax.plot(t, hi, "v", ms=7, mfc="white", mec=BLUE, mew=1.5)
        else:
            ax.errorbar(t, rate, yerr=[[rate - lo], [hi - rate]], fmt="o", ms=6, color=BLUE,
                        mec="white", elinewidth=1, capsize=0)
    nz = [(t, rate) for t, rate, _, _, zero in _rate_points(table) if not zero]
    ax.plot(*zip(*nz), color=BLUE, lw=1.5, alpha=0.6)
    ax.axhline(criterion, color=ORANGE, ls="--", lw=1.5)
    ax.text(0.8, criterion * 1.3, "1 in 5,000", color=INK2, fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 0.3)
    ax.set_ylabel("Recording false-positive\nrate (per window)")
    ax.set_title("E2: effect of the contingency threshold θ", color=INK, fontsize=12, loc="left")
    ax.text(0.99, 0.95, "objects, mirrors: 0 at every θ\n▽ = none observed (95% upper bound)",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)

    ax = axes[1]
    for key, color, name in (("frac_ge1", AQUA, "level ≥ 1"), ("frac_ge2", BLUE, "level ≥ 2"),
                             ("frac_ge3", ORANGE, "level ≥ 3")):
        m = np.array([r[key][0] for r in table])
        h = np.array([r[key][1] for r in table])
        ax.fill_between(th, m - h, m + h, color=color, alpha=0.15, lw=0)
        ax.plot(th, m, "o-", color=color, lw=2, ms=5, mec="white")
        ax.annotate(name, (th[0], m[0]), xytext=(-6, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=9, color=INK)
    ax.set_ylabel("Fraction of minds\nat epoch 300")
    ax.set_ylim(0, None)

    ax = axes[2]
    m = np.array([r["computable"][0] for r in table])
    h = np.array([r["computable"][1] for r in table])
    ax.fill_between(th, m - h, m + h, color=BLUE, alpha=0.15, lw=0)
    ax.plot(th, m, "o-", color=BLUE, lw=2, ms=5, mec="white")
    ax.set_ylabel("Windows with\ncomputable contingency")
    ax.set_xlabel("Contingency threshold θ")
    for a in axes:
        _style(a)
        a.axvline(0.4, color=MUTED, lw=1, ls=":")
    axes[1].set_xlim(-0.12, 0.83)
    axes[0].text(0.405, 0.1, "default θ=0.4", color=INK2, fontsize=8, rotation=90, va="top")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_curves(table, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)
    colors = SEQ[-len(table):] if len(table) <= len(SEQ) else SEQ
    for ax, key, label in ((axes[0], "curve_frac_ge2", "Fraction of minds at level ≥ 2"),
                           (axes[1], "curve_mean_level", "Mean mind level")):
        for r, c in zip(table, colors):
            y = np.asarray(r[key])
            ax.plot(np.arange(len(y)), y, color=c, lw=1.6, label=f"{r['theta']:g}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        _style(ax)
    axes[1].legend(title="θ", fontsize=8, title_fontsize=9, frameon=False, loc="center left",
                   bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("E2: development over time, mean over seeds (light → dark = low → high θ)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_all(table, criterion, out: Path) -> None:
    plot_tradeoff(table, criterion, out / "E2_tradeoff.png")
    plot_by_theta(table, criterion, out / "E2_by_theta.png")
    plot_curves(table, out / "E2_curves.png")
