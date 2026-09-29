"""E1 figures and summary tables from outputs/<tag>/E1.json.

    uv run python -m experiments.E1_plot --tag E1

Writes results/E1_development.png, results/E1_diversity.png,
results/E1_sources.png and prints markdown tables (also saved to
outputs/<tag>/summary.md). Uncertainty is mean ± standard error across seeds.
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

COLORS = {
    "bootstrap+elder": "#2a78d6",
    "elder+elder": "#eb6834",
    "mixed+elder": "#1baf7a",
    "bootstrap (no elder)": "#52514e",
    "mixed (no elder)": "#e87ba4",
}
STYLE = {k: ("--" if "no elder" in k else "-") for k in COLORS}
LEVELS = (1, 2, 3, 4, 5)


def mse(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return float(x.mean()), float(se)


def fmt(x, digits=2) -> str:
    m, s = mse(x)
    if np.isnan(m):
        return "–"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)


def fig_development(results, epochs, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    t = np.arange(1, epochs + 1)
    for ax, key, label in ((axes[0], "mean_level", "Mean level"),
                           (axes[1], "frac_ge2", "Fraction of minds at level ≥ 2")):
        for name in CONDITIONS:
            curves = np.array([r["conditions"][name]["curves"][key] for r in results])
            m = curves.mean(0)
            se = curves.std(0, ddof=1) / np.sqrt(len(curves))
            ax.plot(t, m, color=COLORS[name], ls=STYLE[name], lw=2, label=name)
            ax.fill_between(t, m - se, m + se, color=COLORS[name], alpha=0.18, lw=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        style_axes(ax)
    axes[1].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"E1: development of the 19 non-elder minds (mean ± SE, {len(results)} seeds)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_diversity(results, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(CONDITIONS)
    width = 0.8 / len(names)
    for ax, key, label in ((axes[0], "distinct_flat", "Distinct flat words"),
                           (axes[1], "distinct_core_extras", "Distinct new content")):
        for n_i, name in enumerate(names):
            ms, ses = [], []
            for k in LEVELS:
                vals = [r["conditions"][name]["diversity"][key].get(str(k), 0) for r in results]
                m, s = mse(vals)
                ms.append(m)
                ses.append(s)
            x = np.arange(len(LEVELS)) + (n_i - (len(names) - 1) / 2) * width
            ax.bar(x, ms, width * 0.9, yerr=ses, color=COLORS[name], label=name,
                   hatch="//" if "no elder" in name else None, edgecolor="white", linewidth=0.5,
                   error_kw={"elinewidth": 1, "ecolor": "#52514e"})
        ax.set_xticks(np.arange(len(LEVELS)), [f"L{k}" for k in LEVELS])
        ax.set_xlabel("Record level")
        ax.set_ylabel(label + " per population")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("E1: structural diversity at the final epoch, elder excluded (mean ± SE)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


SOURCE_KEYS = (
    ("bootstrap|other", "Bootstrap with a peer", "#2a78d6"),
    ("elder|other", "Transmission from a non-elder mind", "#eda100"),
    ("elder|elder", "Transmission from agent 0 (the elder, if present)", "#eb6834"),
    ("bootstrap|elder", "Bootstrap with agent 0", "#4a3aa7"),
)


def fig_sources(results, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(CONDITIONS)
    bottom = np.zeros(len(names))
    for key, label, color in SOURCE_KEYS:
        vals = np.array([np.mean([r["conditions"][n]["levelups"].get(key, 0) for r in results]) for n in names])
        ax.bar(names, vals, bottom=bottom, color=color, label=label, edgecolor="white", linewidth=1.5)
        bottom += vals
    ax.set_ylabel("Level-ups of non-elder minds per run (mean)")
    ax.set_title("E1: where level-ups came from", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def tables(results, epochs) -> str:
    L = []
    names = list(CONDITIONS)
    checkpoints = [e for e in (50, 100, 200, epochs) if e <= epochs]
    L.append("### Development (19 non-elder minds)\n")
    L.append("| Condition | " + " | ".join(f"mean level @{e}" for e in checkpoints)
             + " | " + " | ".join(f"frac ≥2 @{e}" for e in checkpoints) + " | frac ≥1 final | max level final |")
    L.append("|---|" + "---|" * (2 * len(checkpoints) + 2))
    for n in names:
        c = [r["conditions"][n]["curves"] for r in results]
        row = [fmt([x["mean_level"][e - 1] for x in c]) for e in checkpoints]
        row += [fmt([x["frac_ge2"][e - 1] for x in c]) for e in checkpoints]
        row += [fmt([x["frac_ge1"][-1] for x in c]), fmt([x["max_level"][-1] for x in c], 1)]
        L.append(f"| {n} | " + " | ".join(row) + " |")

    L.append("\n### Time to first reach a level (epoch, first non-elder mind)\n")
    L.append("Mean ± SE over seeds that reached it; (n) = seeds that reached it within the run.\n")
    L.append("| Condition | " + " | ".join(f"L{k}" for k in (1, 2, 3, 4, 5)) + " | median mind reaches L1 |")
    L.append("|---|" + "---|" * 6)
    for n in names:
        cells = []
        for k in (1, 2, 3, 4, 5):
            v = [r["conditions"][n]["first_any"][str(k)] for r in results]
            got = [x for x in v if x is not None]
            cells.append(f"{fmt(got, 1)} ({len(got)})" if got else f"never (0)")
        cells.append(fmt([r["conditions"][n]["median_mind_time"]["1"] for r in results], 1))
        L.append(f"| {n} | " + " | ".join(cells) + " |")

    L.append("\n### Diversity at the final epoch (elder excluded)\n")
    L.append("distinct flat words / records at that level; then distinct canonical extras.\n")
    L.append("| Condition | " + " | ".join(f"L{k} flat / records" for k in LEVELS) + " | "
             + " | ".join(f"L{k} core" for k in LEVELS) + " |")
    L.append("|---|" + "---|" * (2 * len(LEVELS)))
    for n in names:
        d = [r["conditions"][n]["diversity"] for r in results]
        cells = [f"{fmt([x['distinct_flat'].get(str(k), 0) for x in d], 1)} / {fmt([x['n_records'].get(str(k), 0) for x in d], 1)}"
                 for k in LEVELS]
        cells += [fmt([x["distinct_core_extras"].get(str(k), 0) for x in d], 1) for k in LEVELS]
        L.append(f"| {n} | " + " | ".join(cells) + " |")

    L.append("\n### Level-up sources (non-elder minds, per run)\n")
    L.append("| Condition | " + " | ".join(lbl for _, lbl, _ in SOURCE_KEYS) + " | total | minds taught directly by elder |")
    L.append("|---|" + "---|" * (len(SOURCE_KEYS) + 2))
    for n in names:
        lu = [r["conditions"][n]["levelups"] for r in results]
        cells = [fmt([x.get(k, 0) for x in lu], 1) for k, _, _ in SOURCE_KEYS]
        cells.append(fmt([sum(v for k, v in x.items() if not k.startswith("elder_self")) for x in lu], 1))
        cells.append(fmt([r["conditions"][n]["taught_directly"] for r in results], 1))
        L.append(f"| {n} | " + " | ".join(cells) + " |")

    L.append("\n### Elders and their reach (per seed)\n")
    L.append("| seed | donor seed | donor epochs | elder level | elder learned labels | sync labels at elder's node | elder degree | reachable minds | elder-mode frac ≥1 final | elder-mode frac ≥1 among reachable | elder-mode max level | elder-mode first L1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        e = r["elder"]
        pr = e.get("profile", {})
        c = r["conditions"]["elder+elder"]
        fl = np.array(c["final_levels"])
        fr = np.array(c["final_levels_in_comp"])
        L.append(f"| {r['seed']} | {e['donor_seed']} | {e['donor_epochs']} | {e['elder_level']} | "
                 f"{pr.get('learned_labels', '?')} | {pr.get('sync_available_at_x', '?')} | {c['elder_degree']} | "
                 f"{c['elder_component']} | {(fl >= 1).mean():.2f} | {(fr >= 1).mean() if len(fr) else float('nan'):.2f} | "
                 f"{fl.max()} | {c['first_any']['1']} |")
    by_n: dict = {}
    for r in results:
        by_n.setdefault(r["elder"].get("profile", {}).get("n_learned"), []).append(r)
    L.append("\nElder mode grouped by the elder's number of learned labels:\n")
    L.append("| learned labels | seeds | frac ≥1 final | mean level final | first L1 (epoch) |")
    L.append("|---|---|---|---|---|")
    for n, rs in sorted(by_n.items(), key=lambda kv: str(kv[0])):
        c = [r["conditions"]["elder+elder"] for r in rs]
        L.append(f"| {n} | {len(rs)} | {fmt([x['curves']['frac_ge1'][-1] for x in c])} | "
                 f"{fmt([x['curves']['mean_level'][-1] for x in c])} | {fmt([x['first_any']['1'] for x in c], 1)} |")
    tries = [len(r["elder"]["attempts"]) for r in results]
    L.append(f"\nDonor attempts per seed: {tries}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="E1")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    data = json.loads((Path(args.outputs) / args.tag / "E1.json").read_text())
    results = data["results"]
    epochs = data["meta"]["config"]["epochs"]
    res_dir = Path(args.results)
    res_dir.mkdir(exist_ok=True)
    fig_development(results, epochs, res_dir / "E1_development.png")
    fig_diversity(results, res_dir / "E1_diversity.png")
    fig_sources(results, res_dir / "E1_sources.png")
    md = tables(results, epochs)
    (Path(args.outputs) / args.tag / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
