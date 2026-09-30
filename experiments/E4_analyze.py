"""E4 analysis: tables for results/E4.md and figures results/E4_*.png.

    uv run python -m experiments.E4_analyze
    uv run python -m experiments.E4_analyze --suffix _v03 --p-values 0.1,0.3,0.5,0.8

Reads outputs/E4_pprobe<suffix>/*.npz and outputs/E4_growth<suffix>/*.npz
written by experiments/E4.py; writes results/E4<suffix>_tables.md,
results/E4<suffix>_*.png and outputs/E4<suffix>_summary.json. Uncertainty: mean ± standard error across seeds for
per-run quantities; 95% cluster-bootstrap intervals (resampling seeds) for
pooled rates (hazards, probe success).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("outputs")
RES = Path("results")
ALL_P = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]   # full v0.2 grid; fixes ramp colours
P_VALUES = list(ALL_P)
KMAX_SHOW = 7
BOOT = 2000
RNG = np.random.default_rng(12345)   # analysis-only bootstrap stream

# Reference palette (dataviz skill, light mode).
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#1c5cab", "#0d366b"]   # ordinal blue


def load(tag: str, pattern: str) -> list[dict]:
    runs = []
    for f in sorted((OUT / tag).glob(pattern)):
        seed = int(re.search(r"_s(\d+)\.npz$", f.name).group(1))
        with np.load(f) as z:
            runs.append({"seed": seed, **{k: z[k] for k in z.files}})
    return sorted(runs, key=lambda r: r["seed"])


def fmt_p(p: float) -> str:
    return f"{p:g}"


# ---------------------------------------------------------------- (a) levels

def level_stats(runs: list[dict]) -> dict:
    final = np.array([np.bincount(r["levels"][-1], minlength=11)[:11] / r["levels"].shape[1] for r in runs])
    ge = {k: np.array([(r["levels"][-1] >= k).mean() for r in runs]) for k in (1, 2, 3, 4)}
    maxlvl = np.array([r["levels"][-1].max() for r in runs])
    mean = np.array([r["levels"][-1].mean() for r in runs])

    def first_epoch(r, k, frac):
        n = r["levels"].shape[1]
        hit = np.nonzero((r["levels"] >= k).sum(1) >= max(1, int(np.ceil(frac * n))))[0]
        return int(hit[0]) if len(hit) else None

    times = {}
    for k in (2, 3):
        for frac, name in ((0.0, "first"), (0.25, "25pct")):
            t = [first_epoch(r, k, frac) for r in runs]
            times[f"L{k}_{name}"] = t
    curves = {k: np.array([(r["levels"] >= k).mean(1) for r in runs]) for k in (1, 2, 3)}
    return {"final_dist": final, "ge": ge, "max": maxlvl, "mean": mean, "times": times, "curves": curves}


def se(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def time_summary(t: list) -> str:
    got = [x for x in t if x is not None]
    if not got:
        return f"never (0/{len(t)})"
    return f"{np.median(got):.0f} [{min(got)}–{max(got)}] ({len(got)}/{len(t)})"


# ---------------------------------------------------------------- (b) hazard

def hazard_counts(r: dict, kmax: int = KMAX_SHOW) -> np.ndarray:
    """Per level k: [mind-epochs at k, level-ups, mind-epochs paired with an
    equal-level mind, level-ups among those, multi-level jumps]."""
    L, P = r["levels"].astype(int), r["partner"].astype(int)
    E, n = P.shape
    c = np.zeros((kmax + 1, 5), dtype=np.int64)
    k = L[:-1]
    up = L[1:] > k
    jump = (L[1:] - k) > 1
    pl = np.where(P >= 0, np.take_along_axis(L[:-1], np.clip(P, 0, None), axis=1), -1)
    eq = (P >= 0) & (pl == k)
    for lvl in range(kmax + 1):
        m = k == lvl
        c[lvl] = [m.sum(), (up & m).sum(), (eq & m).sum(), (up & eq & m).sum(), (jump & m).sum()]
    return c


def pooled_rate_ci(counts: np.ndarray, num: int, den: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """counts: (seeds, K, cols). Pooled num/den per K with seed-bootstrap 95% CI."""
    tot = counts.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = tot[:, num] / tot[:, den]
        boots = []
        for _ in range(BOOT):
            s = counts[RNG.integers(len(counts), size=len(counts))].sum(0)
            boots.append(s[:, num] / s[:, den])
    b = np.array(boots)
    lo, hi = np.nanpercentile(b, 2.5, axis=0), np.nanpercentile(b, 97.5, axis=0)
    return est, lo, hi


# ---------------------------------------------------------------- (c) probes

def probe_table(runs: list[dict]) -> dict:
    """Pooled probe outcomes keyed by (level, flat len, n_offers); plus per-seed arrays for CIs."""
    keys = sorted({tuple(row[:3]) for r in runs for row in r["probe_len"]})
    per = np.zeros((len(runs), len(keys), 4), dtype=np.int64)
    for s, r in enumerate(runs):
        for row in r["probe_len"]:
            per[s, keys.index(tuple(row[:3]))] += row[3:]
    return {"keys": keys, "per": per}


def group_rate(pt: dict, keyfun) -> dict:
    groups: dict = {}
    for idx, key in enumerate(pt["keys"]):
        groups.setdefault(keyfun(key), []).append(idx)
    out = {}
    for g, idxs in sorted(groups.items()):
        c = pt["per"][:, idxs].sum(1)          # (seeds, 4)
        starts, succ = c[:, 0].sum(), c[:, 1].sum()
        boots = []
        for _ in range(BOOT):
            cc = c[RNG.integers(len(c), size=len(c))].sum(0)
            boots.append(cc[1] / cc[0] if cc[0] else np.nan)
        out[g] = {"starts": int(starts), "succ": int(succ), "rate": succ / starts if starts else np.nan,
                  "lo": float(np.nanpercentile(boots, 2.5)), "hi": float(np.nanpercentile(boots, 97.5)),
                  "unfinished": int(c[:, 3].sum())}
    return out


# ---------------------------------------------------------------- plotting

def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=TEXT2)
    ax.set_ylabel(ylabel, color=TEXT2)
    if title:
        ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2)


def main() -> None:
    global P_VALUES
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="appended to output tags and result file names")
    ap.add_argument("--p-values", default=",".join(fmt_p(p) for p in ALL_P))
    args = ap.parse_args()
    sfx = args.suffix
    P_VALUES = [float(x) for x in args.p_values.split(",")]
    RES.mkdir(exist_ok=True)
    sweep = {p: load(f"E4_pprobe{sfx}", f"p_probe{fmt_p(p)}_s*.npz") for p in P_VALUES}
    growth_none = load(f"E4_growth{sfx}", "growthnone_s*.npz")
    n_seeds = len(growth_none)
    summary: dict = {}
    md: list[str] = []

    # ---- (a)
    md.append(f"### (a) Final levels and time to level (epoch 500, 20 minds, {n_seeds} seeds)\n")
    md.append("| p_probe | mean level | ≥1 | ≥2 | ≥3 | ≥4 | max (median [range]) | first pair at L2 | 25% at L2 | first pair at L3 | contingency computable |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for p, runs in list(sweep.items()) + [("0.3, growth=none", growth_none)]:
        st = level_stats(runs)
        stats[p] = st
        ge = st["ge"]
        md.append(
            f"| {p if isinstance(p, str) else fmt_p(p)} | {st['mean'].mean():.2f} ± {se(st['mean']):.2f} | "
            + " | ".join(f"{100*ge[k].mean():.0f} ± {100*se(ge[k]):.0f}%" for k in (1, 2, 3, 4))
            + f" | {np.median(st['max']):.0f} [{st['max'].min()}–{st['max'].max()}]"
            + "".join(f" | {time_summary(st['times'][t])}" for t in ("L2_first", "L2_25pct", "L3_first"))
            + f" | {100*np.mean([np.nanmean(r['contingency_computable']) for r in runs]):.0f}%"
            + " |"
        )
        summary[str(p)] = {
            "n_seeds": len(runs),
            "mean_level": [float(st["mean"].mean()), se(st["mean"])],
            **{f"frac_ge{k}": [float(ge[k].mean()), se(ge[k])] for k in (1, 2, 3, 4)},
            "final_dist_mean": st["final_dist"].mean(0)[:7].round(4).tolist(),
            "times": st["times"],
        }
    md.append("\nTime cells: median epoch [range] over seeds that got there (seeds reaching / total).\n")

    # ---- (b)
    hz = {p: np.array([hazard_counts(r) for r in runs]) for p, runs in sweep.items()}
    hz["0.3, growth=none"] = np.array([hazard_counts(r) for r in growth_none])
    md.append("### (b) Per-epoch level-up hazard by current level k\n")
    md.append("h_eq = P(level up this epoch | paired with a mind at the same level); h_raw = P(level up this epoch | at level k). 95% seed-bootstrap CI.\n")
    md.append("| p_probe | k | mind-epochs at k | equal-level pairings | level-ups | h_raw | h_eq [95% CI] |")
    md.append("|---|---|---|---|---|---|---|")
    hz_sum = {}
    for p, c in hz.items():
        raw, _, _ = pooled_rate_ci(c, 1, 0)
        eqr, lo, hi = pooled_rate_ci(c, 3, 2)
        tot = c.sum(0)
        hz_sum[str(p)] = {"raw": raw.tolist(), "eq": eqr.tolist(), "lo": lo.tolist(), "hi": hi.tolist(),
                          "counts": tot.tolist()}
        for k in range(KMAX_SHOW + 1):
            if tot[k, 0] == 0:
                continue
            md.append(f"| {p if isinstance(p, str) else fmt_p(p)} | {k} | {tot[k,0]} | {tot[k,2]} | {tot[k,1]} | "
                      f"{raw[k]:.4f} | {eqr[k]:.4f} [{lo[k]:.4f}, {hi[k]:.4f}] |")
        # sanity: in bootstrap mode, every level-up happens with an equal-level partner
        assert (tot[:, 1] == tot[:, 3]).all(), (p, tot)
    summary["hazard"] = hz_sum
    summary["multi_level_jumps"] = {str(p): int(c.sum(0)[:, 4].sum()) for p, c in hz.items()}

    # ---- (c)
    md.append("\n### (c) Probe success (pooled over all runs with p_probe > 0)\n")
    all_runs = [r for p, runs in sweep.items() if p > 0 for r in runs]
    pt = probe_table(all_runs)
    by_level = group_rate(pt, lambda k: int(k[0]))
    by_len = group_rate(pt, lambda k: int(k[1]))
    by_off = group_rate(pt, lambda k: int(k[2]))
    by_lvl_off = group_rate(pt, lambda k: (int(k[0]), int(k[2])))
    by_lvl_len = group_rate(pt, lambda k: (int(k[0]), int(k[1])))
    for name, g in (("record level", by_level), ("flat word length", by_len), ("offer tokens in flat word", by_off)):
        md.append(f"\n**By {name}**\n\n| {name} | starts | successes | rate [95% CI] |\n|---|---|---|---|")
        for key, v in g.items():
            if v["starts"] < 200:
                continue
            md.append(f"| {key} | {v['starts']} | {v['succ']} | {100*v['rate']:.2f}% [{100*v['lo']:.2f}, {100*v['hi']:.2f}] |")
    md.append("\n**By record level × flat length** (cells with ≥ 200 starts)\n\n| level | flat length | starts | rate |\n|---|---|---|---|")
    for (lv, ln), v in by_lvl_len.items():
        if v["starts"] >= 200:
            md.append(f"| {lv} | {ln} | {v['starts']} | {100*v['rate']:.2f}% |")
    md.append("\n**By record level × offer tokens** (cells with ≥ 200 starts)\n\n| level | offer tokens | starts | rate |\n|---|---|---|---|")
    for (lv, no), v in by_lvl_off.items():
        if v["starts"] >= 200:
            md.append(f"| {lv} | {no} | {v['starts']} | {100*v['rate']:.2f}% |")

    # Token pass rates (per-token conditional survival)
    kinds = ("SELF_OFFER", "SELF_TAU", "SELF_ANY", "OTHER_OFFER", "OTHER_ANY")
    tp = sum(r["token_pass"] for r in all_runs)
    md.append("\n**Per-token pass rate** (probes that reached a token of this kind and got past it)\n\n| token | reached | pass rate |\n|---|---|---|")
    pass_rate = {}
    for i, k in enumerate(kinds):
        pass_rate[k] = 1 - tp[i, 1] / tp[i, 0] if tp[i, 0] else float("nan")
        md.append(f"| {k} | {tp[i,0]} | {100*pass_rate[k]:.1f}% |")
    pos = {}
    for r in all_runs:
        for p_, reached, failed in r["pos_pass"]:
            a = pos.setdefault(int(p_), [0, 0])
            a[0] += reached
            a[1] += failed
    md.append("\n**Pass rate by position in the flat word** (0 = first self token)\n\n| position | reached | pass rate |\n|---|---|---|")
    for p_ in sorted(pos):
        if pos[p_][0] >= 200:
            md.append(f"| {p_} | {pos[p_][0]} | {100*(1-pos[p_][1]/pos[p_][0]):.1f}% |")
    # word composition at end of run
    comp = {}
    for r in all_runs:
        for row in r["word_comp"]:
            c = comp.setdefault(int(row[0]), np.zeros(6, dtype=np.int64))
            c += row[1:]
    md.append("\n**Mean flat-word composition of stored records** (end of run, all p_probe > 0 runs)\n\n| level | records | length | SELF_OFFER | SELF_TAU | SELF_ANY | OTHER_OFFER | OTHER_ANY |\n|---|---|---|---|---|---|---|---|")
    for lv in sorted(comp):
        c = comp[lv]
        md.append(f"| {lv} | {c[0]} | {c[1:].sum()/c[0]:.2f} | " + " | ".join(f"{x/c[0]:.2f}" for x in c[1:]) + " |")
    summary["probe"] = {"by_level": by_level, "by_len": by_len, "by_offers": by_off,
                        "token_pass": pass_rate, "pos_pass": pos}
    # Geometric model: success ≈ q^n_offers, q = geometric mean of the two offer pass rates
    q = float(np.sqrt(pass_rate["SELF_OFFER"] * pass_rate["OTHER_OFFER"]))
    summary["probe"]["q_geo"] = q
    md.append(f"\nGeometric model: q = √(pass(SELF_OFFER)·pass(OTHER_OFFER)) = {q:.3f}; predicted success q^n.\n")
    # Alternative: the first two tokens (positions 0, 1) are the filter; each
    # further offer costs a factor r, fitted by weighted least squares in log space.
    p01 = (1 - pos[0][1] / pos[0][0]) * (1 - pos[1][1] / pos[1][0])
    ns = np.array([no for no, v in by_off.items() if v["starts"] >= 200 and v["succ"] > 0])
    ys = np.array([np.log(by_off[no]["rate"] / p01) for no in ns])
    ws = np.array([by_off[no]["succ"] for no in ns], dtype=float)
    r_fit = float(np.exp(np.sum(ws * ys * (ns - 2)) / np.sum(ws * (ns - 2) ** 2)))
    summary["probe"]["p01"] = p01
    summary["probe"]["r_fit"] = r_fit
    md.append(f"First-exchange model: pass(pos 0) × pass(pos 1) = {100*p01:.2f}%, fitted r = {r_fit:.3f} per further offer.\n")
    md.append("| offer tokens | observed | independent q^n | first-exchange p01·r^(n−2) |\n|---|---|---|---|")
    for no, v in by_off.items():
        if v["starts"] >= 200:
            md.append(f"| {no} | {100*v['rate']:.2f}% | {100*q**no:.3f}% | {100*p01*r_fit**(no-2):.2f}% |")

    # Probe success by record level for each p_probe (does it depend on p_probe?)
    md.append("\n**Probe success by record level, per p_probe**\n\n| p_probe | L1 | L2 | L3 | L4 |\n|---|---|---|---|---|")
    for p, runs in sweep.items():
        if p == 0:
            continue
        g = group_rate(probe_table(runs), lambda k: int(k[0]))
        md.append(f"| {fmt_p(p)} | " + " | ".join(
            f"{100*g[k]['rate']:.1f}% (n={g[k]['starts']})" if k in g else "–" for k in (1, 2, 3, 4)) + " |")

    (RES / f"E4{sfx}_tables.md").write_text("\n".join(md) + "\n")
    (OUT / f"E4{sfx}_summary.json").write_text(json.dumps(summary, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("\n".join(md))

    # ================================================================ figures
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.dpi": 150})

    # Fig 1: development vs p_probe
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for i, p in enumerate(P_VALUES):
        cur = stats[p]["curves"][2].mean(0)
        ax.plot(np.arange(len(cur)), 100 * cur, color=RAMP[ALL_P.index(p)], lw=2, label=f"{fmt_p(p)}")
    style(ax, "epoch", "minds at level ≥ 2 (%)", "Share of minds at level ≥ 2 over time")
    ax.legend(title="p_probe", frameon=False, fontsize=8, title_fontsize=8, ncol=2, loc="center right")
    ax = axes[1]
    for j, k in enumerate((1, 2, 3)):
        m = [100 * stats[p]["ge"][k].mean() for p in P_VALUES]
        e = [100 * se(stats[p]["ge"][k]) for p in P_VALUES]
        ax.errorbar(P_VALUES, m, yerr=e, color=SERIES[j], lw=2, marker="o", ms=5, capsize=3, label=f"level ≥ {k}")
        ax.annotate(f"level ≥ {k}", (P_VALUES[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                    color=TEXT2, va="center", fontsize=9)
    style(ax, "p_probe (probability of starting a probe on a turn)", "minds at epoch 500 (%)",
          f"Final level reached vs probe rate (mean ± SE, {n_seeds} seeds)")
    ax.set_xlim(-0.03, 0.95)
    fig.tight_layout()
    fig.savefig(RES / f"E4{sfx}_development.png")
    plt.close(fig)

    # Fig 2: hazard by level
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    show = [(0.1, SERIES[0]), (0.3, SERIES[1]), (0.8, SERIES[2]), ("0.3, growth=none", SERIES[3])]
    offs = np.linspace(-0.15, 0.15, len(show))
    for (p, col), dx in zip(show, offs):
        h = hz_sum[str(p)]
        cnt = np.array(h["counts"])
        ks = [k for k in range(KMAX_SHOW + 1) if cnt[k, 2] >= 30]
        est = np.array(h["eq"])[ks]
        lo, hi = np.array(h["lo"])[ks], np.array(h["hi"])[ks]
        label = f"p_probe = {p}" if not isinstance(p, str) else "p_probe = 0.3, growth = none"
        ax.errorbar(np.array(ks) + dx, 100 * est, yerr=[100 * (est - lo), 100 * (hi - est)], color=col,
                    marker="o", ms=6, lw=2, capsize=3, label=label)
    style(ax, "current level k", "level-up probability per epoch (%)\ngiven an equal-level partner",
          "Hazard of k → k+1 by current level")
    ax.set_yscale("log")
    ax.set_xticks(range(KMAX_SHOW + 1))
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.text(0.01, 0.02, f"95% seed-bootstrap CIs ({n_seeds} seeds); levels with ≥ 30 equal-level pairings",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=TEXT2)
    fig.tight_layout()
    fig.savefig(RES / f"E4{sfx}_hazard.png")
    plt.close(fig)

    # Fig 3: probe success vs offer tokens
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for j, lv in enumerate((1, 2, 3, 4)):
        pts = [(no, v) for (l_, no), v in by_lvl_off.items() if l_ == lv and v["starts"] >= 200]
        if not pts:
            continue
        x = np.array([no for no, _ in pts]) + (j - 1.5) * 0.06
        y = np.array([v["rate"] for _, v in pts])
        lo = np.array([v["lo"] for _, v in pts])
        hi = np.array([v["hi"] for _, v in pts])
        ax.errorbar(x, 100 * y, yerr=[100 * (y - lo), 100 * (hi - y)], color=SERIES[j], marker="o", ms=6,
                    lw=0, elinewidth=1.5, capsize=3, label=f"level-{lv} record")
    xs = np.arange(2, 14)
    ax.plot(xs, 100 * q ** xs, color=TEXT2, lw=1.5, ls=":",
            label=f"independent offers: q^n, q = {q:.2f}")
    style(ax, "offer tokens in the flattened word (n)", "probe success rate (%)",
          "Probe success by offer tokens and record level")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 100)
    ax.set_xlim(1.5, 13.5)
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, 0.02, f"pooled over all p_probe > 0 runs ({len(all_runs)}); 95% seed-bootstrap CIs; cells ≥ 200 probes",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=TEXT2)
    fig.tight_layout()
    fig.savefig(RES / f"E4{sfx}_probe_success.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
