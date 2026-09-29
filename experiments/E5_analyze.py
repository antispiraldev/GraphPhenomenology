"""E5 analysis: reads outputs/E5/<condition>/s*.json, writes outputs/E5/summary.json
and the figures results/E5_*.png.

    uv run python -m experiments.E5_analyze

Uncertainty is a cluster bootstrap over seeds (each seed is one population);
intervals are 95% percentile intervals from 2000 resamples.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.E5 import OUT

CONDS = ["pairs", "shuffled", "er02"]
NAMES = {"pairs": "Fixed pairs", "shuffled": "Shuffled (ER p=1)", "er02": "Default (ER p=0.2)"}
COLORS = {"pairs": "#2a78d6", "shuffled": "#eb6834", "er02": "#1baf7a"}
CLASSES = ["cobuilt_top", "past", "stranger"]
RESULTS = Path("results")
B = 2000


def load() -> dict[str, list[dict]]:
    return {c: [json.loads(p.read_text()) for p in sorted((OUT / c).glob("s*.json"),
                                                            key=lambda p: int(p.stem[1:]))]
            for c in CONDS}


def boot(runs: list, stat, rng) -> tuple[float, float, float]:
    """Point estimate and 95% interval of stat(list of runs), resampling runs."""
    est = stat(runs)
    vals = []
    for _ in range(B):
        idx = rng.integers(len(runs), size=len(runs))
        v = stat([runs[i] for i in idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return est, float(lo), float(hi)


def fmt(t) -> str:
    return f"{t[0]:.3g} [{t[1]:.3g}, {t[2]:.3g}]"


# ---- per-run extractors ----------------------------------------------------

def probe_table(run: dict, equal_only=True, top_only=True) -> dict[tuple[str, int], np.ndarray]:
    """(class, level) -> [starts, successes, co_starts, co_successes]."""
    out: dict = defaultdict(lambda: np.zeros(4))
    for key, v in run["probes"].items():
        cls, ki, kj, rl = key.split("|")
        ki, kj, rl = int(ki), int(kj), int(rl)
        if equal_only and ki != kj:
            continue
        if top_only and rl != ki:
            continue
        out[(cls, ki)] += np.array(v, dtype=float)
    return out


def enc_table(run: dict, equal_only=True) -> dict[tuple[str, int], np.ndarray]:
    """(class, level) -> [encounters, level-ups]."""
    out: dict = defaultdict(lambda: np.zeros(2))
    for key, v in run["encounters"].items():
        cls, ki, kj = key.split("|")
        if equal_only and ki != kj:
            continue
        out[(cls, int(ki))] += np.array(v, dtype=float)
    return out


def pooled(tables: list[dict], cls: str, col_n: int, col_k: int, levels=None) -> float | None:
    n = k = 0.0
    for t in tables:
        for (c, lvl), v in t.items():
            if c == cls and (levels is None or lvl in levels):
                n += v[col_n]
                k += v[col_k]
    return k / n if n else None


def standardized_ratio(tables: list[dict], a: str, b: str, col_n: int, col_k: int, levels) -> float | None:
    """Observed events of class a / events expected if class a had class b's
    per-level rate (indirect standardization over level)."""
    agg: dict = defaultdict(lambda: np.zeros(2))
    for t in tables:
        for (c, lvl), v in t.items():
            if lvl in levels:
                agg[(c, lvl)] += v[[col_n, col_k]]
    obs = exp = 0.0
    for lvl in levels:
        na, ka = agg[(a, lvl)]
        nb, kb = agg[(b, lvl)]
        if na == 0 or nb == 0:
            continue
        obs += ka
        exp += na * kb / nb
    return obs / exp if exp else None


# ---- main --------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(0)
    data = load()
    S: dict = {"n_seeds": {c: len(v) for c, v in data.items()}}
    epochs = len(data["pairs"][0]["rows"])

    # 1. level distributions and rates
    lv = {}
    for c, runs in data.items():
        d = {}
        for thr in (1, 2, 3, 4):
            d[f"frac_ge{thr}"] = boot(runs, lambda rs, t=thr: float(np.mean([np.mean(np.array(r["final_levels"]) >= t) for r in rs])), rng)
        d["mean_level"] = boot(runs, lambda rs: float(np.mean([np.mean(r["final_levels"]) for r in rs])), rng)
        d["max_level_mean"] = boot(runs, lambda rs: float(np.mean([max(r["final_levels"]) for r in rs])), rng)
        n_agents = len(runs[0]["final_levels"])
        d["levelups_per_agent_epoch"] = boot(runs, lambda rs: float(np.mean([
            sum(x["levelups_bootstrap"] + x["levelups_elder"] for x in r["rows"]) / (n_agents * epochs) for r in rs])), rng)
        d["encounters_per_agent_epoch"] = float(np.mean([
            sum(v[0] for v in r["encounters"].values()) / (n_agents * epochs) for r in runs]))
        d["levelups_per_encounter"] = boot(runs, lambda rs: sum(sum(v[1] for v in r["encounters"].values()) for r in rs)
                                           / sum(sum(v[0] for v in r["encounters"].values()) for r in rs), rng)
        d["frac_encounters_equal_level"] = float(sum(v[0] for r in runs for k, v in r["encounters"].items()
                                                     if k.split("|")[1] == k.split("|")[2])
                                               / sum(v[0] for r in runs for v in r["encounters"].values()))
        # repeat-partner fraction over the whole run
        def rep(rs):
            num = den = 0.0
            for r in rs:
                for x in r["rows"]:
                    ups = x["levelups_bootstrap"] + x["levelups_elder"]
                    if ups and x["repeat_partner_frac"] == x["repeat_partner_frac"]:
                        num += x["repeat_partner_frac"] * ups
                        den += ups
            return num / den if den else None
        d["repeat_partner_frac"] = boot(runs, rep, rng)
        d["final_level_hist"] = np.bincount(np.concatenate([r["final_levels"] for r in runs]), minlength=11).tolist()
        d["seconds_per_run"] = float(np.mean([r["seconds"] for r in runs]))
        lv[c] = d
    S["levels"] = lv

    # 2. per equal-level encounter: level-up probability by starting level
    S["levelup_per_equal_encounter_by_level"] = {}
    for c, runs in data.items():
        tabs = [enc_table(r) for r in runs]
        row = {}
        for k in range(0, 6):
            n = sum(v[0] for t in tabs for (cl, l), v in t.items() if l == k)
            u = sum(v[1] for t in tabs for (cl, l), v in t.items() if l == k)
            if n:
                row[k] = {"encounters": int(n), "levelups": int(u),
                          "rate": boot(tabs, lambda ts, k=k: (lambda n_, u_: u_ / n_ if n_ else None)(
                              sum(v[0] for t in ts for (cl, l), v in t.items() if l == k),
                              sum(v[1] for t in ts for (cl, l), v in t.items() if l == k)), rng)}
        S["levelup_per_equal_encounter_by_level"][c] = row

    # 3. key test: probe success of the top record against an equal-level partner, by partner class
    key = {}
    for c, runs in data.items():
        ptabs = [probe_table(r) for r in runs]
        etabs = [enc_table(r) for r in runs]
        levels = sorted({l for t in ptabs for (_, l) in t if l >= 1})
        d = {"by_class_level": {}}
        for cls in CLASSES:
            for l in levels:
                st = sum(t[(cls, l)][0] for t in ptabs if (cls, l) in t)
                su = sum(t[(cls, l)][1] for t in ptabs if (cls, l) in t)
                en = sum(t[(cls, l)][0] for t in etabs if (cls, l) in t)
                up = sum(t[(cls, l)][1] for t in etabs if (cls, l) in t)
                if st or en:
                    d["by_class_level"][f"{cls}|{l}"] = {"probe_starts": int(st), "probe_successes": int(su),
                                                        "success_rate": su / st if st else None,
                                                        "encounters": int(en), "levelups": int(up),
                                                        "levelup_rate": up / en if en else None}
        for cls in CLASSES:
            d[f"probe_success_{cls}"] = boot(ptabs, lambda ts, cl=cls: pooled(ts, cl, 0, 1, set(levels)), rng)
            d[f"coprobe_frac_success_{cls}"] = boot(ptabs, lambda ts, cl=cls: pooled(ts, cl, 1, 3, set(levels)), rng)
            d[f"coprobe_frac_start_{cls}"] = boot(ptabs, lambda ts, cl=cls: pooled(ts, cl, 0, 2, set(levels)), rng)
        d["ratio_probe_success_cobuilt_vs_stranger_std"] = boot(
            ptabs, lambda ts: standardized_ratio(ts, "cobuilt_top", "stranger", 0, 1, levels), rng)
        d["ratio_levelup_cobuilt_vs_stranger_std"] = boot(
            etabs, lambda ts: standardized_ratio(ts, "cobuilt_top", "stranger", 0, 1, [l for l in levels]), rng)
        # probe success with a co-probing partner vs not (all classes)
        def co_rates(ts):
            st = sum(v[0] for t in ts for v in t.values()); su = sum(v[1] for t in ts for v in t.values())
            cst = sum(v[2] for t in ts for v in t.values()); csu = sum(v[3] for t in ts for v in t.values())
            return csu / cst if cst else None, (su - csu) / (st - cst) if st - cst else None
        d["success_when_partner_probing"] = boot(ptabs, lambda ts: co_rates(ts)[0], rng)
        d["success_when_partner_not_probing"] = boot(ptabs, lambda ts: co_rates(ts)[1], rng)
        d["levels"] = levels
        key[c] = d
    # across conditions: fixed-pair partner vs a stranger in the shuffled population, same level
    def cross(pair_tabs, shuf_tabs, levels):
        obs = exp = 0.0
        for l in levels:
            na = sum(t[("cobuilt_top", l)][0] for t in pair_tabs if ("cobuilt_top", l) in t)
            ka = sum(t[("cobuilt_top", l)][1] for t in pair_tabs if ("cobuilt_top", l) in t)
            nb = sum(t[("stranger", l)][0] for t in shuf_tabs if ("stranger", l) in t)
            kb = sum(t[("stranger", l)][1] for t in shuf_tabs if ("stranger", l) in t)
            if na and nb:
                obs += ka; exp += na * kb / nb
        return obs / exp if exp else None
    pt = [probe_table(r) for r in data["pairs"]]
    stt = [probe_table(r) for r in data["shuffled"]]
    both = list(zip(pt, stt))
    S["cross_ratio_probe_pairs_vs_shuffled_stranger"] = boot(
        both, lambda bs: cross([b[0] for b in bs], [b[1] for b in bs], range(1, 11)), rng)
    et = [enc_table(r) for r in data["pairs"]]
    se = [enc_table(r) for r in data["shuffled"]]
    both_e = list(zip(et, se))
    S["cross_ratio_levelup_pairs_vs_shuffled_stranger"] = boot(
        both_e, lambda bs: cross([b[0] for b in bs], [b[1] for b in bs], range(1, 11)), rng)
    S["key_test"] = key

    # 4. record overlap, co-builders vs other pairs, per level
    ov = {}
    names = ["offered_label_jaccard_extras", "own_label_jaccard_flat", "own_vs_partner_complement_jaccard_flat",
             "interlock_matches_extras", "identical_flat_word"]
    for c, runs in data.items():
        per = {}
        for lvl in ("1", "2", "3"):
            for grp in ("cobuilt", "other"):
                rows = [x for r in runs for x in r["record_overlap"].get(lvl, {}).get(grp, [])]
                if not rows:
                    continue
                arr = np.array([[np.nan if v is None else float(v) for v in x] for x in rows])
                per[f"L{lvl}_{grp}"] = {"n": len(rows), **{n: float(np.nanmean(arr[:, i])) for i, n in enumerate(names)},
                                        "frac_interlock_ge1": float(np.mean(arr[:, 3] >= 1)),
                                        "frac_interlock_ge2": float(np.mean(arr[:, 3] >= 2))}
        so = np.array([r["segment_overlap"] for r in runs]).sum(0)
        per["segment_overlap_frac"] = float(so[0] / so[1]) if so[1] else None
        per["segment_overlap_n"] = int(so[1])
        ov[c] = per
    S["record_overlap"] = ov

    (OUT / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    figures(data, S)
    print(json.dumps(S, indent=1, default=float))


def figures(data, S) -> None:
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: mean level over time and final distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    for c in CONDS:
        curves = np.array([[x["mean_level"] for x in r["rows"]] for r in data[c]])
        m = curves.mean(0)
        se = curves.std(0, ddof=1) / np.sqrt(len(curves))
        ep = np.arange(len(m))
        ax1.fill_between(ep, m - 1.96 * se, m + 1.96 * se, color=COLORS[c], alpha=0.18, linewidth=0)
        ax1.plot(ep, m, color=COLORS[c], lw=2, label=NAMES[c])
        ax1.annotate(NAMES[c], (ep[-1], m[-1]), xytext=(4, 0), textcoords="offset points",
                     va="center", fontsize=9, color="#333333")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean level of the 20 minds")
    ax1.set_title("Mean level over time (band: 95% CI over 30 seeds)", fontsize=11)
    ax1.set_xlim(0, len(m) * 1.22)
    ax1.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax1.legend(frameon=False, loc="upper left")
    width = 0.27
    for n, c in enumerate(CONDS):
        h = np.array(S["levels"][c]["final_level_hist"], float)
        h = h / h.sum()
        ks = np.arange(len(h))
        ax2.bar(ks + (n - 1) * width, h, width=width - 0.03, color=COLORS[c], label=NAMES[c])
    ax2.set_xlabel("Final level (epoch 500)")
    ax2.set_ylabel("Share of minds")
    ax2.set_title("Final level distribution (600 minds per condition)", fontsize=11)
    ax2.set_xticks(range(11))
    ax2.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "E5_levels.png", dpi=150)
    plt.close(fig)

    # Figure 2: key test — per-attempt probe success and per-encounter level-up, by partner, per level
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    groups = [("pairs", "cobuilt_top", "Fixed pair: own partner"),
              ("shuffled", "cobuilt_top", "Shuffled: co-builder of top record"),
              ("shuffled", "stranger", "Shuffled: stranger")]
    gcol = ["#2a78d6", "#eb6834", "#1baf7a"]
    for ax, metric, ylabel in ((axes[0], "probe", "Probe success rate (top record)"),
                               (axes[1], "levelup", "Level-ups per encounter")):
        for g, (c, cls, label) in enumerate(groups):
            xs, ys, lo, hi = [], [], [], []
            for lvl in range(1, 6):
                runs = data[c]
                tabs = [probe_table(r) if metric == "probe" else enc_table(r) for r in runs]
                n = sum(t[(cls, lvl)][0] for t in tabs if (cls, lvl) in t)
                if n < 200:
                    continue
                k = sum(t[(cls, lvl)][1] for t in tabs if (cls, lvl) in t)
                rng = np.random.default_rng(lvl)
                bs = []
                for _ in range(1000):
                    idx = rng.integers(len(tabs), size=len(tabs))
                    nn = sum(tabs[i][(cls, lvl)][0] for i in idx if (cls, lvl) in tabs[i])
                    kk = sum(tabs[i][(cls, lvl)][1] for i in idx if (cls, lvl) in tabs[i])
                    if nn:
                        bs.append(kk / nn)
                xs.append(lvl + (g - 1) * 0.18); ys.append(k / n)
                lo.append(k / n - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - k / n)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=8, color=gcol[g], capsize=3, lw=1.5, label=label)
        ax.set_xlabel("Level of both minds at the start of the encounter")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 6))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    axes[0].set_title("Per probe attempt, equal-level partner", fontsize=11)
    axes[1].set_title("Per encounter, equal-level partner", fontsize=11)
    axes[0].legend(frameon=False, fontsize=9)
    fig.text(0.5, -0.01, "Points with fewer than 200 attempts/encounters are omitted. Bars: 95% CI, bootstrap over seeds.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout()
    fig.savefig(RESULTS / "E5_partner_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
