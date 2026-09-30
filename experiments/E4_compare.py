"""E4 v0.2 vs v0.3 comparison and extra diagnostics (G4c).

    uv run python -m experiments.E4_compare --v02 <dir with E4_pprobe/ and E4_growth/> \
        --v03 outputs --suffix _v03

Reads the npz files written by experiments/E4.py for both spec versions and
reports, on the common grid (p_probe 0.1, 0.3, 0.5, 0.8 and growth=none at 0.3):
  1. final level shares and h_eq(k) side by side;
  2. a *fixed-cohort* hazard: minds that finished at level >= K, their
     equal-level hazard at each k < K. The same minds contribute at every k,
     so between-mind selection cannot make the hazard rise with k;
  3. probe success by record level, pooled over the common grid;
  4. repeat-partner share of level-ups beyond level 1 (as experiments/E4_repeat);
  5. share of level-ups after epoch 100 (does development freeze early?).
Writes results/E4<suffix>_compare.md, results/E4<suffix>_compare.png and
outputs/E4<suffix>_compare.json. 95% intervals resample seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.E4_analyze import SERIES, TEXT2, hazard_counts, style  # noqa: E402

P_COMMON = [0.1, 0.3, 0.5, 0.8]
BOOT = 2000
RNG = np.random.default_rng(2024)
KMAX = 7


def arms(root: Path, sfx: str) -> dict:
    out = {}
    for p in P_COMMON:
        out[f"{p:g}"] = load_dir(root / f"E4_pprobe{sfx}", f"p_probe{p:g}_s*.npz")
    out["none"] = load_dir(root / f"E4_growth{sfx}", "growthnone_s*.npz")
    return out


def load_dir(d: Path, pattern: str) -> list[dict]:
    runs = []
    for f in d.glob(pattern):
        with np.load(f) as z:
            runs.append({"seed": int(f.stem.rsplit("_s", 1)[1]), **{k: z[k] for k in z.files}})
    return sorted(runs, key=lambda r: r["seed"])


def boot_ratio(num: np.ndarray, den: np.ndarray) -> tuple[float, float, float]:
    """num, den: per-seed arrays. Pooled ratio with seed-bootstrap 95% CI."""
    if den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    b = []
    for _ in range(BOOT):
        i = RNG.integers(len(num), size=len(num))
        b.append(num[i].sum() / den[i].sum() if den[i].sum() else np.nan)
    return float(num.sum() / den.sum()), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def cohort_counts(r: dict, K: int) -> np.ndarray:
    """For minds with final level >= K: per k < K, [equal-level pairings at k, level-ups from k, epochs at k]."""
    L, P = r["levels"].astype(int), r["partner"].astype(int)
    keep = L[-1] >= K
    k = L[:-1]
    up = L[1:] > k
    pl = np.where(P >= 0, np.take_along_axis(L[:-1], np.clip(P, 0, None), axis=1), -1)
    eq = (P >= 0) & (pl == k)
    c = np.zeros((K, 3), dtype=np.int64)
    for lvl in range(K):
        m = (k == lvl) & keep[None, :]
        c[lvl] = [(eq & m).sum(), (up & eq & m).sum(), m.sum()]
    return c


def repeat_partner(runs: list[dict]) -> tuple[int, int]:
    same = total = 0
    for r in runs:
        L, P = r["levels"].astype(int), r["partner"].astype(int)
        for i in range(L.shape[1]):
            prev = None
            for e in np.nonzero(L[1:, i] > L[:-1, i])[0]:
                if L[e, i] >= 1 and prev is not None:
                    total += 1
                    same += int(P[e, i] == prev)
                prev = P[e, i]
    return same, total


def late_share(runs: list[dict], after: int = 100) -> np.ndarray:
    """Per seed: fraction of all level-steps that happen in epochs >= after."""
    out = []
    for r in runs:
        d = np.diff(r["levels"].astype(int), axis=0).clip(min=0)
        out.append(d[after:].sum() / d.sum() if d.sum() else np.nan)
    return np.array(out)


def probe_by_level(runs: list[dict]) -> dict:
    per: dict[int, np.ndarray] = {}
    for s, r in enumerate(runs):
        for row in r["probe_len"]:
            a = per.setdefault(int(row[0]), np.zeros((len(runs), 2), dtype=np.int64))
            a[s] += row[3:5]
    return {lv: boot_ratio(a[:, 1], a[:, 0]) + (int(a[:, 0].sum()),) for lv, a in sorted(per.items())}


def pct(x: tuple, d: int = 1) -> str:
    return f"{100*x[0]:.{d}f} [{100*x[1]:.{d}f}, {100*x[2]:.{d}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v02", required=True)
    ap.add_argument("--v03", default="outputs")
    ap.add_argument("--suffix", default="_v03")
    args = ap.parse_args()
    data = {"v0.2": arms(Path(args.v02), ""), "v0.3": arms(Path(args.v03), args.suffix)}
    md, js = [], {}

    # 1. levels + hazard
    md.append("### 1. Final level shares (mean ± SE over seeds) and h_eq(k) [95% CI], %\n")
    md.append("| arm | version | seeds | mean level | ≥2 | ≥3 | ≥5 | max (median) | h 0→1 | h 1→2 | h 2→3 | h 3→4 | h 4→5 | h 5→6 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    hz_all = {}
    for arm in list(data["v0.2"]):
        for ver in data:
            runs = data[ver][arm]
            fin = np.array([r["levels"][-1] for r in runs]).astype(int)
            def ms(x):
                return f"{100*x.mean():.0f} ± {100*x.std(ddof=1)/np.sqrt(len(x)):.0f}"
            c = np.array([hazard_counts(r, KMAX) for r in runs])   # (seeds, K, 5)
            hz = [boot_ratio(c[:, k, 3], c[:, k, 2]) for k in range(KMAX + 1)]
            hz_all[(arm, ver)] = (hz, c.sum(0)[:, 2])
            cells = []
            for k in range(6):
                n_eq = c.sum(0)[k, 2]
                cells.append("–" if n_eq < 30 else (pct(hz[k], 2) if hz[k][0] < 0.01 else pct(hz[k], 1)))
            mean_lvl = fin.mean(1)
            md.append(f"| {arm} | {ver} | {len(runs)} | {mean_lvl.mean():.2f} ± {mean_lvl.std(ddof=1)/np.sqrt(len(runs)):.2f} | "
                      f"{ms((fin >= 2).mean(1))} | {ms((fin >= 3).mean(1))} | {ms((fin >= 5).mean(1))} | "
                      f"{np.median(fin.max(1)):.0f} | " + " | ".join(cells) + " |")
            js.setdefault(arm, {})[ver] = {
                "ge2": float((fin >= 2).mean()), "ge3": float((fin >= 3).mean()), "ge5": float((fin >= 5).mean()),
                "hazard": [list(h) for h in hz], "eq_pairings": c.sum(0)[:, 2].tolist(),
                "level0": float((fin == 0).mean()),
            }
    md.append("\nArms are p_probe values under growth = generalize; `none` is growth = none at p_probe = 0.3. "
              "Hazard cells with < 30 equal-level pairings are omitted.\n")

    # 2. fixed cohort
    md.append("### 2. Fixed-cohort hazard: minds that finished at level ≥ K, h_eq(k) for k < K, % [95% CI]\n")
    md.append("Same minds at every k, so a rise with k is not between-mind selection. Pooled over p_probe 0.3, 0.5, 0.8 (growth = generalize).\n")
    md.append("| version | K | cohort minds | k=1 | k=2 | k=3 | k=4 | mean epochs at k=1 / 2 / 3 / 4 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ver in data:
        runs = [r for p in ("0.3", "0.5", "0.8") for r in data[ver][p]]
        for K in (3, 5):
            c = np.array([cohort_counts(r, K) for r in runs])
            nmind = int(sum((r["levels"][-1] >= K).sum() for r in runs))
            cells = []
            for k in range(1, 5):
                cells.append(pct(boot_ratio(c[:, k, 1], c[:, k, 0]), 1) if k < K and c[:, k, 0].sum() else "–")
            soj = " / ".join(f"{c[:, k, 2].sum()/nmind:.0f}" if k < K and nmind else "–" for k in range(1, 5))
            md.append(f"| {ver} | {K} | {nmind} | " + " | ".join(cells) + f" | {soj} |")
            js.setdefault("cohort", {})[f"{ver}_K{K}"] = {"minds": nmind, "counts": c.sum(0).tolist()}
    md.append("")

    # 3. probe success
    md.append("### 3. Probe success by record level, pooled over p_probe 0.1–0.8 (generalize), % [95% CI]\n")
    md.append("| version | L1 | L2 | L3 | L4 | L5 | L6+ (pooled) |")
    md.append("|---|---|---|---|---|---|---|")
    pb = {}
    for ver in data:
        runs = [r for p in ("0.1", "0.3", "0.5", "0.8") for r in data[ver][p]]
        g = probe_by_level(runs)
        pb[ver] = g
        # L6+ pooled
        per = np.zeros((len(runs), 2), dtype=np.int64)
        for s, r in enumerate(runs):
            for row in r["probe_len"]:
                if row[0] >= 6:
                    per[s] += row[3:5]
        cells = [pct(g[lv][:3], 2) + f" (n={g[lv][3]/1e3:.0f}k)" if lv in g else "–" for lv in (1, 2, 3, 4, 5)]
        cells.append(pct(boot_ratio(per[:, 1], per[:, 0]), 2) if per[:, 0].sum() else "–")
        md.append(f"| {ver} | " + " | ".join(cells) + " |")
        js.setdefault("probe_by_level", {})[ver] = {str(k): list(v) for k, v in g.items()}
    md.append("")

    # 4/5. repeat partner, late level-ups, stranded
    md.append("### 4. Same partner as the previous level-up (level-ups from k ≥ 1), and late development\n")
    md.append("| arm | version | same partner | level-steps after epoch 100 (mean ± SE over seeds) | minds left at level 0 |")
    md.append("|---|---|---|---|---|")
    for arm, lab in (("gen", "generalize, p_probe 0.1–0.8"), ("none", "none, p_probe 0.3")):
        for ver in data:
            runs = [r for p in ("0.1", "0.3", "0.5", "0.8") for r in data[ver][p]] if arm == "gen" else data[ver]["none"]
            s, t = repeat_partner(runs)
            ls = late_share(runs)
            l0 = np.array([(r["levels"][-1] == 0).mean() for r in runs])
            md.append(f"| {lab} | {ver} | {s}/{t} = {100*s/max(t,1):.0f}% | {100*np.nanmean(ls):.0f} ± {100*np.nanstd(ls, ddof=1)/np.sqrt(len(ls)):.0f}% | "
                      f"{100*l0.mean():.0f} ± {100*l0.std(ddof=1)/np.sqrt(len(l0)):.0f}% |")
            js.setdefault("repeat", {})[f"{arm}_{ver}"] = [s, t, float(np.nanmean(ls)), float(l0.mean())]
    md.append("")

    # growth=none is untouched by the v0.3 change (no nodes are grown): check hash-identical levels.
    same = all(np.array_equal(a["levels"], b["levels"]) for a, b in zip(data["v0.2"]["none"], data["v0.3"]["none"]))
    md.append(f"growth = none arm identical level trajectories in v0.2 and v0.3: **{same}** (expected: no growth, no change).\n")
    js["none_identical"] = same

    Path(f"results/E4{args.suffix}_compare.md").write_text("\n".join(md) + "\n")
    Path(f"outputs/E4{args.suffix}_compare.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md))

    # ---- figure: hazard by level, v0.2 vs v0.3 (p=0.3 and 0.5, and none); probe success by level
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    show = [("0.3", SERIES[1]), ("0.5", SERIES[0]), ("none", SERIES[3])]
    for (arm, col), dx in zip(show, (-0.12, 0, 0.12)):
        for ver, ls_, mk in (("v0.2", ":", "o"), ("v0.3", "-", "s")):
            hz, neq = hz_all[(arm, ver)]
            ks = [k for k in range(KMAX + 1) if neq[k] >= 30]
            est = np.array([hz[k][0] for k in ks])
            lo, hi = np.array([hz[k][1] for k in ks]), np.array([hz[k][2] for k in ks])
            fill = col if ver == "v0.3" else "white"
            name = f"p_probe {arm}" if arm != "none" else "growth none"
            ax.errorbar(np.array(ks) + dx, 100 * est, yerr=[100 * (est - lo), 100 * (hi - est)], color=col, ls=ls_,
                        marker=mk, mfc=fill, ms=6, lw=1.8, capsize=3, label=f"{name}, {ver}")
    style(ax, "current level k", "level-up probability per epoch (%)\ngiven an equal-level partner",
          "Hazard of k → k+1")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=7.5, ncol=1, loc="upper right")
    ax = axes[1]
    for ver, col, mk in (("v0.2", TEXT2, "o"), ("v0.3", SERIES[0], "s")):
        g = pb[ver]
        lv = [k for k in sorted(g) if g[k][3] >= 1000]
        est = np.array([g[k][0] for k in lv])
        lo, hi = np.array([g[k][1] for k in lv]), np.array([g[k][2] for k in lv])
        ax.errorbar(lv, 100 * est, yerr=[100 * (est - lo), 100 * (hi - est)], color=col, marker=mk, ms=6, lw=1.8,
                    capsize=3, label=ver)
    style(ax, "record level", "probe success (%)", "Probe success by record level")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    xs = np.arange(len(P_COMMON) + 1)
    for j, (ver, col) in enumerate((("v0.2", "#9a9893"), ("v0.3", SERIES[0]))):
        m = [100 * js[a][ver]["ge2"] for a in list(data[ver])]
        ax.bar(xs + (j - 0.5) * 0.38, m, width=0.36, color=col, label=ver)
    ax.set_xticks(xs, [f"p {p:g}" for p in P_COMMON] + ["none\np 0.3"])
    style(ax, "arm", "minds at level ≥ 2 at epoch 500 (%)", "Level ≥ 2 by arm")
    ax.grid(False, axis="x")
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "means over seeds; hollow/dotted = v0.2, solid = v0.3; 95% seed-bootstrap CIs",
             fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=3)
    fig.savefig(f"results/E4{args.suffix}_compare.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
