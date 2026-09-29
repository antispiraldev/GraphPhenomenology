"""E2 credulity sweep (spec §10): theta vs false recognition of controls and development.

    uv run python -m experiments.E2 all        # both sweeps, then analysis (~25 min on 3 workers)
    uv run python -m experiments.E2 run        # sweeps only
    uv run python -m experiments.E2 analyze    # tables + figures from outputs/E2_*

Two sweeps of experiments/E2.yaml through experiments/sweep.py, 3 workers each:
  E2_coarse: theta in COARSE, seeds 0-9;
  E2_dense:  theta in DENSE (around the recording criterion), seeds 0-31.
Then experiments/E2_diag.py (C distribution of would-be recognitions at θ = 0).
Per-run directories are kept so per-epoch contingency computability can be
averaged over the whole run (sweep.json keeps only the final epoch).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.config import Config

CONFIG = "experiments/E2.yaml"
COARSE = [0.0, 0.1, 0.2, 0.25, 0.5, 0.6, 0.7, 0.8]
DENSE = [0.3, 0.35, 0.4, 0.45]
COARSE_SEEDS, DENSE_SEEDS, WORKERS = 10, 32, 3
CRITERION = 1 / 5000  # §11 recording limit (NOTEBOOK D8)
KINDS = ("object", "recording", "mirror")
SPEED_EPOCH = 25  # early-development snapshot
OUT = Path("outputs")
RESULTS = Path("results")


def run_sweeps() -> None:
    from experiments.sweep import sweep

    base = Config.from_yaml(CONFIG)
    sweep(base, COARSE_SEEDS, [{"theta": t} for t in COARSE], "E2_coarse", keep_runs=True, workers=WORKERS)
    sweep(base, DENSE_SEEDS, [{"theta": t} for t in DENSE], "E2_dense", keep_runs=True, workers=WORKERS)
    from experiments.E2_diag import diagnose

    diagnose(seeds=6, epochs=base.epochs)   # outputs/E2_diag.json (~1 min)


# ---------------------------------------------------------------- statistics

def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space (k is small here)."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    lg = math.lgamma
    return min(1.0, sum(
        math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) + i * lp + (n - i) * lq) for i in range(k + 1)
    ))


def _solve(f, target: float) -> float:
    """p in [0, 1] with f(p) = target, f decreasing in p (bisection in log p)."""
    lo, hi = 1e-12, 1.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided (1 - alpha) binomial confidence interval."""
    if n == 0:
        return float("nan"), float("nan")
    lower = 0.0 if k == 0 else _solve(lambda p: _binom_cdf(k - 1, n, p), 1 - alpha / 2)
    upper = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p), alpha / 2)
    return lower, upper


def mean_ci(x: list[float]) -> tuple[float, float]:
    """Mean and the half-width of a normal 95% interval over seeds."""
    a = np.asarray(x, dtype=float)
    half = float(1.96 * a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    return float(a.mean()), half


# ---------------------------------------------------------------- analysis

def load() -> dict[float, list[dict]]:
    by_theta: dict[float, list[dict]] = defaultdict(list)
    for tag in ("E2_coarse", "E2_dense"):
        data = json.loads((OUT / tag / "sweep.json").read_text())
        for r in data["results"]:
            run_dir = OUT / tag / f"theta{r['point']['theta']}" / f"s{r['seed']}"
            with open(run_dir / "metrics.csv") as f:
                rows = list(csv.DictReader(f))
            r["computable_mean"] = float(np.nanmean([float(x["contingency_computable_frac"]) for x in rows]))
            by_theta[float(r["point"]["theta"])].append(r)
    return dict(sorted(by_theta.items()))


def summarize(by_theta: dict[float, list[dict]]) -> list[dict]:
    table = []
    for th, runs in by_theta.items():
        row: dict = {"theta": th, "seeds": len(runs)}
        for kind in KINDS:
            k = sum(r["totals"][f"one_sided_vs_{kind}"] for r in runs)
            n = sum(r["totals"][f"windows_vs_{kind}"] for r in runs)
            lo, hi = clopper_pearson(k, n)
            row[kind] = {"k": k, "n": n, "rate": k / n if n else float("nan"), "lo": lo, "hi": hi}
        levels = [np.asarray(r["mind_levels"]) for r in runs]
        for g in (1, 2, 3):
            row[f"frac_ge{g}"] = mean_ci([float((lv >= g).mean()) for lv in levels])
        row["mean_level"] = mean_ci([float(lv.mean()) for lv in levels])
        row["max_level"] = mean_ci([float(lv.max()) for lv in levels])
        row["computable"] = mean_ci([r["computable_mean"] for r in runs])
        row["computable_final"] = mean_ci([r["final"]["contingency_computable_frac"] for r in runs])
        row["levelups_per_run"] = mean_ci([r["totals"]["levelups_bootstrap"] for r in runs])
        row["one_sided_vs_mind_per_run"] = mean_ci([r["totals"]["one_sided_vs_mind"] for r in runs])
        row["curve_frac_ge2"] = np.mean([r["curve_frac_ge2"] for r in runs], axis=0).tolist()
        row["curve_mean_level"] = np.mean([r["curve_mean_level"] for r in runs], axis=0).tolist()
        # speed: mean level early in the run, and when the seed-averaged mean level first reaches 1
        row["mean_level_e25"] = mean_ci([r["curve_mean_level"][SPEED_EPOCH] for r in runs])
        ml = np.asarray(row["curve_mean_level"])
        row["epoch_mean_level_1"] = int(np.argmax(ml >= 1.0)) if (ml >= 1.0).any() else None
        table.append(row)
    return table


def crossing(table: list[dict], kind: str = "recording", limit: float = CRITERION) -> dict:
    """Where the rate falls below the limit: the first theta whose point estimate
    is below it, the first whose 95% upper bound is below it, and a log-linear
    interpolation of the point estimate between the bracketing thetas."""
    first_point = next((r["theta"] for r in table if r[kind]["rate"] < limit), None)
    first_upper = next((r["theta"] for r in table if r[kind]["hi"] < limit), None)
    interp = None
    for a, b in zip(table, table[1:]):
        ra, rb = a[kind]["rate"], b[kind]["rate"]
        if ra >= limit > rb and rb > 0:
            f = (math.log(ra) - math.log(limit)) / (math.log(ra) - math.log(rb))
            interp = a["theta"] + f * (b["theta"] - a["theta"])
            break
    return {"first_theta_point_below": first_point, "first_theta_upper95_below": first_upper,
            "interpolated_theta": interp}


def print_table(table: list[dict]) -> None:
    def rate(c: dict) -> str:
        return f"{c['k']}/{c['n']} = {100 * c['rate']:.4f}% [{100 * c['lo']:.4f}, {100 * c['hi']:.4f}]"

    def mc(x: tuple[float, float], d: int = 3) -> str:
        return f"{x[0]:.{d}f} ± {x[1]:.{d}f}"

    print("| θ | seeds | recording one-sided (k/n, %, 95% CI) | object | mirror | level ≥1 | level ≥2 | level ≥3 "
          "| mean level | max level | computable | bootstrap level-ups/run | mean level @25 | epoch mean level ≥1 |")
    print("|" + "---|" * 14)
    for r in table:
        print(f"| {r['theta']:.2f} | {r['seeds']} | {rate(r['recording'])} | {r['object']['k']}/{r['object']['n']} "
              f"| {r['mirror']['k']}/{r['mirror']['n']} | {mc(r['frac_ge1'])} | {mc(r['frac_ge2'])} "
              f"| {mc(r['frac_ge3'])} | {mc(r['mean_level'], 2)} | {mc(r['max_level'], 1)} "
              f"| {mc(r['computable'])} | {mc(r['levelups_per_run'], 1)} | {mc(r['mean_level_e25'], 2)} "
              f"| {r['epoch_mean_level_1'] if r['epoch_mean_level_1'] is not None else '—'} |")


def analyze() -> None:
    from experiments.E2_plots import plot_all

    table = summarize(load())
    print_table(table)
    cross = crossing(table)
    print("recording criterion crossing:", cross)
    out = OUT / "E2_summary.json"
    out.write_text(json.dumps({"table": table, "crossing": cross}, indent=1))
    print(f"wrote {out}")
    diag_path = OUT / "E2_diag.json"
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        print(f"diagnostic (θ = 0 runs, {diag['seeds']} seeds): share of would-be recognitions with C >= θ")
        print("| θ | " + " | ".join(f"{k} ({grp})" for k in ("mind", "recording")
                                    for grp in ("all", "level 0", "level ≥1")) + " |")
        print("|" + "---|" * 7)
        for t in diag["by_kind"]["mind"]["survival"]:
            cells = [diag["by_kind"][k][key].get(t, float("nan")) for k in ("mind", "recording")
                     for key in ("survival", "survival_level0", "survival_level_ge1")]
            print(f"| {t} | " + " | ".join(f"{c:.4f}" for c in cells) + " |")
        for k in ("mind", "recording"):
            v = diag["by_kind"][k]
            print(f"{k}: n = {v['n']}, level-0 share {v['frac_level0']:.2f}")
    RESULTS.mkdir(exist_ok=True)
    plot_all(table, CRITERION, RESULTS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("run", "analyze", "all"))
    args = ap.parse_args()
    if args.step in ("run", "all"):
        run_sweeps()
    if args.step in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
