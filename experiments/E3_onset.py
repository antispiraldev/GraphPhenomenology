"""E3 onset sharpness vs system size: logistic fits with a seed bootstrap.

    uv run python -m experiments.E3_onset --prefix E3v3 [--boot 400]

For N = 10 / 20 / 40 (tags <prefix>_rho_N10, <prefix>_rho_coarse + _rho_fine
restricted to rho <= 0.6, <prefix>_rho_N40) fits
    f(rho) = A / (1 + exp(-(rho - rho_c) / w))
to the per-seed fraction at level >= 1 and at level >= 2 (A free in [0, 1]).
A collective transition predicts the width w shrinks as N grows; a
per-encounter crossover predicts w independent of N. Reports rho_c and w with
95% percentile intervals from resampling seeds within each grid point.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.E3_analyze import load


def per_seed(tags: list[str], key: str, rho_max: float = 0.6) -> dict[float, np.ndarray]:
    _, results = load(tags)
    out: dict[float, list[float]] = {}
    for r in results:
        x = r["point"].get("rho", r["base"]["rho"])
        if x > rho_max + 1e-9 or x < 0.15 - 1e-9:
            continue
        lv = np.array(r["mind_levels"])
        out.setdefault(x, []).append(float(np.mean(lv >= (1 if key == "ge1" else 2))))
    return {x: np.array(v) for x, v in sorted(out.items())}


XC_GRID = np.linspace(0.05, 0.8, 301)
W_GRID = np.geomspace(0.003, 0.5, 200)


def fit(data: dict[float, np.ndarray]) -> tuple[float, float, float] | None:
    """Least-squares fit on a (rho_c, w) grid; A solved in closed form, clipped to [0, 1]."""
    x = np.array(list(data))
    y = np.array([v.mean() for v in data.values()])
    g = 1.0 / (1.0 + np.exp(-(x[None, None, :] - XC_GRID[:, None, None]) / W_GRID[None, :, None]))
    A = np.clip((g * y).sum(-1) / np.maximum((g * g).sum(-1), 1e-12), 0.0, 1.0)
    sse = ((A[..., None] * g - y) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sse), sse.shape)
    if A[i, j] <= 0:
        return None
    return float(A[i, j]), float(XC_GRID[i]), float(W_GRID[j])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="E3v3")
    ap.add_argument("--boot", type=int, default=400)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sets = {10: [f"{args.prefix}_rho_N10"], 20: [f"{args.prefix}_rho_coarse", f"{args.prefix}_rho_fine"],
            40: [f"{args.prefix}_rho_N40"]}
    print(f"{'key':>4} {'N':>3}  {'A':>5}  {'rho_c [95%]':>22}  {'w [95%]':>22}")
    for key in ("ge1", "ge2"):
        for n, tags in sets.items():
            tags = [t for t in tags if (Path("outputs") / t / "sweep.json").exists()]
            if not tags:
                continue
            data = per_seed(tags, key)
            best = fit(data)
            if best is None:
                print(f"{key:>4} {n:>3}  fit failed")
                continue
            boots = []
            for _ in range(args.boot):
                bd = {x: rng.choice(v, size=len(v), replace=True) for x, v in data.items()}
                b = fit(bd)
                if b is not None:
                    boots.append(b)
            bt = np.array(boots)
            lo, hi = np.percentile(bt, [2.5, 97.5], axis=0)
            print(f"{key:>4} {n:>3}  {best[0]:5.2f}  {best[1]:6.3f} [{lo[1]:5.3f}, {hi[1]:5.3f}]"
                  f"  {best[2]:6.3f} [{lo[2]:5.3f}, {hi[2]:5.3f}]")


if __name__ == "__main__":
    main()
