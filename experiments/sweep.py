"""Run a config over many seeds (and optional parameter grid) in parallel.

    uv run python -m experiments.sweep --config experiments/E0.yaml --seeds 100 --tag E0
    uv run python -m experiments.sweep --config X.yaml --seeds 10 --grid rho=0.1,0.3,0.5 --tag E3_rho

Writes outputs/<tag>/sweep.json: for every (grid point, seed) the final
epoch row, column totals over all epochs, and the trace hash. Per-run
directories are written only with --keep-runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

from model.config import Config
from sim.runner import run

SUM_KEYS_PREFIX = ("levelups_", "probe_starts", "probe_successes", "one_sided_", "windows_vs_", "mind_control_windows")


def _one(job: tuple[dict, dict, str | None]) -> dict:
    cfg_dict, point, out = job
    cfg = Config(**cfg_dict)
    res = run(cfg, out)
    totals = {
        k: sum(r[k] for r in res.rows)
        for k in res.rows[0]
        if k.startswith(SUM_KEYS_PREFIX)
    }
    levels = sorted(a.level for a in res.agents if a.kind == "mind")
    return {
        "point": point, "seed": cfg.seed, "final": res.rows[-1], "totals": totals,
        "mind_levels": levels, "trace_hash": res.trace_hash,
        "curve_frac_ge2": [r["frac_level_ge2"] for r in res.rows],
        "curve_mean_level": [r["mean_level"] for r in res.rows],
    }


def parse_grid(specs: list[str]) -> list[dict]:
    axes = []
    for spec in specs:
        key, vals = spec.split("=", 1)
        axes.append([(key, yaml.safe_load(v)) for v in vals.split(",")])
    return [dict(combo) for combo in itertools.product(*axes)] if axes else [{}]


def sweep(base: Config, seeds: int, grid: list[dict], tag: str, keep_runs: bool = False, workers: int | None = None) -> list[dict]:
    out = Path(base.output_dir) / tag
    jobs = []
    for point in grid:
        for seed in range(seeds):
            cfg = dataclasses.replace(base, seed=seed, **point)
            run_dir = None
            if keep_runs:
                name = "_".join(f"{k}{v}" for k, v in point.items()) or "base"
                run_dir = str(out / name / f"s{seed}")
            jobs.append((cfg.to_dict(), point, run_dir))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, jobs))
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps({"base": base.to_dict(), "results": results}))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--grid", nargs="*", default=[], help="key=v1,v2,... (cartesian product)")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--keep-runs", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    base = Config.from_yaml(args.config)
    over = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    base = dataclasses.replace(base, **over)
    results = sweep(base, args.seeds, parse_grid(args.grid), args.tag, args.keep_runs, args.workers)
    print(f"{len(results)} runs -> {Path(base.output_dir) / args.tag / 'sweep.json'}")


if __name__ == "__main__":
    main()
