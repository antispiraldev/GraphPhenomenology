"""E3 analysis: summarize sweep.json files into per-point order-parameter tables.

    uv run python -m experiments.E3_analyze TAG [TAG ...] [--axis rho] [--json OUT]

For each grid point (pooled over all TAGs given, so a coarse and a refined
sweep of the same axis merge) prints, over seeds:
  phi    = fraction of minds at level >= 2 at the final epoch (order parameter)
  se     = standard error of phi across seeds
  var    = variance of phi across seeds (ddof=1)
  chi    = N_agents * var  (susceptibility-style; comparable across sizes)
  ge1    = fraction at level >= 1 at the final epoch (also ge1_se, ge1_var)
  comp   = contingency-computable fraction of windows in the final epoch
  ups    = total level-ups per mind over the run
  iso    = fraction of minds with no contact partner (contact graph degree 0)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from model.actions import Alphabet
from model.config import Config
from sim.runner import build_contact, build_population


def isolated_fraction(cfg: Config) -> float:
    """Replays the runner's RNG stream up to the contact graph (sim.runner.run)
    and returns the fraction of minds with degree 0."""
    rng = np.random.default_rng(cfg.seed)
    agents, _ = build_population(cfg, Alphabet(cfg.m), rng)
    adj = build_contact(len(agents), cfg, rng)
    deg = adj.sum(axis=1)[: cfg.N_agents]
    return float((deg == 0).mean())


def load(tags: list[str], out_dir: str = "outputs") -> tuple[dict, list[dict]]:
    base, results = None, []
    for tag in tags:
        d = json.loads((Path(out_dir) / tag / "sweep.json").read_text())
        base = base or d["base"]
        for r in d["results"]:
            r["base"] = d["base"]
            results.append(r)
    return base, results


def summarize(tags: list[str], axis: str, with_isolation: bool = False) -> list[dict]:
    _, results = load(tags)
    by_x: dict[float, list[dict]] = defaultdict(list)
    for r in results:
        x = r["point"].get(axis, r["base"][axis])
        by_x[x].append(r)
    rows = []
    for x in sorted(by_x):
        rs = by_x[x]
        n_agents = rs[0]["base"]["N_agents"] if "N_agents" not in rs[0]["point"] else rs[0]["point"]["N_agents"]
        phi = np.array([r["final"]["frac_level_ge2"] for r in rs])
        ge1 = np.array([np.mean(np.array(r["mind_levels"]) >= 1) for r in rs])
        comp = np.array([r["final"]["contingency_computable_frac"] for r in rs], dtype=float)
        ups = np.array([(r["totals"]["levelups_bootstrap"] + r["totals"]["levelups_elder"]) / n_agents for r in rs])
        half = np.array([r["curve_frac_ge2"][len(r["curve_frac_ge2"]) // 2] for r in rs])
        row = {
            "x": x, "n": len(rs), "N_agents": n_agents,
            "phi": float(phi.mean()),
            "se": float(phi.std(ddof=1) / np.sqrt(len(phi))) if len(phi) > 1 else float("nan"),
            "var": float(phi.var(ddof=1)) if len(phi) > 1 else float("nan"),
            "phi_half": float(half.mean()),
            "ge1": float(ge1.mean()),
            "ge1_se": float(ge1.std(ddof=1) / np.sqrt(len(ge1))) if len(ge1) > 1 else float("nan"),
            "ge1_var": float(ge1.var(ddof=1)) if len(ge1) > 1 else float("nan"),
            "comp": float(np.nanmean(comp)) if np.isfinite(comp).any() else float("nan"),
            "ups": float(ups.mean()),
        }
        row["chi"] = n_agents * row["var"]
        if with_isolation:
            iso = []
            for r in rs:
                cfg = Config(**{**r["base"], **r["point"], "seed": r["seed"]})
                iso.append(isolated_fraction(cfg))
            row["iso"] = float(np.mean(iso))
        rows.append(row)
    return rows


def print_table(rows: list[dict], axis: str) -> None:
    cols = ["x", "n", "phi", "se", "var", "chi", "phi_half", "ge1", "comp", "ups"] + (["iso"] if "iso" in rows[0] else [])
    print(f"{axis:>8} " + " ".join(f"{c:>8}" for c in cols[1:]))
    for r in rows:
        print(" ".join(f"{r[c]:8.3f}" if isinstance(r[c], float) else f"{r[c]:8d}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--axis", required=True)
    ap.add_argument("--isolation", action="store_true", help="also compute fraction of minds with no contact")
    ap.add_argument("--json", default=None, help="write the summary rows here")
    args = ap.parse_args()
    rows = summarize(args.tags, args.axis, args.isolation)
    print_table(rows, args.axis)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
