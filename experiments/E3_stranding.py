"""E3 diagnostic: are minds 'stranded' at the end of a run?

Under bootstrap mode a mind can only level up with a contact neighbour at the
*same* level (spec §5.4). A mind with no equal-level neighbour is frozen for
good: its neighbours can only move further away. This script runs the E3 base
config and, at the final epoch, counts the minds that still have at least one
equal-level contact neighbour ("live") versus none ("stranded").

    uv run python -m experiments.E3_stranding [--seeds 6] [key=value ...]
"""

from __future__ import annotations

import argparse
import dataclasses
from collections import Counter

import numpy as np
import yaml

from model.actions import Alphabet
from model.config import Config
from sim.runner import build_contact, build_population, run


def contact_graph(cfg: Config) -> np.ndarray:
    """Replay the runner's RNG stream up to the contact graph (as in sim.runner.run)."""
    rng = np.random.default_rng(cfg.seed)
    agents, _ = build_population(cfg, Alphabet(cfg.m), rng)
    return build_contact(len(agents), cfg, rng)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/E3.yaml")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    base = Config.from_yaml(args.config)
    over = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    base = dataclasses.replace(base, **over)
    live_by_level: Counter = Counter()
    total_by_level: Counter = Counter()
    last_levelup_epoch = []
    for seed in range(args.seeds):
        cfg = dataclasses.replace(base, seed=seed)
        adj = contact_graph(cfg)
        res = run(cfg)
        levels = {a.id: a.level for a in res.agents if a.kind == "mind"}
        for i, L in levels.items():
            nbrs = np.nonzero(adj[i])[0]
            live = any(levels.get(int(j)) == L for j in nbrs)
            total_by_level[L] += 1
            live_by_level[L] += live
        ups = [r["epoch"] for r in res.rows if r["levelups_bootstrap"] + r["levelups_elder"] > 0]
        last_levelup_epoch.append(ups[-1] if ups else -1)
    n = sum(total_by_level.values())
    print(f"{args.seeds} seeds, {n} minds, {cfg.epochs} epochs")
    print("level  minds  live(has equal-level neighbour)")
    for L in sorted(total_by_level):
        print(f"{L:5d} {total_by_level[L]:6d} {live_by_level[L]:6d}")
    print(f"stranded fraction: {1 - sum(live_by_level.values()) / n:.3f}")
    print(f"last epoch with any level-up, per seed: {last_levelup_epoch}")


if __name__ == "__main__":
    main()
