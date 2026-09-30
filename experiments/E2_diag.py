"""E2 diagnostic: the contingency value C in windows where a mind would recognize its partner at θ = 0.

    uv run python -m experiments.E2_diag [--seeds 6] [--epochs 300]

Runs experiments/E2.yaml with θ = 0, so every window that passes the structural
part of §5.3 (joint cycle or probe + extra) and has computable contingency with
C >= 0 counts as a one-sided recognition. For each such window it logs C and the
partner kind. The fraction of those windows with C >= θ is then a first-order
prediction of how one-sided recognition of minds and of recordings falls with θ
(ignoring feedback from θ on development). run_episode is wrapped here, read-only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import sim.runner as runner
from model.actions import Alphabet
from model.config import Config
from sim.recognition import contingency

THETAS = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]


def _job(args: tuple[int, int]) -> list[tuple[str, int, float]]:
    seed, epochs = args
    cfg = dataclasses.replace(Config.from_yaml("experiments/E2.yaml"), theta=0.0, seed=seed, epochs=epochs)
    alphabet = Alphabet(cfg.m)
    log: list[tuple[str, int, float]] = []   # (partner kind, mind level before episode, C)
    original = runner.run_episode

    def wrapped(a, b, pa, pb, alph, c, rng):
        levels = {a.id: a.level, b.id: b.level}
        res = original(a, b, pa, pb, alph, c, rng)
        stats = iter(res.window_stats)
        for w in range(c.T_ep // c.W):
            ticks = res.trace[w * c.W:(w + 1) * c.W]
            for me, other in ((a, b), (b, a)):
                if me.kind != "mind":
                    continue
                st = next(stats)
                if st.one_sided:
                    log.append((other.kind, levels[me.id], contingency(ticks, me.id, alphabet).value))
        return res

    runner.run_episode = wrapped
    try:
        runner.run(cfg)
    finally:
        runner.run_episode = original
    return log


def diagnose(seeds: int = 6, epochs: int = 300, workers: int = 3,
             out_path: Path | str = "outputs/E2_diag.json") -> dict:
    out_path = Path(out_path)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        logs = [x for part in ex.map(_job, [(s, epochs) for s in range(seeds)]) for x in part]
    out: dict = {"seeds": seeds, "epochs": epochs, "by_kind": {}}
    for kind in ("mind", "recording"):
        c = np.array([v for k, _, v in logs if k == kind])
        lv = np.array([lvl for k, lvl, _ in logs if k == kind])
        out["by_kind"][kind] = {
            "n": int(len(c)),
            "quantiles": {q: float(np.quantile(c, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            "survival": {t: float((c >= t).mean()) for t in THETAS},
            "survival_level0": {t: float((c[lv == 0] >= t).mean()) for t in THETAS} if (lv == 0).any() else {},
            "survival_level_ge1": {t: float((c[lv >= 1] >= t).mean()) for t in THETAS} if (lv >= 1).any() else {},
            "frac_level0": float((lv == 0).mean()),
        }
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    out = diagnose(args.seeds, args.epochs)
    for kind, v in out["by_kind"].items():
        print(kind, v["n"], "windows; share with C >= theta:", {t: round(x, 4) for t, x in v["survival"].items()})


if __name__ == "__main__":
    main()
