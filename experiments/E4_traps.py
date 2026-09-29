"""E4 diagnostic: are minds trapped in absorbing nodes created by growth (§6)?

    uv run python -m experiments.E4_traps

A node is *absorbing* when every out-edge (any label) is a self-loop. Under
growth='generalize' the node activated on unlock has only self-loops (τ plus
the learned labels), so a mind that steps onto it can never leave. Runs the
default config (500 epochs) for seeds 0-5 under both growth modes, 3 workers,
and reports, at the end: fraction of minds sitting on an absorbing node, sync
labels available there vs elsewhere, and level by trapped status.
"""

from __future__ import annotations

import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from model.config import Config
from sim.runner import run

SEEDS = range(6)


def absorbing(graph, v: int) -> bool:
    out = graph.A[:, v, :].copy()
    out[:, v] = False
    return not out.any()


def one(job: tuple[str, int]) -> dict:
    growth, seed = job
    cfg = dataclasses.replace(Config.from_yaml("experiments/default.yaml"), growth=growth, seed=seed)
    res = run(cfg)
    out = []
    for a in res.agents:
        g = a.graph
        n_sync = int(sum(1 for lab in g.available() if lab != g.alphabet.tau))
        out.append({"level": a.level, "trapped": absorbing(g, g.x), "n_sync": n_sync})
    comp = float(np.nanmean([r["contingency_computable_frac"] for r in res.rows[-100:]]))
    return {"growth": growth, "seed": seed, "minds": out, "computable_last100": comp}


def main() -> None:
    jobs = [(g, s) for g in ("generalize", "none") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(one, jobs))
    report = {}
    for growth in ("generalize", "none"):
        ms = [m for r in results if r["growth"] == growth for m in r["minds"]]
        tr = [m for m in ms if m["trapped"]]
        fr = [m for m in ms if not m["trapped"]]
        rep = {
            "minds": len(ms),
            "trapped_frac": len(tr) / len(ms),
            "mean_sync_labels_trapped": float(np.mean([m["n_sync"] for m in tr])) if tr else None,
            "mean_sync_labels_free": float(np.mean([m["n_sync"] for m in fr])) if fr else None,
            "level_hist_trapped": np.bincount([m["level"] for m in tr], minlength=4).tolist() if tr else [],
            "level_hist_free": np.bincount([m["level"] for m in fr], minlength=4).tolist() if fr else [],
            "contingency_computable_last100": float(np.mean([r["computable_last100"] for r in results if r["growth"] == growth])),
        }
        report[growth] = rep
    Path("outputs/E4_traps").mkdir(parents=True, exist_ok=True)
    Path("outputs/E4_traps/report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
