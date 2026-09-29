"""E4 — Autocatalysis (spec §10): instrumented runs over a parameter grid.

    uv run python -m experiments.E4 --tag E4_pprobe --seeds 8 --workers 3 \
        --grid p_probe=0,0.05,0.1,0.2,0.3,0.5,0.8
    uv run python -m experiments.E4 --tag E4_growth --seeds 8 --workers 3 \
        --grid growth=generalize,none

The loop mirrors sim.runner.run exactly (same RNG draws, same trace hash) and
additionally records, per run:
  - levels[e, i]: level of mind i at the *start* of epoch e (e = 0..epochs;
    row `epochs` is the final state);
  - partner[e, i]: partner id of mind i in epoch e (-1 if unmatched);
  - per-probe outcomes aggregated by (record level, flat word length,
    number of offer tokens):
    starts, successes, aborts, unfinished (probe still running at episode end);
  - per-token pass/fail counts by token kind, to explain probe failure.

Writes outputs/<tag>/<point>_s<seed>.npz plus outputs/<tag>/index.json.
Analysis: experiments/E4_analyze.py.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from analysis.metrics import EpochAccumulator, epoch_row
from experiments.sweep import parse_grid
from model.actions import Alphabet
from model.config import Config
from model.records import flat
from model.trace import ProbeEventKind
from sim.episode import run_episode
from sim.runner import build_contact, build_population, sample_matching

TOKEN_KINDS = ("SELF_OFFER", "SELF_TAU", "SELF_ANY", "OTHER_OFFER", "OTHER_ANY")


def n_offers(word) -> int:
    """Constrained tokens: SELF_OFFER / OTHER_OFFER (each needs an accepted offer)."""
    return sum(t.kind.value in ("SELF_OFFER", "OTHER_OFFER") for t in word)


def run_instrumented(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    alphabet = Alphabet(cfg.m)
    agents, policies = build_population(cfg, alphabet, rng)
    minds = [a for a in agents if a.kind == "mind"]
    n = len(minds)
    adj = build_contact(len(agents), cfg, rng)
    partner_history: dict[int, set[int]] = defaultdict(set)
    hasher = hashlib.sha256()
    rows = []
    levels = np.zeros((cfg.epochs + 1, n), dtype=np.int8)
    partner = np.full((cfg.epochs, n), -1, dtype=np.int16)
    # (record level, flat length, #offer tokens) -> [starts, successes, aborts, unfinished]
    probe_len: dict[tuple[int, int], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    # token kind -> [reached, failed]
    token_pass = {k: [0, 0] for k in TOKEN_KINDS}
    # position index in flat word (0-based) -> [reached, failed], pooled
    pos_pass: dict[int, list[int]] = defaultdict(lambda: [0, 0])

    for epoch in range(cfg.epochs):
        levels[epoch] = [a.level for a in minds]
        acc = EpochAccumulator()
        for i, j in sample_matching(adj, rng):
            if i < n:
                partner[epoch, i] = j
            if j < n:
                partner[epoch, j] = i
            a, b = agents[i], agents[j]
            result = run_episode(a, b, policies[i], policies[j], alphabet, cfg, rng)
            for tick in result.trace:
                hasher.update(repr(tick).encode())
            acc.add_episode(result, partner_history)
            # Probe bookkeeping. Records are immutable and ids stable, so the
            # post-episode library gives the right flat word for any record id.
            active: dict[int, tuple] = {}
            for tick in result.trace:
                for ev in tick.probe_events:
                    ag = agents[ev.agent_id]
                    if ev.kind is ProbeEventKind.START:
                        word = flat(ag.library[ev.record_id], ag.library)
                        active[ev.agent_id] = (ag.library[ev.record_id].level, word)
                        continue
                    lvl, word = active[ev.agent_id]
                    key = (lvl, len(word), n_offers(word))
                    if ev.kind is ProbeEventKind.ADVANCE:
                        continue
                    del active[ev.agent_id]
                    probe_len[key][0] += 1
                    if ev.kind is ProbeEventKind.SUCCESS:
                        probe_len[key][1] += 1
                        passed = len(word)
                    else:
                        probe_len[key][2] += 1
                        passed = ev.pos
                        tk = word[ev.pos].kind.value
                        token_pass[tk][0] += 1
                        token_pass[tk][1] += 1
                        pos_pass[ev.pos][0] += 1
                        pos_pass[ev.pos][1] += 1
                    for p in range(passed):
                        token_pass[word[p].kind.value][0] += 1
                        pos_pass[p][0] += 1
            for lvl, word in active.values():   # cut off by the episode end
                probe_len[(lvl, len(word), n_offers(word))][0] += 1
                probe_len[(lvl, len(word), n_offers(word))][3] += 1
        rows.append(epoch_row(epoch, minds, acc, cfg))
    levels[cfg.epochs] = [a.level for a in minds]

    # Composition of flat words per record level at the end of the run.
    comp: dict[int, list[int]] = defaultdict(lambda: [0] * (len(TOKEN_KINDS) + 1))
    for a in minds:
        for r in a.library:
            w = flat(r, a.library)
            c = comp[r.level]
            c[0] += 1
            for t in w:
                c[1 + TOKEN_KINDS.index(t.kind.value)] += 1

    return {
        "levels": levels,
        "partner": partner,
        "probe_len": np.array([[*k, *v] for k, v in sorted(probe_len.items())], dtype=np.int64).reshape(-1, 7),
        "token_pass": np.array([token_pass[k] for k in TOKEN_KINDS], dtype=np.int64),
        "pos_pass": np.array([[p, *v] for p, v in sorted(pos_pass.items())], dtype=np.int64).reshape(-1, 3),
        "word_comp": np.array([[k, *v] for k, v in sorted(comp.items())], dtype=np.int64).reshape(-1, 2 + len(TOKEN_KINDS)),
        "probe_starts_total": np.array([sum(r["probe_starts"] for r in rows)]),
        "probe_successes_total": np.array([sum(r["probe_successes"] for r in rows)]),
        "contingency_computable": np.array([r["contingency_computable_frac"] for r in rows]),
        "trace_hash": hasher.hexdigest(),
        "rows": rows,
    }


def point_name(point: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in point.items()) or "base"


def _job(job: tuple[dict, dict, str]) -> dict:
    cfg_dict, point, out = job
    cfg = Config(**cfg_dict)
    t0 = time.time()
    res = run_instrumented(cfg)
    secs = time.time() - t0
    path = Path(out) / f"{point_name(point)}_s{cfg.seed}.npz"
    np.savez_compressed(path, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)})
    return {"point": point, "seed": cfg.seed, "file": path.name, "trace_hash": res["trace_hash"],
            "seconds": secs, "final": res["rows"][-1]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/default.yaml")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    base = Config.from_yaml(args.config)
    over = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    base = dataclasses.replace(base, **over)
    out = Path(base.output_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (dataclasses.replace(base, seed=seed, **point).to_dict(), point, str(out))
        for point in parse_grid(args.grid) for seed in range(args.seeds)
    ]
    # Longest jobs first so the pool drains evenly.
    jobs.sort(key=lambda j: -float(j[1].get("p_probe", 0.3)))
    index = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(_job, jobs):
            index.append(r)
            print(f"{point_name(r['point'])} s{r['seed']}: {r['seconds']:.0f}s "
                  f"mean level {r['final']['mean_level']:.2f}", flush=True)
    (out / "index.json").write_text(json.dumps({"base": base.to_dict(), "runs": index}, indent=1))


if __name__ == "__main__":
    main()
