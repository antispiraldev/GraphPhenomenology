"""E5 — Private languages (spec §10, hypothesis §14).

Compares three contact structures (fixed pairs, fully shuffled ER p=1, default
ER p=0.2) with an instrumented copy of the population loop in sim/runner.py.
The loop consumes the RNG in exactly the same order as `sim.runner.run`, so a
run here has the same trace hash as the plain runner (tested in
tests/test_E5.py).

Every episode is seen from both sides. For agent i meeting j, the partner is
classed at episode start as

    cobuilt_top  j is the partner i wrote its current top record with
    past         i leveled up with j before, but not for its current top
    stranger     i never leveled up with j

and every probe START / SUCCESS of i is attributed to that class, to i's level
and j's level at episode start, and to the probed record's level.

    uv run python -m experiments.E5 run --seeds 30 --workers 3
    uv run python -m experiments.E5_analyze
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.metrics import EpochAccumulator, epoch_row
from model.actions import Alphabet
from model.agent import Agent
from model.config import Config
from model.records import Token, TokenKind, extras, flat, own_labels, swap_token
from model.trace import ProbeEventKind, Tick
from sim.episode import run_episode
from sim.runner import build_contact, build_population, sample_matching

CONDITIONS = {
    "pairs": "experiments/E5_pairs.yaml",
    "shuffled": "experiments/E5_shuffled.yaml",
    "er02": "experiments/E5_er02.yaml",
}
OUT = Path("outputs/E5")
WILD = (TokenKind.SELF_ANY, TokenKind.OTHER_ANY)


# --------------------------------------------------------------------------
# Instrumented run
# --------------------------------------------------------------------------

def partner_class(agent: Agent, partner_id: int, history: set[int]) -> str:
    top = agent.library.top
    if top is not None and top.partner_id == partner_id:
        return "cobuilt_top"
    return "past" if partner_id in history else "stranger"


def _probe_intervals(trace: list[Tick], ids: tuple[int, int]) -> list[dict]:
    """Every probe in the episode: owner, record id, start tick, outcome, and
    whether the partner had a probe open on any tick of this probe."""
    open_: dict[int, dict] = {}
    done: list[dict] = []
    for tick in trace:
        touched = {a: a in open_ for a in ids}
        closed = []
        for ev in tick.probe_events:
            if ev.kind is ProbeEventKind.START:
                open_[ev.agent_id] = {"agent": ev.agent_id, "rid": ev.record_id, "t0": tick.t,
                                      "success": False, "co": False}
                touched[ev.agent_id] = True
            elif ev.kind in (ProbeEventKind.SUCCESS, ProbeEventKind.ABORT):
                p = open_.pop(ev.agent_id)
                p["success"] = ev.kind is ProbeEventKind.SUCCESS
                closed.append(p)
        for p in list(open_.values()) + closed:
            other = ids[1] if p["agent"] == ids[0] else ids[0]
            if touched[other]:
                p["co"] = True
        done.extend(closed)
    for p in open_.values():          # unfinished at episode end: not a success
        done.append(p)
    return done


def run_instrumented(cfg: Config) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    alphabet = Alphabet(cfg.m)
    agents, policies = build_population(cfg, alphabet, rng)
    minds = [a for a in agents if a.kind == "mind"]
    adj = build_contact(len(agents), cfg, rng)
    partner_history: dict[int, set[int]] = defaultdict(set)   # read/updated by EpochAccumulator
    hasher = hashlib.sha256()
    rows = []
    # key "cls|k_i|k_j|rec_level" -> [starts, successes, co_starts, co_successes]
    probes: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    # key "cls|k_i|k_j" -> [encounters, level-up events of i]
    enc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    seg_overlap = [0, 0]      # bootstrap level-up pairs whose two segments share a tick / total
    contacts_per_agent = int(adj[: cfg.N_agents, : cfg.N_agents].sum(1).mean())

    for epoch in range(cfg.epochs):
        acc = EpochAccumulator()
        for i, j in sample_matching(adj, rng):
            a, b = agents[i], agents[j]
            k0 = {a.id: a.level, b.id: b.level}
            cls = {a.id: partner_class(a, b.id, partner_history[a.id]),
                   b.id: partner_class(b, a.id, partner_history[b.id])}
            result = run_episode(a, b, policies[i], policies[j], alphabet, cfg, rng)
            for tick in result.trace:
                hasher.update(repr(tick).encode())
            ups = defaultdict(int)
            for up in result.level_ups:
                ups[up.agent_id] += 1
            for me, other in ((a, b), (b, a)):
                enc[f"{cls[me.id]}|{k0[me.id]}|{k0[other.id]}"][0] += 1
                enc[f"{cls[me.id]}|{k0[me.id]}|{k0[other.id]}"][1] += ups[me.id]
            for p in _probe_intervals(result.trace, (a.id, b.id)):
                other = b.id if p["agent"] == a.id else a.id
                key = f"{cls[p['agent']]}|{k0[p['agent']]}|{k0[other]}|{p['rid'] + 1}"
                c = probes[key]
                c[0] += 1
                c[1] += p["success"]
                c[2] += p["co"]
                c[3] += p["success"] and p["co"]
            # the two halves of each mutual bootstrap: same window, same pair
            by_window = defaultdict(list)
            for up in result.level_ups:
                by_window[up.qualification.segment.end // cfg.W].append(up.qualification.segment)
            for segs in by_window.values():
                if len(segs) == 2:
                    s1, s2 = segs
                    seg_overlap[0] += max(s1.consumed_from, s2.consumed_from) <= min(s1.end, s2.end)
                    seg_overlap[1] += 1
            acc.add_episode(result, partner_history)
        rows.append(epoch_row(epoch, minds, acc, cfg))

    return {
        "condition_contact": cfg.contact, "p_contact": cfg.p_contact, "seed": cfg.seed,
        "trace_hash": hasher.hexdigest(), "seconds": time.time() - t_start,
        "contacts_per_agent": contacts_per_agent,
        "rows": rows, "probes": dict(probes), "encounters": dict(enc),
        "segment_overlap": seg_overlap,
        "final_levels": [a.level for a in minds],
        "record_overlap": record_overlap(minds, alphabet),
    }


# --------------------------------------------------------------------------
# Record overlap between co-builders and strangers
# --------------------------------------------------------------------------

def offered_labels(tokens) -> set[int]:
    """Labels that were offered in the word, whichever side offered them."""
    return {t.label for t in tokens if t.kind in (TokenKind.SELF_OFFER, TokenKind.OTHER_OFFER)}


def jaccard(x: set, y: set) -> float | None:
    return len(x & y) / len(x | y) if x | y else None


def interlock(e_i: tuple[Token, ...], e_j: tuple[Token, ...]) -> int:
    """Best alignment of swap(e_j) against e_i at an odd phase shift (so j's
    self turns fall on i's other turns). Returns the number of concrete offer
    tokens that coincide at the best conflict-free alignment (0 if none).
    A wildcard is compatible with anything; two concrete tokens must be equal."""
    b = tuple(swap_token(t) for t in e_j)
    best = 0
    for d in range(-len(b) + 1, len(e_i)):
        if d % 2 == 0:
            continue
        match, ok, overlap = 0, True, 0
        for p, tb in enumerate(b):
            q = p + d
            if not 0 <= q < len(e_i):
                continue
            ta = e_i[q]
            overlap += 1
            if ta.kind in WILD or tb.kind in WILD:
                continue
            if ta != tb:
                ok = False
                break
            match += ta.kind in (TokenKind.SELF_OFFER, TokenKind.OTHER_OFFER)
        if ok and overlap >= 2:
            best = max(best, match)
    return best


def record_overlap(minds: list[Agent], alphabet: Alphabet) -> dict:
    """For each record level, compare the level-k records of every pair of
    minds: co-built (each names the other as partner_id) vs not."""
    out: dict[str, dict[str, list]] = {}
    for k in range(1, max((a.level for a in minds), default=0) + 1):
        have = [a for a in minds if a.level >= k]
        stats = {"cobuilt": [], "other": []}
        for x, y in itertools.combinations(have, 2):
            rx, ry = x.library[k - 1], y.library[k - 1]
            group = "cobuilt" if rx.partner_id == y.id and ry.partner_id == x.id else "other"
            ex, ey = extras(rx), extras(ry)
            fx, fy = flat(rx, x.library), flat(ry, y.library)
            own_x, own_y = set(own_labels(fx, alphabet)), set(own_labels(fy, alphabet))
            stats[group].append([
                jaccard(offered_labels(ex), offered_labels(ey)),           # extras, offered labels
                jaccard(own_x, own_y),                                      # flat, own-move labels
                jaccard(own_x, {alphabet.complement(l) for l in own_y}),    # flat, own vs partner's complement
                max(interlock(ex, ey), interlock(ey, ex)),                  # extras, swap alignment
                fx == fy,                                                   # identical flat words
            ])
        out[str(k)] = stats
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _job(args: tuple[str, dict]) -> tuple[str, int]:
    cond, cfg_dict = args
    cfg = Config(**cfg_dict)
    res = run_instrumented(cfg)
    d = OUT / cond
    d.mkdir(parents=True, exist_ok=True)
    (d / f"s{cfg.seed}.json").write_text(json.dumps(res))
    return cond, cfg.seed


def run_all(seeds: int, workers: int, conditions: list[str]) -> None:
    jobs = []
    for seed in range(seeds):             # interleave so partial results cover all conditions
        for cond in conditions:
            cfg = dataclasses.replace(Config.from_yaml(CONDITIONS[cond]), seed=seed)
            jobs.append((cond, cfg.to_dict()))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for cond, seed in ex.map(_job, jobs):
            print(f"{cond} s{seed} done  ({time.time() - t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seeds", type=int, default=16)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "run":
        run_all(args.seeds, args.workers, args.conditions)
    else:
        from experiments.E5_analyze import main as analyze
        analyze()


if __name__ == "__main__":
    main()
