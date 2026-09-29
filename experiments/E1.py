"""E1 — Bootstrap vs elder (spec §10, §5.4, §5.7).

    uv run python -m experiments.E1 --config experiments/E1.yaml --seeds 16 --workers 3 --tag E1
    uv run python -m experiments.E1_plot --tag E1        # figures + summary table

For each seed s:
  1. Donor: a fresh bootstrap-mode population (same config, seed 10_000 + s) is
     run until its best mind reaches `elder_level` (at most --donor-max-epochs;
     if it stalls, retry with donor seed + 1_000). That mind — graph, position
     and library intact — is the elder.
  2. Conditions: a fresh population built from seed s (identical minds and
     contact graph in every condition). In the "+elder" conditions agent 0 is
     replaced by the transplanted elder; in the "no elder" conditions agent 0
     stays an ordinary fresh mind. All metrics exclude agent 0.

Conditions: bootstrap+elder, elder+elder, mixed+elder, and two references
without an elder (bootstrap, mixed). Elder mode without an elder is omitted:
nobody can ever level up there.

Writes outputs/<tag>/E1.json (per-seed, per-condition curves and summaries).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from analysis.metrics import EpochAccumulator, distinct_flat_words
from model.actions import Alphabet
from model.agent import Agent
from model.config import Config
from model.records import TokenKind, extras, flat, own_labels
from sim.episode import run_episode
from sim.runner import build_contact, build_population, sample_matching

ELDER_ID = 0
DONOR_SEED_OFFSET = 10_000
DONOR_ATTEMPTS = 5
LEVELS_TRACKED = (1, 2, 3, 4, 5)

# name -> (mode, elder present)
CONDITIONS: dict[str, tuple[str, bool]] = {
    "bootstrap+elder": ("bootstrap", True),
    "elder+elder": ("elder", True),
    "mixed+elder": ("mixed", True),
    "bootstrap (no elder)": ("bootstrap", False),
    "mixed (no elder)": ("mixed", False),
}


# --- elder --------------------------------------------------------------------

def grow_elder(cfg: Config, target: int, max_epochs: int) -> tuple[Agent, dict]:
    """Run a bootstrap population until some mind reaches `target`; return a
    deep copy of the highest-level mind (lowest id on ties) and donor info."""
    cfg = dataclasses.replace(cfg, mode="bootstrap")
    rng = np.random.default_rng(cfg.seed)
    alphabet = Alphabet(cfg.m)
    agents, policies = build_population(cfg, alphabet, rng)
    adj = build_contact(len(agents), cfg, rng)
    minds = [a for a in agents if a.kind == "mind"]
    epoch = 0
    while epoch < max_epochs and max(a.level for a in minds) < target:
        for i, j in sample_matching(adj, rng):
            run_episode(agents[i], agents[j], policies[i], policies[j], alphabet, cfg, rng)
        epoch += 1
    best = max(minds, key=lambda a: (a.level, -a.id))
    info = {
        "donor_seed": cfg.seed, "donor_epochs": epoch, "donor_agent": best.id,
        "elder_level": best.level, "elder_active_nodes": best.graph.n_active,
        "elder_sources": [r.source for r in best.library],
        "profile": elder_profile(best),
    }
    return copy.deepcopy(best), info


def elder_profile(elder: Agent) -> dict:
    """How broad the elder's repertoire is: labels it learned (self-looped
    everywhere by growth, §6) and the mean number of sync labels available
    per active node. A narrow repertoire limits how contingently the elder can
    answer a novice's offers (§5.2)."""
    g = elder.graph
    alphabet = g.alphabet
    learned = sorted({lab for r in elder.library for lab in own_labels(flat(r, elder.library), alphabet)})
    nodes = np.flatnonzero(g.active)
    n_sync = [sum(1 for lab in g.available(int(u)) if alphabet.is_sync(lab)) for u in nodes]
    return {"learned_labels": learned, "n_learned": len(learned),
            "mean_sync_available": float(np.mean(n_sync)),
            "sync_available_at_x": sum(1 for lab in g.available() if alphabet.is_sync(lab))}


def transplant(elder: Agent, new_id: int) -> Agent:
    """A fresh Agent object carrying the elder's graph (incl. position) and library."""
    return Agent(new_id, copy.deepcopy(elder.graph), library=copy.deepcopy(elder.library))


# --- diversity ------------------------------------------------------------------

def canonical_extras(rec) -> tuple:
    """Content of a record's extras up to perspective and wildcard padding:
    keep only accepted offers as (who, label), drop SELF_TAU / wildcards, and
    take the lexicographic min of the word and its self<->other swap. Two
    records transmitted from the same source word, at any chain depth, map to
    the same key; this separates new content from relabelled copies."""
    seq = []
    for t in extras(rec):
        if t.kind is TokenKind.SELF_OFFER:
            seq.append((0, t.label))
        elif t.kind is TokenKind.OTHER_OFFER:
            seq.append((1, t.label))
    flipped = [(1 - w, lab) for w, lab in seq]
    return min(tuple(seq), tuple(flipped))


def diversity(minds: list[Agent]) -> dict:
    flat_words = distinct_flat_words(minds)
    n_records: Counter = Counter()
    core: dict[int, set] = defaultdict(set)
    src: dict[int, Counter] = defaultdict(Counter)
    for a in minds:
        for r in a.library:
            n_records[r.level] += 1
            core[r.level].add(canonical_extras(r))
            src[r.level][r.source] += 1
    return {
        "distinct_flat": {int(k): v for k, v in flat_words.items()},
        "n_records": {int(k): v for k, v in n_records.items()},
        "distinct_core_extras": {int(k): len(v) for k, v in core.items()},
        "records_by_source": {int(k): dict(v) for k, v in src.items()},
    }


# --- one condition run ------------------------------------------------------------

def component_of(adj: np.ndarray, start: int) -> set[int]:
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adj[u]):
            v = int(v)
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def run_condition(cfg: Config, elder: Agent | None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    alphabet = Alphabet(cfg.m)
    agents, policies = build_population(cfg, alphabet, rng)
    if elder is not None:
        agents[ELDER_ID] = transplant(elder, ELDER_ID)
    adj = build_contact(len(agents), cfg, rng)
    others = [a for a in agents if a.kind == "mind" and a.id != ELDER_ID]
    comp = component_of(adj, ELDER_ID)
    neighbours = {int(v) for v in np.flatnonzero(adj[ELDER_ID])}

    curves = defaultdict(list)
    first_reach: dict[int, dict[int, int]] = {k: {} for k in LEVELS_TRACKED}  # level -> mind -> epoch
    ups = Counter()          # (source, partner is elder) -> n
    taught_directly: set[int] = set()
    partner_history: dict[int, set[int]] = defaultdict(set)
    elder_meetings = 0

    for epoch in range(cfg.epochs):
        acc = EpochAccumulator()
        n_from_elder = 0
        for i, j in sample_matching(adj, rng):
            if ELDER_ID in (i, j):
                elder_meetings += 1
            result = run_episode(agents[i], agents[j], policies[i], policies[j], alphabet, cfg, rng)
            acc.add_episode(result, partner_history)
            for up in result.level_ups:
                if up.agent_id == ELDER_ID:
                    ups[("elder_self", up.source)] += 1
                    continue
                from_elder = up.partner_id == ELDER_ID
                ups[(up.source, "elder" if from_elder else "other")] += 1
                if from_elder:
                    n_from_elder += 1
                    taught_directly.add(up.agent_id)
        levels = np.array([a.level for a in others])
        curves["mean_level"].append(float(levels.mean()))
        for k in (1, 2, 3):
            curves[f"frac_ge{k}"].append(float((levels >= k).mean()))
        curves["max_level"].append(int(levels.max()))
        curves["levelups_bootstrap"].append(acc.levelups_bootstrap)
        curves["levelups_elder"].append(acc.levelups_elder)
        curves["levelups_from_elder"].append(n_from_elder)
        for a in others:
            for k in LEVELS_TRACKED:
                if a.level >= k and a.id not in first_reach[k]:
                    first_reach[k][a.id] = epoch

    others_in_comp = [a for a in others if a.id in comp]
    return {
        "curves": dict(curves),
        "first_any": {k: (min(v.values()) if v else None) for k, v in first_reach.items()},
        "median_mind_time": {
            # median over minds of the epoch they reached level k; minds that never
            # did count as cfg.epochs (censored), so this is an upper-censored median
            k: float(np.median([v.get(a.id, cfg.epochs) for a in others]))
            for k, v in first_reach.items()
        },
        "levelups": {f"{s}|{p}": n for (s, p), n in ups.items()},
        "taught_directly": len(taught_directly),
        "elder_degree": len(neighbours),
        "elder_component": len(comp) - 1,           # other agents reachable from agent 0
        "elder_meetings": elder_meetings,
        "final_levels": sorted(int(a.level) for a in others),
        "final_levels_in_comp": sorted(int(a.level) for a in others_in_comp),
        "elder_final_level": agents[ELDER_ID].level,
        "diversity": diversity(others),
    }


def one_seed(job: tuple[dict, int, int, int]) -> dict:
    cfg_dict, seed, elder_level, donor_max_epochs = job
    base = Config(**cfg_dict)
    t0 = time.time()
    # Some donor populations stall below the target level. Retry with a fresh
    # donor seed (10_000 + seed + 1_000 * attempt) and log every attempt.
    attempts = []
    for attempt in range(DONOR_ATTEMPTS):
        donor_seed = DONOR_SEED_OFFSET + seed + 1_000 * attempt
        elder, info = grow_elder(dataclasses.replace(base, seed=donor_seed), elder_level, donor_max_epochs)
        attempts.append({k: info[k] for k in ("donor_seed", "donor_epochs", "elder_level")})
        if info["elder_level"] >= elder_level:
            break
    info["attempts"] = attempts
    info["donor_seconds"] = time.time() - t0
    out = {"seed": seed, "elder": info, "conditions": {}}
    for name, (mode, with_elder) in CONDITIONS.items():
        t1 = time.time()
        cfg = dataclasses.replace(base, seed=seed, mode=mode)
        res = run_condition(cfg, elder if with_elder else None)
        res["seconds"] = time.time() - t1
        out["conditions"][name] = res
    print(f"seed {seed} done: elder L{info['elder_level']} after {info['donor_epochs']} donor epochs; "
          f"{time.time() - t0:.0f}s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/E1.yaml")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--elder-level", type=int, default=5)
    ap.add_argument("--donor-max-epochs", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="E1")
    args = ap.parse_args()
    base = Config.from_yaml(args.config)
    jobs = [(base.to_dict(), s, args.elder_level, args.donor_max_epochs)
            for s in range(args.first_seed, args.first_seed + args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(one_seed, jobs))
    out = Path(base.output_dir) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": base.to_dict(), "elder_level": args.elder_level,
            "donor_max_epochs": args.donor_max_epochs, "wall_seconds": time.time() - t0}
    (out / "E1.json").write_text(json.dumps({"meta": meta, "results": results}))
    print(f"{len(results)} seeds in {meta['wall_seconds']:.0f}s -> {out / 'E1.json'}")


if __name__ == "__main__":
    main()
