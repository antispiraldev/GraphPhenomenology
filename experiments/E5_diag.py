"""E5 diagnostic: why do fixed pairs stall? Breaks the §5.3 level condition
into its parts, per window, for minds meeting an equal-level partner.

    uv run python -m experiments.E5_diag --seeds 4 --epochs 300

Uses the same RNG stream as sim.runner.run. Only episodes with no level-up are
decomposed (so the agents' levels and top records are fixed through the
episode). Per window and agent it records: top-record probe succeeded; an
extension (probe success + accepted partner offer within L_ext) exists;
contingency computable / holds; and the agent's and partner's available sync
labels at episode start.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from model.actions import Alphabet
from model.config import Config
from model.trace import ProbeEventKind
from sim.episode import run_episode
from sim.recognition import contingency, find_extension
from sim.runner import build_contact, build_population, sample_matching
from experiments.E5 import CONDITIONS, OUT


def _n_sync(agent, alphabet) -> int:
    return sum(1 for l in agent.graph.available() if alphabet.is_sync(int(l)))


def diag_run(args: tuple[str, dict]) -> tuple[str, dict]:
    cond, cfg_dict = args
    cfg = Config(**cfg_dict)
    rng = np.random.default_rng(cfg.seed)
    alphabet = Alphabet(cfg.m)
    agents, policies = build_population(cfg, alphabet, rng)
    adj = build_contact(len(agents), cfg, rng)
    # level -> counters
    stats: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _ in range(cfg.epochs):
        for i, j in sample_matching(adj, rng):
            a, b = agents[i], agents[j]
            k = a.level
            same = a.level == b.level and k >= 1
            avail = {a.id: _n_sync(a, alphabet), b.id: _n_sync(b, alphabet)}
            at_new = {x.id: x.graph.x >= cfg.N0 for x in (a, b)}   # sitting on a node grown by an unlock
            res = run_episode(a, b, policies[i], policies[j], alphabet, cfg, rng)
            if not same or res.level_ups:
                continue
            s = stats[k]
            for w0 in range(0, cfg.T_ep, cfg.W):
                window = range(w0, w0 + cfg.W)
                ext = {}
                for me in (a, b):
                    top = me.library.top.id
                    succ = any(ev.agent_id == me.id and ev.kind is ProbeEventKind.SUCCESS and ev.record_id == top
                               for t in res.trace[w0:w0 + cfg.W] for ev in t.probe_events)
                    e = find_extension(res.trace, me.id, top, max(w0 - cfg.W, 0), w0 + cfg.W - 1, cfg.L_ext)
                    e_ok = e is not None and e.end >= w0
                    c = contingency(res.trace[w0:w0 + cfg.W], me.id, alphabet)
                    comp = c.value is not None and c.n_offers >= cfg.n_min_total and c.n_labels >= 2
                    holds = c.holds(cfg)
                    ext[me.id] = e_ok and holds
                    s["agent_windows"] += 1
                    s["top_success"] += succ
                    s["extension"] += e_ok
                    s["cont_computable"] += comp
                    s["cont_holds"] += holds
                    s["n_offers"] += c.n_offers
                    s["n_labels"] += c.n_labels
                    s["cont_value_sum"] += c.value if c.value is not None else 0.0
                    s["cont_value_n"] += c.value is not None
                    s["qualifies"] += e_ok and holds
                s["pair_windows"] += 1
                s["both_qualify"] += ext[a.id] and ext[b.id]
            s["encounters"] += 1
            s["avail_sync_sum"] += avail[a.id] + avail[b.id]
            s["at_new_node"] += at_new[a.id] + at_new[b.id]
    return cond, {str(k): dict(v) for k, v in stats.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    jobs = [(c, dataclasses.replace(Config.from_yaml(CONDITIONS[c]), seed=s, epochs=args.epochs).to_dict())
            for s in range(args.seeds) for c in ("pairs", "shuffled", "er02")]
    agg: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for cond, st in ex.map(diag_run, jobs):
            for k, v in st.items():
                for key, x in v.items():
                    agg[cond][k][key] += x
    out = {}
    for cond, by_k in agg.items():
        out[cond] = {}
        for k, s in sorted(by_k.items(), key=lambda kv: int(kv[0])):
            aw, pw = s["agent_windows"], s["pair_windows"]
            out[cond][k] = {
                "encounters": int(s["encounters"]),
                "p_top_probe_success_per_window": s["top_success"] / aw,
                "p_extension_per_window": s["extension"] / aw,
                "p_contingency_computable": s["cont_computable"] / aw,
                "p_contingency_holds": s["cont_holds"] / aw,
                "mean_offers": s["n_offers"] / aw,
                "mean_offer_labels": s["n_labels"] / aw,
                "mean_C": s["cont_value_sum"] / max(s["cont_value_n"], 1),
                "p_one_side_qualifies": s["qualifies"] / aw,
                "p_both_qualify": s["both_qualify"] / pw,
                "mean_avail_sync_labels": s["avail_sync_sum"] / (2 * s["encounters"]),
                "frac_on_grown_node": s["at_new_node"] / (2 * s["encounters"]),
            }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diag.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
