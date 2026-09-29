"""Population runner and CLI (spec §4.1, §12).

    uv run python -m sim.runner --config experiments/default.yaml [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from analysis.metrics import (
    EpochAccumulator,
    distinct_flat_words,
    epoch_row,
    library_dump,
    word_lengths_by_level,
)
from model.actions import Alphabet
from model.agent import Agent, MindPolicy, Policy
from model.config import Config
from model.controls import (
    MirrorPolicy,
    ObjectPolicy,
    RecordingPolicy,
    capture_script,
    make_control_graph,
)
from model.graph import Graph
from sim.episode import run_episode


def build_contact(n: int, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Symmetric bool adjacency over all agents.

    pairs: fixed pairs (0,1), (2,3), ...; ring: i ~ i+1 mod n;
    erdos_renyi: each edge with p_contact; small_world: ring lattice with
    2 neighbours per side, each edge rewired with probability p_contact.
    """
    adj = np.zeros((n, n), dtype=bool)
    if cfg.contact == "pairs":
        for i in range(0, n - 1, 2):
            adj[i, i + 1] = True
    elif cfg.contact == "ring":
        for i in range(n):
            adj[i, (i + 1) % n] = True
    elif cfg.contact == "erdos_renyi":
        adj = np.triu(rng.random((n, n)) < cfg.p_contact, k=1)
    elif cfg.contact == "small_world":
        for i in range(n):
            for d in (1, 2):
                j = (i + d) % n
                if rng.random() < cfg.p_contact:
                    choices = [k for k in range(n) if k != i and not (adj[i, k] or adj[k, i])]
                    if choices:
                        j = choices[rng.integers(len(choices))]
                adj[min(i, j), max(i, j)] = True
    else:
        raise ValueError(f"unknown contact {cfg.contact!r}")
    adj = adj | adj.T
    np.fill_diagonal(adj, False)
    return adj


def sample_matching(adj: np.ndarray, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Random maximal matching: shuffle the edges, take each whose ends are free."""
    iu, ju = np.nonzero(np.triu(adj, k=1))
    order = rng.permutation(len(iu))
    used: set[int] = set()
    pairs = []
    for e in order:
        i, j = int(iu[e]), int(ju[e])
        if i not in used and j not in used:
            used.update((i, j))
            pairs.append((i, j))
    return pairs


def _recording_script(alphabet: Alphabet, cfg: Config, rng: np.random.Generator) -> tuple[int, ...]:
    """§8: one episode between two fresh throwaway minds; the first one's labels."""
    minds = [Agent(-1 - k, Graph.random(alphabet, cfg, rng)) for k in range(2)]
    pol = MindPolicy(alphabet, cfg)
    result = run_episode(minds[0], minds[1], pol, pol, alphabet, cfg, rng)
    return capture_script(result.trace, minds[0].id)


def build_population(
    cfg: Config, alphabet: Alphabet, rng: np.random.Generator
) -> tuple[list[Agent], dict[int, Policy]]:
    """Minds get ids 0..N_agents-1; controls follow, in the order object, recording, mirror."""
    agents: list[Agent] = []
    policies: dict[int, Policy] = {}
    mind_policy = MindPolicy(alphabet, cfg)
    for i in range(cfg.N_agents):
        agents.append(Agent(i, Graph.random(alphabet, cfg, rng)))
        policies[i] = mind_policy
    for kind in ("object", "recording", "mirror"):
        for _ in range(cfg.controls.get(kind, 0)):
            aid = len(agents)
            agents.append(Agent(aid, make_control_graph(kind, alphabet, cfg, rng), kind=kind))
            if kind == "object":
                policies[aid] = ObjectPolicy(alphabet)
            elif kind == "mirror":
                policies[aid] = MirrorPolicy(alphabet)
            else:
                policies[aid] = RecordingPolicy(alphabet, _recording_script(alphabet, cfg, rng))
    return agents, policies


@dataclass
class RunResult:
    rows: list[dict]
    agents: list[Agent]
    trace_hash: str


def run(cfg: Config, out_dir: str | Path | None = None, verbose: bool = False) -> RunResult:
    rng = np.random.default_rng(cfg.seed)
    alphabet = Alphabet(cfg.m)
    agents, policies = build_population(cfg, alphabet, rng)
    minds = [a for a in agents if a.kind == "mind"]
    adj = build_contact(len(agents), cfg, rng)
    partner_history: dict[int, set[int]] = defaultdict(set)
    hasher = hashlib.sha256()
    rows: list[dict] = []
    trace_file = None
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if cfg.record_traces:
            trace_file = open(out / "traces.jsonl", "w")

    try:
        for epoch in range(cfg.epochs):
            acc = EpochAccumulator()
            for i, j in sample_matching(adj, rng):
                a, b = agents[i], agents[j]
                result = run_episode(a, b, policies[i], policies[j], alphabet, cfg, rng)
                for tick in result.trace:
                    hasher.update(repr(tick).encode())
                acc.add_episode(result, partner_history)
                if trace_file is not None:
                    trace_file.write(json.dumps({
                        "epoch": epoch, "pair": [i, j],
                        "ticks": [repr(t) for t in result.trace],
                    }) + "\n")
            row = epoch_row(epoch, minds, acc, cfg)
            rows.append(row)
            if verbose and (epoch % 50 == 0 or epoch == cfg.epochs - 1):
                print(f"epoch {epoch}: mean level {row['mean_level']:.2f}, max {row['max_level']}")
    finally:
        if trace_file is not None:
            trace_file.close()

    result = RunResult(rows, agents, hasher.hexdigest())
    if out_dir is not None:
        write_outputs(Path(out_dir), cfg, result)
    return result


def write_outputs(out: Path, cfg: Config, result: RunResult) -> None:
    minds = [a for a in result.agents if a.kind == "mind"]
    (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.rows[0]))
        writer.writeheader()
        writer.writerows(result.rows)
    (out / "library.json").write_text(json.dumps(library_dump(result.agents), indent=1))
    summary = {
        "trace_hash": result.trace_hash,
        "final": result.rows[-1],
        "word_lengths_by_level": word_lengths_by_level(minds),
        "distinct_flat_words": distinct_flat_words(minds),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="output dir (default: <output_dir>/<config stem>_s<seed>)")
    args = ap.parse_args(argv)
    cfg = Config.from_yaml(args.config)
    out = args.out or Path(cfg.output_dir) / f"{Path(args.config).stem}_s{cfg.seed}"
    result = run(cfg, out, verbose=True)
    print(f"wrote {out}  trace_hash={result.trace_hash[:16]}")


if __name__ == "__main__":
    main()
