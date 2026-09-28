"""Agent state, the Policy protocol, and the mind policy with probes (spec §3.4, §4.3).

Agent, ProbeState and Policy are fixed in G0. MindPolicy is implemented in
segment G1 (worktree: agent).

Per-tick call order, driven by sim.episode (G2):
  1. initiator: label, ev1 = policy.act(agent, rng)
  2. if label is sync: responder: resp, ev2 = policy.respond(agent, label, rng)
     accepted iff resp == complement(label) and both labels available.
  3. tick built; for each of the two agents: ev3 = policy.observe(agent, tick)
  4. if accepted, both graphs step along their label; if τ, initiator steps.
All returned ProbeEvents go into tick.probe_events (act/respond events before
observe events). Growth happens only at window end, never mid-tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from model.actions import Alphabet
from model.config import Config
from model.graph import Graph
from model.records import Library, Token
from model.trace import ProbeEvent, Tick

Kind = Literal["mind", "object", "recording", "mirror"]


@dataclass
class ProbeState:
    record_id: int
    word: tuple[Token, ...]   # flat(record): alternating, starts self, ends other
    pos: int = 0              # index of the next expected token


@dataclass
class Agent:
    id: int
    graph: Graph
    kind: Kind = "mind"
    library: Library = field(default_factory=Library)
    probe: ProbeState | None = None

    @property
    def level(self) -> int:
        return self.library.level


class Policy(Protocol):
    def act(self, agent: Agent, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        """Choose my label on my turn. Must return a label in agent.graph.available()."""
        ...

    def respond(
        self, agent: Agent, offer: int, rng: np.random.Generator
    ) -> tuple[int, list[ProbeEvent]]:
        """Choose my response label to the partner's sync offer. Any label in
        Σ may be returned; a label unavailable at my node counts as a decline."""
        ...

    def observe(self, agent: Agent, tick: Tick) -> list[ProbeEvent]:
        """Update internal state (probe) from the resolved tick. Called for both agents."""
        ...

    def reset_episode(self, agent: Agent) -> None:
        """Clear per-episode state (probe). Graph position persists."""
        ...


# --- G1 (agent) -------------------------------------------------------------

class MindPolicy:
    """§4.3 mind policy.

    act: continue probe (SELF_OFFER σ: offer σ, abort+explore if unavailable;
    SELF_TAU: τ; SELF_ANY: explore); else with prob p_probe and non-empty
    library start a probe on a record chosen ∝ exp(β·level); else explore
    uniformly over available(x).

    respond: probe expects OTHER_OFFER(σ) and offer == σ → complement(σ) if
    available; expects OTHER_OFFER(σ′ ≠ σ) → abort, then reactivity rule;
    expects OTHER_ANY → reactivity rule. No probe → reactivity rule:
    with prob ρ, complement(σ) if available, else uniform over available(x).

    observe: advance/abort/succeed the probe per §4.3 (including partner τ).
    Emits SUCCESS after the last token matches and clears the probe.
    """

    def __init__(self, alphabet: Alphabet, cfg: Config) -> None:
        self.alphabet = alphabet
        self.cfg = cfg

    def act(self, agent: Agent, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        raise NotImplementedError

    def respond(
        self, agent: Agent, offer: int, rng: np.random.Generator
    ) -> tuple[int, list[ProbeEvent]]:
        raise NotImplementedError

    def observe(self, agent: Agent, tick: Tick) -> list[ProbeEvent]:
        raise NotImplementedError

    def reset_episode(self, agent: Agent) -> None:
        raise NotImplementedError
