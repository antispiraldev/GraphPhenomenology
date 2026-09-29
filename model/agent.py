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
from model.records import Library, Token, TokenKind, flat
from model.trace import ProbeEvent, ProbeEventKind, Tick

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
        events: list[ProbeEvent] = []
        if agent.probe is None and len(agent.library) > 0:
            if rng.random() < self.cfg.p_probe:
                rid = self._choose_record(agent, rng)
                word = flat(agent.library[rid], agent.library)
                agent.probe = ProbeState(rid, word, 0)
                events.append(ProbeEvent(agent.id, ProbeEventKind.START, rid, 0))
        if agent.probe is not None:
            probe = agent.probe
            tok = probe.word[probe.pos]
            if tok.kind is TokenKind.SELF_OFFER:
                if tok.label in agent.graph.available():
                    return tok.label, events
                events.append(self._abort(agent))
            elif tok.kind is TokenKind.SELF_TAU:
                return self.alphabet.tau, events
            elif tok.kind is TokenKind.SELF_ANY:
                return self._explore(agent, rng), events
            else:  # out of phase: an other token on my turn
                events.append(self._abort(agent))
        return self._explore(agent, rng), events

    def respond(
        self, agent: Agent, offer: int, rng: np.random.Generator
    ) -> tuple[int, list[ProbeEvent]]:
        events: list[ProbeEvent] = []
        probe = agent.probe
        if probe is not None:
            tok = probe.word[probe.pos]
            if tok.kind is TokenKind.OTHER_OFFER:
                if tok.label == offer:
                    # Accept; if complement is unavailable this is a decline
                    # and observe aborts the probe.
                    return self.alphabet.complement(offer), events
                events.append(self._abort(agent))
            # OTHER_ANY (or out of phase): reactivity; observe advances/aborts.
        return self._react(agent, offer, rng), events

    def observe(self, agent: Agent, tick: Tick) -> list[ProbeEvent]:
        probe = agent.probe
        if probe is None:
            return []
        tok = probe.word[probe.pos]
        role = tick.role_of(agent.id)
        if role == "self":
            if tok.kind is TokenKind.SELF_OFFER:
                ok = tick.action == tok.label and tick.accepted
            elif tok.kind is TokenKind.SELF_TAU:
                ok = tick.action == self.alphabet.tau
            else:
                ok = tok.kind is TokenKind.SELF_ANY
        else:
            if tok.kind is TokenKind.OTHER_OFFER:
                ok = tick.action == tok.label and tick.accepted
            else:
                ok = tok.kind is TokenKind.OTHER_ANY
        if not ok:
            return [self._abort(agent)]
        probe.pos += 1
        if probe.pos == len(probe.word):
            agent.probe = None
            return [ProbeEvent(agent.id, ProbeEventKind.SUCCESS, probe.record_id, probe.pos)]
        return [ProbeEvent(agent.id, ProbeEventKind.ADVANCE, probe.record_id, probe.pos)]

    def reset_episode(self, agent: Agent) -> None:
        agent.probe = None

    # --- helpers ---

    def _choose_record(self, agent: Agent, rng: np.random.Generator) -> int:
        levels = np.array([r.level for r in agent.library], dtype=float)
        w = np.exp(self.cfg.beta * (levels - levels.max()))
        return int(rng.choice(len(levels), p=w / w.sum()))

    def _explore(self, agent: Agent, rng: np.random.Generator) -> int:
        avail = agent.graph.available()
        return int(avail[rng.integers(len(avail))])

    def _react(self, agent: Agent, offer: int, rng: np.random.Generator) -> int:
        comp = self.alphabet.complement(offer)
        if rng.random() < self.cfg.rho and comp in agent.graph.available():
            return comp
        return self._explore(agent, rng)

    def _abort(self, agent: Agent) -> ProbeEvent:
        probe = agent.probe
        agent.probe = None
        return ProbeEvent(agent.id, ProbeEventKind.ABORT, probe.record_id, probe.pos)
