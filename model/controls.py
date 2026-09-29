"""Control agents: object, recording, mirror (spec §8).

Frozen signatures; implemented in segment G1 (worktree: controls).
Control policies never probe, so they return no ProbeEvents, and they never
write records.
"""

from __future__ import annotations

import numpy as np

from model.actions import Alphabet
from model.agent import Agent
from model.config import Config
from model.graph import Graph
from model.trace import ProbeEvent, Tick


class ObjectPolicy:
    """Never initiates (τ on its turn); responds uniformly over available labels."""

    def __init__(self, alphabet: Alphabet) -> None:
        self.alphabet = alphabet

    def act(self, agent: Agent, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        return self.alphabet.tau, []

    def respond(self, agent: Agent, offer: int, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        labels = agent.graph.available()
        return int(labels[rng.integers(len(labels))]), []

    def observe(self, agent: Agent, tick: Tick) -> list[ProbeEvent]:
        return []

    def reset_episode(self, agent: Agent) -> None:
        return None


class MirrorPolicy:
    """Never initiates (τ on its turn); always responds complement(σ)."""

    def __init__(self, alphabet: Alphabet) -> None:
        self.alphabet = alphabet

    def act(self, agent: Agent, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        return self.alphabet.tau, []

    def respond(self, agent: Agent, offer: int, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        return self.alphabet.complement(offer), []

    def observe(self, agent: Agent, tick: Tick) -> list[ProbeEvent]:
        return []

    def reset_episode(self, agent: Agent) -> None:
        return None


class RecordingPolicy:
    """Consumes one script label per tick, initiating or responding (§8).
    Unavailable on its turn → τ; unavailable when responding → the label is
    still returned (a decline). The script loops. The script position is NOT
    reset between episodes."""

    def __init__(self, alphabet: Alphabet, script: tuple[int, ...]) -> None:
        self.alphabet = alphabet
        self.script = script
        self.pos = 0

    def _next(self) -> int:
        """Consume and return the next script label (looping). An empty
        script yields τ."""
        if not self.script:
            return self.alphabet.tau
        label = self.script[self.pos % len(self.script)]
        self.pos = (self.pos + 1) % len(self.script)
        return label

    def act(self, agent: Agent, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        label = self._next()
        if label not in agent.graph.available():
            label = self.alphabet.tau
        return label, []

    def respond(self, agent: Agent, offer: int, rng: np.random.Generator) -> tuple[int, list[ProbeEvent]]:
        return self._next(), []

    def observe(self, agent: Agent, tick: Tick) -> list[ProbeEvent]:
        return []

    def reset_episode(self, agent: Agent) -> None:
        # The script position deliberately persists across episodes (§8, D4).
        return None


def make_control_graph(
    kind: str, alphabet: Alphabet, cfg: Config, rng: np.random.Generator
) -> Graph:
    """Graph.random for object/recording; for mirror additionally a self-loop
    for every label at every active node (§8 mirror details)."""
    if kind not in ("object", "recording", "mirror"):
        raise ValueError(f"unknown control kind {kind!r}")
    graph = Graph.random(alphabet, cfg, rng)
    if kind == "mirror":
        graph.add_self_loops(list(alphabet.labels))
    return graph


def capture_script(trace: list[Tick], agent_id: int) -> tuple[int, ...]:
    """Labels emitted by agent_id in trace order: its action on its turns, its
    response on the partner's turns (skipping partner τ ticks)."""
    script: list[int] = []
    for tick in trace:
        if tick.initiator == agent_id:
            script.append(tick.action)
        elif tick.responder == agent_id and tick.response is not None:
            script.append(tick.response)
    return tuple(script)
