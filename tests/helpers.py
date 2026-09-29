"""Builders for synthetic traces, shared by tests across modules."""

from __future__ import annotations

from model.actions import Alphabet
from model.trace import ProbeEvent, Tick


def build_trace(
    moves: list[tuple[int, int | None, bool]],
    a: int = 0,
    b: int = 1,
    first: int = 0,
    events: dict[int, tuple[ProbeEvent, ...]] | None = None,
) -> list[Tick]:
    """Strictly alternating trace between agents a and b.

    moves[t] = (action, response, accepted); response None for τ.
    `first` is the id of the first initiator. Node positions are left at 0.
    """
    events = events or {}
    trace = []
    init, resp = (first, b if first == a else a)
    for t, (action, response, accepted) in enumerate(moves):
        trace.append(
            Tick(t, init, resp, action, response, accepted, 0, 0, events.get(t, ()))
        )
        init, resp = resp, init
    return trace


def accepted_offer(alphabet: Alphabet, sigma: int) -> tuple[int, int, bool]:
    return (sigma, alphabet.complement(sigma), True)


def declined_offer(alphabet: Alphabet, sigma: int) -> tuple[int, int, bool]:
    return (sigma, alphabet.tau, False)


def tau(alphabet: Alphabet) -> tuple[int, None, bool]:
    return (alphabet.tau, None, False)


def probe_event(agent_id: int, kind, record_id: int, pos: int = 0) -> ProbeEvent:
    """Shorthand for a ProbeEvent (kind: a ProbeEventKind)."""
    return ProbeEvent(agent_id, kind, record_id, pos)


def make_agent(agent_id, alphabet, cfg, rng, level: int = 0, kind: str = "mind"):
    """Agent with a random graph and `level` placeholder records appended
    directly (tokens are a single accepted exchange; REF above level 1)."""
    from model.agent import Agent
    from model.graph import Graph
    from model.records import Record, Token

    agent = Agent(agent_id, Graph.random(alphabet, cfg, rng), kind=kind)
    for k in range(1, level + 1):
        body = (Token.self_offer(0), Token.other_offer(0))
        tokens = body if k == 1 else (Token.make_ref(k - 2),) + body
        agent.library.append(
            Record(
                id=k - 1,
                level=k,
                tokens=tokens,
                anchor=0,
                other=None if k == 1 else k - 2,
                partner_id=-1,
                saturated=False,
                source="bootstrap",
            )
        )
    return agent
