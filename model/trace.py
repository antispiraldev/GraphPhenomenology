"""Trace types shared by policies, episodes and recognition (spec §4.4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProbeEventKind(Enum):
    START = "start"
    ADVANCE = "advance"
    SUCCESS = "success"
    ABORT = "abort"


@dataclass(frozen=True)
class ProbeEvent:
    agent_id: int
    kind: ProbeEventKind
    record_id: int
    # Index into the probe's flat word of the next expected token *after*
    # this event. START: 0. SUCCESS: len(word). ABORT: position where it failed.
    pos: int


@dataclass(frozen=True)
class Tick:
    t: int                   # tick index within the episode, from 0
    initiator: int           # agent id whose turn it is
    responder: int
    action: int              # initiator's label (sync or τ)
    response: int | None     # responder's label; None iff action is τ
    accepted: bool           # always False for τ
    x_init_before: int
    x_resp_before: int
    probe_events: tuple[ProbeEvent, ...] = ()

    def role_of(self, agent_id: int) -> str:
        if agent_id == self.initiator:
            return "self"
        if agent_id == self.responder:
            return "other"
        raise ValueError(f"agent {agent_id} not in tick {self.t}")

    def node_before(self, agent_id: int) -> int:
        return self.x_init_before if agent_id == self.initiator else self.x_resp_before
