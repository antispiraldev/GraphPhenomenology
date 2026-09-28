"""Recognition: joint cycles, contingency, level conditions, mutuality (spec §5).

Frozen signatures; implemented in segment G1 (worktree: recognition).
Everything here is a pure function of the trace and agent state, so it can
be tested on hand-built synthetic traces.

Tick ranges are inclusive [start, end] indices into the episode trace
(trace[t].t == t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from model.actions import Alphabet
from model.agent import Agent
from model.config import Config
from model.records import Token
from model.trace import Tick


@dataclass(frozen=True)
class Segment:
    start: int                  # first recorded tick (agent's turn)
    end: int                    # last recorded tick (partner's turn)
    tokens: tuple[Token, ...]   # encode_segment(trace[start:end+1])
    anchor: int                 # agent's node before tick `start`
    consumed_from: int          # first tick used by the qualification (probe start for extensions; == start for joint cycles)


@dataclass(frozen=True)
class ContingencyResult:
    value: float | None         # C_ij, None if not computable
    n_offers: int               # |O|
    n_labels: int               # distinct offer labels in O

    def holds(self, cfg: Config) -> bool:
        return (
            self.value is not None
            and self.n_offers >= cfg.n_min_total
            and self.n_labels >= 2
            and self.value >= cfg.theta
        )


@dataclass(frozen=True)
class Qualification:
    agent_id: int
    partner_id: int
    level_from: int
    kind: Literal["joint_cycle", "extension"]
    segment: Segment
    contingency: float


@dataclass(frozen=True)
class LevelUp:
    agent_id: int
    partner_id: int
    source: Literal["bootstrap", "elder"]
    qualification: Qualification


def encode_segment(ticks: Sequence[Tick], agent_id: int) -> tuple[Token, ...]:
    """§5.5 encoding from agent_id's perspective. Own turns: SELF_OFFER(σ) if
    accepted, SELF_TAU if τ, else SELF_ANY. Partner turns: OTHER_OFFER(σ) if
    accepted, else OTHER_ANY."""
    raise NotImplementedError


def contingency(ticks: Sequence[Tick], agent_id: int, alphabet: Alphabet) -> ContingencyResult:
    """Pooled lift of the partner's responses to agent_id's sync offers (§5.2)."""
    raise NotImplementedError


def find_joint_cycle(
    trace: Sequence[Tick], agent_id: int, lo: int, hi: int, L_max: int
) -> Segment | None:
    """Shortest segment within [lo, hi], length <= L_max, starting on
    agent_id's turn and ending on the partner's turn, containing an accepted
    agent-initiated offer followed (later tick) by an accepted
    partner-initiated offer (§5.1). Ties: earliest start."""
    raise NotImplementedError


def find_extension(
    trace: Sequence[Tick], agent_id: int, record_id: int, lo: int, hi: int, L_ext: int
) -> Segment | None:
    """§5.3 k >= 1: earliest SUCCESS probe event by agent_id for record_id at
    tick s whose matching START is at tick >= lo, followed by the first
    accepted partner-initiated offer at tick e with s < e <= min(s + L_ext, hi).
    Segment covers the extra [s+1, e]; consumed_from is the probe's START tick."""
    raise NotImplementedError


def level_condition(
    trace: Sequence[Tick],
    agent: Agent,
    partner_id: int,
    window: range,
    lo: int,
    alphabet: Alphabet,
    cfg: Config,
) -> Qualification | None:
    """§5.3 for agent at its current level. The qualifying segment must end
    inside `window` and use no tick before `lo` (lo <= window.start enforces
    §5 boundary rules; the episode tracks lo per agent). Contingency is
    computed over the window's ticks only. Returns None if the agent is not a
    mind, is at k_max, or the condition fails."""
    raise NotImplementedError


def resolve_window(
    agent_i: Agent,
    agent_j: Agent,
    q_i: Qualification | None,
    q_j: Qualification | None,
    cfg: Config,
) -> list[LevelUp]:
    """§5.4 mutuality and modes. Only minds level up.
    bootstrap: equal levels and both qualify → both; if require_mutual is
    false, each qualifying agent levels up alone (source 'bootstrap').
    elder: i qualifies and level(j) >= level(i) + 1 → i ('elder').
    mixed: both rules (disjoint by level)."""
    raise NotImplementedError
