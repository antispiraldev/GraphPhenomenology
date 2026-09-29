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
from model.records import OTHER_ANY, SELF_ANY, SELF_TAU, Token
from model.trace import ProbeEventKind, Tick


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
    out: list[Token] = []
    for tick in ticks:
        if tick.initiator == agent_id:
            if tick.accepted:
                out.append(Token.self_offer(tick.action))
            elif tick.response is None:
                out.append(SELF_TAU)
            else:
                out.append(SELF_ANY)
        else:
            tick.role_of(agent_id)  # raises if agent_id is not in the tick
            out.append(Token.other_offer(tick.action) if tick.accepted else OTHER_ANY)
    return tuple(out)


def contingency(ticks: Sequence[Tick], agent_id: int, alphabet: Alphabet) -> ContingencyResult:
    """Pooled lift of the partner's responses to agent_id's sync offers (§5.2)."""
    offers = [
        (tick.action, tick.response)
        for tick in ticks
        if tick.initiator == agent_id and tick.response is not None
    ]
    n = len(offers)
    labels = {sigma for sigma, _ in offers}
    if n == 0:
        return ContingencyResult(None, 0, 0)
    hits = [r == alphabet.complement(sigma) for sigma, r in offers]
    bases: list[float] = []
    for sigma, _ in offers:
        comp = alphabet.complement(sigma)
        others = [r for s, r in offers if s != sigma]
        if others:
            bases.append(sum(r == comp for r in others) / len(others))
    if not bases:
        return ContingencyResult(None, n, len(labels))
    value = sum(hits) / n - sum(bases) / len(bases)
    return ContingencyResult(float(value), n, len(labels))


def find_joint_cycle(
    trace: Sequence[Tick], agent_id: int, lo: int, hi: int, L_max: int
) -> Segment | None:
    """Shortest segment within [lo, hi], length <= L_max, starting on
    agent_id's turn and ending on the partner's turn, containing an accepted
    agent-initiated offer followed (later tick) by an accepted
    partner-initiated offer (§5.1). Ties: earliest start."""
    return _joint_cycle(trace, agent_id, lo, hi, L_max, min_end=lo)


def find_extension(
    trace: Sequence[Tick], agent_id: int, record_id: int, lo: int, hi: int, L_ext: int
) -> Segment | None:
    """§5.3 k >= 1: earliest SUCCESS probe event by agent_id for record_id at
    tick s whose matching START is at tick >= lo, followed by the first
    accepted partner-initiated offer at tick e with s < e <= min(s + L_ext, hi).
    Segment covers the extra [s+1, e]; consumed_from is the probe's START tick."""
    return _extension(trace, agent_id, record_id, lo, hi, L_ext, min_end=lo)


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
    if agent.kind != "mind" or agent.level >= cfg.k_max:
        return None
    hi = min(window.stop - 1, len(trace) - 1)
    if hi < window.start:
        return None
    lo = max(lo, 0)
    if agent.level == 0:
        kind = "joint_cycle"
        seg = _joint_cycle(trace, agent.id, lo, hi, cfg.L_max, min_end=window.start)
    else:
        kind = "extension"
        top = agent.library.top
        seg = _extension(trace, agent.id, top.id, lo, hi, cfg.L_ext, min_end=window.start)
    if seg is None:
        return None
    c = contingency(trace[window.start : hi + 1], agent.id, alphabet)
    if not c.holds(cfg):
        return None
    return Qualification(agent.id, partner_id, agent.level, kind, seg, c.value)


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
    ups: list[LevelUp] = []
    pairs = ((agent_i, agent_j, q_i), (agent_j, agent_i, q_j))

    def can(agent: Agent, q: Qualification | None) -> bool:
        return q is not None and agent.kind == "mind" and agent.level < cfg.k_max

    bootstrap = cfg.mode in ("bootstrap", "mixed")
    elder = cfg.mode in ("elder", "mixed")
    if bootstrap and agent_i.level == agent_j.level:
        if cfg.require_mutual:
            if can(agent_i, q_i) and can(agent_j, q_j):
                ups += [LevelUp(a.id, b.id, "bootstrap", q) for a, b, q in pairs]
            return ups
    for a, b, q in pairs:
        if not can(a, q):
            continue
        if elder and b.level >= a.level + 1:
            ups.append(LevelUp(a.id, b.id, "elder", q))
        elif bootstrap and not cfg.require_mutual:
            ups.append(LevelUp(a.id, b.id, "bootstrap", q))
    return ups


# --- private helpers ----------------------------------------------------------

def _joint_cycle(
    trace: Sequence[Tick], agent_id: int, lo: int, hi: int, L_max: int, min_end: int
) -> Segment | None:
    """find_joint_cycle restricted to segments ending at tick >= min_end.

    The shortest segment containing (accepted own offer a, later accepted
    partner offer b) is [a, b] itself, so it suffices to pair each b with the
    latest a < b."""
    lo = max(lo, 0)
    hi = min(hi, len(trace) - 1)
    best: tuple[int, int] | None = None  # (length, start)
    last_self: int | None = None
    for t in range(lo, hi + 1):
        tick = trace[t]
        if not tick.accepted:
            continue
        if tick.initiator == agent_id:
            last_self = t
        elif last_self is not None and t >= min_end:
            cand = (t - last_self + 1, last_self)
            if cand[0] <= L_max and (best is None or cand < best):
                best = cand
    if best is None:
        return None
    length, start = best
    end = start + length - 1
    return Segment(
        start=start,
        end=end,
        tokens=encode_segment(trace[start : end + 1], agent_id),
        anchor=trace[start].node_before(agent_id),
        consumed_from=start,
    )


def _extension(
    trace: Sequence[Tick],
    agent_id: int,
    record_id: int,
    lo: int,
    hi: int,
    L_ext: int,
    min_end: int,
) -> Segment | None:
    """find_extension restricted to extras ending at tick >= min_end."""
    lo = max(lo, 0)
    hi = min(hi, len(trace) - 1)
    start_tick: int | None = None  # START tick of agent_id's current probe
    for s in range(0, hi + 1):
        for ev in trace[s].probe_events:
            if ev.agent_id != agent_id:
                continue
            if ev.kind is ProbeEventKind.START:
                start_tick = s if ev.record_id == record_id else None
            elif ev.kind is ProbeEventKind.ABORT:
                start_tick = None
            elif ev.kind is ProbeEventKind.SUCCESS:
                probe_start, start_tick = start_tick, None
                if ev.record_id != record_id or probe_start is None or probe_start < lo:
                    continue
                e = next(
                    (
                        u
                        for u in range(s + 1, min(s + L_ext, hi) + 1)
                        if trace[u].accepted and trace[u].initiator != agent_id
                    ),
                    None,
                )
                if e is None or e < min_end:
                    continue
                return Segment(
                    start=s + 1,
                    end=e,
                    tokens=encode_segment(trace[s + 1 : e + 1], agent_id),
                    anchor=trace[s + 1].node_before(agent_id),
                    consumed_from=probe_start,
                )
    return None
