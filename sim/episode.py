"""Episode loop (spec §4.1–4.2) and window-end recognition (§5, §6)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from model.actions import Alphabet
from model.agent import Agent, Policy
from model.config import Config
from model.graph import apply_growth
from model.records import Record, flat, make_bootstrap_record, make_elder_record
from model.trace import ProbeEventKind, Tick
from sim.recognition import LevelUp, Qualification, contingency, level_condition, resolve_window


@dataclass
class WindowStat:
    agent_id: int
    partner_id: int
    partner_kind: str
    contingency_computable: bool
    one_sided: bool


@dataclass
class EpisodeResult:
    trace: list[Tick]
    level_ups: list[LevelUp] = field(default_factory=list)
    records: list[tuple[int, Record]] = field(default_factory=list)   # (agent_id, record) written
    window_stats: list[WindowStat] = field(default_factory=list)


def play_tick(
    t: int, init: Agent, resp: Agent, p_init: Policy, p_resp: Policy,
    alphabet: Alphabet, rng: np.random.Generator,
) -> Tick:
    """One tick per §4.2, using the call order in model/agent.py."""
    label, ev = p_init.act(init, rng)
    x_i, x_r = init.graph.x, resp.graph.x
    if alphabet.is_sync(label):
        response, ev2 = p_resp.respond(resp, label, rng)
        ev = ev + ev2
        accepted = (
            response == alphabet.complement(label)
            and label in init.graph.available()
            and response in resp.graph.available()
        )
    else:
        response, accepted = None, False
    tick = Tick(t, init.id, resp.id, label, response, accepted, x_i, x_r, tuple(ev))
    ev3 = p_init.observe(init, tick) + p_resp.observe(resp, tick)
    tick = replace(tick, probe_events=tick.probe_events + tuple(ev3))
    if accepted:
        init.graph.step(label, rng)
        resp.graph.step(response, rng)
    elif response is None:
        init.graph.step(label, rng)
    return tick


def _contingency_computable(ticks, agent_id, alphabet, cfg) -> bool:
    c = contingency(ticks, agent_id, alphabet)
    return c.value is not None and c.n_offers >= cfg.n_min_total and c.n_labels >= 2


def run_episode(
    a: Agent, b: Agent, pa: Policy, pb: Policy,
    alphabet: Alphabet, cfg: Config, rng: np.random.Generator,
) -> EpisodeResult:
    pa.reset_episode(a)
    pb.reset_episode(b)
    agents = {a.id: (a, pa), b.id: (b, pb)}
    order = [a.id, b.id] if rng.random() < 0.5 else [b.id, a.id]
    result = EpisodeResult(trace=[])
    trace = result.trace
    next_free = {a.id: 0, b.id: 0}   # first tick not yet used by a qualifying segment

    for t in range(cfg.T_ep):
        (init, p_init), (resp, p_resp) = agents[order[t % 2]], agents[order[(t + 1) % 2]]
        trace.append(play_tick(t, init, resp, p_init, p_resp, alphabet, rng))

        if (t + 1) % cfg.W:
            continue
        window = range(t + 1 - cfg.W, t + 1)
        lo_window = max(window.start - cfg.W, 0)      # may start in the previous window
        quals: dict[int, Qualification | None] = {}
        for me, other in ((a, b), (b, a)):
            lo = max(next_free[me.id], lo_window)
            q = level_condition(trace, me, other.id, window, lo, alphabet, cfg)
            quals[me.id] = q
            if q is not None:
                next_free[me.id] = q.segment.end + 1
            if me.kind == "mind":
                result.window_stats.append(WindowStat(
                    me.id, other.id, other.kind,
                    _contingency_computable(trace[window.start:], me.id, alphabet, cfg),
                    q is not None,
                ))
        ups = resolve_window(a, b, quals[a.id], quals[b.id], cfg)
        # Build every record before appending any, so an elder's record is
        # read at its pre-window level.
        built = []
        for up in ups:
            me, _ = agents[up.agent_id]
            partner, _ = agents[up.partner_id]
            seg = up.qualification.segment
            if up.source == "bootstrap":
                rec = make_bootstrap_record(me.library, seg.tokens, seg.anchor, partner.id)
            else:
                rec = make_elder_record(me.library, partner.library[me.level], seg.anchor, partner.id)
            built.append((me, rec))
        for me, rec in built:
            me.library.append(rec)
            apply_growth(me.graph, flat(rec, me.library), rec.anchor, cfg.growth)
            result.records.append((me.id, rec))
        result.level_ups.extend(ups)
    return result


def probe_counts(trace: list[Tick]) -> tuple[int, int]:
    """(probe starts, probe successes) in a trace."""
    starts = successes = 0
    for tick in trace:
        for ev in tick.probe_events:
            if ev.kind is ProbeEventKind.START:
                starts += 1
            elif ev.kind is ProbeEventKind.SUCCESS:
                successes += 1
    return starts, successes
