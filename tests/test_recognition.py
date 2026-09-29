"""G1 tests: recognition (spec §5)."""

from dataclasses import replace

import numpy as np
import pytest

from model.config import Config
from model.records import OTHER_ANY, SELF_ANY, SELF_TAU, Token
from model.trace import ProbeEventKind as K
from sim.recognition import (
    ContingencyResult,
    Qualification,
    Segment,
    contingency,
    encode_segment,
    find_extension,
    find_joint_cycle,
    level_condition,
    resolve_window,
)
from tests.helpers import (
    accepted_offer,
    build_trace,
    declined_offer,
    make_agent,
    probe_event,
    tau,
)

ME, PARTNER = 0, 1


# --- builders -----------------------------------------------------------------

def base_moves(alphabet, n):
    """n ticks, me first. My turns: offer answered with the complement but not
    accepted (a contingency hit that is encoded SELF_ANY and never closes a
    joint cycle). Partner turns: τ."""
    moves = []
    for t in range(n):
        if t % 2 == 0:
            s = (t // 2) % (2 * alphabet.m)
            moves.append((s, alphabet.complement(s), False))
        else:
            moves.append(tau(alphabet))
    return moves


def accept_at(alphabet, moves, *ticks):
    for t in ticks:
        s = moves[t][0] if moves[t][0] != alphabet.tau else 1
        moves[t] = accepted_offer(alphabet, s)
    return moves


def small_cfg(**kw):
    base = dict(W=10, T_ep=40, n_min_total=3, L_max=8, L_ext=6)
    base.update(kw)
    return Config(**base)


def fake_q(agent, partner, level=None):
    seg = Segment(0, 1, (Token.self_offer(0), Token.other_offer(0)), 0, 0)
    lvl = agent.level if level is None else level
    return Qualification(agent.id, partner.id, lvl, "joint_cycle", seg, 1.0)


@pytest.fixture
def agents(alphabet, rng):
    def build(level_i=0, level_j=0, kind_i="mind", kind_j="mind", cfg=None):
        cfg = cfg or Config()
        return (
            make_agent(ME, alphabet, cfg, rng, level_i, kind_i),
            make_agent(PARTNER, alphabet, cfg, rng, level_j, kind_j),
        )

    return build


# --- encoding -----------------------------------------------------------------

def test_encode_every_turn_type(alphabet):
    moves = [
        accepted_offer(alphabet, 0),   # me: accepted       -> SELF_OFFER(0)
        accepted_offer(alphabet, 4),   # partner: accepted  -> OTHER_OFFER(4)
        tau(alphabet),                 # me: τ              -> SELF_TAU
        declined_offer(alphabet, 2),   # partner: declined  -> OTHER_ANY
        declined_offer(alphabet, 1),   # me: declined       -> SELF_ANY
        tau(alphabet),                 # partner: τ         -> OTHER_ANY
        (3, alphabet.complement(3), False),  # me: complement but unavailable -> SELF_ANY
        (5, 0, False),                 # partner: wrong response -> OTHER_ANY
    ]
    trace = build_trace(moves)
    assert encode_segment(trace, ME) == (
        Token.self_offer(0), Token.other_offer(4), SELF_TAU, OTHER_ANY,
        SELF_ANY, OTHER_ANY, SELF_ANY, OTHER_ANY,
    )
    # From the partner's perspective roles swap.
    assert encode_segment(trace, PARTNER) == (
        Token.other_offer(0), Token.self_offer(4), OTHER_ANY, SELF_ANY,
        OTHER_ANY, SELF_TAU, OTHER_ANY, SELF_ANY,
    )
    assert encode_segment([], ME) == ()
    with pytest.raises(ValueError):
        encode_segment(trace, 7)


# --- contingency --------------------------------------------------------------

def test_contingency_perfect_responder(alphabet):
    trace = build_trace(base_moves(alphabet, 50))
    c = contingency(trace, ME, alphabet)
    assert c.value == 1.0
    assert c.n_offers == 25 and c.n_labels == 6
    assert c.holds(Config())


def test_contingency_ignores_partner_offers_and_tau(alphabet):
    moves = base_moves(alphabet, 50)
    moves[1] = (0, 5, False)  # partner's offer: not in O for me
    trace = build_trace(moves)
    assert contingency(trace, ME, alphabet).n_offers == 25
    # From the partner's side O is its single offer: one label, so None.
    assert contingency(trace, PARTNER, alphabet) == ContingencyResult(None, 1, 1)


def test_contingency_random_responder_near_zero(alphabet):
    """§11: an offer-independent responder scores |C| < 0.1 on average."""
    rng = np.random.default_rng(1)
    values = []
    for _ in range(400):
        moves = []
        for t in range(50):  # one default window: 25 offers
            if t % 2 == 0:
                s = int(rng.integers(2 * alphabet.m))
                r = int(rng.integers(alphabet.size))
                moves.append((s, r, r == alphabet.complement(s)))
            else:
                moves.append(tau(alphabet))
        c = contingency(build_trace(moves), ME, alphabet)
        if c.value is not None:
            values.append(c.value)
    assert len(values) > 390
    assert np.mean(np.abs(values)) < 0.1
    assert abs(np.mean(values)) < 0.03


def test_contingency_hand_computed(alphabet):
    # Offers: σ=0 answered 3 (hit), σ=0 answered 6 (miss), σ=1 answered 3.
    moves = [(0, 3, True), tau(alphabet), (0, 6, False), tau(alphabet), (1, 3, False)]
    c = contingency(build_trace(moves), ME, alphabet)
    # hit mean = 1/3. base: t0 (σ=0, comp 3): others {σ=1→3} → 1; t2 same → 1;
    # t4 (σ=1, comp 4): others {3, 6} → 0. base mean = 2/3.
    assert c.value == pytest.approx(1 / 3 - 2 / 3)
    assert (c.n_offers, c.n_labels) == (3, 2)


def test_contingency_none_and_too_few(alphabet):
    # No offers at all.
    c = contingency(build_trace([tau(alphabet)] * 6), ME, alphabet)
    assert c == ContingencyResult(None, 0, 0) and not c.holds(Config())
    # One label only: O' empty → None.
    moves = [accepted_offer(alphabet, 2), tau(alphabet)] * 20
    c = contingency(build_trace(moves), ME, alphabet)
    assert c.value is None and c.n_offers == 20 and c.n_labels == 1
    assert not c.holds(Config())
    # Perfect but too few offers for n_min_total.
    c = contingency(build_trace(base_moves(alphabet, 20)), ME, alphabet)
    assert c.value == 1.0 and c.n_offers == 10
    assert not c.holds(Config())
    assert c.holds(Config(n_min_total=10))
    # Below θ.
    assert not ContingencyResult(0.29, 30, 3).holds(Config())
    assert ContingencyResult(0.3, 15, 2).holds(Config())


# --- joint cycle --------------------------------------------------------------

def test_joint_cycle_shortest(alphabet):
    moves = accept_at(alphabet, base_moves(alphabet, 20), 0, 5, 12, 15)
    trace = build_trace(moves)
    seg = find_joint_cycle(trace, ME, 0, 19, 8)
    # Candidates: [0,5] (len 6) and [12,15] (len 4).
    assert (seg.start, seg.end, seg.consumed_from) == (12, 15, 12)
    assert seg.tokens == encode_segment(trace[12:16], ME)
    assert seg.tokens[0].kind.name == "SELF_OFFER" and seg.tokens[-1].kind.name == "OTHER_OFFER"


def test_joint_cycle_uses_latest_own_offer(alphabet):
    moves = accept_at(alphabet, base_moves(alphabet, 12), 0, 4, 7)
    seg = find_joint_cycle(build_trace(moves), ME, 0, 11, 8)
    assert (seg.start, seg.end) == (4, 7)


def test_joint_cycle_tie_earliest_start(alphabet):
    moves = accept_at(alphabet, base_moves(alphabet, 20), 2, 5, 10, 13)
    seg = find_joint_cycle(build_trace(moves), ME, 0, 19, 8)
    assert (seg.start, seg.end) == (2, 5)


def test_joint_cycle_order_matters(alphabet):
    # Partner's accepted offer precedes mine: no cycle.
    moves = accept_at(alphabet, base_moves(alphabet, 10), 1, 4)
    assert find_joint_cycle(build_trace(moves), ME, 0, 9, 8) is None


def test_joint_cycle_L_max(alphabet):
    moves = accept_at(alphabet, base_moves(alphabet, 10), 2, 7)  # length 6
    trace = build_trace(moves)
    assert find_joint_cycle(trace, ME, 0, 9, 5) is None
    assert (find_joint_cycle(trace, ME, 0, 9, 6).start) == 2


def test_joint_cycle_lo_hi(alphabet):
    moves = accept_at(alphabet, base_moves(alphabet, 10), 2, 5)
    trace = build_trace(moves)
    assert find_joint_cycle(trace, ME, 3, 9, 8) is None
    assert find_joint_cycle(trace, ME, 0, 4, 8) is None
    assert find_joint_cycle(trace, ME, 2, 5, 8) is not None
    assert find_joint_cycle(trace, ME, 0, 500, 8) is not None  # hi clamped


def test_joint_cycle_anchor(alphabet):
    moves = accept_at(alphabet, base_moves(alphabet, 10), 2, 5)
    trace = build_trace(moves)
    trace[2] = replace(trace[2], x_init_before=7, x_resp_before=9)
    assert find_joint_cycle(trace, ME, 0, 9, 8).anchor == 7


def test_joint_cycle_mirror_like_partner(alphabet):
    """A partner that answers perfectly but never initiates (τ on its turn)."""
    moves = []
    for t in range(50):
        moves.append(accepted_offer(alphabet, (t // 2) % 6) if t % 2 == 0 else tau(alphabet))
    trace = build_trace(moves)
    assert contingency(trace, ME, alphabet).value == 1.0
    assert find_joint_cycle(trace, ME, 0, 49, 8) is None


# --- extension ----------------------------------------------------------------

def ext_trace(alphabet, n=20, start=2, success=5, accepts=(7,), record=0, agent=ME, extra_events=None):
    moves = accept_at(alphabet, base_moves(alphabet, n), *accepts)
    events = {
        start: (probe_event(agent, K.START, record, 0),),
        success: (probe_event(agent, K.SUCCESS, record, 4),),
    }
    events.update(extra_events or {})
    return build_trace(moves, events=events)


def test_extension_basic(alphabet):
    trace = ext_trace(alphabet)
    seg = find_extension(trace, ME, 0, 0, 19, 6)
    assert (seg.start, seg.end, seg.consumed_from) == (6, 7, 2)
    assert seg.tokens == (SELF_ANY, Token.other_offer(accepted_offer(alphabet, 1)[0]))


def test_extension_first_accepted_partner_offer(alphabet):
    seg = find_extension(ext_trace(alphabet, accepts=(9, 11)), ME, 0, 0, 19, 6)
    assert seg.end == 9


def test_extension_requires_success_of_right_record(alphabet):
    assert find_extension(ext_trace(alphabet, record=1), ME, 0, 0, 19, 6) is None
    assert find_extension(ext_trace(alphabet, record=1), ME, 1, 0, 19, 6) is not None
    # The partner's success doesn't count for me.
    assert find_extension(ext_trace(alphabet, agent=PARTNER), ME, 0, 0, 19, 6) is None
    # START without SUCCESS (aborted) doesn't count.
    moves = accept_at(alphabet, base_moves(alphabet, 20), 7)
    trace = build_trace(moves, events={
        2: (probe_event(ME, K.START, 0),),
        4: (probe_event(ME, K.ABORT, 0, 2),),
    })
    assert find_extension(trace, ME, 0, 0, 19, 6) is None


def test_extension_start_must_be_at_or_after_lo(alphabet):
    trace = ext_trace(alphabet)
    assert find_extension(trace, ME, 0, 2, 19, 6).consumed_from == 2
    assert find_extension(trace, ME, 0, 3, 19, 6) is None


def test_extension_matching_start_is_latest(alphabet):
    # An earlier START for the same record was aborted; the SUCCESS matches the re-start.
    trace = ext_trace(alphabet, start=4, success=9, accepts=(11,), extra_events={
        0: (probe_event(ME, K.START, 0),),
        2: (probe_event(ME, K.ABORT, 0, 1),),
    })
    seg = find_extension(trace, ME, 0, 0, 19, 6)
    assert (seg.consumed_from, seg.start, seg.end) == (4, 10, 11)


def test_extension_L_ext_and_hi(alphabet):
    # success at 5: extra must end at e <= 5 + L_ext.
    trace = ext_trace(alphabet, accepts=(11,))
    assert find_extension(trace, ME, 0, 0, 19, 6).end == 11
    assert find_extension(trace, ME, 0, 0, 19, 5) is None
    assert find_extension(trace, ME, 0, 0, 10, 6) is None


def test_extension_earliest_success(alphabet):
    trace = ext_trace(alphabet, accepts=(7, 15), extra_events={
        10: (probe_event(ME, K.START, 0),),
        13: (probe_event(ME, K.SUCCESS, 0, 4),),
    })
    assert find_extension(trace, ME, 0, 0, 19, 6).end == 7
    # With lo past the first probe, the second one qualifies.
    seg = find_extension(trace, ME, 0, 6, 19, 6)
    assert (seg.consumed_from, seg.start, seg.end) == (10, 14, 15)


# --- level condition and window boundaries ------------------------------------

def test_level_condition_level0(alphabet, agents):
    cfg = small_cfg()
    i, _ = agents(cfg=cfg)
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 12, 15))
    q = level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg)
    assert q.kind == "joint_cycle" and q.level_from == 0
    assert (q.agent_id, q.partner_id) == (ME, PARTNER)
    assert (q.segment.start, q.segment.end) == (12, 15)
    assert q.contingency == 1.0


def test_level_condition_levelk(alphabet, agents):
    cfg = small_cfg()
    i, _ = agents(level_i=2, cfg=cfg)
    top = i.library.top.id
    moves = accept_at(alphabet, base_moves(alphabet, 40), 17)
    events = {12: (probe_event(ME, K.START, top),), 15: (probe_event(ME, K.SUCCESS, top, 4),)}
    trace = build_trace(moves, events=events)
    q = level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg)
    assert q.kind == "extension" and q.level_from == 2
    assert (q.segment.consumed_from, q.segment.start, q.segment.end) == (12, 16, 17)
    # A lower (closed-shell) record doesn't qualify.
    events = {12: (probe_event(ME, K.START, 0),), 15: (probe_event(ME, K.SUCCESS, 0, 4),)}
    trace = build_trace(moves, events=events)
    assert level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg) is None


def test_window_segment_may_start_in_previous_window(alphabet, agents):
    cfg = small_cfg()
    i, _ = agents(cfg=cfg)
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 8, 11))
    q = level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg)
    assert (q.segment.start, q.segment.end) == (8, 11)
    # ...but not before lo (previous episode or already-consumed ticks).
    assert level_condition(trace, i, PARTNER, range(10, 20), 10, alphabet, cfg) is None
    assert level_condition(trace, i, PARTNER, range(10, 20), 9, alphabet, cfg) is None


def test_window_segment_must_end_in_window(alphabet, agents):
    cfg = small_cfg()
    i, _ = agents(cfg=cfg)
    # A short cycle that ended in the previous window doesn't count now...
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 4, 5))
    assert level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg) is None
    # ...and doesn't hide a longer one that ends inside the window.
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 4, 5, 12, 17))
    q = level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg)
    assert (q.segment.start, q.segment.end) == (12, 17)
    # Nor one ending after the window.
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 18, 21))
    assert level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg) is None


def test_window_extension_must_end_in_window(alphabet, agents):
    cfg = small_cfg()
    i, _ = agents(level_i=1, cfg=cfg)
    moves = accept_at(alphabet, base_moves(alphabet, 40), 9)
    events = {2: (probe_event(ME, K.START, 0),), 5: (probe_event(ME, K.SUCCESS, 0, 4),)}
    trace = build_trace(moves, events=events)
    assert level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg) is None
    moves = accept_at(alphabet, base_moves(alphabet, 40), 11)
    trace = build_trace(moves, events=events)
    q = level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg)
    assert (q.segment.consumed_from, q.segment.start, q.segment.end) == (2, 6, 11)


def test_window_contingency_uses_window_only(alphabet, agents):
    cfg = small_cfg()
    i, _ = agents(cfg=cfg)
    moves = accept_at(alphabet, base_moves(alphabet, 40), 12, 15)
    # Ruin contingency outside the window: no effect.
    for t in range(0, 10, 2):
        moves[t] = (moves[t][0], alphabet.tau, False)
    trace = build_trace(moves)
    assert level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg) is not None
    # Ruin it inside the window (except the cycle's own accepted offer).
    for t in (10, 14, 16, 18):
        moves[t] = (moves[t][0], alphabet.tau, False)
    trace = build_trace(moves)
    assert level_condition(trace, i, PARTNER, range(10, 20), 0, alphabet, cfg) is None
    # Too few offers in the window.
    assert level_condition(
        build_trace(accept_at(alphabet, base_moves(alphabet, 40), 12, 15)),
        i, PARTNER, range(10, 20), 0, alphabet, small_cfg(n_min_total=6),
    ) is None


def test_level_condition_non_minds_and_k_max(alphabet, agents):
    cfg = small_cfg(k_max=2)
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 12, 15))
    for kind in ("object", "recording", "mirror"):
        c, _ = agents(kind_i=kind, cfg=cfg)
        assert level_condition(trace, c, PARTNER, range(10, 20), 0, alphabet, cfg) is None
    capped, _ = agents(level_i=2, cfg=cfg)
    top = capped.library.top.id
    events = {12: (probe_event(ME, K.START, top),), 13: (probe_event(ME, K.SUCCESS, top, 2),)}
    trace = build_trace(accept_at(alphabet, base_moves(alphabet, 40), 15), events=events)
    assert level_condition(trace, capped, PARTNER, range(10, 20), 0, alphabet, cfg) is None
    below, _ = agents(level_i=2, cfg=cfg)
    assert level_condition(trace, below, PARTNER, range(10, 20), 0, alphabet, small_cfg(k_max=3)) is not None


# --- resolve_window -----------------------------------------------------------

def ups(result):
    return sorted((u.agent_id, u.partner_id, u.source) for u in result)


def test_bootstrap_mutual(agents):
    cfg = Config()
    i, j = agents(1, 1)
    qi, qj = fake_q(i, j), fake_q(j, i)
    out = resolve_window(i, j, qi, qj, cfg)
    assert ups(out) == [(ME, PARTNER, "bootstrap"), (PARTNER, ME, "bootstrap")]
    assert {u.qualification for u in out} == {qi, qj}
    assert resolve_window(i, j, qi, None, cfg) == []
    assert resolve_window(i, j, None, qj, cfg) == []
    assert resolve_window(i, j, None, None, cfg) == []


def test_bootstrap_unequal_levels(agents):
    i, j = agents(1, 2)
    assert resolve_window(i, j, fake_q(i, j), fake_q(j, i), Config()) == []


def test_bootstrap_require_mutual_false(agents):
    cfg = Config(require_mutual=False)
    i, j = agents(1, 1)
    assert ups(resolve_window(i, j, fake_q(i, j), None, cfg)) == [(ME, PARTNER, "bootstrap")]
    assert ups(resolve_window(i, j, None, fake_q(j, i), cfg)) == [(PARTNER, ME, "bootstrap")]
    assert len(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == 2
    # Independent of the partner's level (one-sided recognition writes a record).
    i, j = agents(0, 3)
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [
        (ME, PARTNER, "bootstrap"), (PARTNER, ME, "bootstrap"),
    ]


def test_elder_mode(agents):
    cfg = Config(mode="elder")
    i, j = agents(1, 2)
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [(ME, PARTNER, "elder")]
    assert resolve_window(i, j, None, fake_q(j, i), cfg) == []
    i, j = agents(0, 3)  # any higher partner
    assert ups(resolve_window(i, j, fake_q(i, j), None, cfg)) == [(ME, PARTNER, "elder")]
    i, j = agents(4, 1)  # j is the learner
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [(PARTNER, ME, "elder")]
    # Equal levels: no bootstrap in elder mode.
    i, j = agents(1, 1)
    assert resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg) == []
    assert resolve_window(i, j, fake_q(i, j), fake_q(j, i), Config(mode="elder", require_mutual=False)) == []


def test_mixed_mode(agents):
    cfg = Config(mode="mixed")
    i, j = agents(1, 1)
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [
        (ME, PARTNER, "bootstrap"), (PARTNER, ME, "bootstrap"),
    ]
    assert resolve_window(i, j, fake_q(i, j), None, cfg) == []
    i, j = agents(1, 2)
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [(ME, PARTNER, "elder")]
    # require_mutual=false: learner still takes elder; the higher agent bootstraps alone.
    cfg = Config(mode="mixed", require_mutual=False)
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [
        (ME, PARTNER, "elder"), (PARTNER, ME, "bootstrap"),
    ]


@pytest.mark.parametrize("mode", ["bootstrap", "elder", "mixed"])
@pytest.mark.parametrize("require_mutual", [True, False])
@pytest.mark.parametrize("kind", ["object", "recording", "mirror"])
def test_controls_never_level_up(agents, mode, require_mutual, kind):
    cfg = Config(mode=mode, require_mutual=require_mutual)
    for li, lj in ((0, 0), (0, 1), (1, 0)):
        mind, ctrl = agents(li, lj, "mind", kind)
        out = resolve_window(mind, ctrl, fake_q(mind, ctrl), fake_q(ctrl, mind), cfg)
        assert all(u.agent_id == mind.id for u in out)
        out = resolve_window(ctrl, mind, fake_q(ctrl, mind), fake_q(mind, ctrl), cfg)
        assert all(u.agent_id == mind.id for u in out)
    # Under mutual bootstrap a control can never cause a level-up.
    if mode == "bootstrap" and require_mutual:
        mind, ctrl = agents(0, 0, "mind", kind)
        assert resolve_window(mind, ctrl, fake_q(mind, ctrl), fake_q(ctrl, mind), cfg) == []


def test_k_max_cap(agents):
    for mode in ("bootstrap", "mixed"):
        cfg = Config(mode=mode, k_max=2)
        i, j = agents(2, 2)
        assert resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg) == []
        assert resolve_window(i, j, fake_q(i, j), fake_q(j, i), Config(mode=mode, k_max=3)) != []
    cfg = Config(mode="bootstrap", require_mutual=False, k_max=2)
    i, j = agents(2, 1)
    assert ups(resolve_window(i, j, fake_q(i, j), fake_q(j, i), cfg)) == [(PARTNER, ME, "bootstrap")]
    cfg = Config(mode="elder", k_max=2)
    i, j = agents(2, 3)
    assert resolve_window(i, j, fake_q(i, j), None, cfg) == []
