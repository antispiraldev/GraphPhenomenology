"""E5 instrumented loop: same RNG stream as the runner, and helper sanity."""

from model.config import Config
from model.records import OTHER_ANY, SELF_ANY, Token
from experiments.E5 import interlock, run_instrumented
from sim.runner import run


def test_instrumented_loop_matches_runner_trace_hash():
    for contact, p in (("pairs", 0.2), ("erdos_renyi", 1.0)):
        cfg = Config(N_agents=8, epochs=12, contact=contact, p_contact=p, seed=2)
        res = run_instrumented(cfg)
        assert res["trace_hash"] == run(cfg).trace_hash
        assert res["final_levels"] == [a.level for a in run(cfg).agents]
        n_enc = sum(v[0] for v in res["encounters"].values())
        assert n_enc > 0 and n_enc % 2 == 0


def test_interlock_detects_perspective_swap():
    # i: offers 0, j offers 1;  j's view of the same exchange, phase-shifted by one turn
    e_i = (Token.self_offer(0), Token.other_offer(1))
    e_j = (SELF_ANY, Token.other_offer(0), Token.self_offer(1), OTHER_ANY)
    assert interlock(e_i, e_j) == 2
    assert interlock(e_i, (Token.self_offer(2), Token.other_offer(0))) == 0
