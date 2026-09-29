"""G0 tests: actions, config, token/record types, library, trace helpers."""

import pytest

from model.actions import Alphabet
from model.config import Config
from model.records import (
    OTHER_ANY,
    SELF,
    SELF_ANY,
    Library,
    Record,
    Token,
    TokenKind,
)
from tests.helpers import accepted_offer, build_trace, tau


def test_alphabet_complement_is_involution(alphabet):
    for s in alphabet.sync:
        assert alphabet.complement(s) != s
        assert alphabet.complement(alphabet.complement(s)) == s
    with pytest.raises(ValueError):
        alphabet.complement(alphabet.tau)
    assert alphabet.size == 7
    assert [alphabet.name(x) for x in alphabet.labels] == ["a1", "a2", "a3", "ā1", "ā2", "ā3", "τ"]


def test_config_defaults_and_yaml(tmp_path):
    cfg = Config.from_yaml("experiments/default.yaml")
    assert cfg == Config()
    assert cfg.q_edge == 0.08
    p = tmp_path / "bad.yaml"
    p.write_text("n_min: 5\n")
    with pytest.raises(ValueError):
        Config.from_yaml(p)
    with pytest.raises(ValueError):
        Config(T_ep=210, W=50)


def test_token_validation():
    assert Token.self_offer(2).is_self
    assert OTHER_ANY.is_other and not OTHER_ANY.is_self
    ref = Token.make_ref(0)
    assert not ref.is_self and not ref.is_other
    with pytest.raises(ValueError):
        Token(TokenKind.SELF_OFFER)
    with pytest.raises(ValueError):
        Token(TokenKind.SELF_ANY, label=1)
    assert Token.self_offer(1) == Token.self_offer(1)
    assert hash(SELF_ANY) == hash(Token(TokenKind.SELF_ANY))


def _rec(id, level, **kw):
    base = dict(tokens=(Token.self_offer(0), Token.other_offer(1)), anchor=0,
                other=None, partner_id=1, saturated=False, source="bootstrap")
    base.update(kw)
    return Record(id=id, level=level, **base)


def test_library_append_only():
    lib = Library()
    assert lib.level == 0 and lib.top is None and lib.next_id == 0
    lib.append(_rec(0, 1))
    with pytest.raises(ValueError):
        lib.append(_rec(1, 3))
    with pytest.raises(ValueError):
        lib.append(_rec(5, 2))
    lib.append(_rec(1, 2, other=0))
    assert lib.level == 2 and lib.top.id == 1 and len(lib) == 2
    with pytest.raises(AttributeError):
        lib[0].level = 9


def test_self_sentinel_singleton():
    import pickle
    assert pickle.loads(pickle.dumps(SELF)) is SELF


def test_build_trace_alternates(alphabet):
    trace = build_trace([accepted_offer(alphabet, 0), tau(alphabet), accepted_offer(alphabet, 4)])
    assert [t.initiator for t in trace] == [0, 1, 0]
    assert trace[1].response is None and not trace[1].accepted
    assert trace[2].role_of(1) == "other"
