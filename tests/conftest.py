import numpy as np
import pytest

from model.actions import Alphabet
from model.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config()


@pytest.fixture
def alphabet() -> Alphabet:
    return Alphabet(3)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)
