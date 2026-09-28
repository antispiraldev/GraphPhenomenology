"""Simulation config (spec §9). Loaded from YAML; unknown keys are an error."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

import yaml

Contact = Literal["pairs", "ring", "erdos_renyi", "small_world"]
Mode = Literal["bootstrap", "elder", "mixed"]
Growth = Literal["generalize", "none"]


@dataclass
class Config:
    m: int = 3
    N_max: int = 64
    N0: int = 16
    q_edge: float = 0.05
    N_agents: int = 20
    contact: Contact = "erdos_renyi"
    p_contact: float = 0.2
    epochs: int = 500
    T_ep: int = 200
    W: int = 50
    L_max: int = 8
    L_ext: int = 6
    rho: float = 0.5
    p_probe: float = 0.3
    beta: float = 1.0
    theta: float = 0.3
    n_min_total: int = 15
    mode: Mode = "bootstrap"
    require_mutual: bool = True
    growth: Growth = "generalize"
    k_max: int = 10
    seed: int = 0
    # Population extras (not in §9): number of control agents of each kind,
    # added on top of N_agents minds, e.g. {"mirror": 2}.
    controls: dict[str, int] = field(default_factory=dict)
    output_dir: str = "outputs"
    record_traces: bool = False

    def __post_init__(self) -> None:
        if self.N0 > self.N_max:
            raise ValueError("N0 must be <= N_max")
        if self.T_ep % self.W != 0:
            raise ValueError("T_ep must be a multiple of W")
        if self.mode not in ("bootstrap", "elder", "mixed"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.growth not in ("generalize", "none"):
            raise ValueError(f"unknown growth {self.growth!r}")
        for kind in self.controls:
            if kind not in ("object", "recording", "mirror"):
                raise ValueError(f"unknown control kind {kind!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
