"""Action alphabet (spec §3.1).

Labels are ints:
  0 .. m-1      a_1 .. a_m
  m .. 2m-1     ā_1 .. ā_m
  2m            τ (internal / solo move)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Alphabet:
    m: int

    def __post_init__(self) -> None:
        if self.m < 1:
            raise ValueError("m must be >= 1")

    @property
    def tau(self) -> int:
        return 2 * self.m

    @property
    def size(self) -> int:
        """|Σ| = 2m + 1."""
        return 2 * self.m + 1

    @property
    def sync(self) -> range:
        return range(2 * self.m)

    @property
    def labels(self) -> range:
        return range(self.size)

    def is_sync(self, label: int) -> bool:
        return 0 <= label < 2 * self.m

    def complement(self, label: int) -> int:
        """Involutive complement on sync labels. Raises on τ."""
        if not self.is_sync(label):
            raise ValueError(f"complement undefined for label {label}")
        return (label + self.m) % (2 * self.m)

    def name(self, label: int) -> str:
        if label == self.tau:
            return "τ"
        if 0 <= label < self.m:
            return f"a{label + 1}"
        if self.m <= label < 2 * self.m:
            return f"ā{label - self.m + 1}"
        raise ValueError(f"unknown label {label}")
