"""Per-epoch metrics and end-of-run summaries (spec §10)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from model.agent import Agent
from model.config import Config
from model.records import SELF, Record, flat
from sim.episode import EpisodeResult, probe_counts


@dataclass
class EpochAccumulator:
    levelups_bootstrap: int = 0
    levelups_elder: int = 0
    repeat_levelups: int = 0
    probe_starts: int = 0
    probe_successes: int = 0
    mind_windows: int = 0
    contingency_computable: int = 0
    one_sided_vs_mind: int = 0
    one_sided_vs_control: int = 0
    mind_control_windows: int = 0

    def add_episode(self, result: EpisodeResult, partner_history: dict[int, set[int]]) -> None:
        """Fold one episode in. partner_history (agent → partners it has
        leveled up with) is read, then updated with this episode's level-ups."""
        for up in result.level_ups:
            if up.source == "bootstrap":
                self.levelups_bootstrap += 1
            else:
                self.levelups_elder += 1
            if up.partner_id in partner_history[up.agent_id]:
                self.repeat_levelups += 1
        for up in result.level_ups:
            partner_history[up.agent_id].add(up.partner_id)
        starts, successes = probe_counts(result.trace)
        self.probe_starts += starts
        self.probe_successes += successes
        for w in result.window_stats:
            self.mind_windows += 1
            self.contingency_computable += w.contingency_computable
            if w.partner_kind == "mind":
                self.one_sided_vs_mind += w.one_sided
            else:
                self.mind_control_windows += 1
                self.one_sided_vs_control += w.one_sided


def _frac(num: int, den: int) -> float:
    return num / den if den else float("nan")


def epoch_row(epoch: int, minds: list[Agent], acc: EpochAccumulator, cfg: Config) -> dict:
    levels = np.array([a.level for a in minds])
    hist = np.bincount(levels, minlength=cfg.k_max + 1)
    levelups = acc.levelups_bootstrap + acc.levelups_elder
    row = {
        "epoch": epoch,
        "mean_level": float(levels.mean()) if len(levels) else 0.0,
        "max_level": int(levels.max()) if len(levels) else 0,
        "frac_level_ge2": float((levels >= 2).mean()) if len(levels) else 0.0,
        "levelups_bootstrap": acc.levelups_bootstrap,
        "levelups_elder": acc.levelups_elder,
        "repeat_partner_frac": _frac(acc.repeat_levelups, levelups),
        "probe_starts": acc.probe_starts,
        "probe_successes": acc.probe_successes,
        "probe_success_rate": _frac(acc.probe_successes, acc.probe_starts),
        "saturated": sum(r.saturated for a in minds for r in a.library),
        "contingency_computable_frac": _frac(acc.contingency_computable, acc.mind_windows),
        "one_sided_vs_mind": acc.one_sided_vs_mind,
        "one_sided_vs_control": acc.one_sided_vs_control,
        "mind_control_windows": acc.mind_control_windows,
        "false_positive_rate": _frac(acc.one_sided_vs_control, acc.mind_control_windows),
    }
    for k, n in enumerate(hist):
        row[f"n_level_{k}"] = int(n)
    return row


def record_to_dict(rec: Record) -> dict:
    other = "SELF" if rec.other is SELF else rec.other
    return {
        "id": rec.id,
        "level": rec.level,
        "tokens": [repr(t) for t in rec.tokens],
        "anchor": rec.anchor,
        "other": other,
        "partner_id": rec.partner_id,
        "saturated": rec.saturated,
        "source": rec.source,
    }


def library_dump(agents: list[Agent]) -> dict:
    return {
        str(a.id): {"kind": a.kind, "level": a.level, "records": [record_to_dict(r) for r in a.library]}
        for a in agents
    }


def word_lengths_by_level(minds: list[Agent]) -> dict[int, dict[str, float]]:
    """Mean flat vs compressed word length per record level (§10)."""
    by_level: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a in minds:
        for r in a.library:
            by_level[r.level].append((len(flat(r, a.library)), len(r.tokens)))
    return {
        k: {
            "n": len(v),
            "flat_mean": float(np.mean([f for f, _ in v])),
            "compressed_mean": float(np.mean([c for _, c in v])),
        }
        for k, v in sorted(by_level.items())
    }


def distinct_flat_words(minds: list[Agent]) -> dict[int, int]:
    """Structural diversity: number of distinct flattened words per level (E1)."""
    words: dict[int, set] = defaultdict(set)
    for a in minds:
        for r in a.library:
            words[r.level].add(flat(r, a.library))
    return {k: len(v) for k, v in sorted(words.items())}
