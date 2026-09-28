# Lab Notebook — Mutual Recognition Graph Model

Running log for the project. Entries are appended, never rewritten. Where an entry is superseded, a later entry says so.

- **Experiments:** every run that produces results, including calibration and failed runs.
- **Decisions:** design choices, parameter changes, and project process.
- **Glossary:** terms coined for this project, with where they're defined.

Spec: `mutual-recognition-model-spec.md` (currently v0.2).

---

## Experiments

| ID | Date | Segment | Config / commit | Seeds | Question | Result | Notes |
|---|---|---|---|---|---|---|---|
| _none yet_ | | | | | | | |

---

## Decisions

### D1 — 2026-09-28 — Spec v0.1 → v0.2
Reviewed v0.1 before building. Changes:

1. **Controls judged one-sided.** Under mutual bootstrap, a control can never cause a level-up, so counting level-ups made E0/E2 pass trivially. False recognition is now measured by the mind's one-sided condition (§5.4).
2. **Pooled contingency.** A 50-tick window has about 20 offers over 6 labels, so the per-label minimum of 5 was rarely met. Lift is now pooled across all offers, and `n_min` is removed (§5.2).
3. **Growth adds nodes.** Previously nothing activated new nodes, so the N_max budget was unused. Each unlock now activates one node reached from the record's anchor (§6).
4. **q_edge 0.15 → 0.05.** At 0.15 about 93% of labels were already available at any node, leaving growth nothing to add. At 0.05 it's about 56%.
5. **`SELF_ANY` token.** Declined self offers previously had no encoding (§3.3).
6. **Elder transmission defined.** A plain self↔other swap broke token alternation, and REFs pointed into the elder's library. Now the swapped word is phase-shifted with wildcard padding, and the REF points to the learner's own record (§5.7).
7. **Edge cases settled:** partner τ during a probe, segments crossing window boundaries, recording and mirror behavior (§4.3, §5, §8).
8. **Calibration run C0 added** as a gate before E0–E5 (§10).

v0.1 was backed up only to a session scratchpad and not kept in the project.

### D2 — 2026-09-28 — Project process
- This notebook logs experiments, decisions and glossary terms.
- Work is split into gated segments. Within a segment, independent work runs in parallel with subagents, each in its own git worktree. A segment's gate must pass before the next begins.

### D3 — 2026-09-28 — Segment plan and gate approvals

| Segment | Work | Parallel | Gate |
|---|---|---|---|
| G0 Foundation | Repo, env, config, types, frozen interfaces | serial | Tests run; interfaces frozen |
| G1 Core modules | graph · records · recognition · agent policy · controls · product analysis | 6 worktrees | All unit and invariant tests pass; merged |
| G2 Integration | Episode, runner, metrics, outputs | serial | Reproducibility, isolation, smoke run |
| G3 Calibration C0 | Tune to the pass criterion | serial | **User approval** |
| G4a Controls E0 | Mind vs each control | serial | Zero one-sided recognitions |
| G4b Experiments E1–E5 | One per experiment | 5 worktrees | Results logged here |
| G5 Synthesis | Plots and write-up | serial | **User approval** |

The user approves gates only at G3 and G5. Other gates pass automatically when their criteria are met.

### D4 — 2026-09-28 — G0 implementation choices
Filling gaps the spec leaves open:
- Python 3.14 via uv, dense numpy bool arrays only (no sparse path; N_max = 64).
- Labels are ints; τ = 2m. Record ids are library indices, and each library holds exactly one record per level.
- `other` on a record stores a record id, not an object reference (§12, no object cycles).
- The trace `Tick` gains a `responder` field.
- A policy protocol (`act` / `respond` / `observe` / `reset_episode`) is shared by minds and controls. Probe advancement happens in `observe`, after the tick resolves.
- Joint-cycle ties (same length) are broken by earliest start.
- Anchor of a level ≥ 2 record = the agent's node at the start of the extra segment.
- A recording's script position persists across episodes; a recording "declines" by returning its script label even when unavailable.
- Config adds `controls` (kind → count), `output_dir` and `record_traces`.
- Git identity for this repo is set locally to Luca / the user's email.

---

## Glossary

| Term | Meaning | Defined |
|---|---|---|
| Loop record / tech-tree item | Immutable encoding of a recognized self → other → self loop, reusable as a probe | spec §3.3 |
| Probe | Executing a stored record's word against a partner to try to reproduce the loop | §4.3 |
| Joint cycle | A trace segment where my accepted offer is followed by the partner's accepted offer | §5.1 |
| Contingency (C) | Pooled lift: how much more often the partner answers an offer correctly than it would answer a different offer the same way | §5.2 |
| Extra | The segment after a successful probe that ends with an accepted partner-initiated offer; the new material in a level-up | §5.3 |
| Closed shell | A non-top record: usable as a probe, but it can't qualify an agent for the next level | §5.3 |
| Bootstrap | Mutual level-up of two equal-level agents in the same window | §5.4 |
| Elder transmission | A learner levels up by receiving a higher-level partner's record | §5.4, §5.7 |
| One-sided recognition | An agent satisfies its own level condition against a partner, whether or not the partner does; used to measure false recognition | §5.4 |
| Pairing by analogy | A new record's `other` slot points to the agent's own lower record | §5.5 |
| Saturation | A level whose extras repeat the previous level's; the common-knowledge fixed point | §5.6 |
| Phase shift (transmission) | Padding a perspective-swapped word with `SELF_ANY … OTHER_ANY` so it starts on self and ends on other | §5.7 |
| Generalize (growth) | On unlock: add one node, and add self-loops for learned labels everywhere | §6 |
| Controls: object / recording / mirror | Non-minds that should never trigger recognition | §8 |
| Private language | Interlocking probes between partners who bootstrapped together | §10 E5 |
| Repeat-partner fraction | Share of level-ups with a partner the agent has leveled up with before | §10 |
