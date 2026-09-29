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
| S0 | 2026-09-29 | G2 | smoke: 10 minds + 1 of each control, 20 epochs | 0 | Does the pipeline run end to end? | Yes. 0.5 s; mean level 0.8, max 2; 0 false positives in 8 mind–control windows; contingency computable in 83% of windows | Scratch run, outputs not kept |
| C0 | 2026-09-29 | G3 | C0.yaml (defaults, 100 epochs) | 0–9 | Does development happen at defaults, and is contingency measurable? | **Fail.** Level ≥2: 19% (target 10%, pass). Contingency computable: 78% (target 80%, fail). Probe success 1–2% at every level | outputs/C0 |
| C0-W100 | 2026-09-29 | G3 | C0 + W=100 | 0–9 | Does a longer window fix computability (tuning order: W first)? | **Fail.** Computable 77%, level ≥2 20%, level ≥1 fell to 55% | Longer windows didn't help, so the problem isn't sample size |
| C0-diag | 2026-09-29 | G3 | C0, seed 0, last 10 epochs | 0 | Why is contingency not computable? | 18% of windows: only one offered label; 14%: fewer than 15 offers; the 5th percentile of offers is 0. Agents get trapped in sparse regions of their graph with 0–2 sync labels | Caused by the q_edge 0.05 change in D1 |
| C0-q08 | 2026-09-29 | G3 | C0 + q_edge=0.08 | 0–9 | Does modestly denser wiring fix it? | **Pass.** Computable 89%, level ≥1 89%, level ≥2 32%, level ≥3 13%, mean max level 3.5 | Initial label availability ≈ 74% |
| C0-q10 | 2026-09-29 | G3 | C0 + q_edge=0.10 | 0–9 | Same, denser | **Pass.** Computable 90%, level ≥1 87%, level ≥2 30%, level ≥3 9% | Availability ≈ 81% |
| E0 | 2026-09-29 | G4a | E0.yaml: 20 minds + 4 object, 4 recording, 4 mirror; ER p=0.2; 500 epochs; θ=0.3 | 0–99 | Do minds ever recognize controls? | **Gate fails.** Object 0 / 458,668 windows; mirror 0 / 460,660; **recording 417 / 455,464 (0.092%)**. Minds still develop: 26% at level ≥2, max level 10 | outputs/E0; 100 distinct trace hashes |
| E0-diag | 2026-09-29 | G4a | E0 with 12 recordings | 0–11 | Why do recordings get recognized? | Chance lift: C sits just above θ (median 0.33) with 15–24 offers; 45% of cases are 2-label windows. 55% at level 0, 43% at level 1 | Objects and mirrors are excluded by structure (no initiative); recordings only statistically |
| C0-θ04 | 2026-09-29 | G4a | C0 + θ=0.4 | 0–9 | Does a stricter threshold hurt development? | Barely: level ≥2 30% (vs 32%), computable 87%, still passes | |
| E0-θ04 | 2026-09-29 | G4a | E0 + θ=0.4 | 0–31 | False recognitions at θ=0.4? | Recording 15 / 147,752 (**0.010%**, ~9× lower); object and mirror 0; level ≥2 23% (vs 26%) | outputs/E0_theta04 |

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

### D5 — 2026-09-29 — G1 gate passed; judgment calls by module
Correction to D4: the environment runs **Python 3.12.14** (uv picked it under `requires-python >= 3.12`), not 3.14.

G0 addendum: graph basics, `flat`, `extras`, `is_alternating` and `own_labels` were implemented before G1, so no parallel module was blocked on another.

All six modules were merged into `main`, with 205 tests passing. Choices made where the spec is silent:

- **Graph:** `activate_node` rejects an inactive anchor. Growth mode names are validated. At the node budget, self-loops are still added.
- **Records:** a level-1 record is never saturated. Elder records are checked for saturation against the learner's own top. `transmit` rejects non-alternating input.
- **Recognition:**
  - A qualifying segment must end inside the current window, so an earlier segment can't mask a later valid one.
  - An extension whose first accepted partner offer falls before the window is skipped.
  - With `require_mutual = false`, each qualifying mind levels up regardless of its partner's level (§5.4 "independently").
  - In mixed mode, a lower-level learner who qualifies always takes the elder route.
  - An offer-independent responder scores mean |C| ≈ 0.065 over 50-tick windows.
- **Mind policy:**
  - The reactivity fallback is uniform over available labels and can itself hit the complement. So the effective acceptance rate is ρ + (1−ρ)/|available|, not ρ.
  - A matched `OTHER_OFFER` whose complement is unavailable becomes a decline, and the probe aborts.
  - `SELF_TAU` advances only on an actual τ.
- **Controls:** an object's uniform response can be τ, which declines. A recording keeps its own τ moves in its script, and an empty script plays τ. Capture skips only ticks where the partner played τ.
- **Product analysis:** the full identity is used in I⊗A, so pairs of an active and an inactive node get τ edges. Those states are unreachable, so results are unaffected. A declined-offer pass needs only the mover to have a sync label.

### D6 — 2026-09-29 — G2 gate passed; integration choices
The episode loop, runner, metrics and CLI are done, with 215 tests passing. The reproducibility, isolation, monotonicity, record well-formedness and control invariants are all tested.
- **Window lookback:** `lo = max(first unused tick, start of previous window)`. After any qualifying segment, whether or not it led to a level-up, the agent's first unused tick moves past the segment end.
- **Record writing:** within a window, all level-up records are built before any is appended, so an elder's record is read at its pre-window level.
- **Contact graphs:** `pairs` = (0,1), (2,3), …; `small_world` = a ring lattice with 2 neighbours per side, each edge rewired with probability `p_contact`. Per-epoch matching is a random maximal matching over the contact edges.
- **Population:** controls join the contact graph like minds. A recording's script comes from one throwaway episode between two fresh minds, drawn from the run's single RNG stream.
- **Outputs** per run go to `outputs/<config>_s<seed>/`: `metrics.csv`, `library.json`, `summary.json` (trace hash, word lengths by level, distinct flat words), `config.yaml`, and optionally `traces.jsonl`.

### D7 — 2026-09-29 — G3 gate passed (user approved): q_edge = 0.08
C0 failed at q_edge 0.05 because contingency was computable in only 78% of windows. The cause was agents trapped in sparse graph regions, not window length (C0-W100, C0-diag). The user approved **q_edge = 0.08** as the new default: it passes with 89% computable and 32% of agents at level ≥2. This partly reverts D1 item 4. Initial label availability is ≈ 74%, which still leaves room for growth. The other defaults are unchanged (W = 50). Spec §6 and §9 are updated.

Observation to follow up in E4: probe success is low (1–3%) and falls with record level.

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
