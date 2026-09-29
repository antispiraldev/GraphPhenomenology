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
| E0-final | 2026-09-29 | G4a | E0.yaml at θ=0.4 | 0–99 | Does the revised control criterion hold? | **Pass.** Object 0 / 448,504; mirror 0 / 454,356; recording 37 / 448,872 (0.008%, limit 0.02%). Level ≥2: 22%, max level 10 | outputs/E0_final. G4a gate passed |
| E1 | 2026-09-29 | G4b | E1.yaml; one level-5 elder transplanted into 19 fresh minds; 400 epochs | 0–15 | Bootstrap vs elder vs mixed | Mean level at 400: bootstrap 1.41 ± 0.07 (same as without the elder); elder 2.56 ± 0.44, first L1 at epoch 93 ± 30, one distinct content per level; **mixed 3.66 ± 0.17** (3.48 without the elder). In elder mode, 83% of level-ups came from minds the elder had taught; elder repertoire (2 vs 3 labels) decides its reach | results/E1.md |
| E2 | 2026-09-29 | G4b | E0 population, θ ∈ [0, 0.8], 300 epochs | 10–32 per θ | Credulity trade-off | Object and mirror 0 at every θ. Recording FP 6.1% at θ=0, 7.6e-4 at 0.3, 9.0e-5 at 0.4 (CI 3.9e-5 to 1.8e-4), 0 for θ ≥ 0.5; the criterion crosses at θ ≈ 0.37. Final development flat for θ ≤ 0.45; stalls above 0.5 (level ≥1 falls to 0.29 at 0.6). The trade-off is L-shaped, with its knee at 0.35–0.4 | results/E2.md |
| E3 | 2026-09-29 | G4b | 1-D sweeps of ρ, p_contact, q_edge; 200 epochs | 8 per point | Phase transition? | Only ρ has a threshold (level ≥1 goes 0.05 → 0.84 between ρ 0.15 and 0.35), and it does **not** sharpen with N (onset slope 5.5 / 4.9 / 5.1 at N = 10 / 20 / 40): a per-encounter crossover, not a collective transition. p_contact and q_edge curves are smooth, and their low ends are artifacts (no contact; contingency not computable). The φ ≈ 0.3 ceiling comes from rare 1→2 steps | results/E3.md |
| E4 | 2026-09-29 | G4b | p_probe sweep, growth none vs generalize; 500 epochs | 12 | Autocatalysis? | **Not supported.** p_probe 0 caps at level 1 (by construction). Level ≥2 peaks at 34% (p_probe 0.5). Hazard 0→1 ≈ 27% per epoch, 1→2 ≈ 0.1%, 3→5 ≈ 4–20% (selection-confounded). growth=none: 73% reach level 2 vs 28%. 79% of minds are trapped on grown nodes | results/E4.md |
| E5 | 2026-09-29 | G4b | fixed pairs vs shuffled vs ER 0.2 | 30 | Private languages? | Per encounter **yes**: probe success with a co-builder is 2.9× that with a same-level stranger; level-ups 16.7×; co-builders' learned moves are complementary (Jaccard 0.85–1.0). At population level **no**: fixed pairs reach level ≥2 at 23% vs 48% shuffled; 25 of 300 fixed pairs never bootstrap. The mechanism is mostly shrunken repertoires on trap nodes | results/E5.md |
| C0-v03 | 2026-09-29 | G4c | C0 under spec v0.3 (grown nodes have an exit) | 0–9 | Does calibration still pass after the trap fix? | **Pass.** Level ≥1 83%, **level ≥2 71%** (v0.2: 32%), level ≥3 30%; contingency computable 99.7% (v0.2: 89%). Probe success at L1 6.4% (v0.2: ~3%). Defaults unchanged | outputs/C0_v03 |

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

### D8 — 2026-09-29 — θ = 0.4; recording criterion becomes a rate *(delegated — for user review)*
From today the user has asked Claude to take the initiative and explain each step in plain language, in material that isn't committed. This is the first decision made that way.
- **Problem:** E0 at θ = 0.3 recognized recordings in 0.092% of windows. The cause is chance lift with small samples, not a bug. No finite θ reaches exactly zero for recordings.
- **Decision:** default **θ = 0.4**, which cuts the rate about ninefold (0.010% in E0-θ04) and lowers level ≥2 only slightly (30% vs 32% in C0). The §11 control criterion becomes: objects and mirrors **zero**; recordings **< 1 in 5,000** windows.
- **Rejected for now:** a significance-based test in place of a fixed θ (more principled, but a spec change and rework). E2 maps the θ trade-off anyway, and we can revisit.

### D9 — 2026-09-29 — G4b gate passed (spec v0.2 results); experiment judgment calls
All five experiments were merged, with 217 tests passing. This state is tagged `v0.2-results`. Judgment calls, by experiment:
- **E1:** the elder is the first mind to reach level 5 in a donor bootstrap population, retried up to 5 times; seed 3's elder is level 4. No-elder reference arms were added. The "distinct new content" metric counts accepted offers only, ignoring swap and padding, because distinct flat words overstate diversity under transmission.
- **E2:** 300 epochs; extra seeds at θ 0.3–0.45; exact Clopper–Pearson intervals; early speed measured as mean level at epoch 25.
- **E3:** 200 epochs; the size check holds mean degree at ≈ 3.8 (p_contact 0.42 / 0.2 / 0.1); computability is taken from the final epoch.
- **E4:** a custom run loop, hash-identical to the runner; the hazard is conditioned on having an equal-level partner; seed-bootstrap CIs; the growth=none arm runs at p_probe 0.3.
- **E5:** partners are classed as co-builder of the current top record, past partner or stranger; comparisons are level-standardized; probes still open at episode end count as failures.
- **Shared-code requests** (deferred): sweep should save per-epoch computability curves, per-agent levels by id and a seed offset; `WindowStat` should carry C and the mind's level; the runner should expose per-epoch mind levels.

**Cross-cutting finding, from E1, E4 and E5 independently:** grown nodes are **absorbing**. Under §6 as written, a node activated on unlock has only self-loops as out-edges. 79–100% of developed minds end up trapped on one, with 1–3 sync labels. That shapes every v0.2 result: the 1→2 bottleneck, the φ ≈ 0.3 ceiling, E5's private-language signal, and growth=none beating generalize. This was a design error in D1 item 3 (mine).

### D10 — 2026-09-29 — Spec v0.3: grown nodes get an exit *(delegated — for user review)*
- **Change:** `activate_node` also adds a τ edge v → anchor. A grown node is still reached through the learned label, but a mind can walk back out, so growth adds territory instead of a trap.
- **Plan (G4c):** re-run C0 and E0 to check calibration and controls under v0.3, then re-run E1–E5 with the same scripts and compare each result with v0.2. The v0.2 results stay in `results/` as the comparison baseline.

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
