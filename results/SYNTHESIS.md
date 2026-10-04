# G5 Synthesis — Mutual recognition graph model

Spec v0.3 (`mutual-recognition-model-spec.md`). Code state: tag `v0.3-results`, 218 tests passing. Prepared 2026-10-01 for the G5 review (NOTEBOOK D3).

This document brings the project together in one place. The per-experiment reports in `results/E*_v03.md` hold the full method, tables and caveats. The v0.2 reports (`results/E*.md`, tag `v0.2-results`) are kept as the baseline. All numbers below come from those reports.

![overview](G5_overview.png)

`G5_overview.png` is built by `uv run python -m experiments.G5_overview`, which reads the saved v0.3 outputs (see §7).

## 1. What was built

A population of minds. Each mind is a labelled directed graph, 16 vertices to start, with every edge present with probability q_edge = 0.08. Each edge carries one of three move labels a₁–a₃, its complement ā₁–ā₃, or the solo move τ.

**Encounters.** Pairs meet on a contact graph and take turns. An offer succeeds only if the partner answers with the complementary move and both have the matching edges. The pair therefore walks on the product of the two graphs.

**Recognition** requires three things in the same window:
- a joint cycle (I serve, you return; you serve, I return);
- pooled contingency, meaning the partner's answers depend on my offers (lift C ≥ θ = 0.4);
- mutuality.

**What a recognition does.** Each mind stores the shared loop as a record at its next level. Higher levels are reached by replaying a record (a *probe*) and extending it with a new exchange. Each level-up teaches the mind new moves everywhere (self-loops) and grows one vertex. Under v0.3 that vertex has a τ exit back to its anchor.

**Controls.** Objects never initiate and answer at random, mirrors never initiate but answer perfectly, and recordings replay a real mind but ignore what you do.

## 2. How the spec changed

| Version | Change | Why | Where |
|---|---|---|---|
| v0.1 → v0.2 | Controls judged one-sided; pooled contingency; growth adds nodes; elder transmission defined; calibration gate C0 | Review before building: several criteria were trivially passed or undefined | D1 |
| G3 | q_edge 0.05 → **0.08** (user approved) | At 0.05, contingency was computable in only 78% of windows | D7 |
| G4a | θ 0.3 → **0.4**; recording criterion becomes a rate below 1 in 5,000 windows *(delegated)* | Recordings fooled minds 0.092% of the time by chance; no finite θ reaches exactly zero | D8 |
| v0.2 → v0.3 | Grown vertices get a τ edge back to their anchor *(delegated)* | Three experiments independently found 79–100% of developed minds trapped on absorbing grown vertices. This was a design error introduced in D1 | D9, D10 |

Every v0.2 experiment was re-run under v0.3, and each report lists which conclusions survived, changed or reversed.

## 3. The hypotheses (spec §14)

| Hypothesis | Experiment | Verdict under v0.3 | Key numbers |
|---|---|---|---|
| A sharp threshold in development as coupling increases | E3 | **Partly.** There is a sharp threshold in reactivity ρ, but it is a property of each encounter, not a collective phase transition | φ onset ρ_c ≈ 0.31, width w ≈ 0.034 at N = 10, 20 and 40 (no sharpening); p_contact and q_edge give smooth curves |
| Probing accelerates development autocatalytically | E4 | **Not supported.** Probing is necessary past level 1 and more of it helps, but each step is harder than the last, even for the same minds | Fixed cohort (158 minds reaching L5): hazard 17.5 → 8.6 → 5.8 → 5.0% per equal-level meeting at steps 1→2 … 4→5 |
| Pairs who bootstrapped together form a private language | E5 | **Supported**, and not an artifact of the v0.2 trap | Co-builder vs equal-level stranger: top-probe success 3.7× [3.1, 4.3]; level-ups 4.8× [3.2, 6.9]; above L2 a stranger almost never completes the probe |
| Lowering θ raises development speed and false recognition together | E2 | **Supported for speed, not for final development.** Below θ ≈ 0.5 the final level is flat; above 0.5 development stalls | Recording rate 4.8% at θ = 0, 9e-4 at 0.3, 4.4e-5 at 0.4; level ≥ 2 flat at 0.68–0.73 for θ 0.2–0.5; level ≥ 1 falls 0.80 → 0.35 → 0.03 at θ 0.5 → 0.6 → 0.7 |

## 4. Other results

**E0, controls (gate).** Over 100 worlds, objects were recognized in 0 of 454,604 windows and mirrors in 0 of 453,688. Recordings were recognized in 25 of 451,328 windows (0.006%), inside the D8 limit of 0.02%. Objects and mirrors are excluded by structure: neither ever initiates, so a joint cycle can't close (objects also show no contingency). Recordings are excluded only statistically, which is why they need a rate criterion.

**E1, elder vs bootstrap.**
- Only equals can bootstrap together, so under the default rule a transplanted level-5 elder adds nothing: −0.22 ± 0.15 levels.
- When newcomers may learn from anyone ahead of them ("elder mode"), the group starts at once and every newcomer develops: mean level 3.96 vs 2.38 at epoch 400.
- Allowing both routes is best (4.20), and adding the elder to it changes nothing (0.00 ± 0.12).
- Elder mode copies: exactly one distinct content per level, the elder's. Bootstrap invents the most distinct content at low levels.

**Calibration (C0) under v0.3.** 83% of minds reach level ≥ 1 and 71% level ≥ 2. Contingency is computable in 99.7% of windows.

## 5. What it adds up to

In this toy world, recognition behaves the way the motivating traditions suggest in three respects:
- It comes in discrete steps.
- It needs initiative from both sides (mirrors fail) and contingency (recordings fail): Hegel's mutuality, and the live mother against the replay.
- It produces partner-specific shared structure that strangers can't use.

Two expectations were not borne out:
- **Development does not feed on itself.** Deeper levels need longer shared routines, and a probe has to survive more exchanges, so each step is rarer. Nothing in the model makes a deeper record *easier* to extend.
- **The tipping point lives in the encounter, not the crowd.** Whether a pair can bootstrap depends on how reactive each partner is. Population size only changes how many such pairs there are. What limits the population is *stranding*: 27.5% of minds end with no neighbour at their own level.

A further point is methodological. The v0.2 → v0.3 episode showed how strongly one structural detail (an exit edge) shapes every result. Several v0.2 conclusions reversed, among them "growth hurts", "fixed pairs lose to shuffled contact" and "elders are slow teachers". Conclusions from this model should be read as conditional on the growth rule.

## 6. Limitations

- **Seeds.** v0.3 re-runs used 12 seeds per arm in E1, E4 and E5, and 4–24 per point in E2 and E3, to fit the compute budget. The main contrasts have non-overlapping intervals, but secondary contrasts are noted as "no clear difference" in the reports.
- **Fixed parameters.** All results are at m = 3 labels, N₀ = 16, W = 50, ρ = 0.5 (except E3) and p_probe = 0.3 (except E4).
- **The growth rule.** The τ exit is one of several possible repairs, and the results depend on it (§5).
- **θ is a fixed threshold**, not a significance test. That was deferred in D8.
- **Unexplained:** the elder-mode plateau below the elder's level (E1); why shuffled populations stall at L3 (E5).

## 7. Reproducing

- Tags: `v0.2-results` (spec v0.2), `v0.3-results` (spec v0.3).
- Each `results/E*_v03.md` lists the exact commands.
- Raw outputs are not committed (`outputs/` is gitignored). The G4c worktree outputs were copied into `outputs/archive/` before the worktrees were removed (commit 42bda23). `experiments/G5_overview.py` reads from there and from `outputs/E0_v03/`.

## 8. Decisions for the reviewer

These are the decisions made on the user's behalf after 2026-09-29. Each is logged in NOTEBOOK.md and marked *delegated — for user review*:

1. **D8, θ = 0.4 and the recording rate criterion.** Re-checked under v0.3 in E2: still the lowest θ that meets the criterion, now with more margin.
2. **D10, the τ exit for grown vertices (spec v0.3).** Every headline result is under v0.3.

Possible next questions, if the project continues (none started):
- Why does each level get harder (E4)? Candidates are probe length and the first-exchange filter.
- A significance-based recognition test in place of the fixed θ.
- Remedies for stranding, for example letting minds seek same-level partners.
- Vary the elder's level, and diagnose the elder-mode plateau.
