# Mutual Recognition Graph Model — Specification v0.2

**Changes from v0.1:**
1. Controls are judged by **one-sided recognition** (§5.4), so E0 and E2 can actually fail.
2. Contingency is a **pooled** lift over all offers in the window (§5.2). The per-label minimum `n_min` is removed.
3. Growth also **activates a new node** on each unlock (§6). Default `q_edge` is lowered to 0.05 so availability starts well below 100%.
4. New token `SELF_ANY` encodes my turns that weren't accepted offers or τ moves, such as declined offers (§3.3).
5. Elder transmission is defined precisely, with a phase shift that keeps records well-formed (§5.7).
6. Edge cases are settled: partner τ during a probe (§4.3), segments crossing window boundaries (§5), and control behavior (§8).
7. New calibration run **C0** before the experiments (§10).

## 1. Purpose

A minimal simulation of the hypothesis that some forms of awareness develop only through contact between two distinct minds, in discrete jumps rather than continuously.

Agents are labeled graphs. They interact in coupled pairs. When a pair jointly closes a self → other → self loop that is contingent and mutual, each agent stores an encoding of that loop. The encoding becomes a reusable probe (a "tech-tree item") that makes further recognition more likely.

The model should answer:

- Under what conditions does mutual bootstrapping occur at all?
- Is there a sharp threshold (phase transition) between "no development" and "cascading development" as coupling parameters vary?
- Does probing with unlocked loops produce autocatalytic acceleration?
- Do non-minds (objects, recordings, mirrors) correctly fail to trigger recognition?

Development is assumed **additive and monotonic**. Nothing is ever removed, and stored loops are immutable.

## 2. Concept map

| Concept | Formal element |
|---|---|
| Mind | Agent: labeled Boolean graph + library of loop records |
| Perceptual repertoire | Actions available at the agent's current node |
| Interaction | Coupled pair taking alternating turns |
| Recognition | Qualifying joint cycle + contingency + mutuality (§5) |
| Awareness level / valence | Level of the agent's top loop record |
| Closed inner shells | Lower-level records (immutable, referenced but not re-derived) |
| Husserlian pairing | New record's `other` slot points to the agent's *own* lower record |
| Internalization (Vygotsky) | Unlocked record persists and is usable without the partner |
| Scanning for a pattern | Probe execution (§4.3) |
| Common knowledge | Saturated fixed-point record (§5.6) |
| Private language | Pairs who bootstrapped together have interlocking probes |

## 3. Core objects

### 3.1 Actions

- `m` synchronizing action pairs: Σ_sync = {a₁ … aₘ, ā₁ … āₘ}, with an involutive complement: complement(aᵢ) = āᵢ and complement(āᵢ) = aᵢ.
- One internal action `τ` (solo move).
- Σ = Σ_sync ∪ {τ}.

Any sync action σ can be **offered**. It is **accepted** when the partner responds with complement(σ).

### 3.2 Agent graph

- Fixed node budget `N_max`, with an active mask `V ⊆ {0 … N_max−1}` (initially `N0` active nodes).
- For each σ ∈ Σ, a Boolean matrix `A[σ] ∈ {0,1}^{N_max × N_max}`. `A[σ][u,v] = 1` means an edge u → v labeled σ.
- Edges exist only between active nodes.
- Current position `x ∈ V`.
- `available(x) = { σ : row x of A[σ] has a nonzero entry }`.

Initialization: `N0` random active nodes; each ordered pair (u,v) of active nodes gets an edge for label σ independently with probability `q_edge`. Every active node must have at least one outgoing τ edge; add a τ self-loop if needed.

### 3.3 Loop records

A record is immutable:

```
Record(
  id: int,
  level: int,                    # ≥ 1
  tokens: tuple[Token, ...],     # compressed word
  anchor: int,                   # node id in the owner's graph at segment start
  other: Record | SELF | None,   # None at level 1; own level-(k−1) record for level k; SELF if saturated
  partner_id: int,               # provenance only; never read by the policy
  saturated: bool,
  source: "bootstrap" | "elder"  # provenance only; never read by the policy
)
```

Tokens alternate strictly between self turns and other turns, starting with a self token and ending with an other token:

- `SELF_OFFER(σ)`: on my turn I offered σ and it was accepted.
- `SELF_TAU`: on my turn I made a τ move.
- `SELF_ANY`: my turn, with no constraint (wildcard). Encodes a self turn that was neither an accepted offer nor a τ move, i.e. a declined offer.
- `OTHER_OFFER(σ)`: on the partner's turn they offered σ and I accepted with complement(σ).
- `OTHER_ANY`: the partner's turn, with no constraint (wildcard).
- `REF(record_id)`: stands for the full token sequence of a lower record. It must be followed by a self token, which holds automatically because referenced records end on an other token.

`flat(record)` recursively expands every `REF` into a plain alternating token sequence.

### 3.4 Agent state

```
Agent(
  id, graph (A, V, x),
  library: list[Record],   # append-only
  level: int,              # = max record level, 0 if empty
  probe: ProbeState | None,
  kind: "mind" | "object" | "recording" | "mirror"
)
```

## 4. Interaction dynamics

### 4.1 Contact and episodes

- There are `N_agents` agents on a contact graph (`pairs`, `ring`, `erdos_renyi(p_contact)`, `small_world`).
- In each epoch, sample a matching on the contact graph, so each agent is in at most one pair.
- Each pair runs one **episode** of `T_ep` ticks.
- Positions persist across episodes. Probe state resets at episode start.

### 4.2 Tick semantics

Within an episode, turns alternate strictly: the first initiator is random, then they alternate. On each tick:

1. The **initiator** chooses an action using its policy (§4.3).
2. If the action is a sync offer σ, the **responder** chooses a response label r using its policy.
   - The offer is **accepted** iff r = complement(σ) and both labels are available at the respective current nodes.
   - If accepted, both agents move to a uniformly random successor along their label's edges.
   - If declined, neither agent moves.
3. If the action is τ, the initiator moves along a τ edge. The responder does nothing.
4. The tick is appended to the episode trace.

### 4.3 Policy (kind = "mind")

**On my turn:**
1. If a probe is in progress, execute its next self token:
   - `SELF_OFFER(σ)`: offer σ. If σ is unavailable at my node, abort the probe and fall through to step 3.
   - `SELF_TAU`: make a τ move.
   - `SELF_ANY`: act as in step 3 (explore). The probe advances whatever the outcome.
2. Else, with probability `p_probe` and a non-empty library, start a probe. Choose a record with probability ∝ exp(β · level) and execute its first self token.
3. Else, explore: choose uniformly among `available(x)`.

**On the partner's turn, when they offer σ:**
- If I'm in a probe and the expected token is `OTHER_OFFER(σ)`, accept (respond complement(σ)) if it's available, and advance the probe.
- If I'm in a probe and the expected token is `OTHER_OFFER(σ′)` with σ′ ≠ σ, abort the probe, then respond by the reactivity rule.
- If the expected token is `OTHER_ANY`, respond by the reactivity rule and advance the probe.
- Otherwise, use the **reactivity rule**: with probability `ρ`, respond complement(σ) if available; otherwise respond with a uniformly random label from `available(x)`. Responding τ, or any non-complement label, declines the offer.

**On the partner's turn, when they make a τ move:**
- If the expected token is `OTHER_OFFER(σ)`, abort the probe.
- If the expected token is `OTHER_ANY`, advance the probe.

**Probe success:** every token in the flattened word was matched in order. `SELF_OFFER` and `OTHER_OFFER` tokens require the offer to be accepted. A probe aborts if an offer under a `SELF_OFFER` token is declined. Offers under `SELF_ANY` or `OTHER_ANY` may be declined without aborting.

### 4.4 Trace

Each tick record: `(t, initiator_id, action, response, accepted, x_init_before, x_resp_before, probe_events)`.

`probe_events` logs probe start, advance, success, or abort, with the record id, for both agents.

## 5. Recognition

Recognition is checked at the end of every window of `W` ticks (non-overlapping windows within an episode). Each agent evaluates the conditions from its own perspective. The simulator then applies the mutuality rule (§5.4).

**Window boundaries:** a qualifying segment (joint cycle, probe execution plus extra) counts in the window where it **ends**. It may start in the previous window of the same episode, but never in a previous episode. Contingency (§5.2) uses only the current window's ticks. Each tick can belong to at most one qualifying segment per agent, so a segment that crossed into a window is not counted again later.

### 5.1 Joint cycle (the loop passes through the other)

From agent i's perspective, a **joint cycle** is a trace segment containing:

- at least one tick where i initiated a sync offer that j accepted, **followed by**
- at least one tick where j initiated a sync offer that i accepted.

Closure is defined by turn structure, not by returning to the anchor node. A partner that never initiates can't complete a joint cycle, and this is what excludes mirrors.

### 5.2 Contingency

Over the window, let O be the set of ticks where i offered a sync action σₜ and j responded with rₜ. Lift is pooled over all of them, because a window holds only about 20 offers spread over 2m labels:

```
hit(t)      = 1 if rₜ = complement(σₜ) else 0
base(t)     = fraction of ticks s ∈ O with σₛ ≠ σₜ where rₛ = complement(σₜ)
C_ij        = mean over t ∈ O of hit(t)  −  mean over t ∈ O′ of base(t)
```

O′ is the set of ticks in O for which at least one other tick has a different offer label, so base(t) is defined. The first term measures how often j answers correctly. The second measures how often j would give that same answer to a *different* offer, which is the chance level for an agent that ignores offers.

Contingency holds iff |O| ≥ `n_min_total`, O contains at least two distinct labels, and `C_ij ≥ θ`. An agent whose responses are independent of the offer scores C ≈ 0. A mirror scores C = 1.

### 5.3 Level conditions (agent i at level k)

- **k = 0 → 1:** the window contains a joint cycle segment of length ≤ `L_max`, and contingency holds.
- **k ≥ 1 → k+1:** the window contains a successful execution of i's top record ℓₖ, followed within `L_ext` ticks by at least one accepted **partner-initiated** offer outside that probe (the "extra"). Contingency must also hold.

Only the top record qualifies an agent for the next level. Lower records are closed shells: they can be probed, but they don't lead upward.

### 5.4 Mutuality and modes

`mode` selects the level-up rule:

- **bootstrap** (default): agents i and j both level up iff they are at the **same** level k and both satisfy their §5.3 condition in the **same** window. If `require_mutual = false`, each agent levels up independently whenever its own condition holds. Use this for ablation.
- **elder:** i levels up iff `level(j) ≥ k+1` and i satisfies its condition. i receives a transmitted copy of j's level-(k+1) record (§5.7). This is transmission, with no novelty. j is unchanged.
- **mixed:** both rules are active. They never conflict: bootstrap applies when levels are equal, elder when the partner is higher.

The simulator enforces mutuality because recognition must be reciprocal. Agents cannot verify each other's internal states.

**One-sided recognition.** Agent i has a one-sided recognition of j in a window iff i satisfies its own §5.3 condition against j, regardless of j's level, kind or condition. It is logged every window and never writes a record by itself (except when `require_mutual = false`). This is the quantity used to measure false recognition of controls (E0, E2). Controls never write records, so under mutual bootstrap they can never cause a level-up. Counting level-ups would make the control test pass trivially.

### 5.5 Writing the record (bootstrap)

Agent i appends a new record:

- **level** = k + 1.
- **tokens:**
  - Level 1: the tokens of the shortest qualifying joint-cycle segment, starting on one of i's turns and ending on one of j's.
  - In both cases, encode self turns as `SELF_OFFER(σ)` if accepted, `SELF_TAU` if τ, and otherwise `SELF_ANY`. Encode other turns as `OTHER_OFFER(σ)` if accepted, and otherwise `OTHER_ANY`.
  - Level k+1: `REF(ℓₖ.id)` followed by the extra segment's tokens, from the probe's end through the tick of the accepted partner offer.
- **anchor:** i's node at the start of the segment.
- **other:** None at level 1; otherwise a reference to i's **own** ℓₖ. This is pairing by analogy: i assumes the partner carries what i carries, and it can be wrong.
- **partner_id:** j.
- **source:** "bootstrap".
- **level** field of the agent becomes k + 1.

Level increases by exactly one per event.

### 5.6 Saturation (fixed point)

Define `extras(ℓ₁) = tokens(ℓ₁)` and `extras(ℓₖ₊₁)` = the tokens after the leading `REF`.

If `extras(ℓₖ₊₁) == extras(ℓₖ)`, the new level added nothing new. Mark it `saturated = True` and set `other = SELF`. This is the common-knowledge fixed point.

Saturated records behave normally in the policy. Saturation is primarily an analysis signal.

### 5.7 Writing the record (elder transmission)

Learner i (level k) receives elder j's level-(k+1) record ℓ′. Let e = extras(ℓ′), which alternates self/other and starts with a self token.

**Perspective swap** maps each token to what it looked like from i's side:

| j's token | i's token |
|---|---|
| `SELF_OFFER(σ)` | `OTHER_OFFER(σ)` |
| `SELF_TAU`, `SELF_ANY` | `OTHER_ANY` |
| `OTHER_OFFER(σ)` | `SELF_OFFER(σ)` |
| `OTHER_ANY` | `SELF_ANY` |

swap(e) starts with an other token and ends with a self token, so it is **phase-shifted** by one turn:

```
transmitted(e) = (SELF_ANY,) + swap(e) + (OTHER_ANY,)
```

This alternates correctly, starting with self and ending with other. The record written is:

- **level** = k + 1.
- **tokens:** at level 1, transmitted(e); at level k+1 > 1, `REF(i's own ℓₖ.id)` + transmitted(e). REF always points into i's own library, never j's.
- **anchor:** i's node at the start of the qualifying segment.
- **other:** as in §5.5.
- **partner_id:** j. **source:** "elder".
- Saturation (§5.6) and growth (§6) apply as usual.

Composition (§11) holds for transmitted records too, because the REF is i's own.

## 6. Growth on unlock

When a record is written, apply `growth` (default `generalize`):

- **generalize:**
  1. **New node.** If |V| < N_max, activate one new node v. Add an edge anchor → v labeled with the first own-move label in `flat(record)`, or τ if there is none. Give v a τ self-loop. The record thus opens new territory in the graph.
  2. **Self-loops.** For every label used by my own moves in `flat(record)`, add a self-loop at every active node, including v: `A[label][u,u] = 1`. That includes σ in `SELF_OFFER(σ)` and complement(σ) in `OTHER_OFFER(σ)`. `SELF_TAU` and the wildcards contribute no label. A learned move then becomes usable anywhere, the tech-tree property, and later probes of this record are always feasible.
- **none:** the graph is unchanged. Use this for ablation.

All growth is monotone: edges and active nodes are only ever added. Once |V| = N_max, step 1 is skipped. With N0 = 16 and N_max = 64, an agent can grow for 48 unlocks, far beyond k_max.

The default `q_edge = 0.05` makes a label available at a fresh node with probability about 1 − 0.95¹⁶ ≈ 0.56. At the v0.1 value of 0.15 it was ≈ 0.93, which left growth almost nothing to add.

## 7. Analysis layer (not used in the main loop)

For static compatibility analysis of a pair (i, j), build the product transition relation on V_i × V_j:

```
M = A_i[τ] ⊗ I  +  I ⊗ A_j[τ]  +  Σ_{σ ∈ Σ_sync} A_i[σ] ⊗ A_j[complement(σ)]
```

This uses Kronecker products over the Boolean semiring (OR, AND).

- Use it to compute reachable product states from (x_i, x_j) via Boolean transitive closure, and **pair compatibility** = the number of reachable product states with at least one outgoing sync edge.
- To model turn alternation exactly, augment the state with a parity bit, doubling the dimension.
- Never materialize M inside the simulation loop.

## 8. Control agents (sanity tests)

| Kind | Initiates? | Responds | Expected outcome |
|---|---|---|---|
| object | never (τ on its turn) | uniform random over available labels | no recognition (fails 5.1 and 5.2) |
| recording | offers the next script label | the next script label, whatever was offered | no recognition (C ≈ 0) |
| mirror | never (τ on its turn) | always complement(σ) (ρ = 1) | no recognition (fails 5.1 despite C = 1) |

Controls have their own graphs. They never write records. "No recognition" means zero **one-sided** recognitions by the mind (§5.4).

**Recording details.** The script is the sequence of labels one real mind emitted, on its own turns and as responses, over one full episode against another mind, captured in a separate pre-run with the same seed stream. The recording consumes one script label per tick, whether it is initiating or responding. If the label is unavailable at its node, it makes a τ move on its turn or declines when responding. The script still advances. When the script ends, it loops.

**Mirror details.** A mirror's graph has a self-loop for every label at every node, so complement(σ) is always available and its contingency is always perfect.

## 9. Parameters

| Name | Default | Meaning |
|---|---|---|
| m | 3 | sync action pairs (\|Σ\| = 2m + 1) |
| N_max / N0 | 64 / 16 | node budget / initial active nodes |
| q_edge | 0.05 | initial edge probability per label per node pair |
| N_agents | 20 | population size |
| contact | "erdos_renyi" | contact graph type |
| p_contact | 0.2 | ER edge probability |
| epochs | 500 | number of epochs |
| T_ep / W | 200 / 50 | ticks per episode / window length |
| L_max / L_ext | 8 / 6 | max level-1 segment length / max extra length |
| ρ | 0.5 | reactivity |
| p_probe / β | 0.3 / 1.0 | probe rate / bias toward higher-level records |
| θ | 0.3 | contingency threshold |
| n_min_total | 15 | minimum offers per window for contingency |
| mode | "bootstrap" | bootstrap \| elder \| mixed |
| require_mutual | true | enforce reciprocity |
| growth | "generalize" | generalize \| none |
| k_max | 10 | level cap |
| seed | 0 | RNG seed |

## 10. Measurements and experiments

**Per-epoch metrics:**
- level histogram, mean and max level;
- level-up events by mode;
- probe start and success rates;
- flat vs compressed word length per level;
- saturation count;
- repeat-partner fraction: the share of level-ups with a partner the agent previously leveled up with (a private-language signal);
- one-sided recognition count, and whether the partner was a mind or a control;
- false-positive rate against controls: one-sided recognitions of controls per mind-control window.

**Calibration (run first):**
- **C0 — Does anything develop?** Run the default config for 100 epochs over 10 seeds. Report the fraction of agents reaching levels 1, 2 and 3, the probe start and success rates by record level, and the fraction of windows where contingency is computable (|O| ≥ n_min_total with two or more labels). **Pass criterion:** at least 10% of agents reach level 2 by epoch 100, and contingency is computable in at least 80% of windows. If it fails, tune in this order: `W`, `p_probe`, `T_ep`, `ρ`. Write the tuned values back into §9 as the new defaults, noting the change, before running E0–E5.

**Experiments:**
- **E0 — Controls.** Pair a mind with each control kind across many seeds. Expect zero one-sided recognitions at default θ. Run minds at several levels, since higher-level minds probe more.
- **E1 — Bootstrap vs elder.** Seed one high-level agent in elder mode. Compare development speed and structural diversity (distinct flattened words) against bootstrap mode.
- **E2 — Credulity sweep.** Sweep θ from 0 to 0.8. Look for the trade-off between false recognition of controls (one-sided rate) and stalled development among minds.
- **E3 — Phase transition.** Sweep ρ, p_contact and q_edge. Order parameter: the fraction of agents at level ≥ 2 at the final epoch. Look for a sharp threshold.
- **E4 — Autocatalysis.** Compare p_probe = 0 with p_probe > 0. Probes should accelerate development.
- **E5 — Private languages.** Compare fixed pairs with a shuffled contact graph, and measure the repeat-partner fraction and level-up rates.

## 11. Invariants and tests

- **Monotonicity:** after every tick, `A[σ](t) ≤ A[σ](t+1)` elementwise and `V(t) ⊆ V(t+1)`. The library is append-only, and records are frozen.
- **Isolation:** a single agent with no contact never levels up.
- **Discreteness:** every level-up increments level by exactly 1.
- **Composition:** `flat(ℓₖ₊₁) == flat(ℓₖ) + tokens after the REF`.
- **Controls:** E0 holds statistically (e.g. zero one-sided recognitions of controls in 100 seeds × 500 epochs). A unit test also checks that a mind with `require_mutual = false` *would* level up against another mind in the same setup, so the control test can fail.
- **Contingency calibration:** on synthetic traces, an offer-independent responder scores |C| < 0.1 on average and a perfect responder scores C = 1.
- **Elder transmission:** transmitted records satisfy token alternation and composition, and their REF points into the learner's own library.
- **Growth:** after each unlock with |V| < N_max, |V| increases by exactly 1.
- **Reproducibility:** identical config and seed produce an identical trace hash.
- **Token alternation:** every flattened word alternates self and other, starting with self and ending with other.

## 12. Implementation notes

- Python 3.11+, numpy. Use dense Boolean arrays for N_max ≤ 256, otherwise scipy.sparse CSR. Seed a single `numpy.random.Generator` and pass it down.
- Suggested layout: `model/actions.py`, `model/graph.py`, `model/records.py`, `model/agent.py` (policy and probes), `model/controls.py`, `sim/episode.py`, `sim/recognition.py`, `sim/runner.py`, `analysis/product.py`, `analysis/metrics.py`, `experiments/E0…E5.yaml`, `tests/`.
- Config: a dataclass loaded from YAML. CLI: `python -m sim.runner --config experiments/E3.yaml`.
- Outputs: per-epoch metrics as CSV or Parquet, the full record library as JSON with record references by id, and optional episode traces for debugging. Use matplotlib for sweep plots.
- Records reference other records by id within the owning agent's library. Keep them serializable with no object cycles; represent `SELF` as a sentinel.

## 13. Provisional decisions (review before building)

1. Strict turn alternation within a pair.
2. Loop closure is defined by turn structure (self-initiated then other-initiated), not by returning to the anchor node.
3. Mutuality is enforced by the simulator, not verified by agents.
4. Distinctness of the other is operationalized as **initiative**: mirrors never initiate. An "echo" mirror that replays my last offer on its turn is an open harder test.
5. Bootstrap requires equal levels.
6. Growth rule `generalize` adds self-loops, so learned moves are available everywhere.
7. The `other` slot points to the agent's own lower record (pairing by analogy).
8. Contingency is a pooled lift statistic per window, with threshold θ.
9. False recognition is measured one-sided, from the mind's perspective only.
10. Declined self offers are recorded as `SELF_ANY` wildcards, not as a constraint to reproduce.
11. Elder transmission phase-shifts the elder's word by one turn, padding with wildcards.
12. Each unlock activates one new node, reached from the record's anchor.

## 14. Hypotheses the model should reveal (not built in)

- A sharp threshold in development as coupling increases (E3).
- Probing accelerates development autocatalytically (E4).
- Pairs who bootstrapped together climb faster together: their probes interlock, forming a private language (E5).
- Lowering θ raises development speed and false recognition together (E2).
