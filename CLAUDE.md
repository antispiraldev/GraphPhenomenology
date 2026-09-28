# GraphPhenomenology

Simulation of the mutual recognition graph model. The spec in `mutual-recognition-model-spec.md` is the source of truth; section numbers (§) in docstrings refer to it.

## Commands
- Tests: `uv run pytest -q`
- Run: `uv run python -m sim.runner --config experiments/default.yaml` (from G2 on)

## Conventions
- Labels are ints (`model/actions.py`): a_i = 0..m-1, ā_i = m..2m-1, τ = 2m.
- A single `numpy.random.Generator` is created by the runner and passed down. Never use global numpy or `random` state.
- Record ids are indices into the owning agent's `Library`, and REF tokens point into the owner's own library only.
- Tick ranges are inclusive `[start, end]` trace indices.
- Signatures and docstrings in `model/`, `sim/recognition.py` and `analysis/product.py` are **frozen interfaces**. Implement them; don't change them. If an interface is wrong, stop and report it rather than editing another module's contract.
- Shared test builders live in `tests/helpers.py`. Each module gets its own `tests/test_<module>.py`.

## Process
- `NOTEBOOK.md` logs every experiment, decision and coined glossary term. Append; don't rewrite.
- Work proceeds in gated segments (see NOTEBOOK D3). Parallel work happens in separate git worktrees and is merged into `main` at the gate.
