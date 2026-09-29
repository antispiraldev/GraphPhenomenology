"""C0 calibration: run experiments/C0.yaml over 10 seeds and check the pass criterion (spec §10).

    uv run python -m experiments.run_c0 [--config experiments/C0.yaml] [--tag NAME] [key=value ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import yaml

from model.config import Config
from sim.runner import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/C0.yaml")
    ap.add_argument("--tag", default="C0")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = ap.parse_args()
    base = Config.from_yaml(args.config)
    over = {k: yaml.safe_load(v) for k, v in (o.split("=", 1) for o in args.overrides)}
    base = dataclasses.replace(base, **over)
    out = Path(base.output_dir) / args.tag
    per_seed = []
    for seed in range(args.seeds):
        cfg = dataclasses.replace(base, seed=seed)
        res = run(cfg, out / f"s{seed}")
        levels = np.array([a.level for a in res.agents if a.kind == "mind"])
        rows = res.rows
        per_seed.append({
            "seed": seed,
            "frac_ge1": float((levels >= 1).mean()),
            "frac_ge2": float((levels >= 2).mean()),
            "frac_ge3": float((levels >= 3).mean()),
            "max_level": int(levels.max()),
            "contingency_computable": float(np.nanmean([r["contingency_computable_frac"] for r in rows])),
            "probe": {
                k: [sum(r[f"probe_starts_L{k}"] for r in rows), sum(r[f"probe_successes_L{k}"] for r in rows)]
                for k in range(1, 5)
            },
        })
    agg = {
        key: float(np.mean([s[key] for s in per_seed]))
        for key in ("frac_ge1", "frac_ge2", "frac_ge3", "max_level", "contingency_computable")
    }
    agg["probe_success_rate_by_level"] = {
        k: (lambda st, su: su / st if st else None)(
            sum(s["probe"][k][0] for s in per_seed), sum(s["probe"][k][1] for s in per_seed)
        )
        for k in range(1, 5)
    }
    agg["pass"] = agg["frac_ge2"] >= 0.10 and agg["contingency_computable"] >= 0.80
    report = {"config": dataclasses.asdict(base), "seeds": per_seed, "aggregate": agg}
    out.mkdir(parents=True, exist_ok=True)
    (out / "c0_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
