"""E5 v0.2 vs v0.3: print the headline numbers of two E5_analyze summaries side by side.

    uv run python -m experiments.E5_compare outputs/E5_v02_s16/summary.json outputs/E5/summary.json

Each cell is "estimate [95% CI]" as stored by E5_analyze (cluster bootstrap over seeds).
"""

from __future__ import annotations

import json
import sys

CONDS = ["pairs", "shuffled", "er02"]


def f(t, pct=False, nd=3) -> str:
    if t is None:
        return "n/a"
    if isinstance(t, (int, float)):
        t = (t, None, None)
    k = 100 if pct else 1
    e, lo, hi = t
    if e is None:
        return "n/a"
    s = f"{k * e:.{nd}g}"
    if lo is not None and lo == lo:
        s += f" [{k * lo:.{nd}g}, {k * hi:.{nd}g}]"
    return s


def rows(S: dict) -> dict[str, str]:
    out = {}
    for c in CONDS:
        L = S["levels"][c]
        for thr in (1, 2, 3, 4):
            out[f"{c} level>={thr} (%)"] = f(L[f"frac_ge{thr}"], pct=True)
        out[f"{c} mean level"] = f(L["mean_level"])
        out[f"{c} mean max level"] = f(L["max_level_mean"])
        out[f"{c} still level 0 (%)"] = f(1 - L["frac_ge1"][0], pct=True)
        out[f"{c} level-ups/mind/epoch"] = f(L["levelups_per_agent_epoch"])
        out[f"{c} level-ups/encounter"] = f(L["levelups_per_encounter"])
        out[f"{c} equal-level encounter frac"] = f(L["frac_encounters_equal_level"])
        out[f"{c} repeat-partner frac"] = f(L["repeat_partner_frac"])
        out[f"{c} seconds/run"] = f(L["seconds_per_run"])
        for k, v in S["levelup_per_equal_encounter_by_level"][c].items():
            out[f"{c} equal-level level-up rate from {k}"] = f(v["rate"]) + f" (n={v['encounters']})"
    for c in ("shuffled", "er02", "pairs"):
        K = S["key_test"][c]
        for cls in ("cobuilt_top", "past", "stranger"):
            out[f"{c} top-probe success {cls} (%)"] = f(K[f"probe_success_{cls}"], pct=True)
        out[f"{c} probe ratio cobuilt/stranger (std)"] = f(K["ratio_probe_success_cobuilt_vs_stranger_std"])
        out[f"{c} level-up ratio cobuilt/stranger (std)"] = f(K["ratio_levelup_cobuilt_vs_stranger_std"])
        out[f"{c} success partner probing (%)"] = f(K["success_when_partner_probing"], pct=True)
        out[f"{c} success partner not probing (%)"] = f(K["success_when_partner_not_probing"], pct=True)
        for cls in ("cobuilt_top", "stranger"):
            out[f"{c} co-probe frac of starts {cls}"] = f(K[f"coprobe_frac_start_{cls}"])
    out["cross probe pairs vs shuffled stranger"] = f(S["cross_ratio_probe_pairs_vs_shuffled_stranger"])
    out["cross level-up pairs vs shuffled stranger"] = f(S["cross_ratio_levelup_pairs_vs_shuffled_stranger"])
    for c in CONDS:
        O = S["record_overlap"][c]
        for lvl in ("1", "2", "3"):
            for g in ("cobuilt", "other"):
                d = O.get(f"L{lvl}_{g}")
                if d:
                    out[f"{c} L{lvl} {g} n / compl-Jacc / offered-Jacc / interlock>=1"] = (
                        f"{d['n']} / {d['own_vs_partner_complement_jaccard_flat']:.2f} / "
                        f"{d['offered_label_jaccard_extras']:.2f} / {d['frac_interlock_ge1']:.2f}")
        out[f"{c} segment overlap frac"] = f(O["segment_overlap_frac"]) + f" (n={O['segment_overlap_n']})"
    return out


def main() -> None:
    a, b = (json.loads(open(p).read()) for p in sys.argv[1:3])
    ra, rb = rows(a), rows(b)
    print(f"{'metric':58s} | {'A: ' + sys.argv[1]:40s} | B: {sys.argv[2]}")
    for k in dict.fromkeys(list(ra) + list(rb)):
        print(f"{k:58s} | {ra.get(k, '-'):40s} | {rb.get(k, '-')}")


if __name__ == "__main__":
    main()
