"""E4 diagnostic: are level-ups beyond level 1 with the same partner as the previous level-up?

    uv run python -m experiments.E4_repeat

Reads outputs/E4_pprobe and outputs/E4_growth (from experiments/E4.py). A
mind's level-up in epoch e is attributed to its partner in that epoch.
"""
import glob
import re

import numpy as np

for pattern, name in (("outputs/E4_pprobe/p_probe*_s*.npz", "generalize, p_probe>0"),
                      ("outputs/E4_growth/growthnone_s*.npz", "none, p_probe=0.3")):
    same = total = 0
    for f in glob.glob(pattern):
        if re.search(r"p_probe0_s", f):
            continue
        z = np.load(f)
        L, P = z["levels"].astype(int), z["partner"].astype(int)
        for i in range(L.shape[1]):
            prev = None
            for e in np.nonzero(L[1:, i] > L[:-1, i])[0]:
                if L[e, i] >= 1 and prev is not None:
                    total += 1
                    same += P[e, i] == prev
                prev = P[e, i]
    print(f"growth={name}: level-ups from k>=1 with the previous level-up partner: {same}/{total} = {same/total:.1%}")
