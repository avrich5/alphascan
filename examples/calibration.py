import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import alphascan as A
from tests.test_core import synth
# false pass rate at cost 0 and 12 on pure random walk, current gate
for cost in (0.0, 12.0):
    p=t=0
    for seed in range(15):
        r = A.run_gate(synth(20000, seed), cost, kmax=48, nnull=200, seed=seed)
        r = r[r.verdict != "too_rare"]; t += len(r); p += (r.verdict == "enters").sum()
    print("cost", cost, "false pass", p, "/", t)
# planted edge: after big drops, push price up by +20/10k over next 10 bars -> must be found
from alphascan import registry as R
hit = 0
for seed in range(10):
    d = synth(20000, 100 + seed)
    m = R.candidates(d)["big_dn_2std"]
    c = d.close.to_numpy().copy()
    for i in np.where(m)[0]:
        c[i+1:] *= 1.0020 if i + 1 < len(c) else 1
    d2 = d.copy(); d2["close"] = c
    r = A.run_gate(d2, 12.0, kmax=48, seed=seed, include_registry=False, extra={"planted": m})
    hit += r.verdict.iloc[0] == "enters"; print(r[["n","peak","edge","edge_lo","verdict"]].to_string(header=False, index=False))
print("planted edge found", hit, "/ 10")
