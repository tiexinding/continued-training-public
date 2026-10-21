#!/usr/bin/env python3
"""Format flips on PubMed -> GSM8K (Appendix C.1): problems whose unindented status changes between two conditions
within one data order, against between two data orders within one condition, for the three conditions that keep a
second moment (keep v,t; log-RC v; permuted v). Written 2026-10-19 for the appendix revision."""
import itertools, json
from pathlib import Path
import numpy as np
D = Path(__file__).resolve().parents[2] / "DATA"
f7 = json.load(open(D / "f/diag25/f1/heval_f7.json"))
code = {"vt": "E_vt", "vfac": "E_vfac", "vperm": "E_vperm"}
U = {}
for c, k in code.items():
    U[(c, 17)] = np.array([x == "unind" for x in f7[k]["cats"]])
    for s in (31, 47):
        d = json.load(open(D / f"h/diag25/f1/heval_h_PG_{c}_s{s}.json"))
        U[(c, s)] = np.array([x == "unind" for x in d[f"PG_{c}_s{s}"]["cats"]])
n = len(U[("vt", 17)])
cond = {f"{o}:{a}-{b}": int((U[(a, o)] != U[(b, o)]).sum()) for o in (17, 31, 47) for a, b in itertools.combinations(code, 2)}
order = {f"{c}:{a}-{b}": int((U[(c, a)] != U[(c, b)]).sum()) for c in code for a, b in itertools.combinations((17, 31, 47), 2)}
out = dict(n_problems=n, condition_flips=cond, order_flips=order,
           condition_range_pct=[100 * min(cond.values()) / n, 100 * max(cond.values()) / n], condition_mean=float(np.mean(list(cond.values()))),
           order_range_pct=[100 * min(order.values()) / n, 100 * max(order.values()) / n], order_mean=float(np.mean(list(order.values()))))
(D / "underspec").mkdir(exist_ok=True)
json.dump(out, open(D / "underspec/format_flip_means.json", "w"), indent=1)
print({k: v for k, v in out.items() if "range" in k or "mean" in k})
