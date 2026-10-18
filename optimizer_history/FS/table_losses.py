#!/usr/bin/env python3
"""Tables tab:losses (Section 6.1) and tab:heldout (Appendix C): probe, held-out GSM8K and Dolly losses after stage B.

Inputs: DATA/h/summary_h.json (per-run end-point losses, per answer token; nll_* and dnll_* relative to the start),
and the starting checkpoints' per-problem NLL (segnll) for the held-out sets:
MG start = DATA/diag25/b1/segnll/M.json, GM start = DATA/d/diag25/d4/segnll/A_g.json, PG start = DATA/e/diag25/e4/segnll/B_g.json.
Output: DATA/release/table_losses.json (ranges over data orders 17, 31, 47, rounded as printed)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "DATA"; OUT = R / "release"; OUT.mkdir(parents=True, exist_ok=True)
C = json.load(open(R / "h/summary_h.json"))["cells"]
START = {"MG": "diag25/b1/segnll/M.json", "GM": "d/diag25/d4/segnll/A_g.json", "PG": "e/diag25/e4/segnll/B_g.json"}
CONDS = ("vt", "vfac", "vperm", "0"); ORDERS = (17, 31, 47)
tok = lambda v: sum(v["s"]) / sum(v["c"])
rng = lambda xs, f="%.3f": [f % min(xs), f % max(xs)]

out = {"tab_losses": {}, "tab_heldout": {}}
for sw, f in START.items():
    cells = {c: [C[f"{sw}|{c}|{s}"] for s in ORDERS] for c in CONDS}
    x = cells["vt"][0]
    out["tab_losses"][sw] = {"start_new": "%.3f" % (x["nll_new"] - x["dnll_new"]), "start_old": "%.3f" % (x["nll_old"] - x["dnll_old"]),
                             **{c: {"new_end": rng([r["nll_new"] for r in v]), "old_change": rng([r["dnll_old"] for r in v], "%+.3f")}
                                for c, v in cells.items()}}
    d = json.load(open(R / f))["nll"]; s = {k: tok(d[k]) for k in ("heldout_gsm8k", "heldout_dolly_qa")}
    out["tab_heldout"][sw] = {"start_gsm8k_test": "%.3f" % s["heldout_gsm8k"], "start_dolly_qa": "%.3f" % s["heldout_dolly_qa"],
                              **{c: {"gsm8k_test": rng([r["seg_heldout_gsm8k_tok"] - s["heldout_gsm8k"] for r in v], "%+.3f"),
                                     "dolly_qa": rng([r["seg_heldout_dolly_qa_tok"] - s["heldout_dolly_qa"] for r in v], "%+.3f")}
                                 for c, v in cells.items()}}
json.dump(out, open(OUT / "table_losses.json", "w"), indent=1)
for t, v in out.items():
    print(t)
    for sw, row in v.items(): print(" ", sw, row)
