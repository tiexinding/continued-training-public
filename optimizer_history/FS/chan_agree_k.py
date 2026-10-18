#!/usr/bin/env python3
"""10-11（论文 v1.3 §4 补证据）：成分消融四条件（MG，顺序 17）逐神经元改动能量的一致性，CPU。
比较对（每对：down 列 eW_col、gate 行与 up 行 eW_row；逐层 Spearman 与 top-1/5/10% 集合重合，取 24 层中位数）：
  keep m,v,t vs keep v,t；keep t vs reset all；keep v,t vs reset all；keep m,v,t vs reset all；
  参照：keep v,t 顺序 17 vs 顺序 31（同条件跨顺序）。
用法：python chan_agree_k.py --fields mvt=<npz> vt=<npz> t=<npz> 0=<npz> vt31=<npz> --out <json>"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--fields", nargs="+", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
FILES = dict(s.split("=", 1) for s in a.fields)
KEYS = {"down": "mlp.down_proj.weight|eW_col", "gate": "mlp.gate_proj.weight|eW_row", "up": "mlp.up_proj.weight|eW_row"}
def load(lab):
    F = np.load(FILES[lab]); L = 0
    while f"model.layers.{L}.{KEYS['down']}" in F.files: L += 1
    return {m: [F[f"model.layers.{i}.{k}"].astype(np.float64) for i in range(L)] for m, k in KEYS.items()}
D = {k: load(k) for k in FILES}
rk = lambda x: np.argsort(np.argsort(x)); sp = lambda x, y: float(np.corrcoef(rk(x), rk(y))[0, 1])
def topov(x, y, f): k = max(1, int(f * len(x))); return len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k])) / k
res = {}
for l1, l2 in (("mvt", "vt"), ("t", "0"), ("vt", "0"), ("mvt", "0"), ("vt", "vt31"), ("mvt", "t")):
    if l1 not in D or l2 not in D: continue
    out = {}
    for m in KEYS:
        rows = [dict(spearman=sp(x, y), top1=topov(x, y, .01), top5=topov(x, y, .05), top10=topov(x, y, .10)) for x, y in zip(D[l1][m], D[l2][m])]
        out[m] = {q: float(np.median([r[q] for r in rows])) for q in rows[0]} | {"spearman_min_layer": float(min(r["spearman"] for r in rows))}
    res[f"{l1}-{l2}"] = out; print(l1, "vs", l2, {m: (round(out[m]["spearman"], 3), round(out[m]["top5"], 3)) for m in out}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
