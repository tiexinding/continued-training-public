#!/usr/bin/env python3
"""10-10 I4（文档 43）：同条件跨数据顺序的通道级一致性。
输入：各格子的成对分析场 A_B_fields.npz，标签形如 MG_vt_s17（切换_条件_s种子）。
对每个 切换 × 条件，三对顺序（17–31、17–47、31–47）之间比较通道改动能量：
  down 列（eW_col）、gate 行与 up 行（eW_row）——逐层 Spearman，及 top-1/5/10% 集合重合（除以集合大小），取 24 层中位数。
对照：同顺序、不同条件之间的同一指标（vt 对 vperm、vt 对 vfac、vt 对 0），区分"顺序带来的变化"与"条件带来的变化"。
用法：python cross_order_channels.py --fields MG_vt_s17=<npz> ... --out <json>"""
import argparse, itertools, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--fields", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
FILES = dict(s.split("=", 1) for s in a.fields)
KEYS = {"down": "mlp.down_proj.weight|eW_col", "gate": "mlp.gate_proj.weight|eW_row", "up": "mlp.up_proj.weight|eW_row"}
FR = (0.01, 0.05, 0.10)
_cache = {}


def load(lab):
    if lab not in _cache:
        F = np.load(FILES[lab]); L = 0
        while f"model.layers.{L}.{KEYS['down']}" in F.files: L += 1
        _cache[lab] = {m: [F[f"model.layers.{i}.{k}"].astype(np.float64) for i in range(L)] for m, k in KEYS.items()}
    return _cache[lab]


def rk(x): return np.argsort(np.argsort(x))
def sp(x, y): return float(np.corrcoef(rk(x), rk(y))[0, 1])


def topov(x, y, f):
    k = max(1, int(f * len(x))); return len(set(np.argsort(-x)[:k]) & set(np.argsort(-y)[:k])) / k


def compare(l1, l2):
    A, B = load(l1), load(l2); out = {}
    for m in KEYS:
        rows = [dict(spearman=sp(x, y), **{f"top{int(f * 100)}": topov(x, y, f) for f in FR}) for x, y in zip(A[m], B[m])]
        out[m] = {q: float(np.median([r[q] for r in rows])) for q in rows[0]}
    return out


cells = {}
for lab in FILES:
    sw, cd, sd = lab.split("_"); cells.setdefault((sw, cd), {})[sd] = lab
res = {"across_orders": {}, "across_conditions": {}}
for (sw, cd), by in sorted(cells.items()):
    for s1, s2 in itertools.combinations(sorted(by), 2):
        r = compare(by[s1], by[s2]); res["across_orders"][f"{sw}_{cd}|{s1}-{s2}"] = r
        print(f"顺序 {sw}_{cd} {s1}-{s2}: down ρ {r['down']['spearman']:.3f} top5 {r['down']['top5']:.3f} | gate ρ {r['gate']['spearman']:.3f} | up ρ {r['up']['spearman']:.3f}", flush=True)
for sw in sorted({k[0] for k in cells}):
    for c2 in ("vperm", "vfac", "0"):
        for sd in ("s17", "s31", "s47"):
            l1, l2 = cells.get((sw, "vt"), {}).get(sd), cells.get((sw, c2), {}).get(sd)
            if l1 and l2:
                r = compare(l1, l2); res["across_conditions"][f"{sw}|vt-{c2}|{sd}"] = r
                print(f"条件 {sw} vt-{c2} {sd}: down ρ {r['down']['spearman']:.3f} top5 {r['down']['top5']:.3f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
