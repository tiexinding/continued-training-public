#!/usr/bin/env python3
"""10-11 J 轮判读（文档 46 第 15 节，规则写于开跑前）。本地运行，读 results/j（回传包解开处）与已有的基线（H/E 轮）。
三组，每组三个顺序：
  base  PG 留 v·t 原顺序 17/31/47（PG_vt_s17 = E_vt；H 轮 31、47）
  tail  共享末段：17 = 原运行；31、47 = J1
  dec   末段衰减：J2 的 17/31/47
读数：无缩进率（及极差）、逐对翻转率（无缩进、原始通过、修复后通过）、新任务（gsm8k）/旧任务（bio）探针逐题 ΔNLL 的跨顺序 RMS 差、
      跨顺序 down 列通道排名相关。判读阈值见 RULES。
用法：python analyze_j.py [--res <results 根>]"""
import argparse, itertools, json, re
from pathlib import Path
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser(); ap.add_argument("--res", default=str(Path(__file__).resolve().parents[1] / "DATA")); a = ap.parse_args()
RES = Path(a.res); H = RES / "h"; J = RES / "j"; OUT = RES / "underspec"; OUT.mkdir(exist_ok=True)
HE = list(pd.read_parquet(H / "humaneval.parquet").itertuples())
RULES = dict(tail_ok_range=0.07, tail_ok_churn=0.06, tail_no_range=0.15, tail_no_churn=0.12, dec_ok_range=0.107, dec_ok_churn=0.084, item_half=0.014, chan_min=0.95)

def fmt_cat(o, ep):
    m = re.search(r"```(?:python)?\n(.*?)```", o, re.S); b = m.group(1) if m else o
    if ("def " + ep) in b: return "full"
    first = next((l for l in b.split("\n") if l.strip()), ""); return "unind" if first and not first.startswith((" ", "\t")) else "ind"
def J_(p): return json.load(open(p))

def heval(group, s):
    if (group in ("base", "tail") and s == 17):
        d = J_(RES / "e/diag25/e4/humaneval_e.json")["E_vt"]; rs = J_(RES / "e/humaneval_e_rescore.json")["E_vt"]
        outs, p2, p3 = d["outs"], d["passed"], rs["v3_autoindent"]["passed"]
    else:
        n = f"PG_vt_s{s}" if group == "base" else f"PG_{group}_s{s}"; root = H if group == "base" else J
        d = J_(root / f"diag25/f1/heval_{'h' if group == 'base' else 'j'}_{n}.json")[n]; outs, p2, p3 = d["outs"], d["passed_v2"], d["passed"]
    return dict(unind=np.array([fmt_cat(o, q.entry_point) == "unind" for o, q in zip(outs, HE)]), pass_raw=np.array(p2, bool), pass_rep=np.array(p3, bool))

def probe(group, s):
    if group == "base" or (group == "tail" and s == 17): d = H / f"runs_h/PG_vt_s{s}/records"
    else: d = J / f"runs_j/PG_{group}_s{s}/records"
    out = {}
    for k in (0, 125):
        x = torch.load(d / f"probe_full_{k:04d}.pt", map_location="cpu", weights_only=False)
        out[k] = {t: np.array(x[t]["sums"], float) / np.maximum(np.array(x[t]["cnts"], float), 1) for t in ("gsm8k", "bio")}
    return {t: out[125][t] - out[0][t] for t in ("gsm8k", "bio")}

res = {"rules": RULES, "groups": {}}
X = J_(J / "diag25/j/cross_order_channels_j.json")["across_orders"]
for g in ("base", "tail", "dec"):
    E = {s: heval(g, s) for s in (17, 31, 47)}; Pp = {s: probe(g, s) for s in (17, 31, 47)}
    pairs = list(itertools.combinations((17, 31, 47), 2))
    r = {"unind": [float(E[s]["unind"].mean()) for s in (17, 31, 47)],
         "pass_raw": [float(E[s]["pass_raw"].mean()) for s in (17, 31, 47)], "pass_rep": [float(E[s]["pass_rep"].mean()) for s in (17, 31, 47)]}
    r["unind_range"] = max(r["unind"]) - min(r["unind"])
    for m in ("unind", "pass_raw", "pass_rep"):
        r[f"churn_{m}"] = [float(np.mean(E[s1][m] != E[s2][m])) for s1, s2 in pairs]
        r[f"churn_{m}_mean"] = float(np.mean(r[f"churn_{m}"]))
    for t in ("gsm8k", "bio"):
        r[f"item_D_{t}"] = float(np.mean([np.sqrt(np.mean((Pp[s1][t] - Pp[s2][t]) ** 2)) for s1, s2 in pairs]))
    r["chan_down_rho"] = [v["down"]["spearman"] for k, v in X.items() if k.startswith(f"PG_{g}|")]
    res["groups"][g] = r
b, t, d = (res["groups"][k] for k in ("base", "tail", "dec"))
def verdict_tail():
    if t["unind_range"] <= RULES["tail_ok_range"] and t["churn_unind_mean"] <= RULES["tail_ok_churn"]: return "成立"
    if t["unind_range"] >= RULES["tail_no_range"] and t["churn_unind_mean"] >= RULES["tail_no_churn"]: return "否定"
    return "部分成立"
res["J1"] = verdict_tail(); res["J1_item"] = "支持" if t["item_D_gsm8k"] <= RULES["item_half"] else "不支持"
res["J2"] = "成立" if (d["unind_range"] <= RULES["dec_ok_range"] and d["churn_unind_mean"] <= RULES["dec_ok_churn"]) else "不成立"
res["chan_ok"] = {g: bool(min(res["groups"][g]["chan_down_rho"] or [0]) >= RULES["chan_min"]) for g in ("base", "tail", "dec")}
json.dump(res, open(OUT / "j_verdict.json", "w"), ensure_ascii=False, indent=1)
for g, r in res["groups"].items():
    print(f"{g:4} 无缩进 {['%.3f' % x for x in r['unind']]} 极差 {r['unind_range']:.3f} | 翻转 无缩进 {r['churn_unind_mean']:.3f} 原始 {r['churn_pass_raw_mean']:.3f} 修复 {r['churn_pass_rep_mean']:.3f}"
          f" | 逐题差 gsm8k {r['item_D_gsm8k']:.4f} bio {r['item_D_bio']:.4f} | 通道 ρ {['%.3f' % x for x in r['chan_down_rho']]}"
          f" | 原始通过 {['%.3f' % x for x in r['pass_raw']]}")
print("J1:", res["J1"], "| J1 逐题:", res["J1_item"], "| J2:", res["J2"], "| 通道:", res["chan_ok"])
