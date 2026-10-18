#!/usr/bin/env python3
"""附录 C 表 tab:humaneval（PubMed→GSM8K，HumanEval 配对比较）的生成脚本（10-16，文档 49 L2）。

输入（逐题 pass 向量，164 题）：
  order 17：raw = results/e/diag25/e4/humaneval_e.json 的 passed；indent repair = results/e/humaneval_e_rescore.json 的 v3_autoindent.passed
  order 31/47：results/h/diag25/f1/heval_h_PG_<cond>_s<order>.json 的 passed_v2（raw）与 passed（indent repair）
  起点（start）= B_g 的两种协议向量，与顺序无关。
统计：每个顺序 × 协议内 6 项比较，精确 McNemar（双侧二项检验，n = 得 + 失，p = 0.5），Holm 校正。
输出：results/release/table_humaneval_holm.json 与 paper 表格的 LaTeX 行（results/release/tab_humaneval_rows.tex）。"""
import json
from pathlib import Path
from scipy.stats import binomtest

R = Path(__file__).resolve().parents[1] / "DATA"; OUT = R / "release"; OUT.mkdir(parents=True, exist_ok=True)
E4 = json.load(open(R / "e/diag25/e4/humaneval_e.json")); RS = json.load(open(R / "e/humaneval_e_rescore.json"))
ARM17 = {"vt": "E_vt", "vfac": "E_vfac", "vperm": "E_vperm", "0": "E_0"}
COMP = [("vfac", "vt"), ("vfac", "0"), ("vperm", "0"), ("vperm", "vt"), ("vt", "0"), ("vt", "start")]   # 表中行序
NAME = {"vt": r"\cond{keep v,t}", "vfac": r"\cond{log-RC v}", "vperm": r"\cond{permuted v}", "0": r"\cond{reset all}", "start": "start"}


def vec(cond, order, proto):
    if cond == "start":
        return E4["B_g"]["passed"] if proto == "raw" else RS["B_g"]["v3_autoindent"]["passed"]
    if order == 17:
        a = ARM17[cond]
        return E4[a]["passed"] if proto == "raw" else RS[a]["v3_autoindent"]["passed"]
    n = f"PG_{cond}_s{order}"; d = json.load(open(R / f"h/diag25/f1/heval_h_{n}.json"))[n]
    return d["passed_v2"] if proto == "raw" else d["passed"]


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def fmt_p(p):
    if p >= 0.995: return "1"
    s = f"{p:.2g}" if p < 0.1 else f"{p:.2f}"
    return s.lstrip("0") if s.startswith("0.") else s


res = {}
for proto in ("raw", "repair"):
    for order in (17, 31, 47):
        rows = []
        for a, b in COMP:
            va, vb = vec(a, order, proto), vec(b, order, proto); assert len(va) == len(vb) == 164
            gain = sum(x and not y for x, y in zip(va, vb)); loss = sum(y and not x for x, y in zip(va, vb))
            p = binomtest(gain, gain + loss, 0.5).pvalue if gain + loss else 1.0
            rows.append(dict(a=a, b=b, gained=gain, lost=loss, p_exact=p))
        for r, q in zip(rows, holm([r["p_exact"] for r in rows])): r["p_holm"] = q
        res[f"{proto}|{order}"] = rows
json.dump(res, open(OUT / "table_humaneval_holm.json", "w"), indent=1)
lines = []
for k, (a, b) in enumerate(COMP):
    cells = [f"{res[f'{pr}|{o}'][k]['gained']}/{res[f'{pr}|{o}'][k]['lost']} ({fmt_p(res[f'{pr}|{o}'][k]['p_holm'])})"
             for pr in ("raw", "repair") for o in (17, 31, 47)]
    lines.append(f"{NAME[a]} vs.\\ {NAME[b]} & " + " & ".join(cells) + r" \\")
(OUT / "tab_humaneval_rows.tex").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
