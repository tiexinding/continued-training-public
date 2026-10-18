#!/usr/bin/env python3
"""10-11 文档 46 §9.2 的 C3 / C4（本地 CPU，只读已有结果，不开卡）。

C3 两个层面、同一套统计量的跨顺序稳定性：
  参数层面   down/gate/up 通道排名的跨顺序 Spearman 与 top-5% 一致率（I4 cross_order_channels.json，直接读）；
  逐题损失   探针逐题 NLL 相对起点的变化 Δ_i，跨顺序 Spearman（连续量）；
  逐题生成   HumanEval 164 题 × {无缩进, 原始组装通过, 缩进修复通过}，跨顺序逐题一致率、翻转率、Cohen κ。
  每层同时算"同一顺序、不同条件"的同一统计量，给出 顺序/条件 比 = 顺序带来的变化 ÷ 条件带来的变化
  （排名：(1−ρ_顺序)/(1−ρ_条件)；逐题：翻转率_顺序 / 翻转率_条件）。

事先固定的判读（写于运行前）：
  R-C3a  顺序/条件 比 < 0.2 记"被固定"，0.2–0.5 记"部分固定"，≥ 0.5 记"欠定"（顺序带来的变化与条件效应同量级）。
         "状态固定结构、行为仍欠定"成立 ⇔ 保留 v 的条件在参数层面 < 0.2，且在逐题生成层面 ≥ 0.5。
  R-C3b  v 是否也固定行为：PG 无缩进的跨顺序 κ，若全清零在三对顺序上都低于保留 v 三个条件的最小值，记"v 部分固定行为"；
         否则记"未见 v 固定行为的证据"。通过率在三个切换上同样判，多数（≥2/3 切换）为准。
C4 Multi-Bootstrap（Sellam 等 2022）：对 (数据顺序, 题目) 两层同时有放回重采样，同一顺序下两个条件配对、同一组题目，
  统计量 = 重采样顺序与题目上的平均配对差；与只对题目重采样的逐顺序区间、跨顺序 t 区间并列。
  注意：只有 3 个顺序，顺序层的重采样只有 10 种多重集，区间偏粗、偏窄（n=3 的自助方差低估约 (n−1)/n）。
用法：python underspec_c3c4.py [--res <results 根>] [--B 5000]
输出：<res>/underspec/c3c4.json 与 c3c4_tables.md"""
import argparse, json, re, itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, t as tdist

ap = argparse.ArgumentParser()
ap.add_argument("--res", default=str(Path(__file__).resolve().parents[1] / "DATA"))
ap.add_argument("--B", type=int, default=5000)
a = ap.parse_args()
RES = Path(a.res); H = RES / "h"; OUT = RES / "underspec"; OUT.mkdir(exist_ok=True)
SW = ["MG", "GM", "PG"]; CONDS = ["vt", "vfac", "vperm", "0"]; SEEDS = [17, 31, 47]
CN = {"vt": "留 v·t", "vfac": "v 行列近似", "vperm": "v 置换", "0": "全清零"}
KEEPV = ["vt", "vfac", "vperm"]

# ---------------------------------------------------------------- 文件位置（与 analyze_h.py 的 OLD / src 相同）
OLD = {
    ("MG", "vt", 17): dict(heval=("c/diag25/c4/humaneval_c.json", "K_vt", "e/humaneval_c_rescore.json", "K_vt"), segnll="c/diag25/c4/segnll/K_vt.json"),
    ("MG", "0", 17): dict(heval=("b3/humaneval_v2.json", "Gfresh", "e/humaneval_b3_rescore.json", "Gfresh"), segnll="diag25/b1/segnll/Gfresh.json"),
    ("MG", "vt", 31): dict(heval=("c/diag25/c4/humaneval_c.json", "K_vt_s31", "e/humaneval_c_rescore.json", "K_vt_s31"), segnll="c/diag25/c4/segnll/K_vt_s31.json"),
    ("MG", "0", 31): dict(heval=("c/diag25/c4/humaneval_c.json", "K_0_s31", "e/humaneval_c_rescore.json", "K_0_s31"), segnll="c/diag25/c4/segnll/K_0_s31.json"),
    ("GM", "vt", 17): dict(heval=("d/diag25/d4/humaneval_d.json", "R_vt", "e/humaneval_d_rescore.json", "R_vt"), segnll="d/diag25/d4/segnll/R_vt.json"),
    ("GM", "vperm", 17): dict(heval=("e/diag25/e4/humaneval_e.json", "R_vperm_med", "e/humaneval_e_rescore.json", "R_vperm_med"), segnll="e/diag25/e4/segnll/R_vperm_med.json"),
    ("GM", "0", 17): dict(heval=("d/diag25/d4/humaneval_d.json", "R_0", "e/humaneval_d_rescore.json", "R_0"), segnll="d/diag25/d4/segnll/R_0.json"),
}
for c, e in (("vt", "E_vt"), ("vfac", "E_vfac"), ("vperm", "E_vperm"), ("0", "E_0")):
    OLD[("PG", c, 17)] = dict(heval=("e/diag25/e4/humaneval_e.json", e, "e/humaneval_e_rescore.json", e), segnll=f"e/diag25/e4/segnll/{e}.json")
START_SEG = {"MG": "diag25/b1/segnll/M.json", "GM": "d/diag25/d4/segnll/A_g.json", "PG": "e/diag25/e4/segnll/B_g.json"}
# 逐题损失：新任务用留出集或探针，另取一个非训练任务
SEGSETS = {"MG": {"new": "heldout_gsm8k", "old": "probe_mbpp"}, "GM": {"new": "probe_mbpp", "old": "heldout_gsm8k"},
           "PG": {"new": "heldout_gsm8k", "old": "probe_mbpp"}}   # PG 的旧任务 bio 无逐题文件，"old" 位置用 MBPP 探针（非训练任务）

_jc = {}
def J(p):
    p = Path(p)
    if p not in _jc: _jc[p] = json.load(open(p)) if p.exists() else None
    return _jc[p]

import pandas as pd
HE = list(pd.read_parquet(H / "humaneval.parquet").itertuples())
def fmt_cat(o, ep):
    m = re.search(r"```(?:python)?\n(.*?)```", o, re.S); b = m.group(1) if m else o
    if ("def " + ep) in b: return "full"
    first = next((l for l in b.split("\n") if l.strip()), ""); return "unind" if first and not first.startswith((" ", "\t")) else "ind"

def cell(sw, c, s):
    r = {}
    if (sw, c, s) in OLD:
        fr, kr, fs, ks = OLD[(sw, c, s)]["heval"]; d = J(RES / fr); rs = J(RES / fs)
        outs = d[kr]["outs"]; p2 = d[kr]["passed"]; p3 = rs[ks]["v3_autoindent"]["passed"]; seg = J(RES / OLD[(sw, c, s)]["segnll"])
    else:
        n = f"{sw}_{c}_s{s}"; d = J(H / f"diag25/f1/heval_h_{n}.json")[n]
        outs, p2, p3 = d["outs"], d["passed_v2"], d["passed"]; seg = J(H / f"diag25/h/segnll/{n}.json")
    r["unind"] = np.array([fmt_cat(o, q.entry_point) == "unind" for o, q in zip(outs, HE)])
    r["pass_raw"] = np.array(p2, bool); r["pass_rep"] = np.array(p3, bool)
    st = J(RES / START_SEG[sw])
    for role, ds in SEGSETS[sw].items():
        v, v0 = seg["nll"][ds], st["nll"][ds]
        x = np.array(v["s"], float) / np.maximum(np.array(v["c"], float), 1); x0 = np.array(v0["s"], float) / np.maximum(np.array(v0["c"], float), 1)
        r[f"dnll_{role}"] = x - x0
    return r

C = {(sw, c, s): cell(sw, c, s) for sw in SW for c in CONDS for s in SEEDS}
# 核对：无缩进率与论文表 4（PG）
chk = {f"{c}_s{s}": round(float(C[("PG", c, s)]["unind"].mean()), 3) for c in CONDS for s in SEEDS}
print("PG 无缩进率（核对表 4）:", chk)

def kappa(x, y):
    po = np.mean(x == y); p1, p2 = x.mean(), y.mean(); pe = p1 * p2 + (1 - p1) * (1 - p2)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")

GEN = ["unind", "pass_raw", "pass_rep"]
PAIRS = list(itertools.combinations(SEEDS, 2))
res = {"check_PG_unind": chk, "gen": {}, "loss": {}, "ratio": {}, "struct": {}, "C4": {}}

# ---------------------------------------------------------------- C3 逐题生成与逐题损失
for sw in SW:
    for m in GEN:
        for c in CONDS:      # 跨顺序
            rows = []
            for s1, s2 in PAIRS:
                x, y = C[(sw, c, s1)][m], C[(sw, c, s2)][m]
                rows.append(dict(pair=f"{s1}-{s2}", churn=float(np.mean(x != y)), kappa=kappa(x, y), rate=(float(x.mean()), float(y.mean())), only_first=int((x & ~y).sum()), only_second=int((~x & y).sum()),
                                 churn_indep=float(x.mean() * (1 - y.mean()) + y.mean() * (1 - x.mean()))))
            res["gen"][f"{sw}|{m}|{c}|order"] = rows
        for c1, c2 in itertools.combinations(CONDS, 2):   # 同顺序、不同条件
            rows = []
            for s in SEEDS:
                x, y = C[(sw, c1, s)][m], C[(sw, c2, s)][m]
                rows.append(dict(seed=s, churn=float(np.mean(x != y)), kappa=kappa(x, y)))
            res["gen"][f"{sw}|{m}|{c1}-{c2}|cond"] = rows
    for role in SEGSETS[sw]:
        for c in CONDS:
            res["loss"][f"{sw}|{role}|{c}|order"] = [float(spearmanr(C[(sw, c, s1)][f"dnll_{role}"], C[(sw, c, s2)][f"dnll_{role}"])[0]) for s1, s2 in PAIRS]
        for c1, c2 in itertools.combinations(CONDS, 2):
            res["loss"][f"{sw}|{role}|{c1}-{c2}|cond"] = [float(spearmanr(C[(sw, c1, s)][f"dnll_{role}"], C[(sw, c2, s)][f"dnll_{role}"])[0]) for s in SEEDS]

# ---------------------------------------------------------------- 参数层面（I4）
X = J(RES / "i/diag25/i/cross_order_channels.json")
for sw in SW:
    for c in CONDS:
        v = [X["across_orders"][k]["down"] for k in X["across_orders"] if k.startswith(f"{sw}_{c}|")]
        res["struct"][f"{sw}|{c}|order"] = dict(spearman=[x["spearman"] for x in v], top5=[x["top5"] for x in v])
xc = X["across_conditions"]; res["struct_cond_keys_example"] = list(xc)[:6]
def struct_cond(sw, c1, c2):
    out = []
    for k, v in xc.items():
        w, cc, sd = k.split("|")
        if w == sw and set(cc.split("-")) == {c1, c2}: out.append(v["down"]["spearman"])
    return out

# ---------------------------------------------------------------- 顺序/条件 比（R-C3a）
def lvl(r):
    return "被固定" if r < 0.2 else ("部分固定" if r < 0.5 else "欠定")
for sw in SW:
    for c in KEEPV + ["0"]:
        others = [o for o in CONDS if o != c]
        d = {}
        so = np.mean(res["struct"][f"{sw}|{c}|order"]["spearman"])
        sc = [np.mean(struct_cond(sw, c, o)) for o in others if struct_cond(sw, c, o)]
        if sc and np.median(sc) < 0.99: d["参数"] = dict(order=float(so), cond_med=float(np.median(sc)), ratio=float((1 - so) / (1 - np.median(sc))))   # 行列近似只有对留 v 一个对照（ρ≈1），比值无意义
        for role in SEGSETS[sw]:
            lo = np.mean(res["loss"][f"{sw}|{role}|{c}|order"])
            lc = [np.mean(res["loss"][f"{sw}|{role}|{'-'.join(sorted([c, o], key=CONDS.index))}|cond"]) for o in others]
            d[f"损失_{role}"] = dict(order=float(lo), cond_med=float(np.median(lc)), ratio=float((1 - lo) / (1 - np.median(lc))))
        for m in GEN:
            go = np.mean([r["churn"] for r in res["gen"][f"{sw}|{m}|{c}|order"]])
            gc = [np.mean([r["churn"] for r in res["gen"][f"{sw}|{m}|{'-'.join(sorted([c, o], key=CONDS.index))}|cond"]]) for o in others]
            d[f"生成_{m}"] = dict(order=float(go), cond_med=float(np.median(gc)), ratio=float(go / np.median(gc)) if np.median(gc) > 0 else float("nan"))
        for k in d: d[k]["level"] = lvl(d[k]["ratio"]) if np.isfinite(d[k]["ratio"]) else "—"
        res["ratio"][f"{sw}|{c}"] = d

# 敏感性：分母只用差别最大的一对（留 v·t 对 全清零），只对留 v·t 与全清零两条件
sens = {}
for sw in SW:
    for c in ("vt", "0"):
        d = {}
        so = np.mean(res["struct"][f"{sw}|{c}|order"]["spearman"]); sc = np.mean(struct_cond(sw, "vt", "0"))
        d["参数"] = float((1 - so) / (1 - sc))
        for m in GEN:
            go = np.mean([r["churn"] for r in res["gen"][f"{sw}|{m}|{c}|order"]]); gc = np.mean([r["churn"] for r in res["gen"][f"{sw}|{m}|vt-0|cond"]])
            d[f"生成_{m}"] = float(go / gc) if gc > 0 else float("nan")
        for role in SEGSETS[sw]:
            lo = np.mean(res["loss"][f"{sw}|{role}|{c}|order"]); lc = np.mean(res["loss"][f"{sw}|{role}|vt-0|cond"])
            d[f"损失_{role}"] = float((1 - lo) / (1 - lc))
        sens[f"{sw}|{c}"] = d
res["ratio_vs_vt0"] = sens

# R-C3b
rc3b = {}
for sw in SW:
    for m in GEN:
        k0 = [r["kappa"] for r in res["gen"][f"{sw}|{m}|0|order"]]
        kv = [min(res["gen"][f"{sw}|{m}|{c}|order"][i]["kappa"] for c in KEEPV) for i in range(3)]
        rc3b[f"{sw}|{m}"] = dict(kappa_reset=k0, kappa_keepv_min=kv, reset_lower_all=bool(all(a_ < b_ for a_, b_ in zip(k0, kv))))
res["R_C3b"] = rc3b

# ---------------------------------------------------------------- C4 Multi-Bootstrap
rng = np.random.default_rng(0)
def multiboot(sw, m, c1, c2, B):
    D = np.stack([C[(sw, c1, s)][m].astype(float) - C[(sw, c2, s)][m].astype(float) for s in SEEDS])   # (3, 164)
    point = float(D.mean()); per = D.mean(1)
    S = rng.integers(0, 3, (B, 3)); P = rng.integers(0, D.shape[1], (B, D.shape[1]))
    mb = np.array([D[S[b]][:, P[b]].mean() for b in range(B)])
    pb = np.array([[D[i, P[b]].mean() for i in range(3)] for b in range(min(B, 2000))])   # 逐顺序只对题目
    so = rng.integers(0, 3, (B, 3)); ob = per[so].mean(1)                                    # 只对顺序
    tq = tdist.ppf(0.975, 2) * per.std(ddof=1) / np.sqrt(3)
    return dict(point=point, per_order=per.tolist(),
                multi=[float(np.quantile(mb, .025)), float(np.quantile(mb, .975))],
                problems_only_per_order=[[float(np.quantile(pb[:, i], .025)), float(np.quantile(pb[:, i], .975))] for i in range(3)],
                orders_only=[float(np.quantile(ob, .025)), float(np.quantile(ob, .975))],
                t_orders=[point - float(tq), point + float(tq)],
                frac_var_orders=float(np.var(per, ddof=1) / 3 / max(np.var(mb), 1e-12)))
CMP = [("vperm", "vt"), ("vfac", "vt"), ("vt", "0"), ("vfac", "0"), ("vperm", "0")]
for sw in SW:
    for m in GEN:
        if sw != "PG" and m == "unind": continue
        for c1, c2 in CMP: res["C4"][f"{sw}|{m}|{c1}-{c2}"] = multiboot(sw, m, c1, c2, a.B)

# ---------------------------------------------------------------- 10-11 补：脆弱题替代检验（不是 margin 检验）
# 换顺序会翻转的题（同条件三顺序不全同）与 换条件会翻转的题（同顺序四条件不全同）是否是同一批；与独立时的期望重合比较
frag = {}
for sw in SW:
    for m in GEN:
        A = np.stack([[C[(sw, c, s)][m] for s in SEEDS] for c in CONDS])   # (4 条件, 3 顺序, 164)
        f_ord = (A.min(1) != A.max(1)).any(0); f_cnd = (A.min(0) != A.max(0)).any(0)
        always = int(A.all((0, 1)).sum()); never = int((~A).all((0, 1)).sum())
        n, k1, k2, ov = A.shape[2], int(f_ord.sum()), int(f_cnd.sum()), int((f_ord & f_cnd).sum())
        exp = k1 * k2 / n   # 仅作参照；两集合取自同一组 12 次运行，按构造相关，独立零假设不成立，不报 p
        frag[f"{sw}|{m}"] = dict(n_order_fragile=k1, n_cond_fragile=k2, overlap=ov, expected_indep=float(exp), cond_only=int((f_cnd & ~f_ord).sum()), order_only=int((f_ord & ~f_cnd).sum()), always_true_12runs=always, never_true_12runs=never, mixed=n - always - never)
res["fragile"] = frag

json.dump(res, open(OUT / "c3c4.json", "w"), ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- 表
L = ["# C3 / C4 结果表（underspec_c3c4.py 自动生成）", ""]
L += ["## C3-1 顺序/条件 比（R-C3a：<0.2 被固定，0.2–0.5 部分固定，≥0.5 欠定）", "",
      "| 切换 | 条件 | 层面 | 跨顺序 | 同顺序跨条件（中位） | 比 | 判读 |", "|---|---|---|---|---|---|---|"]
for k, d in res["ratio"].items():
    sw, c = k.split("|")
    for lv, v in d.items():
        unit = "ρ" if not lv.startswith("生成") else "翻转率"
        L.append(f"| {sw} | {CN[c]} | {lv}（{unit}） | {v['order']:.3f} | {v['cond_med']:.3f} | {v['ratio']:.2f} | {v['level']} |")
L += ["", "## C3-1b 敏感性：分母只用 留 v·t 对 全清零", "", "| 切换 | 条件 | " + " | ".join(next(iter(sens.values())).keys()) + " |", "|---|---|" + "---|" * len(next(iter(sens.values())))]
for k, d in sens.items():
    sw, c = k.split("|"); L.append(f"| {sw} | {CN[c]} | " + " | ".join(f"{v:.2f}" for v in d.values()) + " |")
L += ["", "## C3-2 逐题生成：跨顺序 κ 与翻转率（三对顺序）", "", "| 切换 | 指标 | 条件 | 翻转率 | 独立时期望翻转率 | κ |", "|---|---|---|---|---|---|"]
for sw in SW:
    for m in GEN:
        for c in CONDS:
            rr = res["gen"][f"{sw}|{m}|{c}|order"]
            L.append(f"| {sw} | {m} | {CN[c]} | {' / '.join(f'{r['churn']:.3f}' for r in rr)} | {' / '.join(f'{r['churn_indep']:.3f}' for r in rr)} | {' / '.join(f'{r['kappa']:.2f}' for r in rr)} |")
L += ["", "## C3-2b 逐对顺序 得到/失去 题数（前一顺序为真、后一顺序为假 / 反之）", "", "| 切换 | 指标 | 条件 | 17-31 | 17-47 | 31-47 | 三顺序比例 |", "|---|---|---|---|---|---|---|"]
for sw in SW:
    for m in GEN:
        for c in CONDS:
            rr = res["gen"][f"{sw}|{m}|{c}|order"]; rates = [C[(sw, c, s)][m].mean() for s in SEEDS]
            L.append(f"| {sw} | {m} | {CN[c]} | " + " | ".join(f"{r['only_first']}:{r['only_second']}" for r in rr) + " | " + " / ".join(f"{x:.3f}" for x in rates) + " |")
L += ["", "## C3-4 脆弱题替代检验（不是 margin 检验）：换顺序翻转的题与换条件翻转的题是否同一批", "",
      "| 切换 | 指标 | 换顺序会翻转 | 换条件会翻转 | 两者都会 | 只随条件翻转 | 只随顺序翻转 | 12 次运行全真 / 全假 / 混合 |", "|---|---|---|---|---|---|---|---|"]
for k, v in frag.items():
    sw, m = k.split("|")
    L.append(f"| {sw} | {m} | {v['n_order_fragile']} | {v['n_cond_fragile']} | {v['overlap']} | {v['cond_only']} | {v['order_only']} | {v['always_true_12runs']} / {v['never_true_12runs']} / {v['mixed']} |")
L += ["", "## C3-3 R-C3b（全清零的跨顺序 κ 是否在三对顺序上都低于保留 v 三条件的最小值）", ""]
for k, v in rc3b.items(): L.append(f"- {k}: 全清零 κ {['%.2f' % x for x in v['kappa_reset']]}，保留 v 最小 {['%.2f' % x for x in v['kappa_keepv_min']]} → {'是' if v['reset_lower_all'] else '否'}")
L += ["", "## C4 Multi-Bootstrap（差值单位：题目比例；顺序 17/31/47）", "",
      "| 切换 | 指标 | 对比 | 三顺序均值 | 逐顺序差 | 逐顺序题目区间 | 顺序+题目（Multi-Bootstrap） | 只对顺序 | 跨顺序 t（n=3） |", "|---|---|---|---|---|---|---|---|---|"]
for k, v in res["C4"].items():
    sw, m, cc = k.split("|"); c1, c2 = cc.split("-")
    L.append(f"| {sw} | {m} | {CN[c1]} − {CN[c2]} | {v['point']:+.3f} | {' / '.join(f'{x:+.3f}' for x in v['per_order'])} | "
             f"{' '.join(f'[{lo:+.2f},{hi:+.2f}]' for lo, hi in v['problems_only_per_order'])} | [{v['multi'][0]:+.3f}, {v['multi'][1]:+.3f}] | "
             f"[{v['orders_only'][0]:+.3f}, {v['orders_only'][1]:+.3f}] | [{v['t_orders'][0]:+.3f}, {v['t_orders'][1]:+.3f}] |")
open(OUT / "c3c4_tables.md", "w").write("\n".join(L) + "\n")
print("写出", OUT / "c3c4.json", OUT / "c3c4_tables.md")
