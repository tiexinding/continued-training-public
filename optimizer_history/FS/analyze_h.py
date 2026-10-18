#!/usr/bin/env python3
"""H 轮汇总（本地）：三个切换 × 四个核心条件 × 三个数据顺序（种子 17 / 31 / 47）。
结果包解到 results/h/（保留卡上相对路径：diag25/h/…、runs_h/…、diag25/f1/heval_h_*.json）。旧格子（种子 17，MG 的种子 31 留 v 与全清零）从各轮结果目录读，见 OLD。

读数（每格）：
  结构   gate 行 / up 行 / down 列 R²（等元素改动零模型，24 层中位；A_B.json 的 eqstep_Z*_r2fixed）
  重合   down 列 / gate 行 / up 行 原始改动能量 top-5% ∩ 新任务选择性 top-5%，ρ(改动, 旧任务激活)（neuron_act_*_hall.json，同一版脚本重算全部格子）
  探针   训练日志 eval_full 终点的新 / 旧任务探针 NLL 及相对起点的变化
  功能   HumanEval 原始组装 pass_v2、机械缩进修复 pass_v3、无缩进率（一律由保存的生成文本按 drift_diag.fmt_cat 重算）
  逐题   segnll 的逐题 NLL（两口径：逐题等权、token 加权）
判读（文档 41 §8–9）：每格报告三个顺序的值、均值、范围与跨顺序 t 区间（n=3，训练随机性；与题目自助区间分开）；
  关键对比逐顺序计算差值，方向在顺序之间不一致的读数标 ✗（从主张中撤下）。
另：H2 校准核对、H2′ 逐步能量 vs 净变化、平衡分解收敛。输出 results/h/summary_h.json 与 tables_h.md。"""
import json, math, re
from pathlib import Path
import numpy as np

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--res", default=str(Path(__file__).resolve().parents[1] / "DATA"), help="旧轮结果根目录（卡上为镜像目录）")
_ap.add_argument("--h", default="", help="H 轮文件根（含 diag25/h、runs_h、diag25/f1）；本地默认 <res>/h，卡上为 /root/autodl-tmp/fork")
_ap.add_argument("--he", default="", help="HumanEval 题目 parquet；默认 <h>/humaneval.parquet")
_ap.add_argument("--out", default="", help="输出目录；默认 <h>（本地）")
_a = _ap.parse_args()
RES = Path(_a.res); H = Path(_a.h) if _a.h else RES / "h"; OUT = Path(_a.out) if _a.out else H
SW = ["MG", "GM", "PG"]; CONDS = ["vt", "vfac", "vperm", "0"]; SEEDS = [17, 31, 47]
CN = {"vt": "留 v·t", "vfac": "v 行列近似", "vperm": "v 置换", "0": "全清零"}
SWN = {"MG": "MBPP 起点→GSM8K", "GM": "GSM8K 起点→MBPP", "PG": "PubMed 摘要起点→GSM8K"}
NEWOLD = {"MG": ("gsm8k", "mbpp"), "GM": ("mbpp", "gsm8k"), "PG": ("gsm8k", "bio")}
LAB = {"gsm8k": "GSM8K", "mbpp": "MBPP", "bio": "MBPP"}   # neuron_act 的键名写死为 GSM8K/MBPP；PG 用 --sel gsm8k,bio 时 "MBPP" 位置即 bio
NA = {"MG": "M_hall", "GM": "A_g_hall", "PG": "B_g_hall"}

# ---------------------------------------------------------------- 旧格子的文件位置（相对 RES）
# heval: (原始生成文件, 键, 重判文件, 键)；pass_v3 取重判 v3_autoindent，pass_v2 取原始 pass_rate
OLD = {
    ("MG", "vt", 17): dict(pairs="b2/pairs_Kv", log="runs_b2/Kv", heval=("c/diag25/c4/humaneval_c.json", "K_vt", "e/humaneval_c_rescore.json", "K_vt"), segnll="c/diag25/c4/segnll/K_vt.json"),
    ("MG", "0", 17): dict(pairs="t3/diag25/t3/pairs", log="t3/runs_t3/lr5e-6/G", heval=("b3/humaneval_v2.json", "Gfresh", "e/humaneval_b3_rescore.json", "Gfresh"), segnll="diag25/b1/segnll/Gfresh.json"),
    ("MG", "vt", 31): dict(pairs="c/diag25/c3/pairs_K_vt_s31", log="c/runs_c/K_vt_s31", heval=("c/diag25/c4/humaneval_c.json", "K_vt_s31", "e/humaneval_c_rescore.json", "K_vt_s31"), segnll="c/diag25/c4/segnll/K_vt_s31.json"),
    ("MG", "0", 31): dict(pairs="c/diag25/c3/pairs_K_0_s31", log="c/runs_c/K_0_s31", heval=("c/diag25/c4/humaneval_c.json", "K_0_s31", "e/humaneval_c_rescore.json", "K_0_s31"), segnll="c/diag25/c4/segnll/K_0_s31.json"),
    ("GM", "vt", 17): dict(pairs="d/diag25/d3/pairs_R_vt", log="d/runs_d/R_vt", heval=("d/diag25/d4/humaneval_d.json", "R_vt", "e/humaneval_d_rescore.json", "R_vt"), segnll="d/diag25/d4/segnll/R_vt.json"),
    ("GM", "vperm", 17): dict(pairs="e/diag25/e3/pairs_R_vperm_med", log="e/runs_e/R_vperm_med", heval=("e/diag25/e4/humaneval_e.json", "R_vperm_med", "e/humaneval_e_rescore.json", "R_vperm_med"), segnll="e/diag25/e4/segnll/R_vperm_med.json"),
    ("GM", "0", 17): dict(pairs="d/diag25/d3/pairs_R_0", log="d/runs_d/R_0", heval=("d/diag25/d4/humaneval_d.json", "R_0", "e/humaneval_d_rescore.json", "R_0"), segnll="d/diag25/d4/segnll/R_0.json"),
}
for c, e in (("vt", "E_vt"), ("vfac", "E_vfac"), ("vperm", "E_vperm"), ("0", "E_0")):
    OLD[("PG", c, 17)] = dict(pairs=f"e/diag25/e3/pairs_{e}", log=f"e/runs_e/{e}", heval=("e/diag25/e4/humaneval_e.json", e, "e/humaneval_e_rescore.json", e), segnll=f"e/diag25/e4/segnll/{e}.json")


def tag(sw, c, s): return f"{sw}_{c}_s{s}"


def src(sw, c, s):
    """返回该格的文件位置；新跑优先于旧（PG_vt_s17 新旧逐位相同，取旧以与历史读数一致）。"""
    if (sw, c, s) in OLD: return {k: (RES / v if isinstance(v, str) else v) for k, v in OLD[(sw, c, s)].items()} | dict(old=True)
    n = tag(sw, c, s)
    return dict(pairs=H / f"diag25/h/pairs_{n}", log=H / f"runs_h/{n}", heval=("NEW", n), segnll=H / f"diag25/h/segnll/{n}.json", old=False)


# ---------------------------------------------------------------- HumanEval 格式（与 drift_diag.fmt_cat 相同）
try:
    import pandas as pd
    HE = list(pd.read_parquet(_a.he or H / "humaneval.parquet").itertuples())
except Exception as ex:
    HE = None; print("HumanEval 题目文件缺，无缩进率不重算：", ex)


def fmt_cat(o, ep):
    m = re.search(r"```(?:python)?\n(.*?)```", o, re.S); b = m.group(1) if m else o
    if ("def " + ep) in b: return "full"
    first = next((l for l in b.split("\n") if l.strip()), ""); return "unind" if first and not first.startswith((" ", "\t")) else "ind"


_jc = {}
def J(p):
    p = Path(p)
    if p not in _jc: _jc[p] = json.load(open(p)) if p.exists() else None
    return _jc[p]


def kind(n):
    for k in ("gate_proj", "up_proj", "down_proj"):
        if k in n: return k.replace("_proj", "")


# ---------------------------------------------------------------- 逐格读数
def read_cell(sw, c, s):
    S = src(sw, c, s); r = dict(old=S["old"]); new, old = NEWOLD[sw]
    # 结构
    pm = J(Path(S["pairs"]) / "A_B.json")
    if pm:
        pm = pm["per_matrix"]
        for kd, key in (("gate", "eqstep_Zrow_r2fixed"), ("up", "eqstep_Zrow_r2fixed"), ("down", "eqstep_Zcol_r2fixed")):
            r[f"R2_{kd}"] = float(np.median([x["L3"][key] for n, x in pm.items() if kind(n) == kd]))
        conv = [d for x in pm.values() for d in x.get("decomp_conv", [])]
        if conv: r["decomp"] = dict(n=len(conv), iters_max=max(d["iters"] for d in conv), cv_max=max(max(d["cv_row"], d["cv_col"]) for d in conv))
    # 重合与 ρ
    na = J(H / f"diag25/h/neuron_act_{NA[sw]}.json")
    t = tag(sw, c, s)
    if na and t in na["overlap"]:
        ov = na["overlap"][t]
        for kn, short in (("down列原始能量", "down"), ("gate行原始能量", "gate"), ("up行原始能量", "up")):
            k1 = f"{kn}∩{LAB[new]}选择性"; k2 = f"{kn}~{LAB[old]}激活 ρ"
            if k1 in ov: r[f"ov_{short}"] = ov[k1]["median"]; r[f"rho_old_{short}"] = ov[k2]["median"]
    # 探针（训练日志 eval_full）
    lp = Path(S["log"]) / "log.jsonl"
    if lp.exists():
        L = [json.loads(l) for l in open(lp) if '"eval_full"' in l]
        if L:
            r0, r1 = L[0], L[-1]; r["eval_step"] = r1["step"]
            if r1["step"] < {"GM": 108}.get(sw, 125): r["incomplete"] = True; return r   # 运行中：不当终点读数
            for task, nm in ((new, "new"), (old, "old")):
                if task in r1: r[f"nll_{nm}"] = r1[task]; r[f"dnll_{nm}"] = r1[task] - r0[task] if task in r0 else None
            if "dolly_qa" in r1: r["nll_dolly"] = r1["dolly_qa"]
    # HumanEval
    outs = p2 = p3 = None
    if S["heval"][0] == "NEW":
        f = H / f"diag25/f1/heval_h_{t}.json"; d = J(f)
        if d and t in d: d = d[t]; outs, p2, p3 = d["outs"], d["passed_v2"], d["passed"]
    else:
        fr, kr, fs, ks = S["heval"]; d = J(RES / fr); rs = J(RES / fs)
        if d and kr in d: outs = d[kr]["outs"]; p2 = d[kr]["passed"]
        if rs and ks in rs: p3 = rs[ks]["v3_autoindent"]["passed"]; p2 = p2 if p2 is not None else rs[ks]["v2"]["passed"]
    if p2 is not None: r["pass_v2"] = float(np.mean(p2)); r["_p2"] = [bool(x) for x in p2]
    if p3 is not None: r["pass_v3"] = float(np.mean(p3)); r["_p3"] = [bool(x) for x in p3]
    if outs is not None and HE is not None:
        cats = [fmt_cat(o, q.entry_point) for o, q in zip(outs, HE)]; r["unind"] = float(np.mean([x == "unind" for x in cats])); r["_unind"] = [x == "unind" for x in cats]
    # 逐题 NLL
    sg = J(S["segnll"])
    if sg:
        for ds, v in sg["nll"].items():
            s_, c_ = np.array(v["s"], float), np.array(v["c"], float); ok = c_ > 0
            r[f"seg_{ds}_item"] = float(np.mean(s_[ok] / c_[ok])); r[f"seg_{ds}_tok"] = float(s_.sum() / c_.sum()); r[f"_seg_{ds}"] = (s_[ok] / c_[ok]).tolist() if ok.all() else None
    return r


CELLS = {(sw, c, s): read_cell(sw, c, s) for sw in SW for c in CONDS for s in SEEDS}

# ---------------------------------------------------------------- 汇总：三顺序值、均值、范围、跨顺序 t 区间
READ = ["R2_gate", "R2_up", "R2_down", "ov_down", "rho_old_down", "ov_gate", "rho_old_gate", "ov_up", "rho_old_up",
        "nll_new", "nll_old", "dnll_new", "dnll_old", "pass_v2", "pass_v3", "unind"]
T975 = {2: 12.706, 3: 4.303}


def agg(vals):
    x = np.array([v for v in vals if v is not None], float)
    if len(x) == 0: return None
    o = dict(n=len(x), mean=float(x.mean()), min=float(x.min()), max=float(x.max()))
    if len(x) >= 2:
        h = T975[len(x)] * x.std(ddof=1) / math.sqrt(len(x)); o["t_ci"] = [o["mean"] - h, o["mean"] + h]
    return o


SUM = {}
for sw in SW:
    for c in CONDS:
        for q in READ:
            SUM.setdefault(sw, {}).setdefault(c, {})[q] = dict(by_seed={s: CELLS[(sw, c, s)].get(q) for s in SEEDS}, **(agg([CELLS[(sw, c, s)].get(q) for s in SEEDS]) or {}))

# ---------------------------------------------------------------- 关键对比：逐顺序差值与方向一致性
CONTRASTS = [("ov_down", "vt", "vperm"), ("ov_down", "vfac", "vperm"), ("ov_down", "vt", "0"), ("ov_gate", "vt", "vperm"), ("ov_up", "vt", "vperm"),
             ("rho_old_down", "vperm", "vt"), ("R2_down", "0", "vt"), ("R2_gate", "0", "vt"),
             ("unind", "vfac", "vt"), ("unind", "vperm", "vt"), ("unind", "vt", "0"), ("pass_v2", "vt", "0"), ("pass_v3", "vt", "0"),
             ("dnll_old", "vt", "0"), ("dnll_new", "vt", "0")]


def mcnemar(a, b):
    a, b = np.array(a, bool), np.array(b, bool); g, l = int(np.sum(a & ~b)), int(np.sum(~a & b)); n = g + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(g, l) + 1)) / 2 ** n) if n else 1.0
    return g, l, p


CON = {}
for sw in SW:
    for q, x, y in CONTRASTS:
        d = {}; extra = {}
        for s in SEEDS:
            a, b = CELLS[(sw, x, s)].get(q), CELLS[(sw, y, s)].get(q)
            d[s] = None if a is None or b is None else a - b
            lk = {"unind": "_unind", "pass_v2": "_p2", "pass_v3": "_p3"}.get(q)
            if lk and lk in CELLS[(sw, x, s)] and lk in CELLS[(sw, y, s)]: extra[s] = mcnemar(CELLS[(sw, x, s)][lk], CELLS[(sw, y, s)][lk])
        v = [z for z in d.values() if z is not None]
        cons = None if len(v) < 2 else (all(z > 0 for z in v) or all(z < 0 for z in v))
        CON.setdefault(sw, {})[f"{q}:{x}−{y}"] = dict(by_seed=d, n=len(v), consistent=cons, mean=float(np.mean(v)) if v else None, mcnemar=extra or None)

# ---------------------------------------------------------------- H2 校准、H2′ 能量、分解收敛
CAL = {}
for f in sorted((H / "diag25/h/calib").glob("*.json")):
    d = json.load(open(f)); CAL[f.stem] = dict(c_check=d.get("c_check"), summary={g: {q: d["summary"][g][q] for q in ("med_ratio", "rms_ratio")} for g in d["summary"]},
                                               r2_vT={k: d["r2_vT"][k] for k in ("median_r2_all", "median_r2_body", "pooled_r2_all", "pooled_r2_body", "floor_frac_all", "by_kind")} if "r2_vT" in d else None,
                                               r2_v1={k: d["r2_v1"][k] for k in ("median_r2_all", "median_r2_body", "pooled_r2_all", "pooled_r2_body", "floor_frac_all", "by_kind")} if "r2_v1" in d else None)
EN = {k: v["median"] for k, v in (J(H / "diag25/h/h2p_energy.json") or {}).items()}
DEC = {tag(*k): v["decomp"] for k, v in CELLS.items() if "decomp" in v}

# ---------------------------------------------------------------- 输出
def fmt(v, q):
    if v is None: return "—"
    return f"{v:+.3f}" if q.startswith(("rho", "dnll")) else f"{v:.3f}"


md = ["# H 轮汇总表（analyze_h.py 自动生成）", "",
      "三个数据顺序（种子 17 / 31 / 47）；均值后括号为跨顺序 t 区间（n=3，训练随机性，与题目自助区间不同）。—= 缺。旧格子（种子 17 及 MG 种子 31 的留 v·t、全清零）来自 C/D/E 轮；无缩进率一律由保存的生成文本重算。", ""]
for sw in SW:
    md += [f"## {SWN[sw]}（{sw}）", "", "| 读数 | " + " | ".join(CN[c] for c in CONDS) + " |", "|---" * (len(CONDS) + 1) + "|"]
    for q in READ:
        row = [q]
        for c in CONDS:
            a = SUM[sw][c][q]; vs = " / ".join(fmt(a["by_seed"][s], q) for s in SEEDS)
            row.append(vs + (f"；均 {fmt(a['mean'], q)}" + (f" [{fmt(a['t_ci'][0], q)}, {fmt(a['t_ci'][1], q)}]" if "t_ci" in a else "") if "mean" in a else ""))
        md.append("| " + " | ".join(row) + " |")
    md += ["", "**关键对比（逐顺序差值 17 / 31 / 47；✓ 方向一致，✗ 不一致 → 撤下；HumanEval 附逐题 McNemar 得/失, p）**", "", "| 对比 | 差值 | 一致 | McNemar |", "|---|---|---|---|"]
    for k, v in CON[sw].items():
        mc = "；".join(f"{s}: {g}:{l} p={p:.3g}" for s, (g, l, p) in v["mcnemar"].items()) if v["mcnemar"] else ""
        md.append(f"| {k} | " + " / ".join(fmt(v['by_seed'][s], 'dnll' if 'rho' in k or 'dnll' in k else 'x') if v['by_seed'][s] is not None else "—" for s in SEEDS)
                  + f" | {'—' if v['consistent'] is None else ('✓' if v['consistent'] else '✗')} | {mc} |")
    md.append("")
if CAL:
    md += ["## H2 校准核对", "", "| 条件 | c 逐位一致 | 主体 中位数比 中位 [p10, p90] | 词表 中位数比 | 主体 RMS 比 | 词表 RMS 比 |", "|---|---|---|---|---|---|"]
    for k, v in CAL.items():
        cc = v["c_check"]; s_ = v["summary"]
        g = lambda grp, q: (f"{s_[grp][q]['median']:.3f} [{s_[grp][q]['p10']:.2f}, {s_[grp][q]['p90']:.2f}]" if grp in s_ else "—")
        ccs = "—" if not cc else f"{cc['n_exact']}/{cc['n']}（最大差 {cc['max_abs_diff']:.1e}）"
        md.append(f"| {k} | {ccs} | {g('body','med_ratio')} | {g('vocab','med_ratio')} | {g('body','rms_ratio')} | {g('vocab','rms_ratio')} |")
    md += ["", "**log v 行列加法模型 R²（93% 口径）**", "", "| 起点 | 时刻 | 逐矩阵中位（全部 / 主体） | 元素加权合并（全部 / 主体） | 下限占比 |", "|---|---|---|---|---|"]
    for k, v in CAL.items():
        for when in ("r2_vT", "r2_v1"):
            if v[when]: x = v[when]; md.append(f"| {k} | {'切换时 v_T' if when == 'r2_vT' else '新任务第 1 步后'} | {x['median_r2_all']:.3f} / {x['median_r2_body']:.3f} | {x['pooled_r2_all']:.3f} / {x['pooled_r2_body']:.3f} | {x['floor_frac_all']:.2e} |")
    md.append("")
if EN:
    md += ["## H2′ 逐步更新能量累计 vs 净变化（down 列，24 层中位）", "", "| 运行 | ρ(累计, 净) | top-5% 重合 | 抵消比 Σ净/Σ累计 | 前 10 步份额 | 新任务重合 净 / 累计 | ρ(旧激活) 净 / 累计 |", "|---|---|---|---|---|---|---|"]
    for k, m in sorted(EN.items()):
        md.append(f"| {k} | {m['rho_cum_net']:.3f} | {m['top5_cum_net']:.3f} | {m['cancel']:.3f} | {m['early10']:.3f} | {m['ov_new_net']:.3f} / {m['ov_new_cum']:.3f} | {m['rho_old_net']:+.3f} / {m['rho_old_cum']:+.3f} |")
    md.append("")
if DEC:
    md += ["## 平衡分解收敛（新跑）", "", f"{len(DEC)} 条运行；最大迭代次数 {max(d['iters_max'] for d in DEC.values())}，最大终 CV {max(d['cv_max'] for d in DEC.values()):.1e}（容差 1e-7）。", ""]

OUT.mkdir(parents=True, exist_ok=True)
clean = {f"{sw}|{c}|{s}": {k: v for k, v in CELLS[(sw, c, s)].items() if not k.startswith("_")} for (sw, c, s) in CELLS}
json.dump(dict(cells=clean, summary=SUM, contrasts=CON, calib=CAL, energy=EN, decomp=DEC), open(OUT / "summary_h.json", "w"), ensure_ascii=False, indent=1)
open(OUT / "tables_h.md", "w", encoding="utf-8").write("\n".join(md))
print("\n".join(md)); print("\n->", OUT / "summary_h.json", OUT / "tables_h.md")
