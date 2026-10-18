#!/usr/bin/env python3
"""D 轮汇总（本地）：results/d/。第二方向 A_g(→GSM8K 起点) →MBPP：新任务 = MBPP，所以"新任务选择性"= −sel_gsm_vs_mbpp（neuron_act 的 sel 定义为 GSM8K−MBPP）。
输出 summary_d.json 与控制台表：结构读数、选择性重合（翻转后）、功能、离线反事实（修正版、三口径）、干预 v3（含逐题配对区间）。"""
import json, glob, math
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / "DATA"; D = R / "d"; S = {}
def kind(n):
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed"):
        if k in n: return k.replace("_proj", "")
def mcn(a, b):
    a = np.array(a, bool); b = np.array(b, bool); g = int(np.sum(a & ~b)); l = int(np.sum(~a & b)); n = g + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(g, l) + 1)) / 2 ** n) if n else 1.0; return g, l, round(p, 3)
ARMS = ["R_vt", "R_vfac", "R_vperm", "R_0"]
print("== 1 结构读数（A_g→MBPP 四臂；24 层中位）gate 行 R² / up / down 列 / cvZ/零(up) / ε(gate)")
S["structure"] = {}
for a in ARMS + ["R_c"]:
    p = D / f"diag25/d3/pairs_{a}/A_B.json" if a != "R_c" else D / "diag25/d3/pairs_R_vt/A_contA.json"
    if not p.exists(): print(a, "缺"); continue
    pm = json.load(open(p))["per_matrix"]; r = {}
    for kd in ("gate", "up", "down"):
        rs = [x for n, x in pm.items() if kind(n) == kd]
        r[kd] = dict(r2row=float(np.median([x["L3"]["eqstep_Zrow_r2fixed"] for x in rs])), r2col=float(np.median([x["L3"]["eqstep_Zcol_r2fixed"] for x in rs])), cv=float(np.median([x["L3"]["shuffle"]["cvZ_row"] / x["L3"]["shuffle"]["cvZ_row_null_Wshuffle"] for x in rs])), eps=float(np.median([x["L3"]["eps"] for x in rs])))
    S["structure"][a] = r; print(f"{a:8} {r['gate']['r2row']:.2f} / {r['up']['r2row']:.2f} / {r['down']['r2col']:.2f} / {r['up']['cv']:.2f} / {r['gate']['eps']:.4f}")
print("\n== 2 选择性重合（翻转：新任务 = MBPP）：down 列能量 top-5% ∩ MBPP 相对选择性 / ∩ GSM8K 选择性 / ρ(down 能量, GSM8K 激活=旧任务) / ρ(., MBPP 激活)")
p = D / "diag25/b4/neuron_act_A_g_d.json"
if p.exists():
    J = json.load(open(p))["overlap"]; S["overlap"] = {}
    for a in J:
        m = {k: v["median"] for k, v in J[a].items()}
        S["overlap"][a] = dict(new_sel=m["down列原始能量∩MBPP选择性"], old_sel=m["down列原始能量∩GSM8K选择性"], rho_old=m["down列原始能量~GSM8K激活 ρ"], rho_new=m["down列原始能量~MBPP激活 ρ"])
        print(f"{a:8} {m['down列原始能量∩MBPP选择性']:.3f} / {m['down列原始能量∩GSM8K选择性']:.3f} / {m['down列原始能量~GSM8K激活 ρ']:+.3f} / {m['down列原始能量~MBPP激活 ρ']:+.3f}")
print("\n== 3 功能：GSM8K 留出 relaxed（旧任务）/ MBPP pass@1（新任务）/ HumanEval / Dolly 召回 ； 对 A_g 配对")
S["gen"] = {}
Ag_g = Ag_m = None
f = D / "diag25/d4/A_g_max400.jsonl"
if f.exists(): Ag_g = [json.loads(l)["relaxed"] for l in open(f)]; Ag_m = json.load(open(D / "diag25/d4/A_g_mbpp.json"))["passed"]
H = json.load(open(D / "diag25/d4/humaneval_d.json")) if (D / "diag25/d4/humaneval_d.json").exists() else {}
Dl = json.load(open(D / "diag25/d4/dolly_gen_d.json")) if (D / "diag25/d4/dolly_gen_d.json").exists() else {}
for a in ["A_g"] + ARMS + ["R_c"]:
    f = D / f"diag25/d4/{a}_max400.jsonl"
    if not f.exists(): continue
    g = [json.loads(l)["relaxed"] for l in open(f)]; m = json.load(open(D / f"diag25/d4/{a}_mbpp.json"))["passed"]
    S["gen"][a] = dict(gsm8k=float(np.mean(g)), mbpp=float(np.mean(m)), humaneval=H.get(a, {}).get("pass_rate"), dolly_r=Dl.get(a, {}).get("mean_r"), dolly_len=Dl.get(a, {}).get("med_len"), vsAg_gsm=mcn(g, Ag_g) if Ag_g else None, vsAg_mbpp=mcn(m, Ag_m) if Ag_m else None)
    s_ = S["gen"][a]; print(f"{a:8} {s_['gsm8k']:.3f} / {s_['mbpp']:.2f} / {s_['humaneval']} / {s_['dolly_r']} ({s_['dolly_len']} 词)  对 A_g {s_['vsAg_gsm']} / {s_['vsAg_mbpp']}")
print("\n== 4 探针 NLL（GSM8K 探针=旧 / MBPP 探针=新 / Dolly / GSM8K 留出）")
S["nll"] = {}
for f in sorted(glob.glob(str(D / "diag25/d4/segnll/*.json"))):
    n = Path(f).stem; v = json.load(open(f))["nll"]; r = {ds: sum(v[ds]["s"]) / sum(v[ds]["c"]) for ds in v}; S["nll"][n] = r
    print(f"{n:8} {r['probe_gsm8k']:.3f} / {r['probe_mbpp']:.3f} / {r['probe_dolly_qa']:.3f} / {r['heldout_gsm8k']:.3f}")
print("\n== 5 修正版离线反事实（匹配 RMS 后，down 列；逐批份额均值 新任务组/旧任务组 | 总平方 | 合成 | cos 对 orig | 相对误差 | 秩ρ | 原始 RMS 比）")
S["offline"] = {}
for tag, f, newk, oldk in (("M 起点（新=GSM8K）", D / "diag25/d1/adam_offline_v2_M.json", "gsm", "mbpp"), ("A_g 起点（新=MBPP）", D / "diag25/d1/adam_offline_v2_Ag.json", "mbpp", "gsm")):
    if not f.exists(): print(tag, "缺"); continue
    J = json.load(open(f))["variants"]; S["offline"][tag] = {}; print("--", tag)
    for v, r in J.items():
        m = r["matched"]["median"]["down_proj"]; raw = r["raw"]["median"]["down_proj"]
        S["offline"][tag][v] = dict(matched=m, raw=raw)
        print(f"{v:16} {m[f'perbatch_share_{newk}_mean']:.3f}±{m[f'perbatch_share_{newk}_sd']:.3f}/{m[f'perbatch_share_{oldk}_mean']:.3f} | {m[f'sumsq_share_{newk}']:.3f}/{m[f'sumsq_share_{oldk}']:.3f} | {m[f'comb_share_{newk}']:.3f}/{m[f'comb_share_{oldk}']:.3f} | {m['cos_to_orig']:.3f} | {m['rel_err_to_orig']:.3f} | {m['chan_rank_corr_to_orig']:.3f} | {raw['rms_over_orig']:.3f}")
print("\n== 6 干预 v3（K_mvt 终态；纯 NLL；MBPP 探针回退量 + 逐题配对区间；独立代码集 HumanEval 规范解 NLL 回退量）")
p = D / "diag25/d5/intervene_Gkeep_v3.json"
if p.exists():
    J = json.load(open(p)); B = J["B"]; A_ = J["A"]; rng = np.random.default_rng(0); S["intervene_v3"] = {}
    for k, v in J.items():
        if k in ("A", "B", "pairwise"): continue
        d = np.array(B["nll_per_item"]["probe_mbpp"]) - np.array(v["nll_per_item"]["probe_mbpp"]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        he = (B["nll"].get("heldout_humaneval", np.nan) - v["nll"].get("heldout_humaneval", np.nan)) if "heldout_humaneval" in B["nll"] else float("nan")
        S["intervene_v3"][k] = dict(e=v["reverted_energy_frac"], mbpp_rec=float(d.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], he_rec=float(he), gsm_rec=float(B["nll"]["probe_gsm8k"] - v["nll"]["probe_gsm8k"]))
        print(f"{k:26} 能量 {v['reverted_energy_frac']*100:4.1f}%  MBPP 回退 {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  HumanEval 回退 {he:+.4f}  GSM8K 探针回退 {B['nll']['probe_gsm8k']-v['nll']['probe_gsm8k']:+.4f}")
json.dump(S, open(D / "summary_d.json", "w"), ensure_ascii=False, indent=1); print("\nsummary ->", D / "summary_d.json")
