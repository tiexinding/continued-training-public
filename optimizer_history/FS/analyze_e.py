#!/usr/bin/env python3
"""E 轮汇总（本地）：results/e/。E1 离线 v3（含 rowonly/colonly/randfac）；E2 Bio→GSM8K 四臂（新任务 = GSM8K，旧任务 = bio，旧任务损失取训练日志 eval_full 的 bio）；R_vperm_med 复跑；E3 回放两臂（对 K_vt/K_∅）；E5 独立同域确认。"""
import json, glob, math
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / "DATA"; E = R / "e"; S = {}
def kind(n):
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed"):
        if k in n: return k.replace("_proj", "")
def mcn(a, b):
    a = np.array(a, bool); b = np.array(b, bool); g = int(np.sum(a & ~b)); l = int(np.sum(~a & b)); n = g + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(g, l) + 1)) / 2 ** n) if n else 1.0; return g, l, round(p, 3)
ARMS = ["E_vt", "E_vfac", "E_vperm", "E_0"]
print("== 1 结构读数（B_g→GSM8K 四臂 + R_vperm_med）gate 行 R² / up / down 列 / cvZ / ε"); S["structure"] = {}
for a in ARMS + ["R_vperm_med"]:
    p = E / f"diag25/e3/pairs_{a}/A_B.json"
    if not p.exists(): print(a, "缺"); continue
    pm = json.load(open(p))["per_matrix"]; r = {}
    for kd in ("gate", "up", "down"):
        rs = [x for n, x in pm.items() if kind(n) == kd]
        r[kd] = dict(r2row=float(np.median([x["L3"]["eqstep_Zrow_r2fixed"] for x in rs])), r2col=float(np.median([x["L3"]["eqstep_Zcol_r2fixed"] for x in rs])), cv=float(np.median([x["L3"]["shuffle"]["cvZ_row"] / x["L3"]["shuffle"]["cvZ_row_null_Wshuffle"] for x in rs])), eps=float(np.median([x["L3"]["eps"] for x in rs])))
    S["structure"][a] = r; print(f"{a:12} {r['gate']['r2row']:.2f} / {r['up']['r2row']:.2f} / {r['down']['r2col']:.2f} / {r['up']['cv']:.2f} / {r['gate']['eps']:.4f}")
print("\n== 2 选择性重合（B_g 激活图；新任务 = GSM8K：∩GSM8K 选择性 / ∩MBPP 选择性 / ρ(down, GSM8K 激活) / ρ(down, MBPP 激活)）—— 注意：B_g 的旧任务是 bio，neuron_act 的选择性只在 GSM8K/MBPP 间定义；bio 激活图另算")
for f, tag in ((E / "diag25/b4/neuron_act_B_g_e.json", "B_g"), (E / "diag25/b4/neuron_act_A_g_e.json", "A_g(R_vperm_med)")):
    if f.exists():
        J = json.load(open(f))["overlap"]; S.setdefault("overlap", {})
        for a in J:
            m = {k: v["median"] for k, v in J[a].items()}; S["overlap"][a] = m
            print(f"{tag:16} {a:12} {m['down列原始能量∩GSM8K选择性']:.3f} / {m['down列原始能量∩MBPP选择性']:.3f} / {m['down列原始能量~GSM8K激活 ρ']:+.3f} / {m['down列原始能量~MBPP激活 ρ']:+.3f}")
print("\n== 3 训练日志 eval_full 终点（gsm8k / mbpp / dolly / bio 探针 NLL）"); S["eval_full"] = {}
for f in sorted(glob.glob(str(E / "runs_e/*/log.jsonl"))):
    a = Path(f).parent.name; L = [json.loads(l) for l in open(f) if '"eval_full"' in l]
    if not L: continue
    r0, r1 = L[0], L[-1]; S["eval_full"][a] = dict(start=r0, end=r1)
    print(f"{a:12} 起 {r0.get('gsm8k',float('nan')):.3f}/{r0.get('mbpp',float('nan')):.3f}/{r0.get('dolly_qa',float('nan')):.3f}/{r0.get('bio',float('nan')):.3f} → 终 {r1.get('gsm8k',float('nan')):.3f}/{r1.get('mbpp',float('nan')):.3f}/{r1.get('dolly_qa',float('nan')):.3f}/{r1.get('bio',float('nan')):.3f}")
print("\n== 4 功能：GSM8K 留出 / MBPP / HumanEval / Dolly 召回（词数）"); S["gen"] = {}
H = json.load(open(E / "diag25/e4/humaneval_e.json")) if (E / "diag25/e4/humaneval_e.json").exists() else {}
Dl = json.load(open(E / "diag25/e4/dolly_gen_e.json")) if (E / "diag25/e4/dolly_gen_e.json").exists() else {}
G = {}; M = {}
for f in sorted(glob.glob(str(E / "diag25/e4/*_max400.jsonl"))):
    a = Path(f).name.replace("_max400.jsonl", ""); G[a] = [json.loads(l)["relaxed"] for l in open(f)]; M[a] = json.load(open(E / f"diag25/e4/{a}_mbpp.json"))["passed"]
    S["gen"][a] = dict(gsm8k=float(np.mean(G[a])), mbpp=float(np.mean(M[a])), humaneval=H.get(a, {}).get("pass_rate"), dolly_r=Dl.get(a, {}).get("mean_r"), dolly_len=Dl.get(a, {}).get("med_len"))
    s_ = S["gen"][a]; print(f"{a:12} {s_['gsm8k']:.3f} / {s_['mbpp']:.2f} / {s_['humaneval']} / {s_['dolly_r']} ({s_['dolly_len']})")
for a, b in (("E_vt", "E_0"), ("E_vfac", "E_vt"), ("E_vperm", "E_vt"), ("E_vt", "B_g"), ("P_vt", "P_0")):
    if a in G and b in G: print(f"配对 {a} vs {b}: GSM8K {mcn(G[a], G[b])}  MBPP {mcn(M[a], M[b])}")
# 回放两臂对无回放（K_vt / K_∅ 的审计在 b2 与 t3）
try:
    Kvt = [json.loads(l)["relaxed"] for l in open(R / "b2/Kvfinal_max400.jsonl")]; K0 = [json.loads(l)["relaxed"] for l in open(R / "t3/diag25/t3/Gfreshfinal_max400.jsonl")]
    Kvt_m = json.load(open(R / "b2/Kvfinal_mbpp.json"))["passed"]; K0_m = json.load(open(R / "t3/diag25/t3/Gfreshfinal_mbpp.json"))["passed"]
    if "P_vt" in G: print(f"回放 P_vt vs 无回放 K_vt: GSM8K {mcn(G['P_vt'], Kvt)} MBPP {mcn(M['P_vt'], Kvt_m)}   P_0 vs K_∅: GSM8K {mcn(G['P_0'], K0)} MBPP {mcn(M['P_0'], K0_m)}")
except Exception as e: print("回放对照缺", e)
print("\n== 5 探针 NLL（segnll）"); S["nll"] = {}
for f in sorted(glob.glob(str(E / "diag25/e4/segnll/*.json"))):
    n = Path(f).stem; v = json.load(open(f))["nll"]; r = {ds: sum(v[ds]["s"]) / sum(v[ds]["c"]) for ds in v}; S["nll"][n] = r
    print(f"{n:12} gsm {r['probe_gsm8k']:.3f} / mbpp {r['probe_mbpp']:.3f} / dolly {r['probe_dolly_qa']:.3f} / gsm 留出 {r['heldout_gsm8k']:.3f}")
print("\n== 6 离线 v3（匹配后 down 列=中间神经元 逐批份额 新/旧 | cos | 秩ρ）"); S["offline"] = {}
for tag, f, newk, oldk in (("M（新=GSM8K）", E / "diag25/e1/adam_offline_v3_M.json", "gsm", "mbpp"), ("A_g（新=MBPP）", E / "diag25/e1/adam_offline_v3_Ag.json", "mbpp", "gsm")):
    if not f.exists(): print(tag, "缺"); continue
    J = json.load(open(f))["variants"]; S["offline"][tag] = {}; print("--", tag)
    for kd in ("down_proj", "gate_proj"):
        print("  ", kd)
        for v, r in J.items():
            m = r["matched"]["median"][kd]; S["offline"][tag].setdefault(v, {})[kd] = m
            print(f"   {v:14} {m[f'perbatch_share_{newk}_mean']:.3f}/{m[f'perbatch_share_{oldk}_mean']:.3f} | {m['cos_to_orig']:.3f} | {m['chan_rank_corr_to_orig']:.3f}")
print("\n== 7 独立同域确认（E5；估计集 = MBPP 探针前 50，确认集 = 后 50）"); 
p = E / "diag25/e5/intervene_Gkeep_v3_confirm.json"
if p.exists():
    J = json.load(open(p)); B = J["B"]; rng = np.random.default_rng(0); S["confirm"] = {}
    sel = np.array(J["sel_mbpp@0.05"]["nll_per_item"]["probe_mbpp"]); Bi = np.array(B["nll_per_item"]["probe_mbpp"])
    for k in ["sel_mbpp@0.05"] + [f"mbpp_dual_matched{r}@0.05" for r in range(5)] + [f"mbpp_matched_random{r}@0.05" for r in range(5)] + ["interp@0.05"]:
        if k not in J: continue
        v = np.array(J[k]["nll_per_item"]["probe_mbpp"]); dC = (Bi - v)[50:]; dE = (Bi - v)[:50]; dd = ((Bi - sel) - (Bi - v))[50:]
        bs = [rng.choice(dC, len(dC)).mean() for _ in range(3000)]; bs2 = [rng.choice(dd, len(dd)).mean() for _ in range(3000)]
        S["confirm"][k] = dict(est=float(dE.mean()), confirm=float(dC.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], diff_vs_sel_confirm=float(dd.mean()), diff_ci=[float(np.percentile(bs2, 2.5)), float(np.percentile(bs2, 97.5))])
        print(f"{k:26} 估计集回退 {dE.mean():+.4f}  确认集回退 {dC.mean():+.4f} [{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  确认集 sel−该组 {dd.mean():+.4f} [{np.percentile(bs2,2.5):+.4f},{np.percentile(bs2,97.5):+.4f}]")
json.dump(S, open(E / "summary_e.json", "w"), ensure_ascii=False, indent=1); print("\nsummary ->", E / "summary_e.json")
