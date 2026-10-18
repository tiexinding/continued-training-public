#!/usr/bin/env python3
"""文档 28 队列回传后的汇总（本地）：读 results/c/{diag25/c1,c3,c4,c5,c6, runs_c/*/log.jsonl}，输出 results/c/summary_c.json 与控制台表。"""
import json, glob, math, os
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parents[1] / "DATA"; C = R / "c"
def kind(n):
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed"):
        if k in n: return k.replace("_proj", "")
def mcn(a, b):
    a = np.array(a, bool); b = np.array(b, bool); g = int(np.sum(a & ~b)); l = int(np.sum(~a & b)); n = g + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(g, l) + 1)) / 2 ** n) if n else 1.0; return g, l, round(p, 3)
S = {}
ARMS = ["K_mvt_rec", "K_vt_s31", "K_0_s31", "K_vperm", "K_vfac", "K_vconst", "S_style", "S_official"]
# 1 结构读数（新臂）+ 旧臂参照
print("== 1 结构读数（24 层中位）：gate 行 R² / up 行 R² / down 列 R² / cvZ/零(up) / ε(gate)")
ref = {"K_mvt": R / "t2/diag25/t2/lr5e-6/pairs/A_B.json", "K_∅": R / "t3/diag25/t3/pairs/A_B.json"}
for a in ARMS: ref[a] = C / f"diag25/c3/pairs_{a}/A_B.json"
S["structure"] = {}
for a, p in ref.items():
    if not p.exists(): print(a, "缺"); continue
    pm = json.load(open(p))["per_matrix"]; r = {}
    for kd in ("gate", "up", "down"):
        rs = [x for n, x in pm.items() if kind(n) == kd]
        r[kd] = dict(r2row=float(np.median([x["L3"]["eqstep_Zrow_r2fixed"] for x in rs])), r2col=float(np.median([x["L3"]["eqstep_Zcol_r2fixed"] for x in rs])), cv=float(np.median([x["L3"]["shuffle"]["cvZ_row"] / x["L3"]["shuffle"]["cvZ_row_null_Wshuffle"] for x in rs])), eps=float(np.median([x["L3"]["eps"] for x in rs])))
    S["structure"][a] = r; print(f"{a:11} {r['gate']['r2row']:.2f} / {r['up']['r2row']:.2f} / {r['down']['r2col']:.2f} / {r['up']['cv']:.2f} / {r['gate']['eps']:.4f}")
# 2 选择性重合（neuron_act_M_c.json）
p = C / "diag25/b4/neuron_act_M_c.json"
if p.exists():
    J = json.load(open(p))["overlap"]; S["overlap"] = {a: {k: v["median"] for k, v in J[a].items()} for a in J}
    print("\n== 2 down 列能量 top-5% ∩ GSM8K 选择性 / ∩ MBPP 选择性 / ρ(down, MBPP 激活)")
    for a in J: print(f"{a:11} {J[a]['down列原始能量∩GSM8K选择性']['median']:.3f} / {J[a]['down列原始能量∩MBPP选择性']['median']:.3f} / {J[a]['down列原始能量~MBPP激活 ρ']['median']:+.3f}")
# 3 功能：gen_audit（GSM8K 留出 200 relaxed，MBPP 100）
print("\n== 3 生成：GSM8K relaxed / MBPP pass@1 （对 M 配对得/失 p）")
M_g = [json.loads(l)["relaxed"] for l in open(R / "t1/lr1e-5/final_1e-5_max400.jsonl")]; M_m = json.load(open(R / "t1/lr1e-5/final_1e-5_mbpp.json"))["passed"]
S["gen"] = {}
for a in ARMS:
    f = C / f"diag25/c4/{a}_max400.jsonl"
    if not f.exists(): continue
    g = [json.loads(l)["relaxed"] for l in open(f)]; m = json.load(open(C / f"diag25/c4/{a}_mbpp.json"))["passed"]
    S["gen"][a] = dict(gsm8k=float(np.mean(g)), mbpp=float(np.mean(m)), vsM_gsm=mcn(g, M_g), vsM_mbpp=mcn(m, M_m)); print(f"{a:11} {np.mean(g):.3f} / {np.mean(m):.2f}   对 M {mcn(g, M_g)} / {mcn(m, M_m)}")
# 4 HumanEval / Dolly
for name, f in (("humaneval", C / "diag25/c4/humaneval_c.json"), ("dolly", C / "diag25/c4/dolly_gen.json")):
    if f.exists():
        J = json.load(open(f)); S[name] = {k: (v["pass_rate"] if "pass_rate" in v else v["mean_f"]) for k, v in J.items()}
        print(f"\n== 4 {name}:", {k: round(v, 3) for k, v in S[name].items()})
# 5 segnll（探针 NLL 总体）
print("\n== 5 探针 NLL（GSM8K 探针 / MBPP 探针 / Dolly 探针 / GSM8K 留出）")
S["nll"] = {}
for f in sorted(glob.glob(str(C / "diag25/c4/segnll/*.json"))):
    n = Path(f).stem; v = json.load(open(f))["nll"]; r = {ds: sum(v[ds]["s"]) / sum(v[ds]["c"]) for ds in v}; S["nll"][n] = r
    print(f"{n:11} {r['probe_gsm8k']:.3f} / {r['probe_mbpp']:.3f} / {r['probe_dolly_qa']:.3f} / {r['heldout_gsm8k']:.3f}")
# 6 干预 v2
p = C / "diag25/c5/intervene_Gkeep_v2.json"
if p.exists():
    J = json.load(open(p)); B = J["B"]; S["intervene_v2"] = {k: dict(gsm8k=v.get("gsm8k_acc"), mbpp=v.get("mbpp_acc"), nll=v["nll"], e=v.get("reverted_energy_frac")) for k, v in J.items() if k != "pairwise"}
    print("\n== 6 干预 v2（GSM8K acc / MBPP / MBPP NLL / 能量）")
    for k, v in J.items():
        if k == "pairwise": continue
        print(f"{k:28} {v.get('gsm8k_acc', float('nan')):.3f} / {v.get('mbpp_acc', float('nan')):.2f} / {v['nll']['probe_mbpp']:.4f} / {v.get('reverted_energy_frac', float('nan')):.3f}")
    rnd = [v["gsm8k_acc"] for k, v in J.items() if k.startswith("random") and "@" in k]; print("随机 5 种子 GSM8K acc：", rnd)
    S["intervene_v2_pairwise"] = J.get("pairwise", {})
# 7 mask
print("\n== 7 梯度 mask（GSM8K 批间余弦，gate/down 中位）")
S["mask"] = {}
for f in sorted(glob.glob(str(C / "diag25/c6/grad_mask_*.json"))):
    J = json.load(open(f)); n = Path(f).stem
    g = np.median([r["within_gsm8k"] for k, r in J.items() if "gate_proj" in k]); d = np.median([r["within_gsm8k"] for k, r in J.items() if "down_proj" in k]); S["mask"][n] = dict(gate=float(g), down=float(d)); print(f"{n:28} gate {g:.3f}  down {d:.3f}")
# 8 adam_offline
p = C / "diag25/c1/adam_offline.json"
if p.exists():
    J = json.load(open(p))["variants"]; S["adam_offline"] = {v: J[v]["median"] for v in J}
    print("\n== 8 离线 Adam（down 列份额 GSM8K 组 / MBPP 组 / top5∩GSM8K）")
    for v in J: m = J[v]["median"]["down_proj"]; print(f"{v:7} {m['share_gsm']:.3f} / {m['share_mbpp']:.3f} / {m['top5_in_gsm']:.3f}")
json.dump(S, open(C / "summary_c.json", "w"), ensure_ascii=False, indent=1); print("\nsummary ->", C / "summary_c.json")
