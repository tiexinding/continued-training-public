#!/usr/bin/env python3
"""正文中由数据文件推导或解析计算、此前未单独存档的数（10-16，文档 49 L3、L6；表 12 MG 一列见 predictors_mg.py）。每项写出数值与论文位置，
并对印刷值做断言（舍入到论文的位数）。输出 results/release/derived_numbers.json。

  §4.3 / 摘要      置换使高于随机的重合下降 down 45–48%、gate 52–60%、up 46–50%（三顺序均值，未舍入）            ← results/h/summary_h.json
  §5.3            N/O 保留的压低与增强比例（式 retained）                         ← results/e/diag25/e1/adam_offline_v3_{M,Ag}.json
  §5.4            余弦 0.992–0.997 对应相对向量误差 √(2−2cos) ≈ 0.08–0.13          ← 同上（rowcol 的 cos_to_orig）
  附录 B          keep t 数据项相对 reset all 的偏差校正因子（MG，t_A = 108）       解析
  附录 C          重置 t 的偏差校正因子（PG，t = 125）                               解析 + 训练调度 forklib.lr_at
  §8 讨论          行列因子取值数 5,760 对 4.36M（约 760 倍）；β₂^k（k = 108–125）0.88–0.90  解析
  附录 A          审计闭式更新与训练优化器在第 1 批上的最大相对偏差 1.5e-7          ← results/k/logs/k_smoke.log"""
import json, math, re
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / "DATA"; OUT = R / "release"; OUT.mkdir(parents=True, exist_ok=True)
B1, B2 = 0.9, 0.999; S = {}


def check(name, value, printed, nd):
    ok = all(round(v, nd) == p for v, p in zip(np.atleast_1d(value), np.atleast_1d(printed)))
    S[name] = dict(value=np.atleast_1d(value).tolist(), printed=np.atleast_1d(printed).tolist(), ok=bool(ok))
    assert ok, (name, value, printed)


# §4.3：三顺序均值（未舍入）上的重合下降
C = json.load(open(R / "h/summary_h.json"))["cells"]
mean = lambda sw, c, mat: float(np.mean([C[f"{sw}|{c}|{s}"][f"ov_{mat}"] for s in (17, 31, 47)]))
for mat, printed in (("down", [45, 48]), ("gate", [52, 60]), ("up", [46, 50])):
    drop = {sw: (mean(sw, "vt", mat) - mean(sw, "vperm", mat)) / (mean(sw, "vt", mat) - 0.05) for sw in ("MG", "GM", "PG")}
    S[f"overlap_drop_by_switch_{mat}"] = drop
    check(f"overlap_drop_range_pct_{mat}", [100 * min(drop.values()), 100 * max(drop.values())], printed, 0)

# §5.3：保留比例。N = 神经元轴因子（down 为列 colonly，gate/up 为行 rowonly），O = 另一轴
ret = {}
for st, new, old in (("M", "gsm", "mbpp"), ("Ag", "mbpp", "gsm")):
    V = json.load(open(R / f"e/diag25/e1/adam_offline_v3_{st}.json"))["variants"]
    for mat in ("down_proj", "gate_proj", "up_proj"):
        g = lambda v, t: V[v]["raw"]["median"][mat][f"perbatch_share_{t}_mean"]
        T, Cc = (g("orig", new), g("orig", old)), (g("const", new), g("const", old))
        for fac, v in (("N", "colonly" if mat == "down_proj" else "rowonly"), ("O", "rowonly" if mat == "down_proj" else "colonly")):
            X = (g(v, new), g(v, old))
            ret[f"{st}|{mat}|{fac}"] = dict(enhancement=(X[0] - Cc[0]) / (T[0] - Cc[0]), suppression=(Cc[1] - X[1]) / (Cc[1] - T[1]))
        ret[f"{st}|{mat}|cos_rc"] = V["rowcol"]["raw"]["median"][mat]["cos_to_orig"]
S["retention"] = ret
pick = lambda fac, key: [ret[f"{st}|{m}|{fac}"][key] for m in ("down_proj", "gate_proj", "up_proj") for st in ("M", "Ag")]
check("N_enhancement_pct_down_gate_up_x_MBPP_GSM8K", [100 * x for x in pick("N", "enhancement")], [83, 108, 52, 77, 68, 53], 0)
check("N_suppression_range_pct", [100 * min(pick("N", "suppression")), 100 * max(pick("N", "suppression"))], [96, 100], 0)
check("O_enhancement_range_pct", [100 * min(pick("O", "enhancement")), 100 * max(pick("O", "enhancement"))], [-1, 33], 0)
S["O_suppression_range_pct"] = [100 * min(pick("O", "suppression")), 100 * max(pick("O", "suppression"))]   # "essentially none"

# §5.4：余弦与相对向量误差（RMS 匹配后 ‖a−b‖/‖b‖ = √(2−2cos)）；down/gate/up × 两个起点
cos = [ret[f"{st}|{m}|cos_rc"] for st in ("M", "Ag") for m in ("down_proj", "gate_proj", "up_proj")]
check("rc_cos_range", [min(cos), max(cos)], [0.992, 0.997], 3)
check("rc_relerr_range", [math.sqrt(2 - 2 * max(cos)), math.sqrt(2 - 2 * min(cos))], [0.08, 0.13], 2)

# 附录 B：keep t 的偏差校正因子（t_A = 108，第 k 步）
f = [((1 - B1 ** k) / (1 - B1 ** (108 + k))) / math.sqrt((1 - B2 ** k) / (1 - B2 ** (108 + k))) for k in range(1, 126)]
S["keep_t_factor"] = dict(step1=f[0], max=max(f), argmax_step=1 + f.index(max(f)), step125=f[-1], mean=float(np.mean(f)))
check("keep_t_factor_step1_max_step125", [f[0], max(f), f[-1]], [1.0, 2.2, 1.3], 1)
check("keep_t_factor_mean", float(np.mean(f)), 1.67, 2)

# 附录 C：重置 t（t = 125 → 0）后数据项公共因子 η√(1−β₂ᵗ)/(1−β₁ᵗ) 相对 keep t 的比值；学习率加权用训练调度
r = [(math.sqrt(1 - B2 ** k) / (1 - B1 ** k)) / (math.sqrt(1 - B2 ** (125 + k)) / (1 - B1 ** (125 + k))) for k in range(1, 126)]
W = math.ceil(125 * 0.05); lr = [(k / W if k < W else 1.0) for k in range(125)]   # forklib.lr_at（k 为 0 起的步号）
S["t_reset_factor"] = dict(min=min(r), max=max(r), lr_weighted=sum(a * b for a, b in zip(lr, r)) / sum(lr))
check("t_reset_factor_range", [min(r), max(r)], [0.42, 0.92], 2)
check("t_reset_factor_lr_weighted", S["t_reset_factor"]["lr_weighted"], 0.6, 1)

# §8：参数计数与历史衰减系数
check("rowcol_count_ratio", (896 * 4864) / (896 + 4864), 760, -1)
check("beta2_k_108_125", [B2 ** 125, B2 ** 108], [0.88, 0.90], 2)

# 附录 A：闭式更新对训练优化器（smoke 运行，第 1 批，逐元素最大差 / 最大元素，取全部前馈矩阵的最大值）
m = re.search(r"smoke 公式核对 max rel ([0-9.eE+-]+)", open(R / "k/logs/k_smoke.log", encoding="utf-8").read())
check("audit_formula_maxrel_e7", float(m.group(1)) * 1e7, 1.5, 1)

# 附录 A 审计精度：主运行第 1 批 T 更新的 fp32 底噪（逐题 ‖Δy32−Δy64‖/‖Δy64‖）、各条件负 KL 计数、fp32/fp64 端到端试比
K = json.load(open(R / "k/k_M.json")); fl = [f / n for r in K["r3"] if r["floor32"] is not None for f, n in zip(r["floor32"], r["normT"])]
check("audit_fp32_floor_median_max_pct", [100 * float(np.median(fl)), 100 * max(fl)], [0.4, 2.3], 1)
S["audit_min_signal_over_floor"] = 1 / max(fl); assert 1 / max(fl) > 10
neg = {}
for r in K["r4"]:
    for c, v in r["conds"].items(): neg.setdefault(f"{r['task']}|{c}", []).append(int(sum(x < 0 for x in v["kl_T_X"])))
S["audit_negative_kl_counts"] = {k: [min(v), max(v)] for k, v in neg.items()}
check("audit_negative_kl_RC_OE_old", [min(neg["old|RC"]), max(neg["old|RC"]), min(neg["old|OE"]), max(neg["old|OE"])], [5, 12, 9, 19], 0)
assert all(max(v) == 0 for k, v in neg.items() if k not in ("old|RC", "old|OE"))
A32, A64 = (json.load(open(R / f"k/diag25/k/trial_M_f{b}.json"))["r4"] for b in (32, 64))
rel = {f"{x['task']}|{c}": abs(np.mean(x["conds"][c]["kl_T_X"]) - np.mean(y["conds"][c]["kl_T_X"])) / abs(np.mean(y["conds"][c]["kl_T_X"]))
       for x, y in zip(A32, A64) for c in x["conds"]}
S["audit_fp32_fp64_kl_rel"] = rel
assert max(rel[k] for k in rel if k.split("|")[1] in ("RC", "N")) < 0.04
check("audit_fp32_fp64_OE_old_pct", 100 * rel["old|OE"], 11, 0)

json.dump(S, open(OUT / "derived_numbers.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v.get("ok", "-") if isinstance(v, dict) else "-" for k, v in S.items()}, indent=1, ensure_ascii=False))
