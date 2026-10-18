#!/usr/bin/env python3
"""K 轮判读（文档 47 v2 第 5 节，规则写于开卡前）。读 connect_audit.py 的 JSON，输出 markdown 表与 JSON 摘要。
R1：各条件 down/gate/up 的"新任务 / 旧任务"5% 神经元份额（逐批均值，层中位数），与文档 36 对照。
R2：各条件与 T 的方向余弦、相对误差（层中位数，逐批）。
R3：每批、每任务：层中位数（每层先取题目中位数）的相对误差 e_X；成对差 e_N−e_O、e_N−e_C；方向一致批数；
    可靠性：‖Δy_T‖ < 10 × fp32 底噪 的（层 × 题）比例，按批标出。
R4：每批、每任务：KL(p_T‖p_X) 的题目均值；KL(p_W‖p_T) 作尺度参照；KL 只在 N/O/C 间比较。
Q2：OE、CE、RE 的 R3/R4 误差（描述，不报信息比例）。
用法：python analyze_k.py --in <k_M.json> --out <前缀>"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
R = json.load(open(a.inp)); cfg = R["config"]; nb = cfg["nb"]; S = {}; L = [f"# K 轮结果：{a.inp}", ""]
med = lambda x: float(np.median(x))

# R1 / R2
L += ["## R1 份额（%，逐批均值后取层中位数）与 R2 方向", "", "| 条件 | 尺度 | 矩阵 | 新任务份额 | 旧任务份额 | cos 对 T | 相对误差 |", "|---|---|---|---|---|---|---|"]
S["r1"] = {}
for key in sorted(R["r1"]):
    c, sc = key.split("|")
    for k in ("down", "gate", "up"):
        per = R["r1"][key][k]
        sn = med([np.mean([d["share_new"] for d in per[l]]) for l in per]); so = med([np.mean([d["share_old"] for d in per[l]]) for l in per])
        cs = med([np.mean([d["cos"] for d in per[l]]) for l in per]); rl = med([np.mean([d["rel"] for d in per[l]]) for l in per])
        S["r1"][f"{key}|{k}"] = dict(new=sn, old=so, cos=cs, rel=rl)
        L.append(f"| {c} | {sc} | {k} | {100 * sn:.1f} | {100 * so:.1f} | {cs:.3f} | {rl:.3f} |")

# R3
def r3_table():
    rows = R["r3"]; conds = list(rows[0]["conds"]); out = {}
    for t in ("new", "old"):
        for b in range(nb):
            rs = [r for r in rows if r["b"] == b and r["task"] == t]
            if not rs: continue
            rel_ok = [np.array(r["normT"]) >= 10 * np.array(r["floor32"]) for r in rs if r.get("floor32") is not None] or [np.array([True])]
            unrel = 1 - float(np.mean(np.concatenate(rel_ok)))
            e = {c: med([med(r["conds"][c]["rel"]) for r in rs]) for c in conds}
            cs = {c: med([med(r["conds"][c]["cos"]) for r in rs]) for c in conds}
            out[(t, b)] = dict(e=e, cos=cs, unreliable_frac=unrel, normT_med=med([med(r["normT"]) for r in rs]))
    return out, conds
if R["r3"]:
    T3, conds = r3_table(); S["r3"] = {f"{t}|{b}": v for (t, b), v in T3.items()}
    L += ["", "## R3 局部响应：相对误差 e_X（层中位数；每层先取题目中位数），逐批", ""]
    for t in ("new", "old"):
        L += [f"### 任务 {t}", "", "| 批 | " + " | ".join(conds) + " | e_N−e_O | e_N−e_C | 不可靠比例 |", "|" + "---|" * (len(conds) + 4)]
        dNO, dNC = [], []
        for b in range(nb):
            if (t, b) not in T3: continue
            v = T3[(t, b)]; dNO.append(v["e"]["N"] - v["e"]["O"]); dNC.append(v["e"]["N"] - v["e"]["C"])
            L.append(f"| {b} | " + " | ".join(f"{v['e'][c]:.3f}" for c in conds) + f" | {dNO[-1]:+.3f} | {dNC[-1]:+.3f} | {v['unreliable_frac']:.2f} |")
        S[f"r3_pair_{t}"] = dict(N_minus_O=dict(median=med(dNO), range=[min(dNO), max(dNO)], n_neg=int(sum(x < 0 for x in dNO)), n=len(dNO)),
                                 N_minus_C=dict(median=med(dNC), range=[min(dNC), max(dNC)], n_neg=int(sum(x < 0 for x in dNC)), n=len(dNC)))
        L += ["", f"成对差：e_N−e_O 中位 {med(dNO):+.3f}（范围 {min(dNO):+.3f} 至 {max(dNO):+.3f}，N 更好的批 {sum(x < 0 for x in dNO)}/{len(dNO)}）；"
                  f"e_N−e_C 中位 {med(dNC):+.3f}（N 更好 {sum(x < 0 for x in dNC)}/{len(dNC)}）", ""]

# R4
if R["r4"]:
    L += ["## R4 全模型：KL(p_T‖p_X) 题目均值（×1e-6），逐批；KL(p_W‖p_T) 为真实一步变化的尺度", ""]
    conds4 = list(R["r4"][0]["conds"])
    for t in ("new", "old"):
        rs = [r for r in R["r4"] if r["task"] == t]
        L += [f"### 任务 {t}", "", "| 批 | KL(W‖T) | " + " | ".join(conds4) + " |", "|" + "---|" * (len(conds4) + 2)]
        for r in rs:
            L.append(f"| {r['b']} | {1e6 * np.mean(r['kl_W_T']):.3f} | " + " | ".join(f"{1e6 * np.mean(r['conds'][c]['kl_T_X']):.3f}" for c in conds4) + " |")
        S[f"r4_{t}"] = {c: med([np.mean(r["conds"][c]["kl_T_X"]) for r in rs]) for c in conds4} | {"kl_W_T": med([np.mean(r["kl_W_T"]) for r in rs])}
        dNO = [np.mean(r["conds"]["N"]["kl_T_X"]) - np.mean(r["conds"]["O"]["kl_T_X"]) for r in rs]
        dNC = [np.mean(r["conds"]["N"]["kl_T_X"]) - np.mean(r["conds"]["C"]["kl_T_X"]) for r in rs]
        S[f"r4_pair_{t}"] = dict(N_minus_O_nneg=int(sum(x < 0 for x in dNO)), N_minus_C_nneg=int(sum(x < 0 for x in dNC)), n=len(rs))
        L += ["", f"KL_N < KL_O 的批 {sum(x < 0 for x in dNO)}/{len(rs)}；KL_N < KL_C 的批 {sum(x < 0 for x in dNC)}/{len(rs)}", ""]

# 判读标签（第 5 节）
def label():
    if "r3_pair_new" not in S: return "无 R3"
    p = [S[f"r3_pair_{t}"] for t in ("new", "old")]
    unrel = med([v["unreliable_frac"] for v in S["r3"].values()])
    if unrel > 0.5: return "T 的变化多处在数值底噪附近：此尺度无法分辨"
    allN = all(q["N_minus_O"]["n_neg"] == q["N_minus_O"]["n"] and q["N_minus_C"]["n_neg"] == q["N_minus_C"]["n"] for q in p)
    allO = all(q["N_minus_O"]["n_neg"] == 0 for q in p)
    if allN: return "N 在两任务、全部批次上都优于 O 与 C：神经元轴近似的保留能力延伸到所测局部输出扰动"
    if allO: return "O 在全部批次上优于 N：优势依赖评价对象（份额 vs 局部输出）"
    rc_best = all(S["r3"][f"{t}|{b}"]["e"]["RC"] < min(S["r3"][f"{t}|{b}"]["e"]["N"], S["r3"][f"{t}|{b}"]["e"]["O"]) for t in ("new", "old") for b in range(nb) if f"{t}|{b}" in S["r3"])
    if rc_best: return "N 与 O 都不如 RC：两轴联合结构对所测输出更有用"
    return "批次或任务间不一致：局部的条件依赖，如实分层报告"
S["label"] = label(); S["checks"] = R["checks"]; S["runtime"] = R.get("runtime")
L += ["## 判读（按事先表格）", "", S["label"], "", f"检查：{R['checks']}", f"运行时间：{R.get('runtime')}"]
open(a.out + ".md", "w").write("\n".join(L) + "\n"); json.dump(S, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
print("\n".join(L[-6:]))
