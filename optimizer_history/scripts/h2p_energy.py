#!/usr/bin/env python3
"""H2′（文档 41 §8，评审第 3 点）：逐步更新能量的累计排名 vs 净变化排名（down 列，24 层）。
对每条带 records/upd_energy.pt 的新跑：
  cum_j = Σ_k ‖ΔW_k[:, j]‖²（每步实际更新，含权重衰减）；net_j = ‖W_end[:, j] − W_start[:, j]‖²（成对分析 A_B_fields 的 eW_col）。
读数（24 层中位）：ρ(cum, net)、两者 top-5% 重合、抵消比 Σnet/Σcum、前 10 步占累计能量的份额；
以及用 cum 代替 net 时的论文读数——top-5% ∩ 新任务选择性 top-5%、ρ(·, 旧任务激活)——与 net 版并列。
激活图取 neuron_act 的 npz（同一起点）。用法：python h2p_energy.py --runs MG_vt_s47=... 见 run_h3.sh。"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True, help="name=sw，sw ∈ MG/GM/PG")
ap.add_argument("--act", nargs="+", required=True, help="sw=neuron_act npz")
ap.add_argument("--runs_dir", default=_OH_RUNS + "/runs_h"); ap.add_argument("--pairs_dir", default=_OH_RUNS + "/diag25/h")
ap.add_argument("--out", required=True)
ap.add_argument("--segments", action="store_true", help="10-10 I3：另按步段（全部 / 去掉前 10 步 / 1–10 / 11–25 / 26–终点）与阈值 1/5/10% 重算")
a = ap.parse_args()
ACT = {k: np.load(v) for k, v in (s.split("=", 1) for s in a.act)}
# 新 / 旧任务（neuron_act：MG、GM 用默认 sel=gsm8k,mbpp；PG 用 sel=gsm8k,bio）
NEWOLD = {"MG": ("gsm8k", "mbpp"), "GM": ("mbpp", "gsm8k"), "PG": ("gsm8k", "bio")}


def rk(x): return np.argsort(np.argsort(x))
def rho(x, y): return float(np.corrcoef(rk(x), rk(y))[0, 1])
def top(x, f=0.05): k = max(1, int(f * len(x))); return set(np.argsort(-x)[:k].tolist()), k


res = {}
for spec in a.runs:
    n, sw = spec.split("=")
    ue_p = Path(a.runs_dir) / n / "records/upd_energy.pt"; f_p = Path(a.pairs_dir) / f"pairs_{n}" / "A_B_fields.npz"
    if not (ue_p.exists() and f_p.exists()): print("缺", n); continue
    U = torch.load(ue_p, map_location="cpu"); E = U["down_col"].double().numpy()   # (steps, L, H)
    F = np.load(f_p); A = ACT[sw]; new, old = NEWOLD[sw]; L = E.shape[1]
    rows = []
    for i in range(L):
        cum = E[:, i].sum(0); net = F[f"model.layers.{i}.mlp.down_proj.weight|eW_col"].astype(np.float64)
        sel = A[f"L{i}_sel_gsm_vs_mbpp"] if sw != "PG" else A[f"L{i}_sel_gsm8k_vs_bio"]
        sel_new = sel if new == "gsm8k" else -sel; act_old = A[f"L{i}_{old}_rms"]
        Sn, k = top(net); Sc, _ = top(cum); Ts, _ = top(sel_new)
        rows.append(dict(rho_cum_net=rho(cum, net), top5_cum_net=len(Sn & Sc) / k, cancel=float(net.sum() / cum.sum()),
                         early10=float(E[:10, i].sum() / cum.sum()),
                         ov_new_net=len(Sn & Ts) / k, ov_new_cum=len(Sc & Ts) / k, rho_old_net=rho(net, act_old), rho_old_cum=rho(cum, act_old)))
    res[n] = dict(sw=sw, steps=int(E.shape[0]), median={q: float(np.median([r[q] for r in rows])) for q in rows[0]}, per_layer=rows)
    if a.segments:
        S_ = E.shape[0]; SEG = {"all": (0, S_), "excl1_10": (10, S_), "s1_10": (0, 10), "s11_25": (10, 25), "s26_end": (25, S_)}
        seg = {}
        for sname, (s0, s1) in SEG.items():
            for f in (0.01, 0.05, 0.10):
                lr_ = []
                for i in range(L):
                    cum = E[s0:s1, i].sum(0); net = F[f"model.layers.{i}.mlp.down_proj.weight|eW_col"].astype(np.float64)
                    sel = A[f"L{i}_sel_gsm_vs_mbpp"] if sw != "PG" else A[f"L{i}_sel_gsm8k_vs_bio"]
                    sel_new = sel if new == "gsm8k" else -sel; act_old = A[f"L{i}_{old}_rms"]
                    Sn, k = top(net, f); Sc, _ = top(cum, f); Ts, _ = top(sel_new, f)
                    lr_.append(dict(rho_cum_net=rho(cum, net), top_cum_net=len(Sn & Sc) / k, ov_new_net=len(Sn & Ts) / k, ov_new_cum=len(Sc & Ts) / k,
                                    rho_old_cum=rho(cum, act_old), share=float(cum.sum() / E[:, i].sum())))
                seg[f"{sname}|{int(f * 100)}pct"] = {q: float(np.median([r[q] for r in lr_])) for q in lr_[0]}
        res[n]["segments"] = seg
        print(f"  {n} 去掉前10步: ρ(cum,net) {seg['excl1_10|5pct']['rho_cum_net']:.3f} 重合 cum {seg['excl1_10|5pct']['ov_new_cum']:.3f} | 1–10 步 ρ {seg['s1_10|5pct']['rho_cum_net']:.3f}", flush=True)
    m = res[n]["median"]
    print(f"{n:16} ρ(cum,net) {m['rho_cum_net']:.3f}  top5 {m['top5_cum_net']:.3f}  抵消比 {m['cancel']:.3f}  前10步 {m['early10']:.3f} | 重合 net/cum {m['ov_new_net']:.3f}/{m['ov_new_cum']:.3f}  ρ旧 net/cum {m['rho_old_net']:+.3f}/{m['rho_old_cum']:+.3f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
