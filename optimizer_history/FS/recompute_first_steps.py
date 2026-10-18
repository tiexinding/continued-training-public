#!/usr/bin/env python3
"""10-11 更正：逐步记录 fields_*.pt 存的是每行/每列的 ln RMS（forklib.lnrms_fields），能量 = exp(2·lnRMS)。
旧文件 b4/first_steps_groups.json 用了 exp(lnRMS)（RMS 份额，偏平），这里按能量重算：
  u（单步更新）、g（梯度）、dW（累计改动）的 down 列能量落在新/旧任务选择性 5% 神经元上的份额（24 层中位数）；
  v̂：每神经元 lnRMS 剖面与旧/新任务激活（log RMS²）的 Spearman（24 层中位数）。
条件：Kv = keep v,t；bridge = keep t（MG，顺序 17）。输出 b4/first_steps_groups_v2.json。
10-18：改读 b2/records/runs_b2/（keep t 完整记录到第 125 步；旧路径 runs_b2/bridge/ 第 25 步为传输残件，之后无文件）。"""
import json, glob
from pathlib import Path
import numpy as np, torch
R = Path(__file__).resolve().parents[1] / "DATA"; A = np.load(R / "b4/neuron_act_M.npz")
def top(x, f=.05): k = max(1, int(f * len(x))); return np.argsort(-x)[:k]
rk = lambda x: np.argsort(np.argsort(x)); sp = lambda x, y: float(np.corrcoef(rk(x), rk(y))[0, 1])
out = {}
for run in ("Kv", "bridge"):
    files = sorted(glob.glob(str(R / f"b2/records/runs_b2/{run}/records/fields_*.pt")))
    out[run] = {}
    for f in files:
        step = int(Path(f).stem.split("_")[-1]); rec = {}
        try: d = torch.load(f, map_location="cpu", weights_only=False)["fields"]
        except Exception as ex: print("skip damaged", f, type(ex).__name__); continue
        for q in ("u", "g", "dW"):
            sh = []
            for l in range(24):
                c = np.asarray(d[f"model.layers.{l}.mlp.down_proj.weight"][q][1], dtype=np.float64); e = np.exp(2 * (c - c.max())); e /= e.sum()
                sel = A[f"L{l}_sel_gsm_vs_mbpp"]; G, M = top(sel), top(-sel); sh.append((e[G].sum(), e[M].sum()))
            sh = np.array(sh); rec[q] = {"new": float(np.median(sh[:, 0])), "old": float(np.median(sh[:, 1]))}
        r_old, r_new = [], []
        for l in range(24):
            c = np.asarray(d[f"model.layers.{l}.mlp.down_proj.weight"]["vhat"][1], dtype=np.float64)
            r_old.append(sp(c, np.log(A[f"L{l}_mbpp_rms"] ** 2 + 1e-30))); r_new.append(sp(c, np.log(A[f"L{l}_gsm8k_rms"] ** 2 + 1e-30)))
        rec["vhat_rho_old_act"] = float(np.median(r_old)); rec["vhat_rho_new_act"] = float(np.median(r_new))
        out[run][str(step)] = rec
    print(run, {s: (round(v["u"]["new"] * 100, 1), round(v["u"]["old"] * 100, 1), round(v["dW"]["new"] * 100, 1), round(v["dW"]["old"] * 100, 1), round(v["vhat_rho_old_act"], 2), round(v["vhat_rho_new_act"], 2)) for s, v in out[run].items()})
json.dump(out, open(R / "b4/first_steps_groups_v2.json", "w"), indent=1)
