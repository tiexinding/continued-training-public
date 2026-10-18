#!/usr/bin/env python3
"""10-17：Fig. 2 组件对照在 gate / up 行上的对应读数（tier B）。
(1) 每臂、每层：gate 行与 up 行改动能量 top-5% 与新任务选择性 5% 的重合，及与旧任务激活的 Spearman ρ；同时重算 down 列作口径核对。
(2) keep v,t（Kv）与 keep t（bridge）逐步：单步更新能量在新/旧任务选择性 5% 上的份额（gate/up 行），v̂ 行 lnRMS 剖面与新/旧任务激活的 Spearman。
口径与 FS/regen/gen_neuron_overlap_sixarms.py、FS/recompute_first_steps.py 相同（24 层中位数；能量 = exp(2·lnRMS)）。
输入目录由环境变量 TIERB 指定（默认 ../DATA）。输出 b4/component_gateup.json。"""
import json, glob, os
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
R = Path(os.environ.get("TIERB", Path(__file__).resolve().parents[1] / "DATA"))
A = np.load(R / "b4/neuron_act_M.npz")
def top(x, f=.05): k = max(1, int(f * len(x))); return np.argsort(-x)[:k], k
rk = lambda x: np.argsort(np.argsort(x)); sp = lambda x, y: float(np.corrcoef(rk(x), rk(y))[0, 1])
fields = {'Gkeep': 't2/fields/A_B_fields.npz', 'Kv': 'b2/fields/Kv_A_B_fields.npz', 'bridge': 'b2/fields/bridge_A_B_fields.npz',
          'Gfresh': 't3/fields/A_B_fields.npz', 'Ckeep': 'b2/fields/Cfresh_A_contA_fields.npz'}
AX = {"gate": "eW_row", "up": "eW_row", "down": "eW_col"}
out = {"overlap": {}, "first_steps": {}}
for a, p in fields.items():
    F = np.load(R / p); res = {}
    for kind, key in AX.items():
        ov, rho = [], []
        for i in range(24):
            e = F[f"model.layers.{i}.mlp.{kind}_proj.weight|{key}"]; sel = A[f"L{i}_sel_gsm_vs_mbpp"]
            S, k = top(e); Tg, _ = top(sel)
            ov.append(len(set(S) & set(Tg)) / k); rho.append(sp(e, A[f"L{i}_mbpp_rms"]))
        res[kind] = dict(overlap=float(np.median(ov)), rho_old=float(np.median(rho)), overlap_per_layer=ov)
    out["overlap"][a] = res
    print(a, {k: (round(v["overlap"], 3), round(v["rho_old"], 3)) for k, v in res.items()})
for run in ("Kv", "bridge"):
    out["first_steps"][run] = {}
    for f in sorted(glob.glob(str(R / f"b2/records/runs_b2/{run}/records/fields_*.pt"))):
        step = int(Path(f).stem.split("_")[-1])
        try: d = torch.load(f, map_location="cpu", weights_only=False)["fields"]
        except Exception as ex: print("skip", f, type(ex).__name__); continue
        rec = {}
        for kind in ("gate", "up", "down"):
            ax = 1 if kind == "down" else 0; sh = []; ro, rn = [], []
            for l in range(24):
                nm = f"model.layers.{l}.mlp.{kind}_proj.weight"; sel = A[f"L{l}_sel_gsm_vs_mbpp"]; G, _ = top(sel); M, _ = top(-sel)
                c = np.asarray(d[nm]["u"][ax], dtype=np.float64); e = np.exp(2 * (c - c.max())); e /= e.sum(); sh.append((e[G].sum(), e[M].sum()))
                vh = np.asarray(d[nm]["vhat"][ax], dtype=np.float64)
                ro.append(sp(vh, np.log(A[f"L{l}_mbpp_rms"] ** 2 + 1e-30))); rn.append(sp(vh, np.log(A[f"L{l}_gsm8k_rms"] ** 2 + 1e-30)))
            sh = np.array(sh)
            rec[kind] = dict(u_new=float(np.median(sh[:, 0])), u_old=float(np.median(sh[:, 1])), vhat_rho_old=float(np.median(ro)), vhat_rho_new=float(np.median(rn)))
        out["first_steps"][run][str(step)] = rec
    print(run, {s: {k: (round(100 * v[k]["u_new"], 1), round(100 * v[k]["u_old"], 1), round(v[k]["vhat_rho_old"], 2), round(v[k]["vhat_rho_new"], 2)) for k in v} for s, v in out["first_steps"][run].items()})
json.dump(out, open(R / "b4/component_gateup.json", "w"), indent=1)
