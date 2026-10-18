#!/usr/bin/env python3
"""附录 B 表 12（tab:predictors）MG 一列（10-16，文档 49 L6）：以 keep v,t 终态 down 列改动能量 top-5% 为目标，
各排序的逐层命中率取 24 层中位（随机 0.05）。GM 一列见 regen/gen_predictor_baselines_dir2.py。
输入：results/b4/neuron_act_M.npz（MBPP 起点上的激活）、results/b2/fields/Kv_A_B_fields.npz（第二层数组）、
      results/b4/early_checkpoint_W4_4.json（Kv 第 10 步累计改动的命中率，第 8 项）。输出 results/release/predictors_MG.json。"""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / "DATA"; OUT = R / "release"; OUT.mkdir(parents=True, exist_ok=True); S = {}


def check(name, value, printed, nd):
    ok = all(round(v, nd) == p for v, p in zip(np.atleast_1d(value), np.atleast_1d(printed)))
    S[name] = dict(value=np.atleast_1d(value).tolist(), printed=np.atleast_1d(printed).tolist(), ok=bool(ok))
    assert ok, (name, value, printed)


# MG 一列（目标：keep v,t 终态 down 列能量 top-5%，逐层命中率的 24 层中位）
A = np.load(R / "b4/neuron_act_M.npz"); Z = np.load(R / "b2/fields/Kv_A_B_fields.npz"); k5 = int(round(0.05 * 4864))
top = lambda x: set(np.argsort(-x)[:k5])
def hit(pred):
    return float(np.median([len(top(pred(L)) & top(Z[f"model.layers.{L}.mlp.down_proj.weight|eW_col"])) / k5 for L in range(24)]))
mg = dict(selectivity=hit(lambda L: A[f"L{L}_sel_gsm_vs_mbpp"]), inverse_old_activation=hit(lambda L: -A[f"L{L}_mbpp_rms"]),
          new_activation=hit(lambda L: A[f"L{L}_gsm8k_rms"]),
          step10_update=json.load(open(R / "b4/early_checkpoint_W4_4.json"))["Kv"]["10"][7])
S["predictors_MG"] = mg
check("predictors_MG_sel_inv_step10", [mg["selectivity"], mg["inverse_old_activation"], mg["step10_update"]], [0.53, 0.12, 0.30], 2)

json.dump(S, open(OUT / "predictors_MG.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(S["predictors_MG"], indent=1))
