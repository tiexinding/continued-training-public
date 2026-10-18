#!/usr/bin/env python3
"""10-11 P5 §4.5 / eq.(9) 的机制检验（论文 v1.3 §5 用），固定起点、真实 v、真实梯度，GPU 几分钟。
P5 eq.(9)：log v_ij = μ + a_i + b_j + ε_ij（最小二乘：a、b 为行、列均值减总均值）。前馈层的"身份侧"（神经元轴）因子记 f_j：
  down 取列因子 b_j，gate/up 取行因子 a_j。
由 g = δ hᵀ：down 第 j 列的 g² = δ_i² h_j²，故 f_j(down) 预期 ≈ log E_旧[h_j²] + 常数（h = down 的输入 = 神经元激活）；
  gate/up 第 j 行的 g² = δ_j² x_i²，f_j 预期反映神经元 j 的反向误差能量（不是前向激活）。
读数（每层，24 层中位数）：
  R_a  Spearman(f_j, log RMS_旧²) 与 Spearman(f_j, log RMS_新²)（RMS 取自 neuron_act）。
  R_b  一步更新（与 connect_audit 同定义，注入一次 g²，不含衰减）的逐神经元能量 E_u,j，与
       (i) 梯度能量 E_g,j，(ii) E_g,j · e^{−f_j}（"梯度能量 ÷ 神经元因子"），(iii) 激活比 RMS_新²/RMS_旧² 的 Spearman。
  R_c  份额：E_u、E_g·e^{−f}、E_g 在新/旧任务选择性 5% 神经元上的份额（逐批平均），与 adam_offline 的 orig / const 对照。
用法：python vfactor_test.py --state <起点> --act <neuron_act.npz> --train <新任务 train.jsonl> --new_key gsm8k --old_key mbpp --out <json> [--nb 4]"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True); ap.add_argument("--act", required=True); ap.add_argument("--train", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--new_key", default="gsm8k"); ap.add_argument("--old_key", default="mbpp"); ap.add_argument("--sel_sign", type=float, default=1.0, help="sel_gsm_vs_mbpp 的符号：新任务为 GSM8K 时 +1，新任务为 MBPP 时 −1")
ap.add_argument("--nb", type=int, default=4); ap.add_argument("--fork_dir", default=_OH_TRAIN); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--deterministic", action="store_true")
a = ap.parse_args(); sys.path.insert(0, a.fork_dir)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
model, tok = load_model(a); named = fl.named_params(model); opt = fl.DecompAdamW(named, lr=5e-6, wd=0.01); fl.load_state(Path(a.state), model, opt); T0 = opt.t
A = np.load(a.act); FF = [(n, p) for n, p in named if p.dim() == 2 and ".mlp." in n]
kind = lambda n: "down" if "down_proj" in n else ("gate" if "gate_proj" in n else "up"); layer = lambda n: int(n.split(".")[2]); NAX = lambda n: 1 if kind(n) == "down" else 0
rows = fl.prompt_completion_rows(a.train, tok, a.max_len); exs = fl.tokenize_train_manual(rows, tok, a.max_len)
steps = fl.epoch_order(len(exs), 1, seed=17 + 99991)[: a.nb]
bc1, bc2 = 1 - fl.BETA1 ** (T0 + 1), 1 - fl.BETA2 ** (T0 + 1)
rk = lambda x: np.argsort(np.argsort(x)); sp = lambda x, y: float(np.corrcoef(rk(x), rk(y))[0, 1])
def top(x, f=.05): k = max(1, int(f * len(x))); return np.argsort(-x)[:k]
fac = {};
for n, _ in FF:
    L = torch.log(opt.state[n]["v"].double().clamp_min(fl.V_FLOOR)); mu = L.mean()
    fac[n] = ((L.mean(0) - mu) if NAX(n) == 1 else (L.mean(1) - mu)).cpu().numpy()
acc = {}
for b, st in enumerate(steps):
    model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [exs[i] for i in st]); opt.clip_(1.0)
    for n, p in FF:
        g = p.grad.detach().double(); v1 = fl.BETA2 * opt.state[n]["v"].double() + (1 - fl.BETA2) * g * g
        u = ((1 - fl.BETA1) * g / bc1) / ((v1 / bc2).sqrt() + fl.EPS)
        ax = 1 - NAX(n); Eu = u.pow(2).sum(ax).cpu().numpy(); Eg = g.pow(2).sum(ax).cpu().numpy()
        acc.setdefault(n, []).append((Eu, Eg))
model.zero_grad(set_to_none=True)
res = {"T_state": T0, "nb": a.nb, "per_kind": {}}
for k in ("down", "gate", "up"):
    R = {q: [] for q in ("f_vs_old_act", "f_vs_new_act", "Eu_vs_Eg", "Eu_vs_Eg_over_f", "Eu_vs_actratio", "share_new_Eu", "share_old_Eu", "share_new_pred", "share_old_pred", "share_new_Eg", "share_old_Eg")}
    for n, _ in FF:
        if kind(n) != k: continue
        l = layer(n); ro = A[f"L{l}_{a.old_key}_rms"] ** 2; rn = A[f"L{l}_{a.new_key}_rms"] ** 2; sel = a.sel_sign * A[f"L{l}_sel_gsm_vs_mbpp"]; G, M = top(sel), top(-sel)
        f = fac[n]; R["f_vs_old_act"].append(sp(f, np.log(ro + 1e-30))); R["f_vs_new_act"].append(sp(f, np.log(rn + 1e-30)))
        Eu = np.mean([x[0] for x in acc[n]], 0); Eg = np.mean([x[1] for x in acc[n]], 0); pred = Eg * np.exp(-f)
        R["Eu_vs_Eg"].append(sp(Eu, Eg)); R["Eu_vs_Eg_over_f"].append(sp(Eu, pred)); R["Eu_vs_actratio"].append(sp(Eu, rn / (ro + 1e-30)))
        for nm, E in (("Eu", None), ("pred", None), ("Eg", None)):
            sh = []
            for Eu_b, Eg_b in acc[n]:
                e = {"Eu": Eu_b, "pred": Eg_b * np.exp(-f), "Eg": Eg_b}[nm]; e = e / e.sum(); sh.append((e[G].sum(), e[M].sum()))
            R[f"share_new_{nm}"].append(float(np.mean([s[0] for s in sh]))); R[f"share_old_{nm}"].append(float(np.mean([s[1] for s in sh])))
    res["per_kind"][k] = {q: float(np.median(v)) for q, v in R.items()} | {"per_layer": {q: [float(x) for x in v] for q, v in R.items()}}
    m = res["per_kind"][k]; print(k, {q: round(m[q], 3) for q in R}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
