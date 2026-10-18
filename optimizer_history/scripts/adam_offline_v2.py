#!/usr/bin/env python3
"""E2 修正版（10-07，按报告 29 审核 P1-1/P1-2）：固定权重、固定一批梯度、固定 m（= 当前梯度）与 t，离线重算不同 v 下的 AdamW 单步更新。
修正：(1) 置换 v 每个种子只生成一次、跨批固定（"每批重采样"单列为 perm_resample 消融）；(2) 同时输出三种能量口径：逐批能量份额（均值与分布）、
总平方能量份额 Σ_b‖u_b‖²、合成向量能量份额 ‖Σ_b u_b‖²；(3) 加只置换行 rperm / 只置换列 cperm_col；(4) 对每个变体输出与 orig 的 cos(u,u')、‖u−u'‖/‖u‖、
通道能量秩相关（匹配 RMS 前与后各一套）；(5) 常数记为"逐矩阵常数"（几何均值）。
用法同 adam_offline.py，另加 --perm_seeds 3。"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--state", required=True); ap.add_argument("--act", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--nb", type=int, default=8); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--train", default=_OH_SUBSETS + "/gsm8k/train.jsonl")
ap.add_argument("--variants", default="orig,rowcol,cperm,rperm,cperm_col,perm,const,fresh,perm_resample"); ap.add_argument("--perm_seeds", type=int, default=3); ap.add_argument("--nlayers", type=int, default=24)
a = ap.parse_args(); A = np.load(a.act)
model, tok = load_model(a); named = fl.named_params(model); opt = fl.DecompAdamW(named, lr=5e-6, wd=0.01)
fl.load_state(Path(a.state), model, opt); t_old = opt.t; print("state t =", t_old)
rows = fl.prompt_completion_rows(a.train, tok, a.max_len); exs = fl.tokenize_train_manual(rows, tok, a.max_len)
steps = fl.epoch_order(len(exs), 1, seed=17 + 99991)[: a.nb]
W = [(n, p) for n, p in named if p.dim() == 2 and ".mlp." in n]; vT = {n: opt.state[n]["v"].detach().float().cpu().clone() for n, _ in W}   # 全部搬到 CPU，避免 8 批梯度 + 更新占满显存
grads = []
for st in steps:
    model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [exs[i] for i in st]); opt.clip_(1.0); grads.append({n: p.grad.detach().float().cpu().clone() for n, p in W})
model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.set_num_threads(8)
def topset(x, frac=0.05): k = max(1, int(frac * len(x))); return set(np.argsort(-x)[:k].tolist()), k
rk = lambda x: np.argsort(np.argsort(x)); sp = lambda x, y: float(np.corrcoef(rk(x), rk(y))[0, 1])
def make_v(ver, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if ver in ("orig", "perm_resample"): return {n: vT[n].clone() for n, _ in W}
    if ver == "fresh": return {n: torch.zeros_like(vT[n]) for n, _ in W}
    return {n: fl.v_ladder(vT[n], ver, gen=gen) for n, _ in W}
def updates(ver, V, seed):
    """返回逐批更新列表 u_b（dict）。perm_resample：每批重新置换。"""
    t = 1 if ver == "fresh" else t_old + 1; bc2 = 1 - fl.BETA2 ** t; out = []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for g in grads:
        Vb = {n: fl.v_ladder(vT[n], "cperm", gen=gen) for n, _ in W} if ver == "perm_resample" else V
        ub = {}
        for n, _ in W:
            vh = (fl.BETA2 * Vb[n] + (1 - fl.BETA2) * g[n] ** 2) / bc2 if ver != "fresh" else (g[n] ** 2 * (1 - fl.BETA2)) / bc2
            ub[n] = g[n] / (vh.sqrt() + fl.EPS)
        out.append(ub)
    return out
def energies(u, n, dim): e = (u.double() ** 2).sum(1 - dim) if dim == 0 else (u.double() ** 2).sum(0); return e.cpu().numpy()
KINDS = (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1))
res = {"t_old": t_old, "nb": a.nb, "variants": {}}
U_orig = updates("orig", make_v("orig", 0), 0)
def analyze(ver, Ub, matched):
    """Ub：逐批更新 dict 列表；matched：是否逐矩阵把每批 u 的 RMS 对齐到 orig 对应批。"""
    if matched:
        Ub = [{n: u[n] * (U_orig[b][n].pow(2).mean().sqrt() / u[n].pow(2).mean().sqrt().clamp_min(1e-30)) for n in u} for b, u in enumerate(Ub)]
    per = []
    for i in range(a.nlayers):
        sel = A[f"L{i}_sel_gsm_vs_mbpp"]; G, k = topset(sel); Mb, _ = topset(-sel); r = {}
        for kind, dim in KINDS:
            n = f"model.layers.{i}.mlp.{kind}.weight"
            Eb = [energies(u[n], n, dim) for u in Ub]; Eb_n = [e / e.sum() for e in Eb]
            Esum = np.sum(Eb, 0); Esum = Esum / Esum.sum()
            Ucomb = sum(u[n] for u in Ub); Ecomb = energies(Ucomb, n, dim); Ecomb = Ecomb / Ecomb.sum()
            Eo = sum(energies(u[n], n, dim) for u in U_orig); Eo = Eo / Eo.sum()
            cos = float(np.mean([float((u[n] * uo[n]).sum() / (u[n].norm() * uo[n].norm() + 1e-30)) for u, uo in zip(Ub, U_orig)]))
            rel = float(np.mean([float((u[n] - uo[n]).norm() / (uo[n].norm() + 1e-30)) for u, uo in zip(Ub, U_orig)]))
            T, _ = topset(Esum)
            r[kind] = dict(perbatch_share_gsm_mean=float(np.mean([e[list(G)].sum() for e in Eb_n])), perbatch_share_gsm_sd=float(np.std([e[list(G)].sum() for e in Eb_n])),
                           perbatch_share_mbpp_mean=float(np.mean([e[list(Mb)].sum() for e in Eb_n])), perbatch_share_mbpp_sd=float(np.std([e[list(Mb)].sum() for e in Eb_n])),
                           sumsq_share_gsm=float(Esum[list(G)].sum()), sumsq_share_mbpp=float(Esum[list(Mb)].sum()), comb_share_gsm=float(Ecomb[list(G)].sum()), comb_share_mbpp=float(Ecomb[list(Mb)].sum()),
                           top5_in_gsm=len(T & G) / k, top5_in_mbpp=len(T & Mb) / k, cos_to_orig=cos, rel_err_to_orig=rel, chan_rank_corr_to_orig=sp(Esum, Eo), rms_over_orig=float(np.mean([float(u[n].pow(2).mean().sqrt() / U_orig[b][n].pow(2).mean().sqrt()) for b, u in enumerate(Ub)])))
        per.append(r)
    med = {kind: {q: float(np.median([r[kind][q] for r in per])) for q in per[0][kind]} for kind in per[0]}
    return dict(median=med, per_layer=per)
for ver in a.variants.split(","):
    seeds = range(a.perm_seeds) if ver in ("cperm", "rperm", "cperm_col", "perm", "perm_resample") else [0]
    for sd in seeds:
        V = make_v(ver, sd); Ub = updates(ver, V, sd); key = ver if len(list(seeds)) == 1 else f"{ver}#s{sd}"
        res["variants"][key] = dict(raw=analyze(ver, Ub, False), matched=analyze(ver, Ub, True))
        m = res["variants"][key]["matched"]["median"]["down_proj"]
        print(f"{key:16} down(匹配后) 逐批份额 GSM {m['perbatch_share_gsm_mean']:.3f}±{m['perbatch_share_gsm_sd']:.3f} / MBPP {m['perbatch_share_mbpp_mean']:.3f}  总平方 {m['sumsq_share_gsm']:.3f}/{m['sumsq_share_mbpp']:.3f}  合成 {m['comb_share_gsm']:.3f}/{m['comb_share_mbpp']:.3f}  cos {m['cos_to_orig']:.3f} 相对误差 {m['rel_err_to_orig']:.3f} 秩ρ {m['chan_rank_corr_to_orig']:.3f} 原始RMS比 {res['variants'][key]['raw']['median']['down_proj']['rms_over_orig']:.3f}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1); print("done")
