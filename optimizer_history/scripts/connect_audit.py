#!/usr/bin/env python3
"""K 轮连接审计（文档 47 v2，10-11）：固定起点、固定梯度批次、固定学习率的一步反事实。
一步更新（第 2 节）：m1=(1-β1)g；v1_X=β2·v_X+(1-β2)g²；d_X=[m1/(1-β1^(T+1))]/[sqrt(v1_X/(1-β2^(T+1)))+ε]；δW_X=-η·d_X，η=5e-6，不含衰减。
近似/置换只作用于历史 v_X，然后统一注入一次 g²（single 口径）。只改前馈矩阵（gate/up/down）；其余参数保持原权重 W。
条件：T 真 v；N 神经元轴近似（down=col，gate/up=row）；O 另一轴近似；RC rowcol；C const；
      PN 神经元身份置换（同层 gate/up 行与 down 列共享同一排列，种子 1000+层号）；
      OE/CE：O/C 的更新逐神经元能量匹配到 T（每个矩阵分别）；RE<k>：随机方向逐神经元能量匹配（生成器种子 k）。
匹配：主口径逐矩阵逐批把 d_X 的 RMS 对齐到 d_T（OE/CE/RE 已逐神经元匹配能量，不再缩放）；R1/R2 另报原始尺度。
读数：R1 逐批份额（同 adam_offline_v3：act 键 L{l}_sel_gsm_vs_mbpp 的正/负 top-5%，逐批归一后求和）；R2 逐矩阵 cos、相对误差；
      R3 局部响应：每层固定基线 W 下实际送入 MLP 的 h_l，fp64 计算 Δy_X = MLP(W+δW_X; h) − MLP(W; h)，逐题范数、相对误差、cos，
         按任务分开、逐层保存；数值底噪 = 同一 Δy_T 用 fp32 计算与 fp64 之差；
      R4 全模型：全部前馈 δW 一起施加，其余参数为 W；每题 KL(p_T‖p_X)，并报 KL(p_W‖p_T)、KL(p_T‖p_W) 与答案 NLL。
用法（卡上）：python connect_audit.py --state <起点> --act <neuron_act.npz> --train <gsm8k train.jsonl> \
          --probe_new <gsm8k probe> --probe_old <mbpp 或 bio probe> --out <json> [--nb 8 --n_eval 32 --r4_dtype float32|float64 --smoke]"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--state", required=True); ap.add_argument("--act", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--train", required=True); ap.add_argument("--probe_new", required=True); ap.add_argument("--probe_old", required=True)
ap.add_argument("--nb", type=int, default=8); ap.add_argument("--n_eval", type=int, default=32); ap.add_argument("--eval_bs", type=int, default=8)
ap.add_argument("--max_tok", type=int, default=128, help="每题最多取的 completion 位置数（R3/R4 同一组位置）")
ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--rand_seeds", type=int, default=3)
ap.add_argument("--r3", type=int, default=1); ap.add_argument("--r4", type=int, default=1); ap.add_argument("--r4_dtype", default="float32")
ap.add_argument("--r3_dtype", default="float64", help="R3 计算精度；float32 时只在第 1 批用 fp64 算 T 的底噪（10-11：试跑 fp32 底噪中位 0.2%、最大 0.9%）")
ap.add_argument("--raw_r34", action="store_true", help="另算 N/O/C 原始尺度的 R3/R4（预算允许时）")
ap.add_argument("--fork_dir", default=_OH_TRAIN); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--tok_mode", default="manual")
ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.fork_dir)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
if a.deterministic:
    torch.use_deterministic_algorithms(True, warn_only=True); torch.backends.cudnn.benchmark = False
T_START = time.time(); RT = {}
def lap(k): RT[k] = round(time.time() - T_START, 1); print(f"[{RT[k]:8.1f}s] {k}", flush=True)
def sha(t): return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

# ---------------------------------------------------------------- 加载
model, tok = load_model(a); named = fl.named_params(model); opt = fl.DecompAdamW(named, lr=a.lr, wd=0.01)
fl.load_state(Path(a.state), model, opt); T0 = opt.t; dev = next(model.parameters()).device
FF = [(n, p) for n, p in named if p.dim() == 2 and ".mlp." in n]
NONFF = [(n, p) for n, p in named if not (p.dim() == 2 and ".mlp." in n)]
def kind(n): return "down" if "down_proj" in n else ("gate" if "gate_proj" in n else "up")
def layer(n): return int(n.split(".")[2])
def NAX(n): return 1 if kind(n) == "down" else 0          # 神经元轴：down 的列（dim 1）、gate/up 的行（dim 0）
def W(l, k): return f"model.layers.{l}.mlp.{k}_proj.weight"
nonff_hash0 = {n: sha(p) for n, p in NONFF}
base32 = {n: p.detach().clone() for n, p in FF}
vT = {n: opt.state[n]["v"].detach().float().cpu().clone() for n, _ in FF}
nL = 1 + max(layer(n) for n, _ in FF); I = base32[W(0, "down")].shape[1]
A = np.load(a.act)
lap("loaded")

# ---------------------------------------------------------------- 历史 v 的条件
perms = {L: torch.randperm(I, generator=torch.Generator().manual_seed(1000 + L)) for L in range(nL)}
VX = {"T": vT,
      "N": {n: fl.v_ladder(vT[n], "col" if kind(n) == "down" else "row") for n, _ in FF},
      "O": {n: fl.v_ladder(vT[n], "row" if kind(n) == "down" else "col") for n, _ in FF},
      "RC": {n: fl.v_ladder(vT[n], "rowcol") for n, _ in FF}, "C": {n: fl.v_ladder(vT[n], "const") for n, _ in FF},
      "PN": {n: (vT[n][:, perms[layer(n)]] if kind(n) == "down" else vT[n][perms[layer(n)]]).clone() for n, _ in FF}}
checks = {}
if a.smoke:   # 注入前：N 只沿神经元轴变化（固定神经元时沿隐藏维恒定）；O 只沿隐藏维变化；PN 与 T 的数值多重集相同
    ok = True
    for n, _ in FF:
        ax = NAX(n); vN, vO = VX["N"][n], VX["O"][n]
        ok &= bool((vN.std(dim=1 - ax) <= 1e-5 * vN.abs().max()).all()) and bool((vO.std(dim=ax) <= 1e-5 * vO.abs().max()).all())
        ok &= bool(torch.equal(vT[n].flatten().sort()[0], VX["PN"][n].flatten().sort()[0]))
    checks["axis_constancy_and_perm_multiset"] = bool(ok); print("smoke 轴恒定/置换多重集:", ok, flush=True)

# ---------------------------------------------------------------- 批次与梯度（fp32，与训练一致；裁剪按原规则，作用于全部参数）
rows = fl.prompt_completion_rows(a.train, tok, a.max_len); exs = fl.tokenize_train_manual(rows, tok, a.max_len)
steps = fl.epoch_order(len(exs), 1, seed=17 + 99991)[: a.nb]
G = []; clipinfo = []
for st in steps:
    model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [exs[i] for i in st]); gn, cf = opt.clip_(1.0)
    G.append({n: p.grad.detach().float().cpu().clone() for n, p in FF}); clipinfo.append([float(gn), float(cf)])
model.zero_grad(set_to_none=True)
lap("grads")
bc1, bc2 = 1 - fl.BETA1 ** (T0 + 1), 1 - fl.BETA2 ** (T0 + 1)
def dvec(v, g):   # 第 2 节公式（无量纲方向 d；δW = -η·d）
    v1 = fl.BETA2 * v + (1 - fl.BETA2) * g * g
    return ((1 - fl.BETA1) * g / bc1) / ((v1 / bc2).sqrt() + fl.EPS)
if a.smoke:   # 公式核对：用训练代码的 DecompAdamW（清零 m、真 v、t=T0）算第 1 批的 T 更新，与 −η·dvec 逐元素比较（此后不再使用 opt）
    model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [exs[i] for i in steps[0]]); opt.clip_(1.0)
    for n_, s_ in opt.state.items(): s_["m"].zero_()
    rec = opt.compute(a.lr, keep=True)
    err = max(float((rec.u_adam[n].float().cpu() + a.lr * dvec(vT[n], G[0][n])).abs().max() / rec.u_adam[n].abs().max().clamp_min(1e-30)) for n, _ in FF)
    checks["formula_vs_DecompAdamW_maxrel"] = err; checks["formula_ok"] = err < 1e-4; print("smoke 公式核对 max rel", err, flush=True)
    model.zero_grad(set_to_none=True)
def neuron_norms(u, n): return u.double().pow(2).sum(dim=1 - NAX(n)).sqrt()
def energy_match(uX, uT, n):
    nx, nt = neuron_norms(uX, n), neuron_norms(uT, n)
    f = torch.where(nx > 0, nt / nx.clamp_min(1e-300), torch.zeros_like(nx)).to(uX.dtype)
    return uX * (f[None, :] if NAX(n) == 1 else f[:, None]), int((nx == 0).sum())
def rand_match(uT, n, gen):
    return energy_match(torch.randn(uT.shape, generator=gen, dtype=uT.dtype), uT, n)[0]

# ---------------------------------------------------------------- 评估输入（固定 completion 位置：预测 completion token 的位置，每题至多 max_tok 个）
def probe_set(path, k):
    prow = fl.prompt_completion_rows(path, tok, a.max_len, filter_ctx=False); return fl.encode_probe(prow, tok, a.max_len)[:k]
EVAL = {"new": probe_set(a.probe_new, a.n_eval), "old": probe_set(a.probe_old, a.n_eval)}
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def batches(task):
    ex = EVAL[task]
    for s in range(0, len(ex), a.eval_bs):
        b = ex[s:s + a.eval_bs]; L = max(len(e.input_ids) for e in b)
        ids = torch.full((len(b), L), pad, dtype=torch.long, device=dev); att = torch.zeros((len(b), L), dtype=torch.long, device=dev)
        rm = torch.zeros((len(b), L), dtype=torch.bool, device=dev)
        for i, e in enumerate(b):
            n_ = len(e.input_ids); ids[i, :n_] = torch.tensor(e.input_ids, device=dev); att[i, :n_] = 1
            rm[i, :n_] = torch.tensor(e.response_mask, dtype=torch.bool, device=dev)
        tm = torch.zeros_like(rm); tm[:, :-1] = rm[:, 1:] & att[:, 1:].bool()
        tm &= (tm.cumsum(1) <= a.max_tok)
        tgt = torch.cat([ids[:, 1:], ids.new_full((ids.shape[0], 1), pad)], 1)
        yield ids, att, tm, tgt

R4DT = torch.float64 if a.r4_dtype == "float64" else torch.float32
model.eval()
if R4DT == torch.float64: model.double()
base = {n: p.detach().clone() for n, p in FF}

def capture():   # H[t][l] = (总 token, hidden) fp64；LENS[t] = 每题 token 数
    H = {t: {l: [] for l in range(nL)} for t in EVAL}; lens = {t: [] for t in EVAL}; cur = {}; hooks = []
    for l in range(nL):
        hooks.append(model.model.layers[l].mlp.register_forward_pre_hook(lambda m, inp, l=l: cur.__setitem__(l, inp[0].detach())))
    with torch.no_grad():
        for t in EVAL:
            for ids, att, tm, _ in batches(t):
                model(input_ids=ids, attention_mask=att)
                lens[t] += [int(x) for x in tm.sum(1)]
                for l in range(nL): H[t][l].append(cur[l][tm].double())
    for h in hooks: h.remove()
    return {t: {l: torch.cat(H[t][l]) for l in range(nL)} for t in EVAL}, lens
def seg_norm(x, lens):   # 每题 Frobenius 范数
    sq = x.pow(2).sum(-1); idx = torch.repeat_interleave(torch.arange(len(lens), device=x.device), torch.tensor(lens, device=x.device))
    return torch.zeros(len(lens), dtype=sq.dtype, device=x.device).index_add_(0, idx, sq).sqrt()
def seg_dot(x, y, lens):
    s = (x * y).sum(-1); idx = torch.repeat_interleave(torch.arange(len(lens), device=x.device), torch.tensor(lens, device=x.device))
    return torch.zeros(len(lens), dtype=s.dtype, device=x.device).index_add_(0, idx, s)
def mlp_out(Wd, h): return F.linear(F.silu(F.linear(h, Wd["gate"])) * F.linear(h, Wd["up"]), Wd["down"])
def lw(l, dW=None, dtype=torch.float64):
    return {k: base32[W(l, k)].to(dtype) + (0 if dW is None else dW[W(l, k)].to(dev, dtype)) for k in ("gate", "up", "down")}
if a.r3:
    H, LENS = capture(); h_hash0 = sha(H["new"][0][:64]) + sha(H["old"][nL - 1][-64:]); lap("captured h")

def apply(dW):   # 全部前馈矩阵施加 δW（其余参数不动）；dW=None 恢复基线（copy，不做加减）
    with torch.no_grad():
        for n, p in FF: p.copy_(base[n] if dW is None else base[n] + dW[n].to(dev, p.dtype))
def logp_list(task, dW):
    apply(dW); out = []
    with torch.no_grad():
        for ids, att, tm, tgt in batches(task):
            lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.to(R4DT), -1)
            for i in range(ids.shape[0]): out.append(lp[i][tm[i]].clone())
    return out
def kl_items(task, lpR, dW):
    """lpR：参照 log p（每题 (n_tok, V)）。施加 dW 后返回每题 KL(p_R‖p_X)、KL(p_X‖p_R)、答案 NLL(X)。"""
    apply(dW); out = []; j = 0
    with torch.no_grad():
        for ids, att, tm, tgt in batches(task):
            lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.to(R4DT), -1)
            for i in range(ids.shape[0]):
                x = lp[i][tm[i]]; r = lpR[j]; j += 1
                out.append((float((r.exp() * (r - x)).sum(-1).mean()), float((x.exp() * (x - r)).sum(-1).mean()),
                            float(-x.gather(-1, tgt[i][tm[i]][:, None]).mean())))
    return out

# ---------------------------------------------------------------- 主循环（逐批）
CONDS_V = ["T", "N", "O", "RC", "C", "PN"]
EXTRA = ["OE", "CE"] + [f"RE{k}" for k in range(a.rand_seeds)]
res = {"config": dict(state=a.state, act=a.act, train=a.train, probe_new=a.probe_new, probe_old=a.probe_old, nb=a.nb, n_eval=a.n_eval, max_tok=a.max_tok,
                      lr=a.lr, T_state=T0, bc1=bc1, bc2=bc2, beta=[fl.BETA1, fl.BETA2], eps=fl.EPS, weight_decay="excluded", clip=1.0,
                      inject="once (single)", r3_dtype=a.r3_dtype, match="rms per matrix per batch; OE/CE/RE per-neuron energy matched", r4_dtype=a.r4_dtype,
                      batch_ids=steps, eval_n={t: len(EVAL[t]) for t in EVAL}, perm_seed="1000+layer", rand_seeds=list(range(a.rand_seeds)), rand_gen="cuda generator, per matrix in FF order (v2 loop)",
                      clip_per_batch=clipinfo, conds=CONDS_V + EXTRA, raw_r34=a.raw_r34),
       "r1": {}, "r3": [], "r4": [], "match_coef": [], "energy_match_dev": [], "zero_norm_neurons": [], "restored": [], "checks": checks}
if a.r3: res["config"]["eval_tokens"] = {t: LENS[t] for t in EVAL}
def topset(x, frac=0.05): k = max(1, int(frac * len(x))); return np.argsort(-x)[:k]
SETS = {l: (topset(A[f"L{l}_sel_gsm_vs_mbpp"]), topset(-A[f"L{l}_sel_gsm_vs_mbpp"])) for l in range(nL)}
def r1r2(cond, scale, b, U, UT):
    for n, _ in FF:
        l, k = layer(n), kind(n); e = U[n].double().pow(2).sum(dim=1 - NAX(n)).numpy(); e = e / e.sum(); Gs, Ms = SETS[l]
        res["r1"].setdefault(f"{cond}|{scale}", {}).setdefault(k, {}).setdefault(str(l), []).append(
            dict(b=b, share_new=float(e[Gs].sum()), share_old=float(e[Ms].sum()),
                 cos=float((U[n].double() * UT[n].double()).sum() / (U[n].double().norm() * UT[n].double().norm() + 1e-300)),
                 rel=float((U[n].double() - UT[n].double()).norm() / (UT[n].double().norm() + 1e-300))))
def r1r2_one(cond, scale, b, n, U, UT):   # GPU 上算单个矩阵的份额、cos、相对误差
    l, k = layer(n), kind(n); Ud, Td = U.double(), UT.double()
    e = Ud.pow(2).sum(dim=1 - NAX(n)); e = (e / e.sum()).cpu().numpy(); Gs, Ms = SETS[l]
    res["r1"].setdefault(f"{cond}|{scale}", {}).setdefault(k, {}).setdefault(str(l), []).append(
        dict(b=b, share_new=float(e[Gs].sum()), share_old=float(e[Ms].sum()),
             cos=float((Ud * Td).sum() / (Ud.norm() * Td.norm() + 1e-300)), rel=float((Ud - Td).norm() / (Td.norm() + 1e-300))))
def kl_eval(task, lpR):
    """当前已施加的权重下，每题 KL(p_R‖p_X)、KL(p_X‖p_R)、答案 NLL(X)；lpR 在 CPU，按题搬上 GPU。"""
    out = []; j = 0
    with torch.no_grad():
        for ids, att, tm, tgt in batches(task):
            lp = F.log_softmax(model(input_ids=ids, attention_mask=att).logits.to(R4DT), -1)
            for i in range(ids.shape[0]):
                x = lp[i][tm[i]]; r = lpR[j].to(dev); j += 1
                out.append((float((r.exp() * (r - x)).sum(-1).mean()), float((x.exp() * (x - r)).sum(-1).mean()),
                            float(-x.gather(-1, tgt[i][tm[i]][:, None]).mean())))
    return out
EXTRA_RAW = (["Nraw", "Oraw", "Craw"] if a.raw_r34 else [])
for b in range(a.nb):
    names_all = CONDS_V + EXTRA + EXTRA_RAW + (["ZERO"] if a.smoke else [])
    dW = {c: {} for c in names_all}; coefs = {c: {} for c in CONDS_V}; z = 0; devmax = 0.0
    gens = [torch.Generator(device=dev).manual_seed(k_) for k_ in range(a.rand_seeds)]
    for n, _ in FF:                                   # 逐矩阵在 GPU 上算全部条件
        g = G[b][n].to(dev)
        Dn = {c: dvec(VX[c][n].to(dev), g) for c in CONDS_V}; DT = Dn["T"]; rT = DT.pow(2).mean().sqrt()
        for c in CONDS_V:
            cf = float(rT / Dn[c].pow(2).mean().sqrt().clamp_min(1e-30)); coefs[c][n] = cf; Um = Dn[c] * cf
            r1r2_one(c, "raw", b, n, Dn[c], DT); r1r2_one(c, "matched", b, n, Um, DT); dW[c][n] = (-a.lr * Um).cpu()
        nT = neuron_norms(DT, n)
        for c, src in (("OE", "O"), ("CE", "C")):
            U, zz = energy_match(Dn[src], DT, n); z += zz
            devmax = max(devmax, float(((neuron_norms(U, n) - nT).abs() / nT.clamp_min(1e-300)).max()))
            r1r2_one(c, "matched", b, n, U, DT); dW[c][n] = (-a.lr * U).cpu()
        for k_ in range(a.rand_seeds):
            U = energy_match(torch.randn(DT.shape, generator=gens[k_], device=dev, dtype=DT.dtype), DT, n)[0]
            devmax = max(devmax, float(((neuron_norms(U, n) - nT).abs() / nT.clamp_min(1e-300)).max()))
            r1r2_one(f"RE{k_}", "matched", b, n, U, DT); dW[f"RE{k_}"][n] = (-a.lr * U).cpu()
        for c in EXTRA_RAW: dW[c][n] = (-a.lr * Dn[c[0]]).cpu()
        if a.smoke: dW["ZERO"][n] = torch.zeros_like(DT).cpu()
        del Dn, g, DT
    res["match_coef"].append({c: {n: coefs[c][n] for n in coefs[c] if layer(n) in (0, nL // 2, nL - 1)} for c in CONDS_V})
    res["zero_norm_neurons"].append(z); res["energy_match_dev"].append(devmax)
    lap(f"updates batch {b}")
    others = [c for c in dW if c != "T"]
    # ---- R3 局部响应（固定 h_l；每层每条件只搬一次）
    if a.r3:
        R3DT = torch.float64 if a.r3_dtype == "float64" else torch.float32
        for l in range(nL):
            Wl = {c: {k: base32[W(l, k)].to(R3DT) + dW[c][W(l, k)].to(dev, R3DT) for k in ("gate", "up", "down")} for c in dW}
            W0 = lw(l, None, R3DT)
            for t in EVAL:
                h = H[t][l].to(R3DT); y0 = mlp_out(W0, h); dyT = mlp_out(Wl["T"], h) - y0
                nT_ = seg_norm(dyT.double(), LENS[t])
                if R3DT == torch.float64 or b == 0:
                    h64 = H[t][l]; d64 = mlp_out(lw(l, dW["T"]), h64) - mlp_out(lw(l), h64)
                    d32 = (mlp_out(lw(l, dW["T"], torch.float32), h64.float()) - mlp_out(lw(l, None, torch.float32), h64.float())).double()
                    floor = seg_norm(d32 - d64, LENS[t]).tolist()
                else:
                    floor = None
                row = dict(b=b, l=l, task=t, normT=nT_.tolist(), floor32=floor, conds={})
                for c in others:
                    dyX = mlp_out(Wl[c], h) - y0; nX = seg_norm(dyX.double(), LENS[t])
                    row["conds"][c] = dict(rel=(seg_norm((dyX - dyT).double(), LENS[t]) / nT_.clamp_min(1e-300)).tolist(),
                                           cos=(seg_dot(dyX.double(), dyT.double(), LENS[t]) / (nX * nT_).clamp_min(1e-300)).tolist(), normX=nX.tolist())
                res["r3"].append(row)
            del Wl
        lap(f"r3 batch {b}")
    # ---- R4 全模型（每个条件只施加一次，跑完两任务全部题目）
    if a.r4:
        lpT = {t: [x.cpu() for x in logp_list(t, dW["T"])] for t in EVAL}
        rows = {t: dict(b=b, task=t, conds={}) for t in EVAL}
        apply(None)
        for t in EVAL:
            r = kl_eval(t, lpT[t]); rows[t].update(kl_T_W=[x[0] for x in r], kl_W_T=[x[1] for x in r], nll_W=[x[2] for x in r])
        apply(dW["T"])
        for t in EVAL: rows[t]["nll_T"] = [x[2] for x in kl_eval(t, lpT[t])]
        for c in others:
            apply(dW[c])
            for t in EVAL:
                r = kl_eval(t, lpT[t]); rows[t]["conds"][c] = dict(kl_T_X=[x[0] for x in r], nll=[x[2] for x in r])
        for t in EVAL: res["r4"].append(rows[t])
        del lpT
        lap(f"r4 batch {b}")
    apply(None)
    res["restored"].append(all(torch.equal(p, base[n]) for n, p in FF))
    del dW
    if dev.type == "cuda": torch.cuda.empty_cache()

# ---------------------------------------------------------------- 收尾检查
res["checks"]["nonff_untouched"] = all(sha(p) == nonff_hash0[n] for n, p in NONFF)
res["checks"]["all_restored"] = all(res["restored"])
res["checks"]["ff_restored_to_loaded"] = all(sha(p) == sha(base32[n]) for n, p in FF)
if a.r3:
    H2, _ = capture(); res["checks"]["h_identical_after"] = (sha(H2["new"][0][:64]) + sha(H2["old"][nL - 1][-64:])) == h_hash0
res["checks"]["energy_match_dev_max"] = max(res["energy_match_dev"])
res["checks"]["T_self_error_zero"] = max(d["rel"] for k in res["r1"]["T|matched"] for l in res["r1"]["T|matched"][k] for d in res["r1"]["T|matched"][k][l]) == 0.0
if a.smoke and a.r3:
    res["checks"]["s0_zero_dy"] = max(max(row["conds"]["ZERO"]["normX"]) for row in res["r3"]) == 0.0
print("checks:", res["checks"], flush=True)
res["runtime"] = dict(RT, peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev.type == "cuda" else None)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w")); lap("done")
