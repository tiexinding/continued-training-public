#!/usr/bin/env python3
"""G 轮：缩进格式分级的三项判别检验（文档 40）。
  decomp  首位置缩进概率的"表示 × 读出"拆分：P(H_x, W_y)，x,y ∈ {B_g, S}；另含只换缩进+code-only 行的读出。
  untie   解开共享：输入嵌入与输出读出分别取 B_g / S（全部行或 592 行），隐状态用 S 的主体。
  onestep 固定 B_g 权重，用真实训练顺序的前 K 批 GSM8K 梯度，按各 v/t 方案模拟 AdamW 状态递推（权重不动），
          一阶预测 Δ mean log P(缩进) = Σ_k <∂J/∂θ, u_k>，按参数组拆分。是固定权重的多步反事实，不是真实轨迹。
"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys, re, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--out", default=_OH_RUNS + "/diag25/g1")
ap.add_argument("--base_state", default=_OH_RUNS + "/runs_e/B_g/state"); ap.add_argument("--states", default=""); ap.add_argument("--K", type=int, default=25)
ap.add_argument("--groups", default=_OH_RUNS + "/diag25/f1/groups.npz"); ap.add_argument("--parquet", default="/tmp/humaneval.parquet")
ap.add_argument("--data", default=_OH_SUBSETS); ap.add_argument("--runs", default=_OH_RUNS + "/runs_e")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--n_prompts", type=int, default=164)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, tok = load_model(a); model.eval(); named = fl.named_params(model); names = [n for n, _ in named]; P = dict(named); dev = next(model.parameters()).device
EMB = "model.embed_tokens.weight"
import pandas as pd; HE = list(pd.read_parquet(a.parquet).itertuples())[: a.n_prompts]
TG = np.load(a.groups, allow_pickle=True)["token"].item(); IND = np.asarray(TG["indent"]); CODE = np.asarray(TG["code_only"]); TGT = np.concatenate([IND, CODE])
V = model.config.vocab_size; RET = np.array([i for i in range(V) if tok.decode([i]).strip() == "return"]); DEF = np.array([i for i in range(V) if tok.decode([i]).strip() == "def"])
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
prefixes = [tok.apply_chat_template([{"role": "user", "content": "Complete the following Python function. Return the full function in a ```python code block.\n\n" + r.prompt}], tokenize=True, add_generation_prompt=True) for r in HE]
def batches(bs=16):
    for i in range(0, len(prefixes), bs):
        b = prefixes[i:i + bs]; L = max(len(p) for p in b)
        yield torch.tensor([[pad] * (L - len(p)) + p for p in b], device=dev), torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in b], device=dev)
def load_w(p): return torch.load(Path(p) / "weights.pt", map_location="cpu", weights_only=False)
def set_w(sd):
    with torch.no_grad():
        for n, p in named: p.copy_(sd[n].to(p.device))
@torch.no_grad()
def hidden():
    hs = []
    for ids, att in batches(): hs.append(model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1].float())
    return torch.cat(hs)
def metrics(H, W):
    lg = H @ W.float().T; lp = torch.log_softmax(lg, -1)
    m = lambda idx: torch.logsumexp(lp[:, torch.as_tensor(idx, device=lp.device)], -1)
    li, lr_, ld = m(IND), m(RET), m(DEF)
    return dict(p_indent=float(li.exp().mean()), logp_indent=float(li.mean()), p_return=float(lr_.exp().mean()), p_def=float(ld.exp().mean()))
def mix_rows(Wbase, Wsrc, rows):
    W = Wbase.clone(); idx = torch.as_tensor(rows); W[idx] = Wsrc[idx]; return W
B = load_w(a.base_state); WB = B[EMB].to(dev)
res = {}
if a.stage == "decomp":
    set_w(B); HB = hidden(); res["B_g"] = metrics(HB, WB)
    for spec in a.states.split(","):
        tag, p = spec.split("="); S = load_w(p); set_w(S); HS = hidden(); WS = S[EMB].to(dev)
        r = {"HB_WB": metrics(HB, WB), "HS_WB": metrics(HS, WB), "HB_WS": metrics(HB, WS), "HS_WS": metrics(HS, WS),
             "HS_WS_but_target_rows_B": metrics(HS, mix_rows(WS, WB, TGT)), "HS_WB_but_target_rows_S": metrics(HS, mix_rows(WB, WS, TGT)),
             "HS_WS_but_indent_rows_B": metrics(HS, mix_rows(WS, WB, IND))}
        k = "logp_indent"; tot = r["HS_WS"][k] - r["HB_WB"][k]
        rep = .5 * ((r["HS_WB"][k] - r["HB_WB"][k]) + (r["HS_WS"][k] - r["HB_WS"][k])); rd = .5 * ((r["HB_WS"][k] - r["HB_WB"][k]) + (r["HS_WS"][k] - r["HS_WB"][k]))
        r["shapley_logp_indent"] = dict(total=tot, representation=rep, readout=rd)
        r["cos_dH"] = float(torch.nn.functional.cosine_similarity(HS - HB, (WB[torch.as_tensor(IND, device=dev)].float().mean(0) - WB.float().mean(0)).expand_as(HS), dim=-1).mean())
        res[tag] = r; print(tag, {kk: round(vv, 4) for kk, vv in r["shapley_logp_indent"].items()}, "P:", {kk: round(r[kk]["p_indent"], 3) for kk in ("HB_WB", "HS_WB", "HB_WS", "HS_WS", "HS_WS_but_target_rows_B", "HS_WB_but_target_rows_S", "HS_WS_but_indent_rows_B")}, flush=True)
    json.dump(res, open(out / "decomp.json", "w"), indent=1)
elif a.stage == "untie":
    for spec in a.states.split(","):
        tag, p = spec.split("="); S = load_w(p); WS = S[EMB].to(dev); r = {}
        for in_name, Win in (("in_S", WS), ("in_B_all", WB), ("in_B_target", mix_rows(WS, WB, TGT))):
            set_w(S)
            with torch.no_grad(): P[EMB].copy_(Win)
            H = hidden()
            for out_name, Wout in (("out_S", WS), ("out_B_all", WB), ("out_B_target", mix_rows(WS, WB, TGT))): r[f"{in_name}|{out_name}"] = metrics(H, Wout)
        res[tag] = r; print(tag, {k: round(v["p_indent"], 3) for k, v in r.items()}, flush=True)
    json.dump(res, open(out / "untie.json", "w"), indent=1)
elif a.stage == "onestep":
    opt = fl.DecompAdamW(named, lr=5e-6, wd=0.01); fl.load_state(Path(a.base_state), model, opt); t_old = opt.t; vB = {n: opt.state[n]["v"].detach().float().cpu() for n in names}; print("B_g t =", t_old, flush=True)
    # 目标梯度：J = mean log P(缩进)（及 return）在首生成位置
    def obj_grad(idx):
        model.zero_grad(set_to_none=True); n = len(prefixes)
        for ids, att in batches(8):
            h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1]; lp = torch.log_softmax(model.lm_head(h).float(), -1)
            (torch.logsumexp(lp[:, torch.as_tensor(idx, device=dev)], -1).sum() / n).backward()
        return {nn: P[nn].grad.detach().float().clone() for nn in names}
    with torch.enable_grad():
        G_ind = obj_grad(IND); G_ret = obj_grad(RET)
    # 真实训练顺序的前 K 批梯度（裁剪后），存 CPU
    rows = fl.prompt_completion_rows(str(Path(a.data) / "gsm8k" / "train.jsonl"), tok, a.max_len); exs = fl.tokenize_train_manual(rows, tok, a.max_len)
    order = fl.epoch_order(len(exs), 1, 17)[: a.K]; T = 125; grads = []
    with torch.enable_grad():
        for st in order:
            model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [exs[i] for i in st]); opt.clip_(1.0); grads.append({n: P[n].grad.detach().float().cpu() for n in names})
    model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); print("grads ready", len(grads), flush=True)
    def load_c(arm):
        f = Path(a.runs) / arm / "records" / "match_c.pt"
        return torch.load(f, map_location="cpu", weights_only=False) if f.exists() else None
    cfac = load_c("E_vfac"); cperm = load_c("E_vperm"); gen = torch.Generator(device="cpu").manual_seed(0)
    def v_variant(kind):
        if kind == "zero": return {n: torch.zeros_like(vB[n]) for n in names}
        if kind == "true": return {n: vB[n].clone() for n in names}
        ver, c = ("rowcol", cfac) if kind == "fac" else ("cperm", cperm)
        out_ = {}
        for n in names:
            v = fl.v_ladder(vB[n], ver, gen=gen) if vB[n].dim() == 2 else vB[n].clone()
            if c is not None and n in c: v = v * float(c[n]) ** 2
            out_[n] = v
        return out_
    VARS = {"E_vt (真 v，留 t)": ("true", t_old), "E_v (真 v，清 t)": ("true", 0), "E_vfac (行列近似，留 t)": ("fac", t_old), "E_vfac_t0 (行列近似，清 t)": ("fac", 0),
            "E_vperm (通道置换，留 t)": ("perm", t_old), "E_vperm_t0 (通道置换，清 t)": ("perm", 0), "E_∅ (全清零)": ("zero", 0)}
    ind_rows = torch.as_tensor(IND, device=dev); code_rows = torch.as_tensor(CODE, device=dev)
    def grp(n): return "emb" if n == EMB else ("mlp" if ".mlp." in n else ("attn" if ".self_attn." in n else "other"))
    for vname, (kind, t0) in VARS.items():
        Vv = v_variant(kind); m = {n: torch.zeros_like(vB[n]) for n in names}; steps = []; cum = {"ind": 0.0, "ret": 0.0}
        for k, g in enumerate(grads):
            t = t0 + k + 1; lr = fl.lr_at(k, T, 5e-6); bc1, bc2 = 1 - fl.BETA1 ** t, 1 - fl.BETA2 ** t; d = {"step": k + 1, "lr": lr, "t": t}
            acc = {"ind": {}, "ret": {}}; un = {"emb_indent": 0.0, "emb_code": 0.0, "emb_other": 0.0}
            for n in names:
                gg = g[n].to(dev); mm = m[n].to(dev).mul_(fl.BETA1).add_(gg, alpha=1 - fl.BETA1); vv = Vv[n].to(dev).mul_(fl.BETA2).addcmul_(gg, gg, value=1 - fl.BETA2)
                m[n] = mm.cpu(); Vv[n] = vv.cpu()
                u = -(lr / bc1) * mm / (vv.sqrt() / math.sqrt(bc2) + fl.EPS)
                for key, G in (("ind", G_ind), ("ret", G_ret)):
                    if n == EMB:
                        pr = (G[n] * u).sum(1); acc[key]["emb_indent"] = float(pr[ind_rows].sum()); acc[key]["emb_code"] = float(pr[code_rows].sum()); acc[key]["emb_other"] = float(pr.sum()) - acc[key]["emb_indent"] - acc[key]["emb_code"]
                    else: acc[key][grp(n)] = acc[key].get(grp(n), 0.0) + float((G[n] * u).sum())
                if n == EMB:
                    rn = (u ** 2).sum(1); un["emb_indent"] = float(rn[ind_rows].sum().sqrt()); un["emb_code"] = float(rn[code_rows].sum().sqrt()); un["emb_other"] = float(rn.sum().sqrt())
            for key in ("ind", "ret"): d[key] = acc[key]; d[key + "_total"] = sum(acc[key].values()); cum[key] += d[key + "_total"]
            d["cum_ind"] = cum["ind"]; d["cum_ret"] = cum["ret"]; d["u_norms"] = un; steps.append(d)
        res[vname] = steps
        print(vname, "累计 Δlog P(缩进)：步 8 %.4f 步 %d %.4f；Δlog P(return)：%.4f" % (steps[min(7, len(steps) - 1)]["cum_ind"], len(steps), steps[-1]["cum_ind"], steps[-1]["cum_ret"]), "| 步 %d 分组" % len(steps), {k_: round(v_, 4) for k_, v_ in steps[-1]["ind"].items()}, flush=True)
    json.dump(res, open(out / "onestep.json", "w"), indent=1, ensure_ascii=False)
