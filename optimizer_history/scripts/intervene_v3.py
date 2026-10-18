#!/usr/bin/env python3
"""W4-3 v3（10-07，按报告 29 审核 P2-2）：随机种子纳入层号；sel_mbpp 加"同数量 + 同能量"双匹配对照；保存逐题 NLL；加 HumanEval 规范解与 Dolly 留出作为独立 NLL 集。
原说明：局部干预（带对照）。从终态 B（默认 G keep）出发，把选定的前馈中间神经元 i 的改动撤回到起点 A（= M）：gate 行 i、up 行 i、down 列 i 复位为 A。
选择方式（每层各取 k = frac × 4864 个）：
  resid    : 按 M→B 的 up 行残差（偏离零模型）排序取最大；
  energy   : 按原始更新能量（gate 行 eW + up 行 eW + down 列 eW）排序取最大；
  sel_gsm  : 按 GSM8K 选择性（neuron_act）取最大；   sel_mbpp : 按 MBPP 选择性取最大；
  random   : 同层同数量随机；
  energy_matched_random : 随机但使撤回的改动能量与 energy 组相同（按能量分层抽样近似）；
  interp   : 全参数插值 W = A + α(B − A)，α 使撤回能量占比与 energy 组相同。
评估：三探针 NLL（各 100）+ GSM8K 留出 200 题生成 + MBPP 100 题 pass@1。输出 diag25/b4/intervene_<B>.json。"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys, tempfile, re, subprocess
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
from state_mix_eval import heldout_rows, generate, norm_num, HASH, NUM, write_rows, run_mbpp, nll_eval  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--A", required=True); ap.add_argument("--B", required=True); ap.add_argument("--tag", default="Gkeep")
ap.add_argument("--fields", required=True); ap.add_argument("--act", required=True); ap.add_argument("--fracs", default="0.01,0.05"); ap.add_argument("--out", default=_OH_RUNS + "/diag25/b4")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--data", default=_OH_SUBSETS)
ap.add_argument("--no_gen", action="store_true"); ap.add_argument("--n_random", type=int, default=5); ap.add_argument("--v2", action="store_true"); ap.add_argument("--v3", action="store_true"); ap.add_argument("--humaneval", default="/tmp/humaneval.parquet"); ap.add_argument("--eval_slice", default="", help="P4：确认集 = 各探针的 a:b 段（与选择性估计集互斥），结果另存 nll_confirm")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(0)
model, tok = load_model(a); model.eval(); named = fl.named_params(model); names = [n for n, _ in named]
SA = torch.load(Path(a.A) / "weights.pt", map_location="cpu"); SB = torch.load(Path(a.B) / "weights.pt", map_location="cpu")
F = np.load(a.fields); ACT = np.load(a.act); L = 24; H = SA["model.layers.0.mlp.up_proj.weight"].shape[0]
# 数据
P = Path(a.data); sets = {}
for t in ("gsm8k", "mbpp", "dolly_qa"):
    rows = [json.loads(l) for l in open(P / t / "probe.jsonl")]; sets[f"probe_{t}"] = (t, rows)
g, dl = heldout_rows(P, 200); sets["heldout_gsm8k"] = ("gsm8k", g); sets["heldout_dolly"] = ("dolly_qa", dl)
if a.v3:
    import pandas as pd; df = pd.read_parquet(a.humaneval)
    he = [{"example_id": r.task_id, "messages": [{"role": "user", "content": "Complete the following Python function.\n\n" + r.prompt}, {"role": "assistant", "content": r.prompt + r.canonical_solution}]} for r in df.itertuples()]
    sets["heldout_humaneval"] = ("mbpp", he)   # 独立代码集（规范解 NLL），token 类别按 mbpp 口径
enc = {}; tmp = Path(tempfile.mkdtemp())
for k, (t, rows) in sets.items():
    write_rows(rows, tmp / f"{k}.jsonl"); pcs = fl.prompt_completion_rows(str(tmp / f"{k}.jsonl"), tok, a.max_len, filter_ctx=False); ke, kr = [], []
    for pc, r in zip(pcs, rows):
        e = fl.encode_probe([pc], tok, a.max_len)
        if e: ke.append(e[0]); kr.append(r)
    enc[k] = (t, ke, kr)
gold = [norm_num(HASH.search(r["messages"][-1]["content"]).group(1)) for r in enc["heldout_gsm8k"][2]]
def energy_of(i, idx):  # 撤回集合的改动能量（gate 行 + up 行 + down 列）
    e = 0.0
    for kind, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
        d = (SB[f"model.layers.{i}.mlp.{kind}.weight"] - SA[f"model.layers.{i}.mlp.{kind}.weight"]).double()
        e += float((d.index_select(dim, torch.tensor(idx)) ** 2).sum())
    return e
def total_energy():
    return sum(float(((SB[n] - SA[n]).double() ** 2).sum()) for n in names)
TOT = total_energy()
def selections(frac):
    k = max(1, int(frac * H)); sel = {}
    for i in range(L):
        resid = F[f"model.layers.{i}.mlp.up_proj.weight|eqstep_resid_Zrow"]
        en = F[f"model.layers.{i}.mlp.gate_proj.weight|eW_row"] + F[f"model.layers.{i}.mlp.up_proj.weight|eW_row"] + F[f"model.layers.{i}.mlp.down_proj.weight|eW_col"]
        s = ACT[f"L{i}_sel_gsm_vs_mbpp"]
        sel.setdefault("resid", {})[i] = np.argsort(-resid)[:k].tolist(); sel.setdefault("energy", {})[i] = np.argsort(-en)[:k].tolist()
        sel.setdefault("sel_gsm", {})[i] = np.argsort(-s)[:k].tolist(); sel.setdefault("sel_mbpp", {})[i] = np.argsort(s)[:k].tolist()
        sel.setdefault("random", {})[i] = rng.choice(H, k, replace=False).tolist()
        # 能量匹配随机：按能量分 10 档，按 energy 组在各档的分布抽样
        order = np.argsort(-en); bins = np.array_split(order, 10); counts = [sum(1 for j in sel["energy"][i] if j in set(b.tolist())) for b in bins]
        pick = []
        for b, c in zip(bins, counts): pick += rng.choice(b, min(c, len(b)), replace=False).tolist()
        sel.setdefault("energy_matched_random", {})[i] = pick
        if a.v2 or a.v3:
            for r_ in range(a.n_random):
                rr = np.random.default_rng(100 + r_ * 1000 + i)   # v3：种子纳入层号，避免各层共用同一随机序列
                sel.setdefault(f"random{r_}", {})[i] = rr.choice(H, k, replace=False).tolist()
                # 能量匹配（旧 v2 口径）：同层随机累计到同能量，数量不限
                target = en[sel["sel_mbpp"][i]].sum(); cand = rr.permutation(H); pick2 = []; tot = 0.0
                for j in cand:
                    if tot >= target or len(pick2) >= 4 * k: break
                    pick2.append(int(j)); tot += en[j]
                sel.setdefault(f"mbpp_matched_random{r_}", {})[i] = pick2
                if a.v3:
                    # 同数量 + 同能量双匹配：从能量分档中按 sel_mbpp 组各档人数抽样（数量 = k，能量分布近似）
                    order = np.argsort(en); bins = np.array_split(order, 20); tgt = set(sel["sel_mbpp"][i]); pick3 = []
                    for b in bins:
                        c = sum(1 for j in b if j in tgt)
                        if c: pick3 += rr.choice(b, min(c, len(b)), replace=False).tolist()
                    sel.setdefault(f"mbpp_dual_matched{r_}", {})[i] = pick3
    return sel, k
def apply_config(cfg_sel=None, alpha=None, ffn_only=False):
    with torch.no_grad():
        for n, p in named:
            w = SB[n].clone()
            if alpha is not None and (not ffn_only or ".mlp." in n): w = SA[n] + alpha * (SB[n] - SA[n])
            elif cfg_sel is not None and ".mlp." in n:
                i = int(n.split(".")[2]); idx = torch.tensor(cfg_sel[i])
                if "down_proj" in n: w[:, idx] = SA[n][:, idx]
                else: w[idx, :] = SA[n][idx, :]
            p.copy_(w.to(p.device))
def evaluate(name):
    per = {k: nll_eval(model, tok, t, exs, rows) for k, (t, exs, rows) in enc.items()}
    r = {"nll": {k: float(np.sum(v["s"]) / np.sum(v["c"])) for k, v in per.items()}, "nll_per_item": {k: [float(x) / max(1, c) for x, c in zip(v["s"], v["c"])] for k, v in per.items()}}
    if a.eval_slice:
        lo, hi = [int(x) for x in a.eval_slice.split(":")]
        r["nll_confirm"] = {k: float(np.sum(v["s"][lo:hi]) / max(1, np.sum(v["c"][lo:hi]))) for k, v in per.items() if k.startswith("probe_")}
        r["nll_estimate"] = {k: float(np.sum(v["s"][:lo] + v["s"][hi:]) / max(1, np.sum(v["c"][:lo] + v["c"][hi:]))) for k, v in per.items() if k.startswith("probe_")}
    if not a.no_gen:
        t, exs, rows = enc["heldout_gsm8k"]; outs = generate(model, tok, exs, 400)
        r["gsm8k_relaxed"] = [bool((ns := NUM.findall(o)) and norm_num(ns[-1]) == y) for o, y in zip(outs, gold)]
        t, exs, rows = enc["probe_mbpp"]; mo = generate(model, tok, exs, 512); r["mbpp_pass"] = [run_mbpp(o, rr) for o, rr in zip(mo, rows)]
        r["gsm8k_acc"] = float(np.mean(r["gsm8k_relaxed"])); r["mbpp_acc"] = float(np.mean(r["mbpp_pass"]))
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in ("nll", "gsm8k_acc", "mbpp_acc")}), flush=True); return r
res = {}
apply_config(); res["B"] = evaluate("B(终态)"); apply_config(alpha=0.0); res["A"] = evaluate("A(起点M)")
for frac in [float(x) for x in a.fracs.split(",")]:
    sel, k = selections(frac)
    keys = ["resid", "energy", "sel_gsm", "sel_mbpp", "random", "energy_matched_random"] + ([f"random{r_}" for r_ in range(a.n_random)] + [f"mbpp_matched_random{r_}" for r_ in range(a.n_random)] if (a.v2 or a.v3) else []) + ([f"mbpp_dual_matched{r_}" for r_ in range(a.n_random)] if a.v3 else [])
    for key in keys:
        e_frac = sum(energy_of(i, sel[key][i]) for i in range(L)) / TOT
        apply_config(cfg_sel=sel[key]); r = evaluate(f"{key}@{frac}"); r["reverted_energy_frac"] = e_frac; res[f"{key}@{frac}"] = r
    e_en = res[f"energy@{frac}"]["reverted_energy_frac"]; alpha = float(np.sqrt(max(0.0, 1 - e_en)))   # 插值使撤回能量占比相同：(1−α)² ≈ e_en → α = 1−√e_en
    alpha = 1 - float(np.sqrt(e_en)); apply_config(alpha=alpha); r = evaluate(f"interp α={alpha:.4f}"); r["alpha"] = alpha; r["reverted_energy_frac"] = e_en; res[f"interp@{frac}"] = r
    if a.v2:   # FFN 等干预平方范数插值：只在前馈三矩阵上插值，使 (1−α_f)²·E_ffn = e_en·E_total
        E_ffn = sum(float(((SB[n] - SA[n]).double() ** 2).sum()) for n in names if ".mlp." in n); e_f = min(1.0, e_en * TOT / E_ffn)
        alpha_f = 1 - float(np.sqrt(e_f)); apply_config(alpha=alpha_f, ffn_only=True); r = evaluate(f"interp_ffn α_f={alpha_f:.4f}"); r["alpha"] = alpha_f; r["reverted_energy_frac"] = e_en; r["ffn_energy_frac"] = E_ffn / TOT; res[f"interp_ffn@{frac}"] = r
import math
def mcn(x, y):
    x = np.array(x, bool); y = np.array(y, bool); g = int(np.sum(x & ~y)); l = int(np.sum(~x & y)); n = g + l
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(g, l) + 1)) / 2 ** n) if n else 1.0; return [g, l, p]
if not a.no_gen:
    pairs = {}
    for k1 in res:
        for k2 in res:
            if k1 < k2 and k1 not in ("A", "B") and k2 not in ("A", "B") and k1.split("@")[-1] == k2.split("@")[-1]:
                pairs[f"{k1}|{k2}"] = dict(gsm8k=mcn(res[k1]["gsm8k_relaxed"], res[k2]["gsm8k_relaxed"]), mbpp=mcn(res[k1]["mbpp_pass"], res[k2]["mbpp_pass"]))
    res["pairwise"] = pairs
json.dump(res, open(out / f"intervene_{a.tag}.json", "w"), ensure_ascii=False); print("done")
