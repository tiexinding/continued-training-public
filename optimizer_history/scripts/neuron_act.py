#!/usr/bin/env python3
"""W4-1 / W4-2：前馈中间神经元的任务条件激活与改动排序的对应。
在状态 S（默认 M）上前向三任务探针各 100 条（只统计回答 token 位置），钩住每层 down_proj 的输入 h = act(gate x) * (up x)：
  逐神经元：RMS_task、|h| 均值、"打开率"（|h| > τ，τ = 该层全 token |h| 的中位数）、任务选择性 sel = (RMS_gsm8k − RMS_mbpp)/(RMS_gsm8k + RMS_mbpp)。
排序对应：对一段改动（默认 M→G keep，用其逐行列场 npz 的 gate/up 行残差、down 列残差，以及原始更新能量 eW），
  计算每层 top-5% 神经元与"GSM8K 选择性最强 5%""MBPP 选择性最强 5%""两任务激活都强 5%"的重合，超几何零值 = 5%。
输出 diag25/b4/neuron_act_<S>.npz（逐层数组）与 .json（汇总）。"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--state", required=True); ap.add_argument("--name", default="M"); ap.add_argument("--fields", nargs="*", default=[], help="tag=path 的逐行列场 npz")
ap.add_argument("--out", default=_OH_RUNS + "/diag25/b4"); ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true"); ap.add_argument("--data", default=_OH_SUBSETS); ap.add_argument("--probe_slice", default="", help="P4：只用探针的 a:b 段估计选择性（估计集），其余留作确认集"); ap.add_argument("--tasks", default="gsm8k,mbpp,dolly_qa", help="E 轮：可加 bio"); ap.add_argument("--sel", default="gsm8k,mbpp", help="选择性 = (new − old)/(new + old) 的两任务名")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, tok = load_model(a); model.eval(); named = fl.named_params(model)
if a.state != "base":
    sd = torch.load(Path(a.state) / "weights.pt", map_location="cpu")
    with torch.no_grad():
        for n, p in named: p.copy_(sd[n].to(p.device))
layers = model.model.layers; L = len(layers); H = layers[0].mlp.down_proj.in_features
acc = {}; cur = {"task": None, "mask": None}
def hook(i):
    def f(mod, inp):
        h = inp[0].detach().float()[0]; m = cur["mask"]
        if m is not None: h = h[m]
        d = acc.setdefault((cur["task"], i), {"sq": torch.zeros(H, device=h.device), "abs": torch.zeros(H, device=h.device), "n": 0, "absall": []})
        d["sq"] += (h ** 2).sum(0); d["abs"] += h.abs().sum(0); d["n"] += h.shape[0]
        if len(d["absall"]) < 8: d["absall"].append(h.abs().flatten()[: 20000].cpu())
    return f
hs = [layers[i].mlp.down_proj.register_forward_pre_hook(hook(i)) for i in range(L)]
TASKS = a.tasks.split(","); NEW, OLD = a.sel.split(",")
for task in TASKS:
    rows = [json.loads(l) for l in open(Path(a.data) / task / "probe.jsonl")]
    tmp = Path(tempfile.mkdtemp());
    with open(tmp / "p.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    pcs = fl.prompt_completion_rows(str(tmp / "p.jsonl"), tok, a.max_len, filter_ctx=False); exs = fl.encode_probe(pcs, tok, a.max_len)
    cur["task"] = task
    with torch.no_grad():
        for e in exs:
            ids = torch.tensor([e.input_ids], device=next(model.parameters()).device); cur["mask"] = torch.tensor(e.response_mask, device=ids.device).bool()
            model(input_ids=ids)
for h in hs: h.remove()
res = {}; arrays = {}
for i in range(L):
    stats = {}
    for task in TASKS:
        d = acc[(task, i)]; rms = (d["sq"] / d["n"]).sqrt().cpu().numpy(); mabs = (d["abs"] / d["n"]).cpu().numpy()
        tau = float(torch.cat(d["absall"]).median()); stats[task] = dict(rms=rms, mabs=mabs, tau=tau)
        arrays[f"L{i}_{task}_rms"] = rms
    g, m = stats[NEW]["rms"], stats[OLD]["rms"]; sel = (g - m) / (g + m + 1e-12); arrays[f"L{i}_sel_gsm_vs_mbpp"] = sel; arrays[f"L{i}_sel_{NEW}_vs_{OLD}"] = sel
    res[f"L{i}"] = dict(tau_new=stats[NEW]["tau"], rms_corr_new_old=float(np.corrcoef(g, m)[0, 1]), sel_pair=[NEW, OLD])
np.savez_compressed(out / f"neuron_act_{a.name}.npz", **arrays)
# ---- 与改动排序的对应
def topset(x, frac=0.05): k = max(1, int(frac * len(x))); return set(np.argsort(-x)[:k].tolist()), k
summary = {}
for spec in a.fields:
    tag, path = spec.split("=", 1); F = np.load(path); rows = []
    for i in range(L):
        g = arrays[f"L{i}_{NEW}_rms"]; m = arrays[f"L{i}_{OLD}_rms"]; sel = arrays[f"L{i}_sel_gsm_vs_mbpp"]; both = np.minimum(g, m)
        keys = {"gate行残差": F[f"model.layers.{i}.mlp.gate_proj.weight|eqstep_resid_Zrow"], "up行残差": F[f"model.layers.{i}.mlp.up_proj.weight|eqstep_resid_Zrow"], "down列残差": F[f"model.layers.{i}.mlp.down_proj.weight|eqstep_resid_Zcol"],
                "gate行原始能量": F[f"model.layers.{i}.mlp.gate_proj.weight|eW_row"], "up行原始能量": F[f"model.layers.{i}.mlp.up_proj.weight|eW_row"], "down列原始能量": F[f"model.layers.{i}.mlp.down_proj.weight|eW_col"]}
        r = {}
        for kn, v in keys.items():
            S, k = topset(v)
            for tn, t in (("GSM8K选择性", sel), ("MBPP选择性", -sel), ("两任务都强", both), ("GSM8K激活", g), ("MBPP激活", m)):
                T, _ = topset(t); r[f"{kn}∩{tn}"] = len(S & T) / k
            r[f"{kn}~GSM8K激活 ρ"] = float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(g)))[0, 1])
            r[f"{kn}~MBPP激活 ρ"] = float(np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(m)))[0, 1])
        rows.append(r)
    summary[tag] = {k: dict(median=float(np.median([r[k] for r in rows])), per_layer=[r[k] for r in rows]) for k in rows[0]}
    print(f"== {tag}（24 层中位；top-5% 重合的零值 = 0.05）")
    for k, v in summary[tag].items(): print(f"  {k:30} {v['median']:.3f}")
json.dump(dict(layers=res, overlap=summary), open(out / f"neuron_act_{a.name}.json", "w"), ensure_ascii=False, indent=1); print("done")
