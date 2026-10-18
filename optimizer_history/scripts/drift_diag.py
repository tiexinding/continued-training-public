#!/usr/bin/env python3
"""F 轮：无监督维度漂移机制诊断（文档 38）。
阶段（--stage，顺序执行，结果写 --out 目录）：
  groups   在 B_g 上定义参数子集：前馈中间神经元按 (bio, gsm8k, code) 探针激活 RMS 分组（code-only / both-used / 其余）；词表行按 token 在 bio 训练集、gsm8k 训练集、HumanEval 文本中的频次分组（indent / code-only / both-used / 其余）；并统计 B_g 优化器 v 在各组的相对大小。
  energy   对若干 (起点, 终态) 对，算 ΔW 能量在各组的份额 / 参数份额（=富集比）。
  logits   在 HumanEval 聊天提示的首生成位置，各状态对 indent 类 token、'return' 类、'def' 类的概率质量。
  interv   把 E_vperm / E_vfac 中的子集参数撤回到 B_g（或把 E_vperm 的子集移植进 E_∅），测首位置 indent 概率、无缩进率与 v3 规则通过率。
  offline  B_g 固定 8 批 GSM8K 梯度的一步更新，v 版本 orig/rowcol/cperm/const/fresh（匹配 RMS），各组能量富集比。
"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys, re, tempfile, subprocess, collections, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--out", default=_OH_RUNS + "/diag25/f1")
ap.add_argument("--base_state", default=_OH_RUNS + "/runs_e/B_g/state"); ap.add_argument("--pairs", default=""); ap.add_argument("--states", default="")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--data", default=_OH_SUBSETS); ap.add_argument("--parquet", default="/tmp/humaneval.parquet"); ap.add_argument("--frac", type=float, default=0.05)
ap.add_argument("--nb", type=int, default=8); ap.add_argument("--tag", default="x"); ap.add_argument("--n_random", type=int, default=2)
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
model, tok = load_model(a); model.eval(); named = fl.named_params(model); names = [n for n, _ in named]; dev = next(model.parameters()).device
L = len(model.model.layers); H = model.model.layers[0].mlp.down_proj.in_features
EMB = "model.embed_tokens.weight"; TIED = getattr(model.config, "tie_word_embeddings", False)
import pandas as pd; df = pd.read_parquet(a.parquet); HE = list(df.itertuples())
def load_w(p): return torch.load(Path(p) / "weights.pt", map_location="cpu")
def set_w(sd):
    with torch.no_grad():
        for n, p in named: p.copy_(sd[n].to(p.device))
def tok_ids(s): return tok(s, add_special_tokens=False)["input_ids"]
# ---------- token 分组 ----------
def token_groups():
    V = model.config.vocab_size; cnt = {}
    def count(texts):
        c = np.zeros(V, dtype=np.int64)
        for t in texts:
            for i in tok_ids(t): c[i] += 1
        return c
    bio = [json.loads(l) for l in open(Path(a.data) / "bio" / "train.jsonl")]; gsm = [json.loads(l) for l in open(Path(a.data) / "gsm8k" / "train.jsonl")]
    txt = lambda rows: [" ".join(m["content"] for m in r["messages"]) for r in rows]
    cnt["bio"] = count(txt(bio)); cnt["gsm8k"] = count(txt(gsm)); cnt["code"] = count([r.prompt + r.canonical_solution for r in HE])
    strs = [tok.decode([i]) for i in range(V)]
    indent = np.array([bool(re.fullmatch(r"\n? {2,}|\t+|\n\t+", s)) for s in strs])
    both_used = (cnt["bio"] > 0) & (cnt["gsm8k"] > 0)
    code_only = (cnt["code"] >= 3) & (cnt["bio"] == 0) & (cnt["gsm8k"] == 0)
    unused = (cnt["bio"] == 0) & (cnt["gsm8k"] == 0) & (cnt["code"] == 0)
    return dict(indent=np.where(indent)[0], code_only=np.where(code_only & ~indent)[0], both_used=np.where(both_used)[0], unused=np.where(unused & ~indent)[0]), {k: int(v.sum()) for k, v in cnt.items()}
# ---------- 神经元分组（激活 RMS） ----------
def neuron_rms(sd):
    set_w(sd); acc = {}; cur = {"task": None, "mask": None}
    def hook(i):
        def f(mod, inp):
            h = inp[0].detach().float()[0]; m = cur["mask"]
            if m is not None: h = h[m]
            d = acc.setdefault((cur["task"], i), [torch.zeros(H, device=h.device), 0]); d[0] += (h ** 2).sum(0); d[1] += h.shape[0]
        return f
    hs = [model.model.layers[i].mlp.down_proj.register_forward_pre_hook(hook(i)) for i in range(L)]
    sets = {}
    for task in ("bio", "gsm8k"):
        rows = [json.loads(l) for l in open(Path(a.data) / task / "probe.jsonl")]; tmp = Path(tempfile.mkdtemp())
        with open(tmp / "p.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        pcs = fl.prompt_completion_rows(str(tmp / "p.jsonl"), tok, a.max_len, filter_ctx=False); sets[task] = [(e.input_ids, e.response_mask) for e in fl.encode_probe(pcs, tok, a.max_len)]
    sets["code"] = [(tok_ids(r.prompt + r.canonical_solution)[: a.max_len], None) for r in HE]
    with torch.no_grad():
        for task, exs in sets.items():
            cur["task"] = task
            for ids, m in exs:
                cur["mask"] = torch.tensor(m, device=dev).bool() if m is not None else None; model(input_ids=torch.tensor([ids], device=dev))
    for h in hs: h.remove()
    return {task: np.stack([(acc[(task, i)][0] / acc[(task, i)][1]).sqrt().cpu().numpy() for i in range(L)]) for task in sets}
def neuron_groups(R, frac):
    k = max(1, int(frac * H)); G = {}
    other = np.maximum(R["bio"], R["gsm8k"]); sel_code = (R["code"] - other) / (R["code"] + other + 1e-12)
    both = np.minimum(R["bio"], R["gsm8k"]) / (R["code"] + 1e-12)
    for i in range(L):
        G.setdefault("code_only", {})[i] = np.argsort(-sel_code[i])[:k].tolist(); G.setdefault("both_used", {})[i] = np.argsort(-both[i])[:k].tolist()
        G.setdefault("low_all", {})[i] = np.argsort(R["bio"][i] + R["gsm8k"][i] + R["code"][i])[:k].tolist()
    return G, dict(sel_code=sel_code, both=both)
GP = out / "groups.npz"
def load_groups():
    z = np.load(GP, allow_pickle=True); return z["neuron"].item(), z["token"].item()
# ---------- 能量富集 ----------
def enrich(dW, NG, TG):
    """返回 {组: 能量份额 / 参数份额}，分前馈神经元组（gate 行 + up 行 + down 列）与词表行组。"""
    r = {}
    tot_ffn = sum(float((dW[n].double() ** 2).sum()) for n in names if ".mlp." in n); n_ffn = sum(dW[n].numel() for n in names if ".mlp." in n)
    for g, sel in NG.items():
        e = 0.0; cnt = 0
        for i in range(L):
            idx = torch.tensor(sel[i])
            for kind, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                d = dW[f"model.layers.{i}.mlp.{kind}.weight"].double(); s = d.index_select(dim, idx); e += float((s ** 2).sum()); cnt += s.numel()
        r[f"ffn_{g}"] = dict(energy_share=e / tot_ffn, param_share=cnt / n_ffn, enrich=(e / tot_ffn) / (cnt / n_ffn))
    E = dW[EMB].double(); tot = float((E ** 2).sum()); rows_e = (E ** 2).sum(1).numpy()
    for g, idx in TG.items():
        if len(idx) == 0: continue
        r[f"vocab_{g}"] = dict(energy_share=float(rows_e[idx].sum() / tot), param_share=len(idx) / E.shape[0], enrich=float(rows_e[idx].sum() / tot) / (len(idx) / E.shape[0]), n=int(len(idx)))
    return r
# ---------- 首位置 logit ----------
def chat_prefix(r): return tok.apply_chat_template([{"role": "user", "content": "Complete the following Python function. Return the full function in a ```python code block.\n\n" + r.prompt}], tokenize=True, add_generation_prompt=True)
def first_pos_probs(TG):
    cls = {"indent": TG["indent"], "return": np.array([i for i in range(model.config.vocab_size) if tok.decode([i]).strip() == "return"]), "def": np.array([i for i in range(model.config.vocab_size) if tok.decode([i]).strip() == "def"]), "fence": np.array([i for i in range(model.config.vocab_size) if tok.decode([i]).strip().startswith("``")])}
    res = {k: [] for k in cls}; top = []
    with torch.no_grad():
        for r in HE:
            ids = torch.tensor([chat_prefix(r)], device=dev); lg = model(input_ids=ids).logits[0, -1].float(); p = torch.softmax(lg, -1).cpu().numpy()
            for k, idx in cls.items(): res[k].append(float(p[idx].sum()))
            top.append(tok.decode([int(p.argmax())]))
    return {k: dict(mean=float(np.mean(v)), per=v) for k, v in res.items()} | {"top1_counter": collections.Counter(top).most_common(6)}
# ---------- 生成与 v3 判分 ----------
def gen(prefixes, max_new=400, bs=16):
    im_end = tok.convert_tokens_to_ids("<|im_end|>"); eos = [t for t in {tok.eos_token_id, im_end} if t is not None]; pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]; outs = []
    for i in range(0, len(prefixes), bs):
        batch = prefixes[i:i + bs]; Lm = max(len(p) for p in batch)
        ids = torch.tensor([[pad] * (Lm - len(p)) + p for p in batch], device=dev); att = torch.tensor([[0] * (Lm - len(p)) + [1] * len(p) for p in batch], device=dev)
        o = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        outs += [tok.decode([t for t in row if t not in eos and t != pad], skip_special_tokens=True) for row in o[:, Lm:].tolist()]
    return outs
def fmt_cat(o, ep):
    m = re.search(r"```(?:python)?\n(.*?)```", o, re.S); b = m.group(1) if m else o
    if ("def " + ep) in b: return "full"
    first = next((l for l in b.split("\n") if l.strip()), ""); return "unind" if first and not first.startswith((" ", "\t")) else "ind"
def run_v3(code, r):
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S); body = m.group(1) if m else code
    header = "\n".join(l for l in r.prompt.split("\n") if l.startswith("import ") or l.startswith("from ")) + "\nfrom typing import *\nimport math\n"
    if ("def " + r.entry_point) in body: prog = header + body
    else:
        first = next((l for l in body.split("\n") if l.strip()), "")
        if first and not first.startswith((" ", "\t")): body = "\n".join(("    " + l if l.strip() else l) for l in body.split("\n"))
        prog = r.prompt + body
    prog += "\n\n" + r.test + f"\n\ncheck({r.entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); fn = f.name
    try: pr = subprocess.run([sys.executable, fn], capture_output=True, timeout=10); return pr.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally: Path(fn).unlink(missing_ok=True)
def run_v2(code, r):
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S); body = m.group(1) if m else code
    header = "\n".join(l for l in r.prompt.split("\n") if l.startswith("import ") or l.startswith("from ")) + "\nfrom typing import *\nimport math\n"
    prog = (header + body if ("def " + r.entry_point) in body else r.prompt + body) + "\n\n" + r.test + f"\n\ncheck({r.entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); fn = f.name
    try: pr = subprocess.run([sys.executable, fn], capture_output=True, timeout=10); return pr.returncode == 0
    except subprocess.TimeoutExpired: return False
    finally: Path(fn).unlink(missing_ok=True)
def humaneval_eval():
    outs = gen([chat_prefix(r) for r in HE]); cats = [fmt_cat(o, r.entry_point) for o, r in zip(outs, HE)]; passed = [run_v3(o, r) for o, r in zip(outs, HE)]; p2 = [run_v2(o, r) for o, r in zip(outs, HE)]
    return dict(unind=float(np.mean([c == "unind" for c in cats])), full=float(np.mean([c == "full" for c in cats])), pass_v3=float(np.mean(passed)), pass_v2=float(np.mean(p2)), cats=cats, passed=passed, passed_v2=p2, outs=outs)
# ---------- 干预 ----------
def patch(sd_target, sd_source, NG, TG, neuron_keys=(), token_keys=(), rand_seed=None):
    """把 sd_target 中指定子集替换为 sd_source 的值。rand_seed 给出时用同数量随机神经元/行。"""
    sd = {n: sd_target[n].clone() for n in names}; rng = np.random.default_rng(rand_seed) if rand_seed is not None else None
    for g in neuron_keys:
        for i in range(L):
            idx = NG[g][i] if rng is None else rng.choice(H, len(NG[g][i]), replace=False).tolist(); idx = torch.tensor(idx)
            for kind, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                n = f"model.layers.{i}.mlp.{kind}.weight"
                if dim == 0: sd[n][idx, :] = sd_source[n][idx, :]
                else: sd[n][:, idx] = sd_source[n][:, idx]
    for g in token_keys:
        idx = TG[g] if rng is None else rng.choice(sd[EMB].shape[0], len(TG[g]), replace=False); idx = torch.tensor(np.asarray(idx))
        sd[EMB][idx, :] = sd_source[EMB][idx, :]
    return sd
# ================= 阶段 =================
if a.stage == "groups":
    base = load_w(a.base_state); R = neuron_rms(base); NG, aux = neuron_groups(R, a.frac); TG, tot = token_groups()
    np.savez_compressed(GP, neuron=np.array(NG, dtype=object), token=np.array(TG, dtype=object), rms_bio=R["bio"], rms_gsm8k=R["gsm8k"], rms_code=R["code"], sel_code=aux["sel_code"])
    # v 统计
    opt = fl.DecompAdamW(named, lr=5e-6, wd=0.01); fl.load_state(Path(a.base_state), model, opt); vs = {}
    def rel(v, idx, dim):
        s = v.index_select(dim, torch.tensor(idx)) if dim is not None else v[idx]; return float(s.median() / v.median())
    for g in NG:
        vals = {"gate_row": [], "down_col": []}
        for i in range(L):
            vals["gate_row"].append(rel(opt.state[f"model.layers.{i}.mlp.gate_proj.weight"]["v"].float().cpu(), NG[g][i], 0)); vals["down_col"].append(rel(opt.state[f"model.layers.{i}.mlp.down_proj.weight"]["v"].float().cpu(), NG[g][i], 1))
        vs[f"ffn_{g}"] = {k: float(np.median(v)) for k, v in vals.items()}
    vE = opt.state[EMB]["v"].float().cpu(); rowmed = vE.median(1).values
    for g, idx in TG.items():
        if len(idx): vs[f"vocab_{g}"] = dict(rel_median=float(rowmed[torch.tensor(idx)].median() / rowmed.median()), frac_zero_rows=float((rowmed[torch.tensor(idx)] <= fl.V_FLOOR * 1.01).float().mean()), n=int(len(idx)))
    json.dump(dict(tied=TIED, token_totals=tot, group_sizes={f"vocab_{g}": int(len(v)) for g, v in TG.items()}, v_relative=vs, rms_corr=dict(code_bio=float(np.corrcoef(R["code"].ravel(), R["bio"].ravel())[0, 1]), code_gsm=float(np.corrcoef(R["code"].ravel(), R["gsm8k"].ravel())[0, 1])), indent_tokens=[tok.decode([int(i)]) for i in TG["indent"][:20]]), open(out / "groups.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(vs, indent=1))
elif a.stage == "energy":
    NG, TG = load_groups(); res = {}
    for spec in a.pairs.split(","):
        tag, A, B = spec.split("=")[0], *spec.split("=")[1].split(":"); SA, SB = load_w(A), load_w(B); dW = {n: SB[n] - SA[n] for n in names}; res[tag] = enrich(dW, NG, TG); print(tag, {k: round(v["enrich"], 2) for k, v in res[tag].items()}, flush=True)
    json.dump(res, open(out / "energy_enrich.json", "w"), indent=1)
elif a.stage == "logits":
    NG, TG = load_groups(); res = {}
    for spec in a.states.split(","):
        tag, p = spec.split("="); set_w(load_w(p)); res[tag] = first_pos_probs(TG); print(tag, {k: round(v["mean"], 3) for k, v in res[tag].items() if k != "top1_counter"}, res[tag]["top1_counter"], flush=True)
    json.dump(res, open(out / "first_pos_probs.json", "w"), indent=1, ensure_ascii=False)
elif a.stage == "interv":
    NG, TG = load_groups(); base = load_w(a.base_state); res = {}
    def ev(tag, sd):
        set_w(sd); r = first_pos_probs(TG); h = humaneval_eval(); res[tag] = dict(indent_p=r["indent"]["mean"], return_p=r["return"]["mean"], **{k: h[k] for k in ("unind", "full", "pass_v3", "cats", "passed")}); print(tag, {k: round(res[tag][k], 3) for k in ("indent_p", "return_p", "unind", "full", "pass_v3")}, flush=True)
    for arm in ("E_vperm", "E_vfac", "E_0"):
        S = load_w(f"{_OH_RUNS}/runs_e/{arm}/snaps/step0125"); ev(f"{arm}", S)
        if arm != "E_0":
            ev(f"{arm}~revert_neuron_code_only", patch(S, base, NG, TG, neuron_keys=("code_only",)))
            ev(f"{arm}~revert_vocab_indent+code", patch(S, base, NG, TG, token_keys=("indent", "code_only")))
            ev(f"{arm}~revert_both", patch(S, base, NG, TG, neuron_keys=("code_only",), token_keys=("indent", "code_only")))
            ev(f"{arm}~revert_neuron_both_used", patch(S, base, NG, TG, neuron_keys=("both_used",)))
            for s in range(a.n_random): ev(f"{arm}~revert_random{s}", patch(S, base, NG, TG, neuron_keys=("code_only",), token_keys=("indent", "code_only"), rand_seed=s))
    Sp = load_w(_OH_RUNS + "/runs_e/E_vperm/snaps/step0125"); S0 = load_w(_OH_RUNS + "/runs_e/E_0/snaps/step0125")
    ev("E_0~transplant_both_from_vperm", patch(S0, Sp, NG, TG, neuron_keys=("code_only",), token_keys=("indent", "code_only")))
    ev("E_0~transplant_random0_from_vperm", patch(S0, Sp, NG, TG, neuron_keys=("code_only",), token_keys=("indent", "code_only"), rand_seed=0))
    json.dump(res, open(out / "intervene_drift.json", "w"), indent=1)
elif a.stage == "interv_energy":
    """10-10 I5（文档 43）：恢复实验的能量匹配随机对照（PG 置换，种子 17）。随机组与目标组按"相对起点的改动能量"
    分层匹配：所有候选按能量分位数分 50 箱，每个目标元素从同箱的非目标元素中无放回抽一个。词表行能量 = ‖ΔE_r‖²；
    神经元能量（逐层）= gate 行 + up 行 + down 列的 ‖Δ‖²。参照：未改动、恢复 592 行、恢复神经元 + 行。"""
    NG, TG = load_groups(); base = load_w(a.base_state); res = {}
    S = load_w(_OH_RUNS + "/runs_e/E_vperm/snaps/step0125")
    def ev(tag, sd, extra=None):
        set_w(sd); r = first_pos_probs(TG); h = humaneval_eval()
        res[tag] = dict(indent_p=r["indent"]["mean"], **{k: h[k] for k in ("unind", "full", "pass_v3", "cats", "passed")}, **(extra or {}))
        print(tag, {k: round(res[tag][k], 4) for k in ("indent_p", "unind", "pass_v3")}, extra or "", flush=True)
    def match(e, tgt, rng, nb=50):
        e = np.asarray(e, dtype=np.float64); tgt = np.asarray(tgt); is_t = np.zeros(len(e), bool); is_t[tgt] = True
        q = np.quantile(e, np.linspace(0, 1, nb + 1)); b = np.clip(np.searchsorted(q, e, side="right") - 1, 0, nb - 1)
        pools = {k: list(rng.permutation(np.where((b == k) & ~is_t)[0])) for k in range(nb)}; out = []
        for t in tgt:
            k = b[t]; d = 0
            while True:
                for kk in (k - d, k + d):
                    if 0 <= kk < nb and pools[kk]: out.append(pools[kk].pop()); break
                else: d += 1; continue
                break
        return np.array(out)
    eR = ((S[EMB] - base[EMB]).double() ** 2).sum(1).numpy(); T = np.concatenate([TG["indent"], TG["code_only"]]).astype(int)
    eN = []
    for i in range(L):
        g_, u_, d_ = (f"model.layers.{i}.mlp.{k}.weight" for k in ("gate_proj", "up_proj", "down_proj"))
        eN.append((((S[g_] - base[g_]).double() ** 2).sum(1) + ((S[u_] - base[u_]).double() ** 2).sum(1) + ((S[d_] - base[d_]).double() ** 2).sum(0)).numpy())
    def revert(rows, neur):
        sd = {n: S[n].clone() for n in names}
        if rows is not None:
            idx = torch.tensor(rows); sd[EMB][idx, :] = base[EMB][idx, :]
        if neur is not None:
            for i in range(L):
                idx = torch.tensor(neur[i])
                for kind, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                    n = f"model.layers.{i}.mlp.{kind}.weight"
                    if dim == 0: sd[n][idx, :] = base[n][idx, :]
                    else: sd[n][:, idx] = base[n][:, idx]
        return sd
    eT_rows = float(eR[T].sum()); eT_neur = float(sum(eN[i][NG["code_only"][i]].sum() for i in range(L)))
    ev("E_vperm", S)
    ev("rows592", revert(T, None), dict(energy_rows=eT_rows))
    ev("rows592+neurons", revert(T, [NG["code_only"][i] for i in range(L)]), dict(energy_rows=eT_rows, energy_neur=eT_neur))
    for s in range(3):
        rng = np.random.default_rng(1000 + s); R_ = match(eR, T, rng)
        ev(f"rows_ematch{s}", revert(R_, None), dict(energy_rows=float(eR[R_].sum()), energy_ratio_rows=float(eR[R_].sum()) / eT_rows))
        Nn = [match(eN[i], np.asarray(NG["code_only"][i], int), rng) for i in range(L)]
        en = float(sum(eN[i][Nn[i]].sum() for i in range(L)))
        ev(f"rows+neurons_ematch{s}", revert(R_, Nn), dict(energy_rows=float(eR[R_].sum()), energy_neur=en, energy_ratio_rows=float(eR[R_].sum()) / eT_rows, energy_ratio_neur=en / eT_neur))
    json.dump(res, open(out / "intervene_energy.json", "w"), indent=1)
elif a.stage == "heval":
    NG, TG = load_groups(); res = {}
    for spec in a.states.split(","):
        tag, p = spec.split("="); set_w(load_w(p)); r = first_pos_probs(TG); h = humaneval_eval()
        res[tag] = dict(indent_p=r["indent"]["mean"], return_p=r["return"]["mean"], def_p=r["def"]["mean"], **h); print(tag, {k: round(res[tag][k], 3) for k in ("indent_p", "return_p", "unind", "full", "pass_v2", "pass_v3")}, flush=True)
    json.dump(res, open(out / f"heval_{a.tag}.json", "w"), indent=1)
elif a.stage == "localize":
    """逐矩阵撤回定位：对每个二维参数（含共享嵌入），单独把它撤回到 B_g，测首位置缩进概率的变化；再对变化最大的矩阵做通道级归因（一阶：ΔW ⊙ ∂log p(indent)/∂W 按行/列求和），并与 B_g 的 v 对照。"""
    NG, TG = load_groups(); base = load_w(a.base_state); idx_ind = torch.tensor(TG["indent"], device=dev)
    prefixes = [chat_prefix(r) for r in HE]; im_end = tok.convert_tokens_to_ids("<|im_end|>"); pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def batches(bs=32):
        for i in range(0, len(prefixes), bs):
            b = prefixes[i:i + bs]; Lm = max(len(p) for p in b)
            yield torch.tensor([[pad] * (Lm - len(p)) + p for p in b], device=dev), torch.tensor([[0] * (Lm - len(p)) + [1] * len(p) for p in b], device=dev)
    def last_logits(ids, att):
        h = model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, -1]; return model.lm_head(h).float()
    def indent_p():
        ps = []
        with torch.no_grad():
            for ids, att in batches():
                lg = last_logits(ids, att); ps += torch.softmax(lg, -1)[:, idx_ind].sum(1).tolist()
        return float(np.mean(ps))
    opt = fl.DecompAdamW(named, lr=5e-6, wd=0.01); fl.load_state(Path(a.base_state), model, opt); vB = {n: opt.state[n]["v"].detach().float().cpu() for n, _ in named}; del opt
    res = {}
    for spec in a.states.split(","):
        tag, p = spec.split("="); S = load_w(p); set_w(S); p0 = indent_p(); per = {}
        for n, prm in named:
            if prm.dim() != 2: continue
            with torch.no_grad(): prm.copy_(base[n].to(dev))
            per[n] = indent_p() - p0
            with torch.no_grad(): prm.copy_(S[n].to(dev))
        top = sorted(per.items(), key=lambda kv: -abs(kv[1]))[:8]; print(tag, "p0=%.3f" % p0, [(k.replace("model.layers.", "L").replace(".weight", ""), round(v, 4)) for k, v in top], flush=True)
        # 累计撤回：按 |Δ| 从大到小累计撤回前 k 个矩阵
        cum = []; set_w(S)
        for k, (n, _) in enumerate(top[:8], 1):
            with torch.no_grad(): dict(named)[n].copy_(base[n].to(dev))
            cum.append(indent_p())
        set_w(S)
        # 通道级一阶归因（前 4 个矩阵）
        chan = {}
        for n, _ in top[:4]:
            prm = dict(named)[n]; prm.requires_grad_(True); model.zero_grad(set_to_none=True); tot = 0.0
            for ids, att in batches(8):
                lg = last_logits(ids, att); lp = torch.log(torch.softmax(lg, -1)[:, idx_ind].sum(1) + 1e-12).sum(); lp.backward(); tot += float(lp); del lg, lp; torch.cuda.empty_cache()
            g = prm.grad.detach().float().cpu(); prm.requires_grad_(False); dW = (S[n] - base[n]).float(); contrib = (dW * g)
            row = contrib.sum(1).numpy(); col = contrib.sum(0).numpy(); vr = vB[n].median(1).values.numpy(); vc = vB[n].median(0).values.numpy()
            def rank_rel(c, v):
                k = max(1, int(0.05 * len(c))); top_i = np.argsort(c)[:k]   # 对缩进概率贡献最负（把缩进压得最狠）的 5%
                return dict(top5_contrib_share=float(c[top_i].sum() / (c.sum() + 1e-30)), v_rel_median_top5=float(np.median(v[top_i]) / (np.median(v) + 1e-30)), frac_floor_top5=float(np.mean(v[top_i] <= fl.V_FLOOR * 10)), frac_floor_all=float(np.mean(v <= fl.V_FLOOR * 10)), rank_rho_contrib_v=float(np.corrcoef(np.argsort(np.argsort(c)), np.argsort(np.argsort(v)))[0, 1]))
            ent = {"row": rank_rel(row, vr), "col": rank_rel(col, vc), "total_first_order": float(contrib.sum())}
            if n == EMB:
                top_rows = np.argsort(row)[:30]; ent["top_rows_tokens"] = [tok.decode([int(i)]) for i in top_rows]; ent["indent_rows_contrib_share"] = float(row[TG["indent"]].sum() / (row.sum() + 1e-30))
            elif ".mlp." in n:
                L_i = int(n.split(".")[2]); key = "row" if ("gate" in n or "up_proj" in n) else "col"; c = row if key == "row" else col; k = max(1, int(0.05 * len(c))); top_i = set(np.argsort(c)[:k].tolist())
                ent["overlap_code_only"] = len(top_i & set(NG["code_only"][L_i])) / k; ent["overlap_both_used"] = len(top_i & set(NG["both_used"][L_i])) / k
            chan[n] = ent; print("  ", n.replace("model.layers.", "L"), json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ent.items() if k in ("row", "col", "overlap_code_only", "overlap_both_used", "indent_rows_contrib_share")}, ensure_ascii=False), flush=True)
            model.zero_grad(set_to_none=True)
        res[tag] = dict(p0=p0, per_matrix=per, top=top, cumulative=cum, channel=chan)
    json.dump(res, open(out / "localize.json", "w"), indent=1, ensure_ascii=False)
elif a.stage == "offline":
    NG, TG = load_groups(); opt = fl.DecompAdamW(named, lr=5e-6, wd=0.01); fl.load_state(Path(a.base_state), model, opt); t_old = opt.t
    rows = fl.prompt_completion_rows(str(Path(a.data) / "gsm8k" / "train.jsonl"), tok, a.max_len); exs = fl.tokenize_train_manual(rows, tok, a.max_len); steps = fl.epoch_order(len(exs), 1, seed=17 + 99991)[: a.nb]
    W = [(n, p) for n, p in named if (".mlp." in n and p.dim() == 2) or n == EMB]; vT = {n: opt.state[n]["v"].detach().float().cpu().clone() for n, _ in W}; grads = []
    for st in steps:
        model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [exs[i] for i in st]); opt.clip_(1.0); grads.append({n: p.grad.detach().float().cpu().clone() for n, p in W})
    model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); res = {}
    gen_ = torch.Generator(device="cpu").manual_seed(0)
    for ver in ("orig", "rowcol", "cperm", "const", "fresh"):
        V = {n: (vT[n].clone() if ver == "orig" else torch.zeros_like(vT[n]) if ver == "fresh" else fl.v_ladder(vT[n], ver, gen=gen_)) for n, _ in W}
        t = 1 if ver == "fresh" else t_old + 1; bc2 = 1 - fl.BETA2 ** t; U = {n: torch.zeros_like(vT[n], dtype=torch.float64) for n, _ in W}
        for g in grads:
            for n, _ in W:
                vh = ((fl.BETA2 * V[n] + (1 - fl.BETA2) * g[n] ** 2) / bc2) if ver != "fresh" else (g[n] ** 2 * (1 - fl.BETA2)) / bc2
                u = g[n] / (vh.sqrt() + fl.EPS); U[n] += (u.double() ** 2)   # 逐批能量累加（份额口径与 v3 一致）
        dW = {n: U[n].sqrt() for n in U}; dW.update({n: torch.zeros(1) for n in names if n not in U})
        r = {}
        tot_ffn = sum(float(U[n].sum()) for n in U if ".mlp." in n); n_ffn = sum(U[n].numel() for n in U if ".mlp." in n)
        for g, sel in NG.items():
            e = 0; c = 0
            for i in range(L):
                idx = torch.tensor(sel[i])
                for kind, dim in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
                    s = U[f"model.layers.{i}.mlp.{kind}.weight"].index_select(dim, idx); e += float(s.sum()); c += s.numel()
            r[f"ffn_{g}"] = (e / tot_ffn) / (c / n_ffn)
        rows_e = U[EMB].sum(1).numpy(); tot = rows_e.sum()
        for g, idx in TG.items():
            if len(idx): r[f"vocab_{g}"] = float(rows_e[idx].sum() / tot) / (len(idx) / len(rows_e))
        res[ver] = r; print(ver, {k: round(v, 2) for k, v in r.items()}, flush=True)
    json.dump(res, open(out / "offline_enrich.json", "w"), indent=1)
