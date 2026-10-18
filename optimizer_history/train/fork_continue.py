#!/usr/bin/env python3
"""文档 22：切换状态的整段训练（补跑状态 A、B）与分叉接续（干预 + 三层记录）。

用法示例
  # 状态 A：从基座训练 GSM8K 整段，保存完整状态
  python fork_continue.py segment --task gsm8k --train data/.../gsm8k/train.jsonl --out runs/stateA --init base
  # 状态 B：从 A 起，全保留方式训练 MBPP 整段
  python fork_continue.py segment --task mbpp --train .../mbpp/train.jsonl --out runs/stateB --init runs/stateA --switch keep
  # 分叉：从 B 起接续 Dolly 50 步，一阶矩注入 s=1
  python fork_continue.py fork --task dolly_qa --train .../dolly_qa/train.jsonl --init runs/stateB --out runs/B-s1 --m_op scale:1 --steps 50

通用：--probe name=path（可多次），--model（模型名或本地目录，或 tiny:<json> 用于冒烟），--tokenizer，--tok_mode trl|manual，
      --device，--deterministic，--order_seed（数据顺序种子）。
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forklib as fl  # noqa: E402


# ----------------------------------------------------------------------------- 参数
def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["segment", "fork"])
    ap.add_argument("--task", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--probe", action="append", default=[], help="name=path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="base", help="base 或状态目录")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--tok_mode", choices=["trl", "manual"], default="trl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--order_seed", type=int, default=17)
    ap.add_argument("--order_file", default="", help="10-11 J 轮：JSON 步序（list[list[int]]，每步样本下标）；给出时取代 epoch_order，order_seed 仍用于校准批次")
    ap.add_argument("--decay_last", type=int, default=0, help="10-11 J 轮：最后 k 步学习率乘 (n_steps-step+1)/(k+1)，线性降向 0；0=不衰减（原行为）")
    ap.add_argument("--max_len", type=int, default=512)
    # 切换处理
    ap.add_argument("--switch", default="keep", help="segment 模式：fresh（全清零）| keep（全保留）")
    ap.add_argument("--m_op", default="zero", help="fork：zero | scale:<s> | signrand")
    ap.add_argument("--v_op", default="keep", help="fork：keep | perm | ladder:<version>[:matched|:matched_med] | reset | clamp:<frac>")
    ap.add_argument("--t_reset", action="store_true", help="fork：步数计数清零（全清零组用，配合 v_op=reset、m_op=zero）")
    ap.add_argument("--match_batches", type=int, default=8)
    # 运行长度与记录
    ap.add_argument("--steps", type=int, default=0, help="fork：接续步数（0 = 整段）")
    ap.add_argument("--full_eval", default="0,10,25,50", help="全探针评估步（另加此后每 25 步）")
    ap.add_argument("--dense", default="1-20,25,30,35,40,45,50", help="小探针评估 + 第①②层记录的步")
    ap.add_argument("--every", type=int, default=25, help="50 步之后的记录间隔")
    ap.add_argument("--small_n", type=int, default=32)
    ap.add_argument("--ledger", action="store_true", help="在记录步计算逐步一阶读数（探针梯度·更新分量）")
    ap.add_argument("--fields", action="store_true", help="记录第①层行列场与 x/δ 能量")
    ap.add_argument("--snap", default="", help="完整权重快照步，如 0,10,25,50")
    ap.add_argument("--full_u_layers", default="", help="10-07：在 dense 步保存这些层 down_proj 的完整更新矩阵 u（逗号分隔层号），用于整向量余弦")
    ap.add_argument("--snap_v", action="store_true", help="快照时另存共享词表矩阵的完整二阶矩（第 6.4 节判据不合格时）")
    ap.add_argument("--identity_last", type=int, default=0, help="segment：最后 N 步记录一阶矩恒等式标量")
    ap.add_argument("--eval_every", type=int, default=25, help="segment：全探针评估间隔")
    ap.add_argument("--upd_energy", action="store_true", help="10-09 H2′：每步记录 down 列的实际更新能量 Σ_i ΔW_ij²（ΔW 含权重衰减，与净变化同口径），存 records/upd_energy.pt")
    ap.add_argument("--upd_stats", action="store_true", help="10-10 I2：每步、每个二维矩阵记录实际更新的统计量（只读），存 records/upd_stats.pt；见 upd_stats_record")
    ap.add_argument("--upd_rows", default="", help="I2：词表行分组 JSON（{组名: [token id…]}），对共享词表矩阵另记各组的更新能量")
    return ap.parse_args()


# ----------------------------------------------------------------------------- I2：实际更新统计（只读）
US_D = ["rms", "med", "p99", "max", "top01_share"]   # 数据项 d = m̂/(√v̂+ε)（不含学习率，第 1 步 lr=0 时也有定义）
US_T = ["rms", "max"]                                  # 实际施加的总更新 ΔW = −lr·d − lr·wd·W（第 1 步为 0）


@torch.no_grad()
def upd_stats_record(named, opt, lr, rows):
    """须在 opt.compute 之后、opt.apply 之前调用。中位数、p99 与 top-0.1% 阈值用跨步采样（与中位数校准相同的
    k = max(1, n // 2e5)），RMS、最大值与能量份额用全部元素。不改任何状态。"""
    bc1, bc2 = 1 - fl.BETA1 ** opt.t, 1 - fl.BETA2 ** opt.t
    D, T, RW = [], [], {}
    for n, p in named:
        if p.dim() != 2: continue
        s = opt.state[n]
        d = (s["m"] / bc1) / ((s["v"].sqrt() / math.sqrt(bc2)).add_(fl.EPS))
        a = d.abs().flatten(); k = max(1, a.numel() // 200_000); smp = a[::k].double()
        e = a.double().pow(2); et = float(e.sum())
        thr = float(torch.quantile(smp, 0.999)) if smp.numel() > 1 else float(a.max())
        D.append([float(e.mean().sqrt()), float(smp.median()), float(torch.quantile(smp, 0.99)) if smp.numel() > 1 else float(a.max()),
                  float(a.max()), float(e[a >= thr].sum()) / max(et, 1e-300)])
        tot = s["_u"] - (lr * opt.wd * p if not fl.is_no_decay(n) else 0.0)
        T.append([float(tot.double().pow(2).mean().sqrt()), float(tot.abs().max())])
        if rows and fl.param_group(n) == "vocab":
            er = d.double().pow(2).sum(1)
            for g, idx in rows.items():
                RW.setdefault(n, {})[g] = [float(er[idx].sum()) / max(et, 1e-300), float(er[idx].mean().sqrt() / max(float(er.mean().sqrt()), 1e-300))]
        del d, a, e
    return torch.tensor(D, dtype=torch.float64), torch.tensor(T, dtype=torch.float64), RW


def parse_steps(spec: str) -> set[int]:
    out = set()
    for part in filter(None, spec.split(",")):
        if "-" in part:
            a, b = map(int, part.split("-")); out |= set(range(a, b + 1))
        else:
            out.add(int(part))
    return out


# ----------------------------------------------------------------------------- 模型
def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.model.startswith("tiny:"):
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
                   max_position_embeddings=1024)
        cfg.update(json.loads(args.model[5:] or "{}"))
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(**cfg))
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.config.use_cache = False
    model.to(args.device).train()
    return model, tok


# ----------------------------------------------------------------------------- 记录工具
class Recorder:
    def __init__(self, out: Path):
        self.out = out; out.mkdir(parents=True, exist_ok=True)
        self.jl = open(out / "log.jsonl", "a", encoding="utf-8")

    def log(self, rec: dict):
        self.jl.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.jl.flush()

    def save(self, name: str, obj):
        d = self.out / "records"; d.mkdir(exist_ok=True)
        torch.save(obj, d / name)


def eval_probes(model, probes, pad_id, which: str, small_n: int) -> dict:
    """which = full | small。返回 {task: {nll, sums, cnts}}；small 用检验用一半中的前 small_n 条。"""
    out = {}
    for name, (exs, sel, val) in probes.items():
        idx = list(range(len(exs))) if which == "full" else val[:small_n]
        s, c = fl.probe_nll(model, [exs[i] for i in idx], pad_id)
        out[name] = {"nll": sum(s) / max(1, sum(c)), "idx": idx, "sums": s, "cnts": c}
    return out


@torch.no_grad()
def field_record(named, opt: fl.DecompAdamW, rec: fl.StepRecord, W0: dict, keep_inj: dict | None) -> dict:
    bc1, bc2 = 1 - fl.BETA1 ** opt.t, 1 - fl.BETA2 ** opt.t
    out = {}
    for n, p in named:
        s = opt.state[n]
        uo = rec.u_old.get(n)
        qs = {"g": p.grad, "mhat": s["m"] / bc1, "vhat": s["v"] / bc2, "u": rec.u_adam.get(n),
              "u_old": uo, "u_new": (rec.u_adam[n] - uo) if uo is not None else None,
              "dW": p.detach() - W0[n].to(p.device, non_blocking=True),
              "u_inj_v": keep_inj.get(n) if keep_inj else None}
        ent = {}
        for k, X in qs.items():
            if X is None:
                continue
            if X.dim() == 2:
                r, c = fl.lnrms_fields(X); ent[k] = (r.cpu(), c.cpu())
            else:
                ent[k] = X.detach().float().cpu()
        if p.dim() == 2:
            ent["v_decomp"] = fl.logv_decomp(s["v"])
        out[n] = ent
    return out


@torch.no_grad()
def scalar_record(named, opt, rec: fl.StepRecord) -> dict:
    bc1 = 1 - fl.BETA1 ** opt.t
    out = {}
    for n, p in named:
        g = p.grad; mh = opt.state[n]["m"] / bc1
        gg = float((g * g).sum())
        e = {"g_norm": gg ** 0.5, "mhat_norm": float(mh.norm()), "id_ratio": float((mh * g).sum()) / max(gg, 1e-30)}
        for k in ("u_adam", "u_old", "u_new"):
            u = (rec.u_adam[n] - rec.u_old[n]) if (k == "u_new" and n in rec.u_old) else getattr(rec, k).get(n)
            if u is not None:
                e[f"{k}_norm"] = float(u.norm())
                e[f"{k}_cos_g"] = float((u * g).sum()) / max(float(u.norm()) * gg ** 0.5, 1e-30)
        out[n] = e
    return out


def ledger_record(model, named, probes, pad_id, rec: fl.StepRecord, small_n: int) -> dict:
    """逐步一阶读数：小探针梯度 · 本步更新分量，按行、按列汇总（不作分支间归因）。"""
    out = {}
    comps = {"u_old": rec.u_old, "u_new": None} if rec.u_old else {"u": rec.u_adam}
    for name, (exs, sel, val) in probes.items():
        fl.probe_grad(model, [exs[i] for i in val[:small_n]], pad_id)
        tab = {}
        with torch.no_grad():
            for n, p in named:
                if p.grad is None: continue
                ent = {}
                for k, U in comps.items():
                    Un = (rec.u_adam[n] - rec.u_old[n]) if U is None else U[n]
                    prod = p.grad * Un
                    ent[k] = (prod.sum(1).cpu(), prod.sum(0).cpu()) if prod.dim() == 2 else float(prod.sum())
                tab[n] = ent
        out[name] = tab
    model.zero_grad(set_to_none=True)
    return out


# ----------------------------------------------------------------------------- 干预
@torch.no_grad()
def apply_switch(args, model, named, opt: fl.DecompAdamW, train_exs, order_seed) -> dict:
    info = {"m_op": args.m_op, "v_op": args.v_op, "t_reset": args.t_reset}
    gen = torch.Generator(device="cpu").manual_seed(20261004)
    # 一阶矩
    m_T = {n: opt.state[n]["m"].clone() for n, _ in named}
    if args.m_op == "zero":
        for n, _ in named: opt.state[n]["m"].zero_()
        opt.m_inj = None
    elif args.m_op.startswith("scale:"):
        s = float(args.m_op.split(":")[1])
        inj = {n: m_T[n] * s for n, _ in named}
        for n, _ in named: opt.state[n]["m"].copy_(inj[n])
        opt.m_inj = inj
    elif args.m_op == "signrand":
        inj = {}
        for n, _ in named:
            sg = (torch.randint(0, 2, m_T[n].shape, generator=gen) * 2 - 1).to(m_T[n].device, m_T[n].dtype)
            inj[n] = m_T[n] * sg
        for n, _ in named: opt.state[n]["m"].copy_(inj[n])
        opt.m_inj = inj
    else:
        raise ValueError(args.m_op)
    del m_T
    opt.k_since = 0
    if opt.m_inj is not None:
        opt.cum_old = {n: torch.zeros_like(p) for n, p in named}
    # 二阶矩
    v_T = {n: opt.state[n]["v"].clone() for n, _ in named}
    vop = args.v_op
    if vop == "keep":
        pass
    elif vop == "reset":
        for n, _ in named: opt.state[n]["v"].zero_()
    elif vop == "perm":
        for n, _ in named:
            flat = v_T[n].flatten(); idx = torch.randperm(flat.numel(), generator=gen).to(flat.device)
            opt.state[n]["v"].copy_(flat[idx].view_as(v_T[n]))
    elif vop.startswith("ladder:"):
        parts = vop.split(":"); ver = parts[1]; matched = len(parts) > 2 and parts[2] in ("matched", "matched_med")
        newv = {n: (fl.v_ladder(v_T[n], ver) if p.dim() == 2 else v_T[n].clone()) for n, p in named}
        if matched:
            c = match_coeffs(model, named, opt, v_T, newv, train_exs, args, robust=(parts[2] == "matched_med"))
            info["match_c"] = {n: float(x) for n, x in c.items()}
            for n in c: newv[n].mul_(c[n] ** 2)
        for n, _ in named: opt.state[n]["v"].copy_(newv[n])
    elif vop.startswith("clamp:"):   # 10-07 F 轮：v ← max(v, frac × 非下限元素中位数)，逐张量；只抬高"几乎没用过"的元素，保留其余排列
        frac = float(vop.split(":")[1]); info["clamp_thr"] = {}
        for n, _ in named:
            v = opt.state[n]["v"]; flat = v.flatten()
            if flat.numel() > 4_000_000: flat = flat[torch.randperm(flat.numel(), generator=gen)[:4_000_000].to(flat.device)]
            nz = flat[flat > fl.V_FLOOR * 10]
            thr = float(frac * nz.float().median()) if nz.numel() > 0 else 0.0
            info["clamp_thr"][n] = thr; v.clamp_(min=thr)
    else:
        raise ValueError(vop)
    if vop == "perm" or vop.startswith("ladder:"):
        opt.v_shadow = v_T     # 影子二阶矩从原始 v_T 出发（全清零组步数计数也清零，影子无对应，不维护）
    else:
        del v_T
    if args.t_reset:
        opt.t = 0
    return info


def match_coeffs(model, named, opt, v_T, newv, train_exs, args, robust=False) -> dict:
    """c = RMS(u_ver)/RMS(u_orig)，u ∝ g/(√v̂+ε)，8 个新任务批次（另一随机顺序）合并计算；不含权重衰减。
    robust=True（10-07，matched_med）：改用逐元素 |u| 中位数之比——置换 v 时 RMS 被少数巨大元素主导（D 轮 R_vperm 的 c 中位 3.1、最大 1056，典型步幅反而小 1/3）。"""
    steps = fl.epoch_order(len(train_exs), 1, seed=args.order_seed + 99991)[: args.match_batches]
    bc2 = 1 - fl.BETA2 ** max(1, opt.t + 1)
    s_o = {n: 0.0 for n, p in named if p.dim() == 2}; s_v = dict(s_o); med_o = {n: [] for n in s_o}; med_v = {n: [] for n in s_o}
    for st in steps:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            fl.micro_losses_backward(model, [train_exs[i] for i in st])
        opt.clip_(args.clip)   # 与训练口径一致：用裁剪后的梯度
        with torch.no_grad():
            for n, p in named:
                if p.dim() != 2: continue
                g = p.grad
                uo = g / ((v_T[n] / bc2).sqrt() + fl.EPS); uv = g / ((newv[n] / bc2).sqrt() + fl.EPS)
                s_o[n] += float((uo ** 2).sum()); s_v[n] += float((uv ** 2).sum())
                if robust:
                    k = max(1, uo.numel() // 200_000); med_o[n].append(float(uo.abs().flatten()[::k].median())); med_v[n].append(float(uv.abs().flatten()[::k].median()))
    model.zero_grad(set_to_none=True)
    if robust:
        return {n: (sum(med_v[n]) / max(sum(med_o[n]), 1e-300)) for n in s_o}
    return {n: (s_v[n] / max(s_o[n], 1e-300)) ** 0.5 for n in s_o}


# ----------------------------------------------------------------------------- 主循环
def main():
    args = parse()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("FORK_THREADS", "4")))
    out = Path(args.out); R = Recorder(out)
    model, tok = load_model(args)
    named = fl.named_params(model)
    opt = fl.DecompAdamW(named, lr=args.lr, wd=args.wd)
    meta_in = {}
    if args.init != "base":
        meta_in = fl.load_state(args.init, model, opt)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # 数据
    rows = fl.prompt_completion_rows(args.train, tok, args.max_len)
    train_exs = fl.tokenize_train_trl(rows, tok, model, args.max_len) if args.tok_mode == "trl" \
        else fl.tokenize_train_manual(rows, tok, args.max_len)
    probes = {}
    for spec in args.probe:
        name, path = spec.split("=", 1)
        prow = fl.prompt_completion_rows(path, tok, args.max_len, filter_ctx=False)
        exs = fl.encode_probe(prow, tok, args.max_len)
        sel, val = fl.split_halves(len(exs))
        probes[name] = (exs, sel, val)
    order = fl.epoch_order(len(train_exs), args.epochs, args.order_seed)
    if args.order_file:
        order = [[int(i) for i in b] for b in json.load(open(args.order_file))]
        assert all(0 <= i < len(train_exs) for b in order for i in b), "order_file 下标越界"
        R_order_sha = __import__("hashlib").sha256(json.dumps(order).encode()).hexdigest()[:16]
    else:
        R_order_sha = ""
    T = len(order)
    R.log({"event": "setup", "mode": args.mode, "task": args.task, "n_train": len(train_exs), "T": T,
           "warmup": fl.warmup_steps(T), "init": args.init, "meta_in": meta_in, "args": vars(args), "order_sha16": R_order_sha})

    # 切换处理
    if args.mode == "segment":
        if args.switch == "fresh":
            for n, _ in named: opt.state[n]["m"].zero_(); opt.state[n]["v"].zero_()
            opt.t = 0
        sw = {"switch": args.switch}
    else:
        sw = apply_switch(args, model, named, opt, train_exs, args.order_seed)
    R.log({"event": "switch", **{k: v for k, v in sw.items() if k != "match_c"}})
    if "match_c" in sw:
        R.save("match_c.pt", sw["match_c"])

    n_steps = T if (args.mode == "segment" or args.steps <= 0) else min(args.steps, T)
    full_at = parse_steps(args.full_eval) | set(range(args.every, n_steps + 1, args.every))
    dense = parse_steps(args.dense) | set(range(50 + args.every, n_steps + 1, args.every))
    snaps = parse_steps(args.snap)
    W0 = {n: p.detach().to("cpu").pin_memory() if p.is_cuda else p.detach().clone() for n, p in named} if args.mode == "fork" else None
    energy = fl.ChannelEnergy(model) if args.fields else None

    def do_eval(step):
        if step in full_at or (args.mode == "segment" and (step % args.eval_every == 0 or step == n_steps)):
            r = eval_probes(model, probes, pad_id, "full", args.small_n)
            R.log({"event": "eval_full", "step": step, **{k: v["nll"] for k, v in r.items()}})
            R.save(f"probe_full_{step:04d}.pt", r)
        elif args.mode == "fork" and step in dense:
            r = eval_probes(model, probes, pad_id, "small", args.small_n)
            R.log({"event": "eval_small", "step": step, **{k: v["nll"] for k, v in r.items()}})
            R.save(f"probe_small_{step:04d}.pt", r)

    def do_snap(step):
        if step in snaps:
            d = out / "snaps" / f"step{step:04d}"; d.mkdir(parents=True, exist_ok=True)
            torch.save({n: p.detach().cpu() for n, p in named}, d / "weights.pt")
            if opt.cum_old is not None:
                torch.save({n: x.cpu() for n, x in opt.cum_old.items()}, d / "cum_old.pt")
            if args.snap_v:
                torch.save({n: opt.state[n]["v"].cpu() for n, p in named if fl.param_group(n) == "vocab"}, d / "v_vocab.pt")

    down_named = [(n, p) for n, p in named if n.endswith("mlp.down_proj.weight")]
    assert all(not fl.is_no_decay(n) for n, _ in down_named)
    ue = []
    us_d, us_t, us_rows = [], [], []
    rows_idx = {}
    if args.upd_rows:
        dev0 = named[0][1].device
        rows_idx = {g: torch.tensor(sorted(set(int(i) for i in v)), dtype=torch.long, device=dev0) for g, v in json.load(open(args.upd_rows)).items()}
    def lr_of(k):   # 10-11 J 轮：末段线性衰减（decay_last=0 时与 fl.lr_at 完全相同）
        lr = fl.lr_at(k, T, args.lr); step = k + 1
        if args.decay_last and step > n_steps - args.decay_last:
            lr *= (n_steps - step + 1) / (args.decay_last + 1)
        return lr

    do_eval(0); do_snap(0)
    t0 = time.time()
    for k in range(n_steps):
        step = k + 1
        rec_on = args.mode == "fork" and step in dense
        id_on = args.mode == "segment" and args.identity_last and step > n_steps - args.identity_last
        lr = lr_of(k)
        if rec_on and energy: energy.start()
        model.zero_grad(set_to_none=True)
        loss = fl.micro_losses_backward(model, [train_exs[i] for i in order[k]])
        ch = energy.stop() if (rec_on and energy) else None
        gnorm, coef = opt.clip_(args.clip)
        keep = rec_on or id_on
        rec = opt.compute(lr, keep=keep)
        rec.loss, rec.grad_norm, rec.clip_coef = loss, gnorm, coef
        inj_v = None
        if rec_on and opt.v_shadow is not None:
            inj_v = {n: rec.u_adam[n] - opt.shadow_update(n, lr) for n, _ in named}
        if rec_on or id_on:
            R.save(f"scalars_{step:04d}.pt", scalar_record(named, opt, rec))
        if rec_on and args.fields:
            R.save(f"fields_{step:04d}.pt", {"fields": field_record(named, opt, rec, W0, inj_v), "energy": ch})
            if args.full_u_layers:
                R.save(f"fullu_{step:04d}.pt", {f"model.layers.{L}.mlp.down_proj.weight": rec.u_adam[f"model.layers.{L}.mlp.down_proj.weight"].detach().cpu().half() for L in [int(x) for x in args.full_u_layers.split(",")]})
        if id_on:   # 一阶矩恒等式：逐行 ⟨m̂,g⟩ 与 ‖g‖²
            with torch.no_grad():
                bc1 = 1 - fl.BETA1 ** opt.t
                R.save(f"identity_{step:04d}.pt", {n: ((opt.state[n]["m"] / bc1 * p.grad).sum(-1).cpu(),
                                                      (p.grad * p.grad).sum(-1).cpu()) for n, p in named})
        if rec_on and args.ledger:
            R.save(f"ledger_{step:04d}.pt", ledger_record(model, named, probes, pad_id, rec, args.small_n))
        if args.upd_energy:   # 只读，不改状态：ΔW = u − lr·wd·W（apply 的两项）
            with torch.no_grad():
                ue.append(torch.stack([(opt.state[n]["_u"] - lr * opt.wd * p).pow(2).sum(0) for n, p in down_named]).cpu())
        if args.upd_stats:   # 只读（I2）
            dd, tt, rw = upd_stats_record(named, opt, lr, rows_idx); us_d.append(dd); us_t.append(tt); us_rows.append(rw)
        opt.apply(lr)
        del rec, inj_v
        R.log({"event": "step", "step": step, "lr": lr, "loss": loss, "grad_norm": gnorm, "clip": coef,
               "t": opt.t, "sec": round(time.time() - t0, 2)})
        do_eval(step); do_snap(step)

    if args.mode == "segment":
        fl.save_state(out / "state", model, opt, {"task": args.task, "T": T, "steps_done": n_steps,
                                                  "order_seed": args.order_seed, "init": args.init})
    if args.upd_energy:
        R.save("upd_energy.pt", {"down_col": torch.stack(ue), "names": [n for n, _ in down_named], "steps": n_steps})
    if args.upd_stats:
        R.save("upd_stats.pt", {"d": torch.stack(us_d), "tot": torch.stack(us_t), "d_stats": US_D, "tot_stats": US_T,
                                "names": [n for n, p in named if p.dim() == 2], "rows": us_rows, "row_stats": ["energy_share", "rms_rel"],
                                "lr": [lr_of(k) for k in range(n_steps)], "steps": n_steps})
    R.log({"event": "done", "sec": round(time.time() - t0, 2)})


if __name__ == "__main__":
    main()
