"""文档 22 阶段 0：切换状态分叉的核心库（数据、探针、手写 AdamW 与更新分解、行列尺度场、状态存取）。

设计要点
- 训练循环与 AdamW 自己实现，以便逐步精确分解更新（旧一阶矩分量 / 新分量 / 权重衰减项）并维护影子状态。
  公式与 torch.optim.AdamW（adamw_torch，非 amsgrad）一致：先 p ← p(1−lr·wd)，再 p ← p − (lr/bc1)·m/(√v/√bc2 + ε)。
- 数据格式化、探针编码与 NLL 口径逐行照搬 Ren 仓库（plasticity_loss_sft.data / probe_confidence_callback），
  保证探针数值与 T7 可比（基座 GSM8K 全探针 0.7506）。
- 训练样本分词有两种：'trl'（云卡上调用 TRL SFTTrainer 自身的预处理，与 T7 完全一致）与 'manual'（本地无 TRL 时的复刻，
  阶段 1 在卡上逐条对照两者）。
- 损失归一化复刻 HF Trainer（梯度累积 8）：每个微批次的 token NLL 之和除以本优化步所有微批次的 completion token 总数。
- 学习率：HF constant_with_warmup，第 k 步（0 起）lr = base·min(1, k/W)，W = ceil(0.05·T)；第 0 步 lr = 0。
"""
from __future__ import annotations

import json, math, os, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
V_FLOOR = 1e-16  # = ε²，log v 加法分解的截断下限（文档 22 第 2.3 节）


# ----------------------------------------------------------------------------- 数据（照搬 Ren data.py）
def _clean_message(m):
    role, content = str(m.get("role", "")).strip(), str(m.get("content", "")).strip()
    if not role or not content:
        raise ValueError(f"Invalid chat message: {m}")
    return {"role": role, "content": content}


def to_messages(row: dict) -> list[dict]:
    if row.get("messages"):
        return [_clean_message(m) for m in row["messages"]]
    for p in ("prompt", "instruction", "question", "input", "text"):
        if row.get(p):
            prompt = row[p]; break
    else:
        raise ValueError(f"cannot convert row: {sorted(row)}")
    for r in ("response", "completion", "answer", "output", "code"):
        if row.get(r):
            return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(row[r])}]
    raise ValueError(f"cannot convert row: {sorted(row)}")


def messages_to_prompt_completion(messages: list[dict], tok) -> dict:
    ai = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
    prompt = tok.apply_chat_template(messages[:ai], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages[: ai + 1], tokenize=False, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise RuntimeError("Chat-template prompt is not a prefix of the full rendered conversation.")
    comp = full[len(prompt):]
    if not comp:
        raise RuntimeError("Rendered assistant completion is empty.")
    return {"prompt": prompt, "completion": comp}


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def prompt_completion_rows(path, tok, max_len=512, filter_ctx=True, min_comp=16) -> list[dict]:
    rows = [messages_to_prompt_completion(to_messages(r), tok) for r in load_jsonl(path)]
    if not filter_ctx:
        return rows
    out = []
    for r in rows:  # Ren filter_prompt_completion_for_context
        pl = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
        cl = len(tok(r["completion"], add_special_tokens=False)["input_ids"])
        if cl >= min_comp and pl <= max_len - min_comp:
            out.append(r)
    return out


@dataclass
class TrainExample:
    input_ids: list[int]
    completion_mask: list[int]


def tokenize_train_manual(rows, tok, max_len=512, add_eos=True) -> list[TrainExample]:
    """TRL 0.2x 'prompt-completion' 预处理的复刻（本地用；卡上以 'trl' 模式为准并逐条对照）。"""
    eos = tok.eos_token
    out = []
    for r in rows:
        comp = r["completion"]
        if add_eos and eos and not comp.endswith(eos):
            comp = comp + eos
        p_ids = tok(r["prompt"])["input_ids"]
        f_ids = tok(r["prompt"] + comp)["input_ids"]
        mask = [0] * len(p_ids) + [1] * (len(f_ids) - len(p_ids))
        out.append(TrainExample(f_ids[:max_len], mask[:max_len]))
    return out


def tokenize_train_trl(rows, tok, model, max_len=512) -> list[TrainExample]:
    """用 TRL SFTTrainer 自身的数据预处理（与 T7 一致）。只构造 trainer 取其 train_dataset，不训练。"""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    args = SFTConfig(output_dir="/tmp/_fork_trl_tok", max_length=max_len, completion_only_loss=True,
                     packing=False, report_to=[], remove_unused_columns=False, per_device_train_batch_size=1)
    tr = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok)
    ds = tr.train_dataset
    out = [TrainExample(list(e["input_ids"]), list(e["completion_mask"])) for e in ds]
    del tr
    return out


# ----------------------------------------------------------------------------- 探针（照搬 Ren probe_confidence_callback 口径）
@dataclass
class ProbeExample:
    input_ids: list[int]
    response_mask: list[int]


def encode_probe(rows, tok, max_len=512) -> list[ProbeExample]:
    out = []
    for r in rows:
        p = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        c = tok(r["completion"], add_special_tokens=False)["input_ids"]
        ids, mask = (p + c)[:max_len], ([0] * len(p) + [1] * len(c))[:max_len]
        if sum(mask) > 0:
            out.append(ProbeExample(ids, mask))
    return out


@torch.no_grad()
def probe_nll(model, examples: list[ProbeExample], pad_id: int, batch_size=8) -> tuple[list[float], list[int]]:
    """逐样本 NLL 之和与 token 数（任务 NLL = Σ和 / Σ数，与 Ren 一致）。"""
    dev = next(model.parameters()).device
    was = model.training; model.eval()
    sums, cnts = [], []
    try:
        for s in range(0, len(examples), batch_size):
            b = examples[s:s + batch_size]; L = max(len(e.input_ids) for e in b)
            ids = torch.full((len(b), L), pad_id, dtype=torch.long, device=dev)
            att = torch.zeros((len(b), L), dtype=torch.long, device=dev)
            rm = torch.zeros((len(b), L), dtype=torch.bool, device=dev)
            for i, e in enumerate(b):
                n = len(e.input_ids)
                ids[i, :n] = torch.tensor(e.input_ids, device=dev); att[i, :n] = 1
                rm[i, :n] = torch.tensor(e.response_mask, dtype=torch.bool, device=dev)
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1, :].float()
            tgt, tm = ids[:, 1:], rm[:, 1:] & att[:, 1:].bool()
            nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
            sums += (nll * tm).sum(1).tolist(); cnts += tm.sum(1).tolist()
    finally:
        if was: model.train()
    return sums, cnts


def probe_grad(model, examples: list[ProbeExample], pad_id: int, batch_size=1) -> None:
    """把"探针 token 平均 NLL"的梯度写入 param.grad（会覆盖原 grad）。"""
    model.zero_grad(set_to_none=True)
    dev = next(model.parameters()).device
    total = sum(sum(e.response_mask[1:]) for e in examples)
    for s in range(0, len(examples), batch_size):
        b = examples[s:s + batch_size]; L = max(len(e.input_ids) for e in b)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long, device=dev)
        att = torch.zeros((len(b), L), dtype=torch.long, device=dev)
        rm = torch.zeros((len(b), L), dtype=torch.bool, device=dev)
        for i, e in enumerate(b):
            n = len(e.input_ids)
            ids[i, :n] = torch.tensor(e.input_ids, device=dev); att[i, :n] = 1
            rm[i, :n] = torch.tensor(e.response_mask, dtype=torch.bool, device=dev)
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1, :].float()
        tm = rm[:, 1:] & att[:, 1:].bool()
        tgt = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view_as(tgt)
        ((nll * tm).sum() / total).backward()


def split_halves(n: int, seed=20261004) -> tuple[list[int], list[int]]:
    """把探针固定对半分：前一半'挑选用'、后一半'检验用'（文档 22 v3+：MBPP 无留出池，统一采用此规则）。"""
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    return sorted(idx[: n // 2]), sorted(idx[n // 2:])


# ----------------------------------------------------------------------------- 训练批次与学习率
def epoch_order(n: int, epochs: int, seed: int) -> list[list[int]]:
    """每轮一个随机排列，按 8 条切成优化步（每轮最后一步可不足 8 条，与 HF 一致）。"""
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).tolist()
        steps += [perm[i:i + 8] for i in range(0, n, 8)]
    return steps


def warmup_steps(total: int, ratio=0.05) -> int:
    return math.ceil(total * ratio)


def lr_at(k: int, total: int, base=1e-4, ratio=0.05) -> float:
    W = warmup_steps(total, ratio)
    return base * (k / max(1, W) if k < W else 1.0)


def micro_losses_backward(model, exs: list[TrainExample]) -> float:
    """一个优化步内的梯度累积：Σ token NLL / Σ completion token 数（HF num_items_in_batch 口径）。返回该步平均损失。"""
    dev = next(model.parameters()).device
    n_tot = sum(sum(e.completion_mask[1:]) for e in exs)
    loss_sum = 0.0
    for e in exs:
        ids = torch.tensor([e.input_ids], device=dev)
        lab = ids.clone(); lab[0, torch.tensor(e.completion_mask, device=dev) == 0] = -100
        logits = model(input_ids=ids).logits[:, :-1, :].float()
        l = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        (l / n_tot).backward(); loss_sum += float(l.detach())
    return loss_sum / max(1, n_tot)


# ----------------------------------------------------------------------------- 参数分组与拓扑
def is_no_decay(name: str) -> bool:
    """HF Trainer：bias 与各类归一化权重不做权重衰减。"""
    n = name.lower()
    return n.endswith("bias") or "norm" in n


def param_group(name: str) -> str:
    n = name.lower()
    if "embed_tokens" in n or "lm_head" in n: return "vocab"
    if "norm" in n: return "norm"
    if "self_attn" in n: return "attn"
    if "mlp" in n: return "ffn"
    return "other"


def named_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """去重（共享词表矩阵只出现一次）。"""
    seen, out = set(), []
    for n, p in model.named_parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); out.append((n, p))
    return out


# ----------------------------------------------------------------------------- 手写 AdamW 与更新分解
@dataclass
class StepRecord:
    lr: float
    clip_coef: float
    grad_norm: float
    loss: float
    u_old: dict = field(default_factory=dict)   # name -> tensor（仅在需要时保留）
    u_new: dict = field(default_factory=dict)
    u_adam: dict = field(default_factory=dict)


class DecompAdamW:
    """与 torch.optim.AdamW 等价的手写实现；支持旧一阶矩分量的精确拆分与影子二阶矩。

    旧一阶矩：切换时 m ← s·m_T（或随机符号版本），记 m_inj = 该注入量；此后第 k 步旧分量为 β1^k·m_inj。
    影子二阶矩：从原始 v_T 出发、用本分支梯度更新；注入部分 = 实际 Adam 更新 − 用影子 v 算的更新。
    """

    def __init__(self, params: list[tuple[str, torch.nn.Parameter]], lr=1e-4, wd=0.01):
        self.params = params; self.wd = wd
        self.state = {n: {"m": torch.zeros_like(p), "v": torch.zeros_like(p)} for n, p in params}
        self.t = 0
        self.m_inj: dict[str, torch.Tensor] | None = None; self.k_since = 0
        self.v_shadow: dict[str, torch.Tensor] | None = None
        self.cum_old: dict[str, torch.Tensor] | None = None   # 累计直接位移（一阶矩干预）

    # 状态存取
    def state_dict(self):
        return {"t": self.t, "state": {n: {"m": s["m"].cpu(), "v": s["v"].cpu()} for n, s in self.state.items()}}

    def load_state_dict(self, sd):
        self.t = int(sd["t"])
        for n, s in self.state.items():
            s["m"].copy_(sd["state"][n]["m"]); s["v"].copy_(sd["state"][n]["v"])

    def clip_(self, max_norm: float) -> tuple[float, float]:
        grads = [p.grad for _, p in self.params if p.grad is not None]
        total = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()
        coef = 1.0
        if max_norm and max_norm > 0:
            coef = min(1.0, max_norm / (total + 1e-6))
            if coef < 1.0:
                for g in grads: g.mul_(coef)
        return total, coef

    @torch.no_grad()
    def compute(self, lr: float, keep: bool) -> StepRecord:
        """推进状态并算出本步更新（尚未施加）。keep=True 时保留分量张量供记录。"""
        self.t += 1
        bc1, bc2 = 1 - BETA1 ** self.t, 1 - BETA2 ** self.t
        rec = StepRecord(lr=lr, clip_coef=1.0, grad_norm=0.0, loss=0.0)
        for n, p in self.params:
            g = p.grad; s = self.state[n]
            s["m"].mul_(BETA1).add_(g, alpha=1 - BETA1)
            s["v"].mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
            denom = (s["v"].sqrt() / math.sqrt(bc2)).add_(EPS)
            u = -(lr / bc1) * s["m"] / denom
            if self.v_shadow is not None:
                vs = self.v_shadow[n]; vs.mul_(BETA2).addcmul_(g, g, value=1 - BETA2)
            if keep:
                rec.u_adam[n] = u
                if self.m_inj is not None:   # 新分量不单独保存（省显存），需要时用 u_adam − u_old 现算
                    rec.u_old[n] = -(lr / bc1) * (BETA1 ** (self.k_since + 1)) * self.m_inj[n] / denom
            if self.cum_old is not None and self.m_inj is not None:
                self.cum_old[n].add_(-(lr / bc1) * (BETA1 ** (self.k_since + 1)) * self.m_inj[n] / denom)
            s["_u"] = u
        self.k_since += 1
        return rec

    @torch.no_grad()
    def apply(self, lr: float) -> None:
        for n, p in self.params:
            if not is_no_decay(n):
                p.mul_(1 - lr * self.wd)
            p.add_(self.state[n].pop("_u"))

    @torch.no_grad()
    def shadow_update(self, n: str, lr: float) -> torch.Tensor:
        """用影子二阶矩算本步 Adam 更新（须在 compute 之后、apply 之前调用）。"""
        bc1, bc2 = 1 - BETA1 ** self.t, 1 - BETA2 ** self.t
        return -(lr / bc1) * self.state[n]["m"] / ((self.v_shadow[n].sqrt() / math.sqrt(bc2)).add_(EPS))


# ----------------------------------------------------------------------------- 行列尺度场与 log v 加法分解
@torch.no_grad()
def lnrms_fields(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """原始 ln RMS（不去中位数）。X: (out, in)。"""
    X2 = X.float().pow(2)   # fp32 足够（均方后取对数）；避免大张量转 fp64 的显存尖峰
    r = 0.5 * torch.log(X2.mean(1).clamp_min(1e-38))
    c = 0.5 * torch.log(X2.mean(0).clamp_min(1e-38))
    return r, c


@torch.no_grad()
def logv_decomp(v: torch.Tensor, floor=V_FLOOR) -> dict:
    """log v = μ + a_i + b_j + e_ij（a、b 中心化）；返回 μ、a、b、残差分位数与截断占比。"""
    L = torch.log(v.float().clamp_min(floor))
    mu = L.mean(dtype=torch.float64).float(); a = L.mean(1) - mu; b = L.mean(0) - mu
    e = L - mu - a[:, None] - b[None, :]
    q = torch.quantile(e.flatten()[:: max(1, e.numel() // 2_000_000)].float(),
                       torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], device=e.device))
    tot = L.var(); r2 = 1 - e.var() / tot if tot > 0 else torch.tensor(float("nan"))
    del L
    return {"mu": float(mu), "a": a.float().cpu(), "b": b.float().cpu(), "e_q": q.cpu(),
            "trunc_frac": float((v < floor).double().mean()), "r2_add": float(r2)}


@torch.no_grad()
def v_ladder(v: torch.Tensor, version: str, floor=V_FLOOR, gen: torch.Generator | None = None) -> torch.Tensor:
    """二阶矩消融阶梯（工具⑭）。只用于二维矩阵；alpha1 恢复原始 v（含被截断元素）。"""
    if version == "orig" or version == "alpha1":
        return v.clone()
    if version == "perm":
        flat = v.flatten(); idx = torch.randperm(flat.numel(), generator=gen).to(v.device)
        return flat[idx].view_as(v).clone()
    if version == "cperm":   # 10-06 新增：通道置换（行再列），保每行/每列内部结构与全局分布，破坏"哪个通道大"
        ri = torch.randperm(v.shape[0], generator=gen).to(v.device); ci = torch.randperm(v.shape[1], generator=gen).to(v.device)
        return v[ri][:, ci].clone()
    if version == "rperm":   # 只置换行（输出通道）
        ri = torch.randperm(v.shape[0], generator=gen).to(v.device); return v[ri].clone()
    if version == "cperm_col":   # 只置换列（输入通道）
        ci = torch.randperm(v.shape[1], generator=gen).to(v.device); return v[:, ci].clone()
    L = torch.log(v.double().clamp_min(floor))
    mu = L.mean(); a = L.mean(1) - mu; b = L.mean(0) - mu
    e = L - mu - a[:, None] - b[None, :]
    if version == "const": out = mu.expand_as(L)
    elif version == "row": out = (mu + a)[:, None].expand_as(L)
    elif version == "col": out = (mu + b)[None, :].expand_as(L)
    elif version == "rowcol": out = mu + a[:, None] + b[None, :]
    elif version.startswith("alpha"):
        al = float(version[5:]); out = mu + a[:, None] + b[None, :] + al * e
    else:
        raise ValueError(version)
    return torch.exp(out).to(v.dtype).contiguous()


# ----------------------------------------------------------------------------- x、δ 通道能量钩子
class ChannelEnergy:
    """线性层输入 x 的列能量（Σx²/数）与输出反向信号 δ 的行能量；只在记录步开启。"""

    def __init__(self, model):
        self.model = model; self.h = []; self.acc = {}

    def start(self):
        self.acc = {}
        for name, mod in self.model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                self.h.append(mod.register_forward_hook(self._fwd(name)))

    def _fwd(self, name):
        def f(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).double()
            a = self.acc.setdefault(name, {"x": 0, "nx": 0, "d": 0, "nd": 0})
            a["x"] = a["x"] + x.pow(2).sum(0); a["nx"] += x.shape[0]
            if out.requires_grad:
                def bh(gr, a=a):
                    d = gr.detach().reshape(-1, gr.shape[-1]).double()
                    a["d"] = a["d"] + d.pow(2).sum(0); a["nd"] += d.shape[0]
                out.register_hook(bh)
        return f

    def stop(self) -> dict:
        for h in self.h: h.remove()
        self.h = []
        out = {}
        for n, a in self.acc.items():
            out[n] = {"x_col_lnrms": (0.5 * torch.log((a["x"] / max(1, a["nx"])).clamp_min(1e-300))).float().cpu(),
                      "delta_row_lnrms": (0.5 * torch.log((a["d"] / max(1, a["nd"])).clamp_min(1e-300))).float().cpu()
                      if isinstance(a["d"], torch.Tensor) else None}
        return out


# ----------------------------------------------------------------------------- 状态存取
def save_state(path, model, opt: DecompAdamW, meta: dict) -> None:
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    torch.save({n: p.detach().cpu() for n, p in named_params(model)}, path / "weights.pt")
    torch.save(opt.state_dict(), path / "optim.pt")
    rng = {"torch": torch.get_rng_state(), "py": random.getstate()}
    if torch.cuda.is_available(): rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, path / "rng.pt")
    (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def load_state(path, model, opt: DecompAdamW) -> dict:
    path = Path(path)
    w = torch.load(path / "weights.pt", map_location="cpu")
    with torch.no_grad():
        for n, p in named_params(model): p.copy_(w[n])
    opt.load_state_dict(torch.load(path / "optim.pt", map_location="cpu"))
    rng = torch.load(path / "rng.pt", weights_only=False)
    torch.set_rng_state(rng["torch"]); random.setstate(rng["py"])
    if "cuda" in rng and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng["cuda"])
    return json.loads((path / "meta.json").read_text())
