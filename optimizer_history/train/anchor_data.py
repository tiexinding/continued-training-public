#!/usr/bin/env python3
"""F10 包容式学习（自锚定回放）：用起点状态自己在"其他情形"上的贪心输出做回放样本（不引入新标签），情形三类：
  mbpp_full   MBPP 训练题（原题干，含断言）→ 起点自己的完整函数输出
  func_cont   由 MBPP 训练题构造的"续写函数体"情形：题干 = 解答首行 def 签名 + 任务描述作文档字符串，要求补全 → 起点自己的续写
  dolly       Dolly 训练题 → 起点自己的回答
输出 jsonl（messages 格式），再由 replay_data.py 按 token 预算混入。"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, re, sys, hashlib
from pathlib import Path
import torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--state", required=True); ap.add_argument("--out", required=True); ap.add_argument("--data", default=_OH_SUBSETS)
ap.add_argument("--n_mbpp", type=int, default=100); ap.add_argument("--n_cont", type=int, default=80); ap.add_argument("--n_dolly", type=int, default=100)
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true")
a = ap.parse_args(); model, tok = load_model(a); model.eval(); named = fl.named_params(model); dev = next(model.parameters()).device
sd = torch.load(Path(a.state) / "weights.pt", map_location="cpu")
with torch.no_grad():
    for n, p in named: p.copy_(sd[n].to(p.device))
im_end = tok.convert_tokens_to_ids("<|im_end|>"); eos = [t for t in {tok.eos_token_id, im_end} if t is not None]; pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
def gen(prompts, max_new=400, bs=16):
    outs = []
    for i in range(0, len(prompts), bs):
        b = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True) for p in prompts[i:i + bs]]; Lm = max(len(x) for x in b)
        ids = torch.tensor([[pad] * (Lm - len(x)) + x for x in b], device=dev); att = torch.tensor([[0] * (Lm - len(x)) + [1] * len(x) for x in b], device=dev)
        o = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        outs += [tok.decode([t for t in row if t not in eos and t != pad], skip_special_tokens=True) for row in o[:, Lm:].tolist()]
    return outs
mb = [json.loads(l) for l in open(Path(a.data) / "mbpp" / "train.jsonl")]; dl = [json.loads(l) for l in open(Path(a.data) / "dolly_qa" / "train.jsonl")]
rows = []
# 1 mbpp_full
P = [r["messages"][0]["content"] for r in mb[: a.n_mbpp]]; O = gen(P)
rows += [{"example_id": hashlib.md5(("mf" + p).encode()).hexdigest()[:16], "messages": [{"role": "user", "content": p}, {"role": "assistant", "content": o}], "metadata": {"anchor": "mbpp_full"}} for p, o in zip(P, O) if o.strip()]
# 2 func_cont：从 MBPP 解答取 def 首行，任务描述做文档字符串
P2 = []
for r in mb[a.n_mbpp: a.n_mbpp + 3 * a.n_cont]:
    sol = r["messages"][1]["content"].replace("\r\n", "\n"); m = re.search(r"^(def .+?:)\s*$", sol, re.M)
    if not m: continue
    desc = r["messages"][0]["content"].split("\n\nTests:")[0].strip()
    P2.append("Complete the following Python function.\n\n" + m.group(1) + '\n    """' + desc + '"""\n')
    if len(P2) >= a.n_cont: break
O2 = gen(P2)
iscode = lambda o: ("def " in o) or ("return" in o) or any(l.startswith("    ") for l in o.split("\n"))
rows += [{"example_id": hashlib.md5(("fc" + p).encode()).hexdigest()[:16], "messages": [{"role": "user", "content": p}, {"role": "assistant", "content": o}], "metadata": {"anchor": "func_cont"}} for p, o in zip(P2, O2) if o.strip() and iscode(o)]
# 3 dolly
P3 = [r["messages"][0]["content"] for r in dl[: a.n_dolly]]; O3 = gen(P3, max_new=200)
rows += [{"example_id": hashlib.md5(("dl" + p).encode()).hexdigest()[:16], "messages": [{"role": "user", "content": p}, {"role": "assistant", "content": o}], "metadata": {"anchor": "dolly"}} for p, o in zip(P3, O3) if o.strip()]
with open(a.out, "w") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
import collections; print("anchor rows", len(rows), collections.Counter(r["metadata"]["anchor"] for r in rows)); print("func_cont 样例:", repr(O2[0][:160]) if O2 else None)
