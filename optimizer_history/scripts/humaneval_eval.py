#!/usr/bin/env python3
"""W3-1b：HumanEval 164 题 pass@1（贪心，补全函数体），作为未参与任何选择的独立代码评估集。
提示 = 聊天模板中的 user 消息："Complete the following Python function.\n\n" + prompt；生成 ≤ 400 token；取 ```python``` 块或原文；
程序 = prompt + 生成（若生成含 def 则直接用生成）+ test + check(entry_point)；子进程 10 s。
用法：python humaneval_eval.py --out diag25/b3/humaneval.json --states W0=base M=... """
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
import torch, pandas as pd
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--states", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--parquet", default="/tmp/humaneval.parquet")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true")
a = ap.parse_args(); df = pd.read_parquet(a.parquet)
model, tok = load_model(a); model.eval(); named = fl.named_params(model); base = {n: p.detach().cpu().clone() for n, p in named}
im_end = tok.convert_tokens_to_ids("<|im_end|>"); eos = [t for t in {tok.eos_token_id, im_end} if t is not None]; pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
prompts = [tok.apply_chat_template([{"role": "user", "content": "Complete the following Python function. Return the full function in a ```python code block.\n\n" + r.prompt}], tokenize=True, add_generation_prompt=True) for r in df.itertuples()]
@torch.no_grad()
def gen(prefixes, max_new=400, bs=16):
    dev = next(model.parameters()).device; outs = []
    for i in range(0, len(prefixes), bs):
        batch = prefixes[i:i + bs]; L = max(len(p) for p in batch)
        ids = torch.tensor([[pad] * (L - len(p)) + p for p in batch], device=dev); att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in batch], device=dev)
        o = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        outs += [tok.decode([t for t in row if t not in eos and t != pad], skip_special_tokens=True) for row in o[:, L:].tolist()]
    return outs
def run(code, r):
    """v2（10-06）：生成体若已含 def entry_point，则在其前面补上题干中的 import 行（v1 漏补 → 微调态输出不带 from typing import List 时 NameError，造成假阴性）；返回 (通过, 失败原因尾部)。"""
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S); body = m.group(1) if m else code
    header = "\n".join(l for l in r.prompt.split("\n") if l.startswith("import ") or l.startswith("from ")) + "\nfrom typing import *\nimport math\n"
    prog = (header + body if ("def " + r.entry_point) in body else r.prompt + body) + "\n\n" + r.test + f"\n\ncheck({r.entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); fn = f.name
    try:
        pr = subprocess.run([sys.executable, fn], capture_output=True, timeout=10); return pr.returncode == 0, pr.stderr.decode(errors="ignore")[-200:]
    except subprocess.TimeoutExpired: return False, "timeout"
    finally: Path(fn).unlink(missing_ok=True)
res = {}
for spec in a.states:
    name, path = spec.split("=", 1); sd = base if path == "base" else torch.load(Path(path) / "weights.pt", map_location="cpu")
    with torch.no_grad():
        for n, p in named: p.copy_(sd[n].to(p.device))
    outs = gen(prompts); rr = [run(o, r) for o, r in zip(outs, df.itertuples())]; passed = [x[0] for x in rr]
    res[name] = dict(pass_rate=sum(passed) / len(passed), passed=passed, outs=outs, errors=[x[1] for x in rr]); print(name, f"HumanEval pass@1 {res[name]['pass_rate']:.3f}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False); print("done")
