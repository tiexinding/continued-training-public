#!/usr/bin/env python3
"""E7：HumanEval 下降原因的判别试验（文档 33 §1b）。对指定状态做四种提示变体的 pass@1：
 base        = 原协议（"Complete the following Python function. Return the full function in a ```python code block."）
 with_tests  = 题干后附 MBPP 式断言（从官方 test 里抽 2 条 assert，改写为 candidate→entry_point）
 restate     = 系统提示：先逐字复述函数的文档字符串与示例，再写完整函数
 terse       = 系统提示：不写文档字符串与注释，只输出函数定义与函数体（给基座用，测"风格"本身的作用）
用法：python humaneval_diag.py --out diag25/e7/humaneval_diag.json --states W0=base M=... --variants base,with_tests,restate,terse"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
import torch, pandas as pd
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
ap = argparse.ArgumentParser(); ap.add_argument("--states", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--parquet", default="/tmp/humaneval.parquet"); ap.add_argument("--variants", default="base,with_tests,restate,terse")
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda"); ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true")
a = ap.parse_args(); df = pd.read_parquet(a.parquet)
model, tok = load_model(a); model.eval(); named = fl.named_params(model); base = {n: p.detach().cpu().clone() for n, p in named}
im_end = tok.convert_tokens_to_ids("<|im_end|>"); eos = [t for t in {tok.eos_token_id, im_end} if t is not None]; pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
def asserts(r, k=2):
    lines = [l.strip() for l in r.test.split("\n") if l.strip().startswith("assert")][:k]
    return "\n".join(l.replace("candidate", r.entry_point) for l in lines)
def prompts(variant):
    out = []
    for r in df.itertuples():
        user = "Complete the following Python function. Return the full function in a ```python code block.\n\n" + r.prompt
        msgs = []
        if variant == "with_tests": user += "\n\nYour code should pass these tests:\n" + asserts(r)
        elif variant == "restate": msgs.append({"role": "system", "content": "Before writing code, restate the function's docstring and examples verbatim inside the function, then write the complete function."})
        elif variant == "terse": msgs.append({"role": "system", "content": "Output only the function definition and body. Do not write a docstring, comments, or explanations."})
        msgs.append({"role": "user", "content": user}); out.append(tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
    return out
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
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S); body = m.group(1) if m else code
    header = "\n".join(l for l in r.prompt.split("\n") if l.startswith("import ") or l.startswith("from ")) + "\nfrom typing import *\nimport math\n"
    prog = (header + body if ("def " + r.entry_point) in body else r.prompt + body) + "\n\n" + r.test + f"\n\ncheck({r.entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); fn = f.name
    try: pr = subprocess.run([sys.executable, fn], capture_output=True, timeout=10); return pr.returncode == 0, pr.stderr.decode(errors="ignore")[-120:]
    except subprocess.TimeoutExpired: return False, "timeout"
    finally: Path(fn).unlink(missing_ok=True)
res = {}
for spec in a.states:
    name, path = spec.split("=", 1); sd = base if path == "base" else torch.load(Path(path) / "weights.pt", map_location="cpu")
    with torch.no_grad():
        for n, p in named: p.copy_(sd[n].to(p.device))
    res[name] = {}
    for v in a.variants.split(","):
        outs = gen(prompts(v)); rr = [run(o, r) for o, r in zip(outs, df.itertuples())]; passed = [x[0] for x in rr]
        res[name][v] = dict(pass_rate=sum(passed) / len(passed), passed=passed, outs=outs, errors=[x[1] for x in rr], docstring_rate=sum('"""' in o for o in outs) / len(outs), med_lines=sorted(len(o.strip().split("\n")) for o in outs)[len(outs) // 2])
        print(name, v, f"pass@1 {res[name][v]['pass_rate']:.3f}  docstring {res[name][v]['docstring_rate']:.2f}  lines {res[name][v]['med_lines']}", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), ensure_ascii=False); print("done")
