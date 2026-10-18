#!/usr/bin/env python3
"""P0 生成审计（评审 10-05 第 2 条）：对指定状态在 GSM8K 留出 200 题上贪心生成，保存全部输出与逐题元信息：
token 数、是否触及生成上限、是否含 ####、抽取到的最后数字、严格/宽松判定、gold。另可指定更长的生成上限以检验截断。
用法（卡上，fork/ 目录）：python gen_audit.py --out diag25/gen_audit --max_new 400 --states W0=base A2e5=runs3/lr2e-5/A/state ...
复用 state_mix_eval 的留出集抽样（seed 20261004，与评估一致）、编码、生成与抽取函数。"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, _OH_TRAIN)
import forklib as fl  # noqa
from fork_continue import load_model  # noqa
from state_mix_eval import heldout_rows, generate, norm_num, HASH, NUM, write_rows, run_mbpp  # noqa
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--states", nargs="+", required=True, help="name=base 或 name=状态目录")
ap.add_argument("--out", required=True); ap.add_argument("--max_new", type=int, default=400); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None); ap.add_argument("--device", default="cuda")
ap.add_argument("--tok_mode", default="manual"); ap.add_argument("--max_len", type=int, default=512); ap.add_argument("--deterministic", action="store_true")
ap.add_argument("--data", default=_OH_SUBSETS); ap.add_argument("--mbpp", action="store_true", help="同时做 MBPP 探针 100 题 pass@1")
args = ap.parse_args()
model, tok = load_model(args); model.eval(); named = fl.named_params(model)
base = {n: p.detach().cpu().clone() for n, p in named}
g, _ = heldout_rows(Path(args.data), args.n)
tmp = Path(tempfile.mkdtemp()); write_rows(g, tmp / "h.jsonl")
pcs = fl.prompt_completion_rows(str(tmp / "h.jsonl"), tok, args.max_len, filter_ctx=False)
exs, rows = [], []
for pc, r in zip(pcs, g):
    e = fl.encode_probe([pc], tok, args.max_len)
    if e: exs.append(e[0]); rows.append(r)
gold = [norm_num(HASH.search(r["messages"][-1]["content"]).group(1)) for r in rows]
mexs, mrows = [], []
if args.mbpp:
    mr = [json.loads(l) for l in open(Path(args.data) / "mbpp" / "probe.jsonl")]
    write_rows(mr, tmp / "m.jsonl"); mp = fl.prompt_completion_rows(str(tmp / "m.jsonl"), tok, args.max_len, filter_ctx=False)
    for pc, r in zip(mp, mr):
        e = fl.encode_probe([pc], tok, args.max_len)
        if e: mexs.append(e[0]); mrows.append(r)
im_end = tok.convert_tokens_to_ids("<|im_end|>")
Path(args.out).mkdir(parents=True, exist_ok=True)
for spec in args.states:
    name, path = spec.split("=", 1)
    sd = base if path == "base" else torch.load(Path(path) / "weights.pt", map_location="cpu")
    with torch.no_grad():
        for n, p in named: p.copy_(sd[n].to(p.device))
    outs = generate(model, tok, exs, args.max_new)
    recs = []
    for r, o, y in zip(rows, outs, gold):
        nt = len(tok(o)["input_ids"]); m = HASH.search(o); ns = NUM.findall(o)
        recs.append(dict(id=r["example_id"], gold=y, out=o, n_tokens=nt, hit_max=nt >= args.max_new - 2, has_hash=bool(m),
                         hash_ans=norm_num(m.group(1)) if m else None, last_num=norm_num(ns[-1]) if ns else None,
                         strict=bool(m and norm_num(m.group(1)) == y), relaxed=bool(ns and norm_num(ns[-1]) == y)))
    fn = Path(args.out) / f"{name}_max{args.max_new}.jsonl"
    with open(fn, "w") as f:
        for x in recs: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    s = sum(x["strict"] for x in recs) / len(recs); rl = sum(x["relaxed"] for x in recs) / len(recs)
    if args.mbpp:
        mo = generate(model, tok, mexs, 512); passed = [run_mbpp(o, r) for o, r in zip(mo, mrows)]
        json.dump(dict(pass_rate=sum(passed) / len(passed), passed=passed, outs=mo), open(Path(args.out) / f"{name}_mbpp.json", "w"), ensure_ascii=False)
        print(name, f"MBPP pass@1 {sum(passed)/len(passed):.3f}", flush=True)
    print(name, f"max_new {args.max_new}: strict {s:.3f} relaxed {rl:.3f} | has#### {sum(x['has_hash'] for x in recs)/len(recs):.2f} | hit_max {sum(x['hit_max'] for x in recs)/len(recs):.2f} | tokens med {sorted(x['n_tokens'] for x in recs)[len(recs)//2]}", flush=True)
print("done")
