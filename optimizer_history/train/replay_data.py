#!/usr/bin/env python3
"""P3 回放数据：新任务训练集中按 token 预算混入旧域样本，总 token 固定为原新任务集的 token 数，旧域占比 r（默认 0.10）；随机抽样种子固定；输出 jsonl 与 meta。
用法：python replay_data.py --new mbpp/train.jsonl --old gsm8k/train.jsonl --out data_e/train_mbpp_replay10.jsonl --ratio 0.10 --tok <tokenizer 路径>"""
import argparse, json, random
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--old", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ratio", type=float, default=0.10); ap.add_argument("--seed", type=int, default=17); ap.add_argument("--tok", default="Qwen/Qwen2.5-0.5B-Instruct")
a = ap.parse_args(); tok = AutoTokenizer.from_pretrained(a.tok); rng = random.Random(a.seed)
def ntok(r): return sum(len(tok(m["content"])["input_ids"]) for m in r["messages"])
new = [json.loads(l) for l in open(a.new)]; old = [json.loads(l) for l in open(a.old)]
T = sum(ntok(r) for r in new); budget_old = a.ratio * T; budget_new = T - budget_old
rng.shuffle(old); keep_old, t = [], 0
for r in old:
    if t >= budget_old: break
    keep_old.append(r); t += ntok(r)
rng.shuffle(new); keep_new, t2 = [], 0
for r in new:
    if t2 >= budget_new: break
    keep_new.append(r); t2 += ntok(r)
mix = keep_new + keep_old; rng.shuffle(mix)
for r in keep_old: r.setdefault("metadata", {})["replay"] = True
with open(a.out, "w") as f:
    for r in mix: f.write(json.dumps(r, ensure_ascii=False) + "\n")
meta = dict(total_tokens_target=T, old_tokens=t, new_tokens=t2, n_old=len(keep_old), n_new=len(keep_new), n_new_dropped=len(new) - len(keep_new), ratio_actual=t / (t + t2))
json.dump(meta, open(a.out.replace(".jsonl", "_meta.json"), "w"), indent=1); print(meta)
