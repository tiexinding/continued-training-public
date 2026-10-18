"""10-12 评审（七）：统计 105 个缩进 token 在 GSM8K 训练监督目标中的出现次数（训练同口径分词与掩码）。
用法：python count_indent_targets.py <forklib 目录> <groups.npz> <gsm8k/train.jsonl>"""
import sys, json, re, collections
sys.path.insert(0, sys.argv[1])
import numpy as np
from transformers import AutoTokenizer
import forklib as fl
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
TG = np.load(sys.argv[2], allow_pickle=True)["token"].item(); IND = set(int(i) for i in TG["indent"])
rows = fl.prompt_completion_rows(sys.argv[3], tok, 512); exs = fl.tokenize_train_manual(rows, tok, 512)
tot = 0; hit = 0; per = collections.Counter(); nex = 0
for e in exs:
    ids = [i for i, m in zip(e.input_ids, e.completion_mask) if m]
    tot += len(ids); h = [i for i in ids if i in IND]; hit += len(h); per.update(h); nex += bool(h)
print(json.dumps(dict(n_examples=len(exs), n_indent_ids=len(IND), supervised_tokens=tot, indent_occurrences=hit, examples_with_indent=nex,
                      top=[(tok.decode([i]), c) for i, c in per.most_common(8)]), ensure_ascii=False))
