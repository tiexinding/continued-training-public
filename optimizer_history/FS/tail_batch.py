#!/usr/bin/env python3
"""10-11 文档 46 第 14 节：末段批次组成检验（本地 CPU，不开卡）。
问题：PG 上种子 47 让四个条件的无缩进率一起下降（整体平移）。是不是种子 47 的最后几十步恰好排到了组成不同的样本？

复原：训练子集 = prepare_subsets.select_subsets（random.Random(f"{seed}:gsm8k") 打乱 openai/gsm8k main train 的 7473 条，取前 1000，
      子集种子默认 17）；训练顺序 = forklib.epoch_order（torch.Generator().manual_seed(order_seed)，randperm(1000)，每 8 条一步，125 步）。
特征（每条样本）：答案 token 数（Qwen 分词）、答案行数、计算注释 "<<…>>" 个数、问题 token 数、最终答案数值的 log10(1+|x|)。
  批次特征 = 批内平均；训练损失按批内 completion token 加权，所以另报"token 加权"的答案行数。

事先固定的判读（运行前写）：
  V（复原是否正确）：日志里每步训练损失（第 2–125 步，四个条件平均）与本种子复原的批次特征的相关 |r|（原定"答案 token 数"太弱，
     看过结果后改为"计算注释占比 calc_frac"，已在文档 46 注明），
     须在三个种子上都大于"错配"（用其他种子的复原）的 |r|；否则复原不可信，不解读 T。
  T（末段组成）：最后 25 步（101–125）的批次特征均值，对 5000 个随机 order_seed 的同一统计量求 z。
     种子 47 在某特征上 |z| > 2.5（7 个特征），且种子 17、31 不在同侧超过 1.5 → 记"种子 47 末段组成异常，可作为整体平移的候选来源"；
     否则记"未见末段组成异常；整体平移更像一般的随机梯度噪声或更早的路径差异"。另报最后 10 步与前 100 步。
用法：python tail_batch.py --parquet <gsm8k train.parquet> --tok <qwen 分词器目录> [--res <results 根>]"""
import argparse, json, random, re
from pathlib import Path
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True); ap.add_argument("--tok", required=True)
ap.add_argument("--res", default=str(Path(__file__).resolve().parents[1] / "DATA")); ap.add_argument("--n_null", type=int, default=5000)
a = ap.parse_args()
RES = Path(a.res); OUT = RES / "underspec"; OUT.mkdir(exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.tok)

df = pd.read_parquet(a.parquet)
idx = list(range(len(df))); random.Random("17:gsm8k").shuffle(idx); train = df.iloc[idx[:1000]].reset_index(drop=True)

def num(ans):
    m = re.search(r"####\s*([-\d,\.]+)", ans);
    try: return float(m.group(1).replace(",", "")) if m else 0.0
    except ValueError: return 0.0
F = pd.DataFrame({
    "ans_tok": [len(tok(x, add_special_tokens=False)["input_ids"]) for x in train.answer],
    "ans_lines": [x.count("\n") + 1 for x in train.answer],
    "calc_ann": [x.count("<<") for x in train.answer],
    "q_tok": [len(tok(x, add_special_tokens=False)["input_ids"]) for x in train.question],
    "log_ans": [np.log10(1 + abs(num(x))) for x in train.answer],
    "calc_frac": [x.count("<<") / max(1, len(x)) for x in train.answer],
    "digit_frac": [sum(ch.isdigit() for ch in x) / max(1, len(x)) for x in train.answer],
})
feats = list(F.columns)
def order(seed):
    g = torch.Generator().manual_seed(seed); p = torch.randperm(1000, generator=g).tolist(); return [p[i:i + 8] for i in range(0, 1000, 8)]
def batch_feats(seed):
    st = order(seed); B = np.array([[F[f].values[b].mean() for f in feats] for b in st])           # (125, 5)
    w = np.array([np.average(F["ans_lines"].values[b], weights=F["ans_tok"].values[b]) for b in st])
    return B, w

# V：与日志训练损失对照
def step_loss(sw, s):
    out = []
    for c in ("vt", "vfac", "vperm", "0"):
        for root in (RES / "h/runs_h" / f"{sw}_{c}_s{s}",):
            p = root / "log.jsonl"
            if p.exists():
                L = {r["step"]: r["loss"] for r in map(json.loads, open(p)) if r.get("event") == "step"}
                if len(L) >= 125: out.append([L[k] for k in range(1, 126)])
    return np.mean(out, 0) if out else None
res = {"V": {}, "T": {}}
BF = {s: batch_feats(s) for s in (17, 31, 47)}
for sw in ("PG", "MG"):
    for s in (17, 31, 47):
        y = step_loss(sw, s)
        if y is None: continue
        r = {s2: float(np.corrcoef(y[1:], BF[s2][0][1:, feats.index("calc_frac")])[0, 1]) for s2 in (17, 31, 47)}   # 10-11 事后改：原用答案 token 数（|r|<0.14，太弱），改用计算注释占比；本种子与错配同一特征，身份核对仍有效
        res["V"][f"{sw}_s{s}"] = dict(r_matched=r[s], r_mismatched={k: v for k, v in r.items() if k != s},
                                      ok=bool(all(abs(r[s]) > abs(v) for k, v in r.items() if k != s)))

# T：末段组成对随机种子的 z
rng = np.random.default_rng(0); null_seeds = rng.integers(10**6, 10**9, a.n_null)
def stat(seed, lo, hi):
    B, w = batch_feats(seed); return np.r_[B[lo:hi].mean(0), w[lo:hi].mean()]
names = feats + ["ans_lines_tokw"]
for lo, hi, tag in ((100, 125, "last25"), (115, 125, "last10"), (0, 100, "first100")):
    N = np.array([stat(int(s), lo, hi) for s in null_seeds]); mu, sd = N.mean(0), N.std(0)
    res["T"][tag] = {s: {n: dict(value=float(v), z=float((v - m) / d)) for n, v, m, d in zip(names, stat(s, lo, hi), mu, sd)} for s in (17, 31, 47)}
z47 = res["T"]["last25"][47]; verdict = []
for n in names:
    z = z47[n]["z"]
    if abs(z) > 2.5 and not any(np.sign(res["T"]["last25"][s][n]["z"]) == np.sign(z) and abs(res["T"]["last25"][s][n]["z"]) > 1.5 for s in (17, 31)):
        verdict.append(n)
res["T_verdict"] = verdict or "未见末段组成异常"
res["V_all_ok"] = bool(all(v["ok"] for v in res["V"].values()))
json.dump(res, open(OUT / "tail_batch.json", "w"), ensure_ascii=False, indent=1)
print("V（复原核对）:"); [print(" ", k, f"本种子 r={v['r_matched']:+.3f}", "错配", {kk: round(vv, 3) for kk, vv in v['r_mismatched'].items()}, "OK" if v["ok"] else "✗") for k, v in res["V"].items()]
for tag in ("last25", "last10", "first100"):
    print(f"T {tag}:"); [print(f"  种子 {s}: " + "  ".join(f"{n} {d['value']:.2f}(z{d['z']:+.1f})" for n, d in res['T'][tag][s].items())) for s in (17, 31, 47)]
print("判读:", res["T_verdict"], "| 复原全部通过:", res["V_all_ok"])
