#!/usr/bin/env python3
"""阶段 2（卡上 CPU）：文档 24 的 0.5B 模型（Qwen2.5-0.5B-Instruct）状态 W0 / A / B / contA 的四层端点对比，口径同文档 25 v2。
对：W0→A（GSM8K 216 步）、A→B（MBPP 216 步，保留状态）、A→contA（GSM8K 再 216 步，保留状态，同起点对照）。
另：B 对 contA 的残差通道对应（同起点、换任务 / 同任务）；近零集合更替率（A 的近零元素在 B / contA 中仍近零的比例，对随机更替）。
矩阵：24 层 × 7 种 + 词表（embed_tokens，Qwen2.5-0.5B 共享输出头）。线程 ≤ 8（卡上 CPU 配额 16 核）。
用法：python phase2_card.py  （布局 /root/autodl-tmp/fork；输出 diag25/）"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(8)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--phase3", default="", help="阶段 3 轨迹目录（含 A/B/C/state/weights.pt）"); ap.add_argument("--out", default=""); ap.add_argument("--device", default="cpu", help="cpu = numpy 后端；cuda = torch 后端"); ap.add_argument("--ab_only", action="store_true", help="10-09 H 轮：只算 A→B 一对（不需要 W0、C），并记录平衡分解收敛")
args = ap.parse_args()
if args.device == "cuda":
    import pairlib_t as pl; pl.DEVICE = "cuda"
else:
    import pairlib as pl
W = Path(_OH_RUNS); OUT = Path(args.out) if args.out else W / "diag25"; OUT.mkdir(parents=True, exist_ok=True)
if args.phase3:
    T = Path(args.phase3)
    SRC = {"W0": W / "runs/W0/weights.pt", "A": T / "A/state/weights.pt", "B": T / "B/state/weights.pt", "contA": T / "C/state/weights.pt"}
else:
    SRC = {"W0": W / "runs/W0/weights.pt", "A": W / "runs/A/state/weights.pt", "B": W / "runs/B/state/weights.pt", "contA": W / "runsAB/contA/snaps/step0216/weights.pt"}
KINDS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"]


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return {k: v.double().numpy() for k, v in d.items() if v.ndim == 2 and ("self_attn" in k or "mlp" in k or "embed_tokens" in k)}


def kind_of(n):
    for k in KINDS:
        if k in n: return k
    return "other"


PAIRS = [("A_B", "A", "B")] if args.ab_only else [("W0_A", "W0", "A"), ("A_B", "A", "B"), ("A_contA", "A", "contA")]
cache = {}


def get(k):
    if k not in cache:
        if len(cache) > 2: cache.pop(next(iter(cache)))
        cache[k] = load(SRC[k])
    return cache[k]


for tag, a, b in PAIRS:
    jp = OUT / f"{tag}.json"
    if jp.exists(): continue
    t0 = time.time(); WA, WB = get(a), get(b); rng = np.random.default_rng(0); res, fields = {}, {}
    for n in sorted(WA):
        big = WA[n].shape[0] > 20000
        if hasattr(pl, "CONV"): pl.CONV.clear()
        if args.device == "cuda":   # GPU 后端：词表矩阵也做完整四层（Gram 路径求子空间），打乱零值 1 次；显存不足时该矩阵退回 CPU torch
            try:
                r, f = pl.analyse_pair(WA[n], WB[n], rng=rng, ranks=(8, 32, 128), with_spectrum=True, shuffle_reps=1 if big else 5)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache(); pl.DEVICE = "cpu"; torch.set_num_threads(12); print("OOM -> cpu for", n, flush=True)
                r, f = pl.analyse_pair(WA[n], WB[n], rng=rng, ranks=(8, 32, 128), with_spectrum=True, shuffle_reps=1 if big else 5)
                pl.DEVICE = "cuda"; torch.set_num_threads(8)
        else:
            r, f = pl.analyse_pair(WA[n], WB[n], rng=rng, ranks=(8, 32, 128), with_spectrum=not big)
            if big:  # 词表矩阵：谱与子空间用列方向（896 维）的协方差路径
                dZ, dW = pl.delta_z_ref(WA[n], WB[n], *pl.decompose(WA[n])[:3])
                r["L3"]["spec_dW"] = pl.spectrum(dW, rng); r["L3"]["spec_dZ"] = pl.spectrum(dZ, rng)
        res[n] = pl.to_jsonable(r)
        if hasattr(pl, "CONV"): res[n]["decomp_conv"] = list(pl.CONV); pl.CONV.clear()
        for k, v in f.items(): fields[f"{n}|{k}"] = v.astype(np.float32)
        print(tag, n, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(dict(group=tag, A=a, B=b, per_matrix=res), open(jp, "w"), indent=0)
    np.savez_compressed(OUT / f"{tag}_fields.npz", **fields)

if args.ab_only:
    print("done"); sys.exit(0)

# ---------------------------------------------------------------- 通道对应：A→B 对 A→contA（同起点）；W0→A 对 A→B（相邻段）
rng = np.random.default_rng(5)


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


cc = {}
for lab, t1, t2 in [("A→B 对 A→contA（同起点，换任务/同任务）", "A_B", "A_contA"), ("W0→A 对 A→B（相邻段）", "W0_A", "A_B"), ("W0→A 对 A→contA", "W0_A", "A_contA")]:
    F1, F2 = np.load(OUT / f"{t1}_fields.npz"), np.load(OUT / f"{t2}_fields.npz"); acc = {}
    for n in sorted({k.split("|")[0] for k in F1.files}):
        for fld, key in (("eZ行残差", "eqstep_resid_Zrow"), ("θ行残差", "theta_resid_row"), ("eZ列残差", "eqstep_resid_Zcol")):
            a, b = F1[f"{n}|{key}"], F2[f"{n}|{key}"]
            null = [spear(a, rng.permutation(b)) for _ in range(10)]
            acc.setdefault((kind_of(n), fld), []).append(dict(rho=spear(a, b), null95=float(np.percentile(np.abs(null), 95)), layer=n))
    cc[lab] = {f"{k}|{fld}": dict(rho_med=float(np.median([d["rho"] for d in v])), rho_min=float(min(d["rho"] for d in v)), rho_max=float(max(d["rho"] for d in v)),
                                 null95=float(np.median([d["null95"] for d in v])), per_layer=[(d["layer"], d["rho"]) for d in v]) for (k, fld), v in acc.items()}
json.dump(cc, open(OUT / "channel_corr.json", "w"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- 近零集合更替率
nz = {}
WA = get("A")
for other in ("B", "contA"):
    WB = get(other); acc = {}
    for n in sorted(WA):
        r = np.sqrt((WA[n] ** 2).mean()); out = {}
        for t in (1e-3, 1e-2, 1e-1):
            sa, sb = np.abs(WA[n]) < t * r, np.abs(WB[n]) < t * r
            pa, pb = sa.mean(), sb.mean(); keep = (sa & sb).sum() / max(sa.sum(), 1)
            out[f"t{t:g}"] = dict(frac_A=float(pa), frac_B=float(pb), keep=float(keep), keep_random=float(pb))  # 随机更替下 keep ≈ pb
        acc.setdefault(kind_of(n), []).append(out)
    nz[f"A→{other}"] = {k: {t: {q: float(np.median([d[t][q] for d in v])) for q in v[0][t]} for t in v[0]} for k, v in acc.items()}
json.dump(nz, open(OUT / "nearzero_turnover.json", "w"), indent=1, ensure_ascii=False)
print("done")
