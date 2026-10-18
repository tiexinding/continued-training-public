#!/usr/bin/env python3
"""H2（文档 41 §3）：v 替换条件的校准核对。
1. 按 fork_continue.main() 的前序流程（加载、读状态、分词、apply_switch）逐位重算校准系数 c，与运行日志里的 match_c.pt 比较；
2. 在 8 个与校准批次不重叠的新任务批次上，算缩放后（v ← c²·v_替换）替换 v 与真 v 的更新比：逐元素 |u| 中位数比（同 matched_med 口径）与 RMS 比；
   按矩阵种类（q/k/v/o/gate/up/down/embed）和网络主体 / 词表两组报告中位数与 10%/90% 分位数；
3. 93% 口径（附录 A.2）：真 v 在切换时刻（v_T）和新任务第 1 步后（v_1 = β2·v_T + (1−β2)·g²，用第 1 个训练批次）上，
   log v 行列加法模型的逐矩阵 R² 中位数、按元素数加权的合并 R²，以及处于下限 V_FLOOR 的元素占比。
用法（卡上，$W 下）：python calib_check.py --init <起点> --task mbpp --v_op ladder:cperm:matched_med --order_seed 17 --match_c <records/match_c.pt> --out <json>"""
import os as _os
_OH_TRAIN = _os.environ.get("OH_TRAIN", _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "train"))
_OH_SUBSETS = _os.environ.get("OH_SUBSETS", "prepared_subsets"); _OH_RUNS = _os.environ.get("OH_RUNS", "runs")   # 发布包：原卡上路径见 DATA_PROVENANCE §2
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--task", required=True); ap.add_argument("--train", default="")
ap.add_argument("--v_op", required=True); ap.add_argument("--order_seed", type=int, default=17)
ap.add_argument("--match_c", default="", help="运行日志里的 match_c.pt（可缺）"); ap.add_argument("--out", required=True)
ap.add_argument("--n_indep", type=int, default=8); ap.add_argument("--r2", action="store_true", help="同时算第 3 项（与 v_op 无关，每个起点算一次即可）")
ap.add_argument("--applied", action="store_true", help="10-10 I1：在同一批独立批次上算第一个实际步（先注入当前 g²、m 从零累积、按 t_T+1 校正）及两次注入的第 2 步，与真 v 比较")
ap.add_argument("--rows", default="", help="I1：词表行分组 JSON（{组名: [token id…]}），另算各组的能量比")
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--fork_dir", default=_OH_TRAIN); ap.add_argument("--data", default=_OH_SUBSETS)
a = ap.parse_args()
sys.path.insert(0, a.fork_dir)
import fork_continue as fc, forklib as fl

train = a.train or f"{a.data}/{a.task}/train.jsonl"
sys.argv = ["fork_continue.py", "fork", "--task", a.task, "--train", train, "--out", "/tmp/_calib_unused", "--init", a.init,
            "--m_op", "zero", "--v_op", a.v_op, "--order_seed", str(a.order_seed), "--tok_mode", "manual", "--deterministic", "--lr", "5e-6"]
args = fc.parse()
# ---- 与 fork_continue.main() 相同的前序（顺序不可改：cperm 用全局随机数）
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); torch.use_deterministic_algorithms(True)
torch.set_num_threads(int(os.environ.get("FORK_THREADS", "4")))
model, tok = fc.load_model(args); named = fl.named_params(model)
opt = fl.DecompAdamW(named, lr=args.lr, wd=args.wd); meta_in = fl.load_state(args.init, model, opt)
rows = fl.prompt_completion_rows(args.train, tok, args.max_len)
train_exs = fl.tokenize_train_manual(rows, tok, args.max_len)
t_T = opt.t
v_T = {n: opt.state[n]["v"].clone() for n, p in named if p.dim() == 2}
sw = fc.apply_switch(args, model, named, opt, train_exs, args.order_seed)
res = {"init": a.init, "v_op": a.v_op, "order_seed": a.order_seed, "t_T": t_T}

# ---- 1. c 逐位核对
c_new = sw.get("match_c", {})
if a.match_c and Path(a.match_c).exists() and c_new:
    c_old = torch.load(a.match_c, map_location="cpu")
    diff = {n: abs(c_new[n] - float(c_old[n])) for n in c_new}
    res["c_check"] = dict(n=len(diff), n_exact=int(sum(d == 0.0 for d in diff.values())), max_abs_diff=max(diff.values()))
res["c"] = c_new

# ---- 2. 独立批次上的更新比（替换 v 已含 c²；与真 v_T 比较）
cal = fl.epoch_order(len(train_exs), 1, seed=args.order_seed + 99991)[: args.match_batches]
cal_idx = {i for b in cal for i in b}
indep = [b for b in fl.epoch_order(len(train_exs), 1, seed=args.order_seed + 424242) if not (set(b) & cal_idx)][: a.n_indep]
res["n_indep_batches"] = len(indep)
bc2 = 1 - fl.BETA2 ** max(1, t_T + 1)
names2 = [n for n, p in named if p.dim() == 2]
med_o = {n: [] for n in names2}; med_v = {n: [] for n in names2}; s_o = {n: 0.0 for n in names2}; s_v = dict(s_o)
for b in indep:
    model.zero_grad(set_to_none=True)
    fl.micro_losses_backward(model, [train_exs[i] for i in b]); opt.clip_(args.clip)
    with torch.no_grad():
        for n, p in named:
            if p.dim() != 2: continue
            g = p.grad
            uo = g / ((v_T[n] / bc2).sqrt() + fl.EPS); uv = g / ((opt.state[n]["v"] / bc2).sqrt() + fl.EPS)
            k = max(1, uo.numel() // 200_000)
            med_o[n].append(float(uo.abs().flatten()[::k].median())); med_v[n].append(float(uv.abs().flatten()[::k].median()))
            s_o[n] += float((uo ** 2).sum()); s_v[n] += float((uv ** 2).sum())
model.zero_grad(set_to_none=True)
per = {n: dict(med_ratio=sum(med_v[n]) / max(sum(med_o[n]), 1e-300), rms_ratio=(s_v[n] / max(s_o[n], 1e-300)) ** 0.5) for n in names2}
res["per_matrix"] = per


def kind(n):
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"):
        if k in n: return k
    return "other"


def summ(vals):
    x = np.asarray(vals); return dict(n=len(x), median=float(np.median(x)), p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)), min=float(x.min()), max=float(x.max()))


groups = {}
for n in names2:
    for g in (kind(n), "vocab" if fl.param_group(n) == "vocab" else "body"):
        groups.setdefault(g, []).append(n)
res["summary"] = {g: {q: summ([per[n][q] for n in ns]) for q in ("med_ratio", "rms_ratio")} for g, ns in groups.items()}

# ---- 4.（10-10 I1）实际步：与代理（第 2 项）用同一组独立批次。数据项 d = m̂/(√v̂+ε)，学习率在比值中约去；
#      含权重衰减的版本 d + wd·W（衰减项两边相同）。单次注入：每个批次单独算第 1 个实际步；
#      两次注入：相邻两个批次 (g1, g2) 依次进入 m、v，对应"第 1 步 lr=0、第 2 步才移动"的实际情形。
if a.applied:
    P = dict(named); W2 = {n: P[n].detach() for n in names2}
    v_R = {n: opt.state[n]["v"] for n in names2}            # 替换后（已含 c²）；真 v 条件下即 v_T
    RW = {g: torch.tensor(sorted(set(int(i) for i in v)), dtype=torch.long, device=W2[names2[0]].device)
          for g, v in json.load(open(a.rows)).items()} if a.rows else {}
    B1, B2 = fl.BETA1, fl.BETA2

    def dterm(mhat, vhat): return mhat / (vhat.sqrt() + fl.EPS)

    def stats(x, smp_idx_k):
        ax = x.abs().flatten(); smp = ax[::smp_idx_k].double(); e = ax.double().pow(2)
        thr = float(torch.quantile(smp, 0.999)) if smp.numel() > 1 else float(ax.max())
        return dict(ss=float(e.sum()), med=float(smp.median()), p99=float(torch.quantile(smp, 0.99)) if smp.numel() > 1 else float(ax.max()),
                    mx=float(ax.max()), top01=float(e[ax >= thr].sum()) / max(float(e.sum()), 1e-300))

    keys = ("single", "double", "proxy", "single_wd")
    acc = {q: {n: {"true": [], "rep": []} for n in names2} for q in keys}
    g2f = {n: {"true": [], "rep": []} for n in names2}; rows_acc = {g: {"true": [], "rep": []} for g in RW}
    prev = None
    for bi, b in enumerate(indep):
        model.zero_grad(set_to_none=True)
        fl.micro_losses_backward(model, [train_exs[i] for i in b]); opt.clip_(args.clip)
        with torch.no_grad():
            cur = {n: P[n].grad.detach().clone() for n in names2}
            for n in names2:
                g = cur[n]; k = max(1, g.numel() // 200_000)
                for tag, vb in (("true", v_T[n]), ("rep", v_R[n])):
                    t1 = t_T + 1; v1 = B2 * vb + (1 - B2) * g * g
                    d1 = dterm((1 - B1) * g / (1 - B1 ** t1), v1 / (1 - B2 ** t1))
                    acc["single"][n][tag].append(stats(d1, k)); acc["single_wd"][n][tag].append(stats(d1 + a.wd * W2[n], k))
                    acc["proxy"][n][tag].append(stats(g / ((vb / (1 - B2 ** t1)).sqrt() + fl.EPS), k))
                    gs = ((1 - B2) * g * g / v1.clamp_min(1e-300)).flatten()[::k].double()
                    g2f[n][tag].append(float(gs.median()))
                    if RW and fl.param_group(n) == "vocab":
                        er = d1.double().pow(2).sum(1); et = float(er.sum())
                        for gname, idx in RW.items(): rows_acc[gname][tag].append(float(er[idx].sum()) / max(et, 1e-300))
                    if prev is not None and bi % 2 == 1:
                        gp = prev[n]; t2 = t_T + 2
                        m2 = B1 * (1 - B1) * gp + (1 - B1) * g; v2 = B2 * (B2 * vb + (1 - B2) * gp * gp) + (1 - B2) * g * g
                        acc["double"][n][tag].append(stats(dterm(m2 / (1 - B1 ** t2), v2 / (1 - B2 ** t2)), k))
                    del v1, d1
            prev = cur if bi % 2 == 0 else None
    model.zero_grad(set_to_none=True)

    def ratio(lst_r, lst_t, f):
        if f == "rms": return (sum(x["ss"] for x in lst_r) / max(sum(x["ss"] for x in lst_t), 1e-300)) ** 0.5
        if f == "med": return sum(x["med"] for x in lst_r) / max(sum(x["med"] for x in lst_t), 1e-300)
        if f == "mx": return max(x["mx"] for x in lst_r) / max(max(x["mx"] for x in lst_t), 1e-300)
        return float(np.mean([x[f] for x in lst_r])) / max(float(np.mean([x[f] for x in lst_t])), 1e-300)

    app = {}
    for q in keys:
        app[q] = {n: {f + "_ratio": ratio(acc[q][n]["rep"], acc[q][n]["true"], f) for f in ("med", "rms", "p99", "mx")} for n in names2}
        for n in names2:
            app[q][n]["top01_rep"] = float(np.mean([x["top01"] for x in acc[q][n]["rep"]])); app[q][n]["top01_true"] = float(np.mean([x["top01"] for x in acc[q][n]["true"]]))
    res["applied"] = dict(per_matrix=app, g2_frac={n: {t: float(np.median(g2f[n][t])) for t in ("true", "rep")} for n in names2},
                          rows={gname: {t: float(np.mean(v[t])) for t in ("true", "rep")} for gname, v in rows_acc.items()},
                          summary={q: {g: {f: summ([app[q][n][f] for n in ns]) for f in ("med_ratio", "rms_ratio", "p99_ratio", "mx_ratio")} for g, ns in groups.items()} for q in keys},
                          note="ratio = 替换 / 真 v；single = 第 1 个实际步（注入一次 g²）；double = 两次注入（相邻两批）；proxy = 不注入 g²（与表 9 同口径）；single_wd = 含权重衰减项")

# ---- 3. log v 行列加法模型 R²（真 v）
if a.r2:
    first = fl.epoch_order(len(train_exs), args.epochs, args.order_seed)[0]
    model.zero_grad(set_to_none=True); fl.micro_losses_backward(model, [train_exs[i] for i in first]); opt.clip_(args.clip)

    def r2_stats(vd):
        out = {}
        for n in names2:
            L = torch.log(vd[n].double().clamp_min(fl.V_FLOOR)); mu = L.mean(); a_ = L.mean(1) - mu; b_ = L.mean(0) - mu
            e = L - mu - a_[:, None] - b_[None, :]; ss_tot = float(((L - mu) ** 2).sum()); ss_res = float((e ** 2).sum())
            out[n] = dict(r2=1 - ss_res / max(ss_tot, 1e-300), ss_tot=ss_tot, ss_res=ss_res, numel=L.numel(), floor_frac=float((vd[n] <= fl.V_FLOOR).double().mean()))
        body = [n for n in names2 if fl.param_group(n) != "vocab"]
        def pooled(ns): return 1 - sum(out[n]["ss_res"] for n in ns) / sum(out[n]["ss_tot"] for n in ns)
        return dict(per_matrix={n: {k: v for k, v in d.items() if k in ("r2", "floor_frac", "numel")} for n, d in out.items()},
                    median_r2_all=float(np.median([out[n]["r2"] for n in names2])), median_r2_body=float(np.median([out[n]["r2"] for n in body])),
                    pooled_r2_all=pooled(names2), pooled_r2_body=pooled(body),
                    floor_frac_all=sum(out[n]["floor_frac"] * out[n]["numel"] for n in names2) / sum(out[n]["numel"] for n in names2),
                    by_kind={k: float(np.median([out[n]["r2"] for n in ns])) for k, ns in groups.items() if k not in ("body", "vocab")},
                    v_floor=fl.V_FLOOR)
    v1 = {n: fl.BETA2 * v_T[n] + (1 - fl.BETA2) * dict(named)[n].grad.double().pow(2).to(v_T[n].dtype) for n in names2}
    res["r2_vT"] = r2_stats(v_T); res["r2_v1"] = r2_stats(v1)
    model.zero_grad(set_to_none=True)

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: res[k] for k in res if k in ("c_check", "n_indep_batches")}, ensure_ascii=False))
for g in ("body", "vocab", "gate_proj", "down_proj", "embed_tokens"):
    if g in res["summary"]: print(g, {q: round(res["summary"][g][q]["median"], 3) for q in ("med_ratio", "rms_ratio")})
if a.applied:
    for q in ("proxy", "single", "double"):
        S = res["applied"]["summary"][q]; print("applied", q, {g: (round(S[g]["rms_ratio"]["median"], 3), round(S[g]["rms_ratio"]["p90"], 2), round(S[g]["rms_ratio"]["max"], 1)) for g in ("body", "vocab") if g in S})
    print("rows", res["applied"]["rows"])
if a.r2: print("R2 vT median/pooled(body)", round(res["r2_vT"]["median_r2_body"], 3), round(res["r2_vT"]["pooled_r2_body"], 3), "| v1", round(res["r2_v1"]["median_r2_body"], 3), round(res["r2_v1"]["pooled_r2_body"], 3))
