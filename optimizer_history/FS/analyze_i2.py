#!/usr/bin/env python3
"""10-10 I2 汇总：轨迹实测步长。读 runs_i/<sw>_<cond>_s17/records/upd_stats.pt，
替换条件（vfac、vperm）对 留 v·t（vt）在同一步号上的比值：
  数据项 d = m̂/(√v̂+ε) 的 RMS、中位数、p99、最大值（不含学习率，第 1 步也有定义）；实际总更新 ΔW 的 RMS（第 1 步 lr=0 除外）。
分段：2–10 / 11–25 / 26–终点 / 11–终点，段内取各步比值的中位数；主体 = 词表以外的 2D 矩阵，取矩阵间中位数与 p90。
另报 592 行（indent、code_only）在共享词表矩阵数据项能量中的占比（各段步中位数）。
用法：python analyze_i2.py --runs_dir <runs_i> --out <json>"""
import argparse, json
from pathlib import Path
import numpy as np, torch

ap = argparse.ArgumentParser(); ap.add_argument("--runs_dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
R = Path(a.runs_dir); SEG = {"2-10": (1, 10), "11-25": (10, 25), "26-end": (25, None), "11-end": (10, None)}
res = {}
for sw in ("MG", "GM", "PG"):
    U = {c: torch.load(R / f"{sw}_{c}_s17/records/upd_stats.pt", map_location="cpu") for c in ("vt", "vfac", "vperm") if (R / f"{sw}_{c}_s17/records/upd_stats.pt").exists()}
    if "vt" not in U: continue
    names = U["vt"]["names"]; voc = [i for i, n in enumerate(names) if "embed_tokens" in n or "lm_head" in n]; body = [i for i in range(len(names)) if i not in voc]
    ds = U["vt"]["d_stats"]; S = U["vt"]["d"].shape[0]
    for c in ("vfac", "vperm"):
        if c not in U: continue
        r = (U[c]["d"] / U["vt"]["d"]).numpy()            # (steps, mats, stats)
        rt = (U[c]["tot"][:, :, 0] / U["vt"]["tot"][:, :, 0]).numpy()
        out = {}
        for sname, (s0, s1) in SEG.items():
            sl = slice(s0, s1 if s1 else S); o = {}
            for j, st in enumerate(ds):
                o[f"vocab_{st}"] = float(np.median(r[sl, voc, j]))
                per_mat = np.median(r[sl][:, body, j], axis=0)
                o[f"body_{st}_median"] = float(np.median(per_mat)); o[f"body_{st}_p90"] = float(np.percentile(per_mat, 90)); o[f"body_{st}_max"] = float(per_mat.max())
            o["vocab_tot_rms"] = float(np.median(rt[sl, voc])); o["body_tot_rms_median"] = float(np.median(np.median(rt[sl][:, body], axis=0)))
            out[sname] = o
        out["step1_vocab_rms"] = float(r[0, voc, 0].mean()); out["step2_vocab_rms"] = float(r[1, voc, 0].mean())
        out["max_body_matrix_11end"] = names[body[int(np.argmax(np.median(r[10:, body, 0], axis=0)))]]
        res[f"{sw}_{c}"] = out
        q = out["11-end"]; print(f"{sw} {c:5} 11–终点 词表 RMS 比 {q['vocab_rms']:.2f}（2–10 {out['2-10']['vocab_rms']:.2f}，26–终 {out['26-end']['vocab_rms']:.2f}）| 主体 中位 {q['body_rms_median']:.2f} p90 {q['body_rms_p90']:.2f} max {q['body_rms_max']:.1f} | 总更新 词表 {q['vocab_tot_rms']:.2f}", flush=True)
    for c in U:
        rows = U[c]["rows"]; vn = [n for n in rows[0]] if rows and rows[0] else []
        if vn:
            for g in rows[0][vn[0]]:
                sh = np.array([rw[vn[0]][g][0] for rw in rows])
                res.setdefault(f"{sw}_rows", {})[f"{c}|{g}"] = {s: float(np.median(sh[s0:(s1 if s1 else S)])) for s, (s0, s1) in SEG.items()}
    if f"{sw}_rows" in res: print(" ", sw, "592 行能量占比（11–终点）", {k: round(v["11-end"], 4) for k, v in res[f"{sw}_rows"].items()})
json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
