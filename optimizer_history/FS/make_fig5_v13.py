#!/usr/bin/env python3
"""论文 v1.3b 图 5（10-12 评审（五、六））：由 make_figs_paper_en.py 的图 5 拆出。
正文图只留 (a) HumanEval 三顺序两协议、(b) PG 无缩进率与首 token 缩进概率（四个核心条件 × 三顺序；清零 t 的柱移出，数值见附录 C.2）；
原 (c) 探针损失配对差移到附录图 figA2_probe_loss。数据与样式不变。"""
import json; from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
R = Path(__file__).resolve().parents[1] / "DATA"; OUT = Path(__file__).resolve().parents[1] / "FIG"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.8, "axes.unicode_minus": True, "legend.fontsize": 6.6, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.markersize": 4.5, "axes.linewidth": .6})
TW = 6.8  # text width (in), figures are drawn at print size
# ---- one colour / marker per condition (all figures)
C = {"keep_mvt": ("#053061", "^"), "keep_vt": ("#2166AC", "o"), "keep_t": ("#92C5DE", "v"), "reset": ("#4D4D4D", "X"),
     "logrc": ("#5AAE61", "s"), "perm": ("#E08214", "D"), "same": ("#BABABA", "P"), "base": ("#000000", "*")}
LBL = {"keep_mvt": "keep $m,v,t$", "keep_vt": "keep $v,t$", "keep_t": "keep $t$", "reset": "reset all", "logrc": "log-RC $v$, keep $t$", "perm": "permuted $v$, keep $t$", "same": "same task"}
def style(ax):
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7, length=2.5, width=.6); ax.grid(alpha=.22, lw=.5)
def title(ax, s): ax.set_title(s, loc="left", pad=4)
def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white"); fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)
jit = lambda n, w, seed=0: (np.random.default_rng(seed).random(n) - .5) * 2 * w
SWN = {"MG": "MBPP → GSM8K", "GM": "GSM8K → MBPP", "PG": "PubMed → GSM8K"}; CK = {"keep_vt": "vt", "logrc": "vfac", "perm": "vperm", "reset": "0"}; SEEDS = (17, 31, 47)
HE = json.load(open(R / "e/humaneval_e_rescore.json")); HD = json.load(open(R / "e/humaneval_d_rescore.json")); NLL = json.load(open(R / "e/nll_two_calibers_e.json"))
FMT = json.load(open(R / "e/humaneval_pairs_e_rescore.json"))["format"]; FP = json.load(open(R / "f/diag25/f1/first_pos_probs.json")); H7 = json.load(open(R / "f/diag25/f1/heval_f7.json")); H14 = json.load(open(R / "f/diag25/f1/heval_f14.json"))
fig = plt.figure(figsize=(TW, 2.75)); gs5t = fig.add_gridspec(1, 2, left=.075, right=.995, top=.92, bottom=.2, wspace=.30)
axs = [fig.add_subplot(gs5t[0, 0]), fig.add_subplot(gs5t[0, 1])]
SH = json.load(open(R / "h/final/h_analysis/out/summary_h.json"))["cells"]
def hcell(sw, c, s): return SH[f"{sw}|{CK[c]}|{s}"]
def ind_p(c, s):   # first-token indent probability: seed 17 from heval_f7 (same drift_diag protocol), seeds 31/47 from the H round
    if s == 17: return H7[{"keep_vt": "E_vt", "logrc": "E_vfac", "perm": "E_vperm", "reset": "E_0"}[c]]["indent_p"]
    n = f"PG_{CK[c]}_s{s}"; return json.load(open(R / f"h/diag25/f1/heval_h_{n}.json"))[n]["indent_p"]
ax = axs[0]; conds5 = ["keep_vt", "logrc", "perm", "reset"]
for g, (sw, src, sk) in enumerate((("PG", HE, "B_g"), ("GM", HD, "A_g"))):
    x0 = g * 6.5; v2 = src[sk]["v2"]["pass_rate"] * 164; v3 = src[sk]["v3_autoindent"]["pass_rate"] * 164
    ax.plot([x0, x0], [v2, v3], color="0", lw=.9); ax.scatter([x0], [v2], s=22, marker="*", facecolor="white", edgecolor="0", lw=.8, zorder=3); ax.scatter([x0], [v3], s=22, marker="*", color="0", zorder=3)
    for i, c in enumerate(conds5):
        for k, s in enumerate(SEEDS):
            x = x0 + 1 + i * 1.3 + (k - 1) * .32; d = hcell(sw, c, s); v2, v3 = d["pass_v2"] * 164, d["pass_v3"] * 164
            ax.plot([x, x], [v2, v3], color=C[c][0], lw=.8); ax.scatter([x], [v2], s=12, marker=C[c][1], facecolor="white", edgecolor=C[c][0], lw=.8, zorder=3); ax.scatter([x], [v3], s=12, marker=C[c][1], color=C[c][0], zorder=3)
ax.set_xticks([2.95, 9.45]); ax.set_xticklabels(["PubMed → GSM8K", "GSM8K → MBPP"], fontsize=6.8); ax.set_ylabel("Problems passed (of 164)"); ax.set_ylim(10, 72); ax.set_xlim(-.8, 12.4)
ax.plot([5.85, 5.85], [10, 58], color="0.45", lw=.7, ls=(0, (4, 3)), zorder=1)   # 不延伸到顶部图例
ax.legend(handles=[Line2D([], [], color="0.3", marker="o", mfc="white", ls="", label="raw assembly"), Line2D([], [], color="0.3", marker="o", ls="", label="indent repair")] + [Line2D([], [], color=("0" if c == "base" else C[c][0]), marker=("*" if c == "base" else C[c][1]), ls="", label=("start" if c == "base" else LBL[c].replace(", keep $t$", ""))) for c in ("base", "keep_vt", "logrc", "perm", "reset")], frameon=False, loc="upper left", ncol=4, mode="expand", bbox_to_anchor=(0, 1.0, 1, 0), borderaxespad=.2, columnspacing=.3, handletextpad=.2, fontsize=6.3)
title(ax, "(a)  HumanEval, three data orders"); style(ax)
ax = axs[1]; xs = []
for i, c in enumerate(["reset", "keep_vt", "logrc", "perm"]):
    for k, s in enumerate(SEEDS):
        x = i + (k - 1) * .26; u = hcell("PG", c, s)["unind"]
        ax.bar(x, u, .23, color=C[c][0] if k != 2 else "white", alpha=1, edgecolor=C[c][0], lw=(.6, .9, .9)[k], zorder=2)
        if k == 1: ax.bar(x, u, .23, color="white", alpha=.55, lw=0, zorder=2.5)   # order 31: light tint of the condition colour
        ax.scatter([x], [ind_p(c, s)], s=14, marker=C[c][1], color="0.1", edgecolor="white", zorder=3, lw=.5)
ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["reset\nall", "keep\n$v,t$", "log-RC\n$v$", "permuted\n$v$"], fontsize=6.8); ax.set_ylim(0, 1.05)
ax.set_ylabel("Rate / probability"); ax.legend(handles=[Patch(fc="0.45", label="unindented, order 17"), Patch(fc="0.75", ec="0.45", lw=.9, label="order 31"), Patch(fc="white", ec="0.45", label="order 47"), Line2D([], [], color="0.1", marker="o", ls="", ms=3.5, label="first-token indent prob.")], frameon=False, loc="upper left", ncol=2, columnspacing=.5, handlelength=1.2)
title(ax, "(b)  Output format, PubMed → GSM8K"); style(ax)
save(fig, "fig5_functional_boundary.png")

fig, ax = plt.subplots(figsize=(TW*.62, 2.6));
items = [("log-RC $v$ (rms) − keep $v,t$, old task", "R_vfac-R_vt|probe_gsm8k", "G→M"), ("permuted $v$ (rms) − keep $v,t$, old task", "R_vperm-R_vt|probe_gsm8k", "G→M"), ("permuted $v$ (median) − keep $v,t$, old task", "R_vperm_med-R_vt|probe_gsm8k", "G→M"),
    ("log-RC $v$ (median) − keep $v,t$, new task", "E_vfac-E_vt|probe_gsm8k", "P→G"), ("reset all − keep $v,t$, new task", "E_0-E_vt|probe_gsm8k", "P→G"), ("replay − no replay (keep $v,t$), old task", "P_vt-K_vt|probe_mbpp", "M→G"), ("replay: keep $v,t$ − reset all, old task", "P_vt-P_0|probe_mbpp", "M→G")]
for i, (lab, k, dd) in enumerate(items):
    d = NLL[k]; ax.errorbar(d["item"], i - .15, xerr=[[d["item"] - d["item_ci"][0]], [d["item_ci"][1] - d["item"]]], fmt="o", color="0.15", ms=3, capsize=1.5, lw=.8)
    ax.errorbar(d["token"], i + .15, xerr=[[d["token"] - d["token_ci"][0]], [d["token_ci"][1] - d["token"]]], fmt="s", mfc="white", color="0.45", ms=3, capsize=1.5, lw=.8)
ax.set_yticks(range(len(items))); ax.set_yticklabels([f"{x[2]}   {x[0]}" for x in items], fontsize=6.6); ax.invert_yaxis(); ax.axvline(0, color="0.3", lw=.8)
ax.set_xlabel("Probe NLL difference (nats/token)"); ax.legend(handles=[Line2D([], [], color="0.15", marker="o", ls="", label="item-weighted"), Line2D([], [], color="0.45", marker="s", mfc="white", ls="", label="token-weighted")], frameon=False, loc="lower right")
title(ax, "Probe loss, paired differences"); style(ax)
save(fig, "figA2_probe_loss.png")
