#!/usr/bin/env python3
"""论文 v1.3（10-11）：重画论文图 4（文件 fig3_one_step_counterfactual：去掉红色，新任务组深蓝实心、旧任务组白底灰斜纹）
与论文图 5（文件 fig4_cross_condition）：(a)(b) 同 v1.2（down 列）；(c) 改为 gate 行、up 行的重合（三顺序均值与范围，h/summary_h.json）；
(d) 改为通道排名的跨顺序一致性（i/diag25/i/cross_order_channels.json：各条件跨顺序 Spearman 与同顺序跨条件对照）。
原 (c)"切换时的预测量"移到附录表。样式同 make_figs_paper_en.py。"""
import json; from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D; from matplotlib.patches import Patch
R = Path(__file__).resolve().parents[1] / "DATA"; OUT = Path(__file__).resolve().parents[1] / "FIG"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.8, "axes.unicode_minus": True, "legend.fontsize": 6.4,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.markersize": 3.6, "lines.linewidth": 1.0})
TW = 6.8
C = {"keep_vt": ("#2166AC", "o"), "logrc": ("#5AAE61", "s"), "perm": ("#E08214", "D"), "reset": ("#4D4D4D", "X")}
LBL = {"keep_vt": "keep $v,t$", "logrc": "log-RC $v$", "perm": "permuted $v$", "reset": "reset all"}
def style(ax): ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7, length=2.5, width=.6); ax.grid(alpha=.22, lw=.5)
def title(ax, s): ax.set_title(s, loc="left", pad=4)
def save(fig, name): fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white"); fig.savefig(OUT / name.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white"); plt.close(fig); print("wrote", name)
jit = lambda n, w, seed=0: (np.random.default_rng(seed).random(n) - .5) * 2 * w

# ===================== 论文图 5：一步反事实（10-18 改为与图 6 同样式的点图） =====================
# 数据：e/diag25/e1/adam_offline_v3_{M,Ag}.json，variants[v]["matched"]["median"][矩阵]（RMS 匹配后，逐批份额批均、24 层中位）；
# 多种子变体（shuffled = randfac、permuted = cperm）取三种子平均。neuron-only：down 取 colonly、gate/up 取 rowonly；hidden-only 相反。
# 面板：(a)(b) 旧任务选择性 5% 的份额，(c)(d) 新任务选择性 5% 的份额；左 = MBPP 起点，右 = GSM8K 起点。矩阵 = 颜色 + 形状，空心。
OFF = {st: json.load(open(R / f"e/diag25/e1/adam_offline_v3_{st}.json"))["variants"] for st in ("M", "Ag")}
VROWS = [("orig", "true $v$"), ("rowcol", "row–column"), ("N", "neuron-only"), ("O", "hidden-only"),
         ("randfac", "shuffled factors"), ("cperm", "permuted $v$"), ("const", "constant $v$"), ("fresh", "$v=0$")]
MATS3 = [("down_proj", "down", "o", "#2166AC"), ("gate_proj", "gate", "s", "#D95F02"), ("up_proj", "up", "^", "#1B9E77")]
def share(st, v, kd, task):
    if v == "N": v = "colonly" if kd == "down_proj" else "rowonly"
    elif v == "O": v = "rowonly" if kd == "down_proj" else "colonly"
    V = OFF[st]; ks = [k for k in V if k == v or k.startswith(v + "#")]
    return 100 * float(np.mean([V[k]["matched"]["median"][kd][f"perbatch_share_{task}_mean"] for k in ks]))
fig, axs = plt.subplots(1, 4, figsize=(TW, 3.0), sharey=True, gridspec_kw=dict(wspace=.12))
yv = {k: i for i, (k, _) in enumerate(VROWS)}; offm = {"down_proj": -.22, "gate_proj": 0, "up_proj": .22}
for p_, (grp, st, sw, ttl) in enumerate((("old", "M", "MBPP → GSM8K", "(a) Old-task share"), ("old", "Ag", "GSM8K → MBPP", "(b) Old-task share"),
                                         ("new", "M", "MBPP → GSM8K", "(c) New-task share"), ("new", "Ag", "GSM8K → MBPP", "(d) New-task share"))):
    ax = axs[p_]; task = {"M": {"new": "gsm", "old": "mbpp"}, "Ag": {"new": "mbpp", "old": "gsm"}}[st][grp]
    ax.axhspan(3.5, 7.6, color="0.93", zorder=0, lw=0)
    for i in range(len(VROWS)): ax.axhline(i, color="0.88", lw=.4, zorder=0)
    ax.axvline(5, color="0.35", lw=.7, ls=(0, (3, 2)), zorder=1)
    for v, _ in VROWS:
        for kd, _, mk, col in MATS3:
            ax.plot(share(st, v, kd, task), yv[v] + offm[kd], mk, ms=4.2, mfc="none", mec=col, mew=1.0, ls="", zorder=3)
    ax.set_title(f"{ttl}\n{sw}", loc="left", pad=3, fontsize=7.3); ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2.2, width=.5, labelsize=6.6); ax.grid(axis="x", alpha=.25, lw=.4)
    ax.set_xlim(0, 6.2) if grp == "old" else ax.set_xlim(3, 20)
axs[0].set_yticks(range(len(VROWS))); axs[0].set_yticklabels([l for _, l in VROWS], fontsize=7); axs[0].set_ylim(7.7, -.65)
axs[0].get_yticklabels()[0].set_fontweight("bold")
axs[0].text(.25, 3.72, "controls", ha="left", va="center", fontsize=5.9, color="0.3", style="italic")
fig.legend(handles=[Line2D([], [], ls="", marker=mk, ms=4.2, mfc="none", mec=col, mew=1.0, label=lab) for _, lab, mk, col in MATS3] +
           [Line2D([], [], color="0.35", lw=.7, ls=(0, (3, 2)), label="uniform (5%)")],
           loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.55, -.1), handletextpad=.3, columnspacing=1.2, fontsize=6.6)
fig.text(.55, -.005, "share of update energy on the selected 5% of neurons (%)", ha="center", fontsize=6.9)
save(fig, "fig3_one_step_counterfactual.png")

# ===================== 论文图 4（10-18 重画）：跨切换、跨顺序复验，2 行 × 3 列 =====================
# 行：down 列重合 / down 列改动能量与旧任务激活的 ρ；列：三个切换；横轴：四个核心条件；每条件三个点 = 顺序 17、31、47 的 24 层中位数。
# 原 (c) gate/up 重合见附录表 replication；原 (d) 排名一致性改为附录表 rankstab（i/diag25/i/cross_order_channels.json）。
HF = R / "h/final/diag25/h"; HALL = {sw: json.load(open(HF / f"neuron_act_{n}_hall.json"))["overlap"] for sw, n in (("MG", "M"), ("GM", "A_g"), ("PG", "B_g"))}
NEWOLD = {"MG": ("GSM8K", "MBPP"), "GM": ("MBPP", "GSM8K"), "PG": ("GSM8K", "MBPP")}
SWN = {"MG": "MBPP → GSM8K", "GM": "GSM8K → MBPP", "PG": "PubMed → GSM8K"}; CK = {"keep_vt": "vt", "logrc": "vfac", "perm": "vperm", "reset": "0"}; SEEDS = (17, 31, 47)
XL = {"keep_vt": "keep\n$v,t$", "logrc": "log-RC\n$v$", "perm": "permuted\n$v$", "reset": "reset\nall"}
def cell(sw, c, s, p):
    new, old = NEWOLD[sw]; d = HALL[sw][f"{sw}_{CK[c]}_s{s}"]
    return d[f"down列原始能量∩{new}选择性" if p == 0 else f"down列原始能量~{old}激活 ρ"]
fig, axs = plt.subplots(2, 3, figsize=(TW, 3.55), sharey="row", gridspec_kw=dict(hspace=.36, wspace=.08))
order = ["keep_vt", "logrc", "perm", "reset"]
for p in range(2):
    for j, sw in enumerate(("MG", "GM", "PG")):
        ax = axs[p, j]
        for i, c in enumerate(order):
            meds = [cell(sw, c, s, p)["median"] for s in SEEDS]
            ax.plot([i - .14, i, i + .14], meds, marker=C[c][1], color=C[c][0], ms=4.2, ls="", mew=.6, mec="white" if c != "reset" else C[c][0])
            ax.plot([i - .26, i + .26], [np.mean(meds)] * 2, color=C[c][0], lw=.9, alpha=.6)
        if p == 0: ax.axhline(243 / 4864, color="0.35", lw=.7, ls="--")
        else: ax.axhline(0, color="0.72", lw=.6, ls="-")
        for i in range(4): ax.axvline(i, color="0.82", lw=.5, ls=(0, (3, 3)), zorder=0)
        ax.set_xticks(range(4)); ax.set_xlim(-.55, 3.55); style(ax); ax.grid(axis="x", visible=False)
        ax.set_xticklabels([XL[c] for c in order] if p == 1 else [], fontsize=6.3); ax.tick_params(axis="y", labelsize=6.5)
        if p == 0: ax.set_title(SWN[sw], fontsize=7.8, pad=4)
    axs[p, 0].set_ylabel(("(a)  Overlap with\nnew-task-selective 5%", "(b)  Spearman $\\rho$ with\nold-task activation")[p], fontsize=7)
axs[0, 0].set_ylim(0, .78); axs[1, 0].set_ylim(-.6, .45)
axs[0, 0].text(3.5, 243 / 4864 + .015, "random set", ha="right", va="bottom", fontsize=6, color="0.35")
save(fig, "fig4_cross_condition.png")
