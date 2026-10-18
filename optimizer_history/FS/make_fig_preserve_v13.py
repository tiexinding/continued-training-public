#!/usr/bin/env python3
"""论文图 6（fig_preserve）："各近似保留了真二阶矩更新的哪些性质"。
数据：DATA/k/summary_k_M.json（K 轮，MBPP 起点，8 批，同一次计算）：
  r1[<cond>|matched|<mat>]：逐批份额（新/旧任务选择性 5%）与 cos（24 层中位，批均）；
  r3[<task>|<b>]["e"][<cond>]：局部前馈输出相对误差（逐题、逐层中位）；r4_new[<cond>]：全模型 KL(p_T‖p_X)，8 批中位。
面板：(a) 旧任务份额 − 真 v（百分点）(b) 新任务份额 − 真 v (c) 1 − cos (d) 局部相对误差（8 批中位与范围）(e) 全模型 KL（新任务）。
10-18 重画（老丁）：加高；纵轴分两组（v 的替换 / 能量匹配对照，对照组说明写在 (a) 灰色区空白处），长标签换行；
矩阵用颜色 + 形状（down 蓝圆、gate 橙方、up 绿三角），任务用颜色（新任务深红、旧任务紫），全部空心；
(c)(d)(e) 横轴写明参照为真 v。"""
import json; from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[1] / "DATA"; OUT = Path(__file__).resolve().parents[1] / "FIG"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "axes.titlesize": 7.3, "axes.labelsize": 6.9, "axes.unicode_minus": True,
                     "legend.fontsize": 6.6, "xtick.labelsize": 6.6, "ytick.labelsize": 7, "axes.linewidth": .6})
TW = 6.8; MUTE = "0.45"
D = json.load(open(R / "k/summary_k_M.json"))
ROWS = [("RC", "row–column"), ("N", "neuron-only"), ("O", "hidden-only"), ("C", "constant $v$"), ("PN", "neuron-\npermuted $v$"),
        ("OE", "matched energy,\nhidden-only dir."), ("CE", "matched energy,\nconstant-$v$ dir."), ("RE", "matched energy,\nrandom dir.")]
MATS = [("down", "o", "#2166AC"), ("gate", "s", "#D95F02"), ("up", "^", "#1B9E77")]
TASK = {"new": "#B2182B", "old": "#6A3D9A"}
y = {k: i for i, (k, _) in enumerate(ROWS)}
def r1(c, m):   # RE：三种子汇总取平均
    if c != "RE": return D["r1"][f"{c}|matched|{m}"]
    seeds = [D["r1"][f"RE{s}|matched|{m}"] for s in range(3)]
    return {key: float(np.mean([v[key] for v in seeds])) for key in seeds[0]}
def r3(c, task):
    if c == "RE": vals = [np.mean([D["r3"][f"{task}|{b}"]["e"][f"RE{s}"] for s in range(3)]) for b in range(8)]
    else: vals = [D["r3"][f"{task}|{b}"]["e"][c] for b in range(8)]
    return np.median(vals), min(vals), max(vals)
def kl(c): return (np.mean([D["r4_new"][f"RE{s}"] for s in range(3)]) if c == "RE" else D["r4_new"][c]) * 1e6
fig, axs = plt.subplots(1, 5, figsize=(TW, 3.5), sharey=True, gridspec_kw=dict(wspace=.14, width_ratios=[1, 1, .95, 1, 1]))
def frame(ax, ttl, xlabel):
    ax.set_title(ttl, loc="left", pad=3); ax.set_xlabel(xlabel); ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.2, width=.5); ax.grid(axis="x", alpha=.25, lw=.4)
    ax.axhspan(4.5, 7.6, color="0.93", zorder=0, lw=0)
    for i in range(len(ROWS)): ax.axhline(i, color="0.88", lw=.4, zorder=0)
MK = dict(ms=4.2, mfc="none", mew=1.0, ls="", zorder=3)
off = {"down": -.22, "gate": 0, "up": .22}
for j, (key, ttl) in enumerate((("old", "(a) Old-task share"), ("new", "(b) New-task share"))):
    ax = axs[j]; frame(ax, ttl, "difference from\ntrue $v$ (points)")
    ax.axvline(0, color=MUTE, lw=.7, zorder=1)
    for c, _ in ROWS:
        for m, mk, col in MATS:
            dv = 100 * (r1(c, m)[key] - r1("T", m)[key]); ax.plot(dv, y[c] + off[m], mk, mec=col, **MK)
    ax.set_xlim(-1.0, 4.2) if key == "old" else ax.set_xlim(-3.4, 0.8)
ax = axs[2]; frame(ax, "(c) Direction", "$1-\\cos$ with\ntrue-$v$ update")
for c, _ in ROWS:
    for m, mk, col in MATS:
        v = max(1 - r1(c, m)["cos"], 1e-3); ax.plot(v, y[c] + off[m], mk, mec=col, **MK)
ax.set_xscale("log"); ax.set_xlim(2e-3, 2)
ax = axs[3]; frame(ax, "(d) Local output", "relative error\nvs. true $v$")
for c, _ in ROWS:
    for task, dy in (("new", -.16), ("old", .16)):
        med, lo, hi = r3(c, task); ax.plot([lo, hi], [y[c] + dy] * 2, color=TASK[task], lw=.9, zorder=2)
        ax.plot(med, y[c] + dy, "D", ms=3.8, mfc="none", mec=TASK[task], mew=1.0, zorder=3)
ax.set_xscale("log"); ax.set_xlim(1.2e-2, 2)
ax = axs[4]; frame(ax, "(e) Next-token KL", "KL from true-$v$ model\n($10^{-6}$ nats)")
ref = D["r4_new"]["kl_W_T"] * 1e6
ax.axvline(ref, color=MUTE, lw=.7, ls=(0, (3, 2)), zorder=1); ax.text(ref * .75, -.35, "start → true update", ha="right", va="center", fontsize=5.8, color="0.3")
for c, _ in ROWS: ax.plot(kl(c), y[c], "D", ms=3.8, mfc="none", mec=TASK["new"], mew=1.0, zorder=3)
ax.set_xscale("log"); ax.set_xlim(.15, 6e4)
axs[0].set_yticks(range(len(ROWS))); axs[0].set_yticklabels([l for _, l in ROWS]); axs[0].set_ylim(7.7, -.65)
axs[0].get_yticklabels()[0].set_fontweight("bold")
# 对照组说明：写在 (a) 灰色区域右侧空白处（该组 (a)(b) 按构造为 0）
axs[0].text(4.1, 6.0, "energy of each\nneuron matched\nto true $v$; only\nthe direction\ndiffers", ha="right", va="center", fontsize=5.7, color="0.3", style="italic", linespacing=1.1)
h1 = [Line2D([], [], ls="", marker=mk, ms=4.2, mfc="none", mec=col, mew=1.0, label=m) for m, mk, col in MATS]
h2 = [Line2D([], [], ls="", marker="D", ms=3.8, mfc="none", mec=TASK["new"], mew=1.0, label="new-task inputs"),
      Line2D([], [], ls="", marker="D", ms=3.8, mfc="none", mec=TASK["old"], mew=1.0, label="old-task inputs")]
fig.legend(handles=h1, title="matrix (a–c)", loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.36, -.16), handletextpad=.2, columnspacing=.9, title_fontsize=6.6)
fig.legend(handles=h2, title="inputs (d, e)", loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.72, -.16), handletextpad=.2, columnspacing=.9, title_fontsize=6.6)
fig.savefig(OUT / "fig_preserve.png", dpi=220, bbox_inches="tight", facecolor="white"); fig.savefig(OUT / "fig_preserve.pdf", bbox_inches="tight", facecolor="white"); print("ok", ref)
