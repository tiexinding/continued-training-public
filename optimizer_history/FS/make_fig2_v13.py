#!/usr/bin/env python3
"""论文图 2、图 3（10-18 按第四节评估拆分）：MBPP→GSM8K，顺序 17。
图 2（§4.1，fig2_core_allocation）：
  (a) 每层 down 列改动能量 top-5% 与新任务选择性 top-5% 神经元的重合（b4/neuron_overlap_sixarms.json：
      Gkeep=keep m,v,t，Kv=keep v,t，bridge=keep t，Gfresh=reset all）；同任务续训移到附录表 9；
  (b) 嵌套对照的逐层配对差：m = keep m,v,t − keep v,t；v = keep v,t − keep t；t = keep t − reset all。
      原图 2(b) 等变化零模型 R² 移附录（表 8 已有全部数值）。
图 3（§4.2，fig3_allocation_dynamics）：原图 2(c)(d)。
  (a) 单步更新在新任务 / 旧任务选择性 5% 神经元上的份额（b4/first_steps_groups_v2.json，down 列，24 层中位；Kv=keep v,t，bridge=keep t）；
  (b) v̂ 列 log-RMS 剖面与起点激活能量的 Spearman。实心 = 新任务，空心 = 旧任务。
keep t 用完整记录（到第 125 步）。"""
import json; from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = Path(__file__).resolve().parents[1] / "DATA"; OUT = Path(__file__).resolve().parents[1] / "FIG"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.8, "axes.unicode_minus": True, "legend.fontsize": 6.4,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.markersize": 2.8, "lines.linewidth": 1.0})
TW = 6.8
C = {"keep_mvt": ("#B35806", "^"), "keep_vt": ("#2166AC", "o"), "keep_t": ("#6BAED6", "v"), "reset": ("#4D4D4D", "s")}
LBL = {"keep_mvt": "keep $m,v,t$", "keep_vt": "keep $v,t$", "keep_t": "keep $t$", "reset": "reset all"}
def style(ax): ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7, length=2.5, width=.6); ax.grid(alpha=.22, lw=.5)
def title(ax, s): ax.set_title(s, loc="left", pad=4)
def fmt(x): t = f"{x:+.2f}"; return "0.00" if t in ("+0.00", "-0.00") else t.replace("-", "\u2212")
def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white"); fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white"); plt.close(fig)
O = json.load(open(R / "b4/neuron_overlap_sixarms.json")); L = np.arange(24)
ov = {c: np.array([p["dg"] for p in O[k]["per_layer"]]) for k, c in (("Gkeep", "keep_mvt"), ("Kv", "keep_vt"), ("bridge", "keep_t"), ("Gfresh", "reset"))}

# ===================== 图 2：终点分配与嵌套配对差 =====================
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.75), gridspec_kw=dict(wspace=.3))
ax = axs[0]
for c, off in (("keep_mvt", -.12), ("keep_vt", .12), ("keep_t", -.12), ("reset", .12)):
    ax.plot(L + off, ov[c], marker=C[c][1], color=C[c][0], mfc="white", mew=.7, ls={"keep_mvt": "--", "keep_vt": "-"}.get(c, ":"), lw=.9, ms=2.6, label=LBL[c])
ax.axhline(243 / 4864, color="0.4", lw=.7, ls="--", label="random set (0.05)")
ax.set_xlabel("Layer"); ax.set_ylabel("Overlap: top-change and\nnew-task-selective 5%"); ax.set_ylim(0, .9)
ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=.8, handlelength=1.8, handletextpad=.3); title(ax, "(a)  Which neurons receive the change"); style(ax)
ax = axs[1]
D = [("v", ov["keep_vt"] - ov["keep_t"], "keep_vt", "retain $v$:  keep $v,t$ $-$ keep $t$"),
     ("m", ov["keep_mvt"] - ov["keep_vt"], "keep_mvt", "retain $m$:  keep $m,v,t$ $-$ keep $v,t$"),
     ("t", ov["keep_t"] - ov["reset"], "keep_t", "retain $t$:  keep $t$ $-$ reset all")]
for _, d, c, lab in D:
    ax.plot(L, d, marker=C[c][1], color=C[c][0], mfc="white", mew=.7, ls="-", lw=.9, ms=2.6, label=f"{lab}  (median {fmt(np.median(d))})")
ax.axhline(0, color="0.3", lw=.7)
ax.set_xlabel("Layer"); ax.set_ylabel("Paired difference in overlap"); ax.set_ylim(-.15, .9)
ax.legend(frameon=False, loc="upper left", handlelength=1.8, handletextpad=.3); title(ax, "(b)  Conditional effects in the nested comparisons"); style(ax)
save(fig, "fig2_core_allocation")
print("paired differences:", {k: (round(float(np.median(d)), 3), int((d > 0).sum()), int((d < 0).sum())) for k, d, _, _ in D})

# ===================== 图 3：逐步动态 =====================
F2 = json.load(open(R / "b4/first_steps_groups_v2.json"))
fig, axs = plt.subplots(1, 2, figsize=(TW, 2.75), gridspec_kw=dict(wspace=.3))
ax = axs[0]
for key, c in (("Kv", "keep_vt"), ("bridge", "keep_t")):
    st = [int(s) for s in F2[key] if int(s) >= 2]
    for grp, mfc, ls in (("new", C[c][0], "-"), ("old", "white", "--")):
        ax.plot(st, [100 * F2[key][str(s)]["u"][grp] for s in st], marker=C[c][1], color=C[c][0], mfc=mfc, ls=ls, lw=.9, ms=2.8)
ax.axhline(5, color="0.4", lw=.7, ls=":"); ax.set_xscale("log"); ax.set_xlabel("Step after the switch (log)"); ax.set_ylabel("Share of update energy (%)"); ax.set_ylim(0, 17)
title(ax, "(a)  Share of each single update"); style(ax)
ax.legend(handles=[Line2D([], [], color=C["keep_vt"][0], marker="o", ls="-", label="keep $v,t$"), Line2D([], [], color=C["keep_t"][0], marker="v", ls="-", label="keep $t$"),
                   Line2D([], [], color="0.3", marker="o", ls="-", label="new-task-selective 5%"), Line2D([], [], color="0.3", marker="o", mfc="white", ls="--", label="old-task-selective 5%")],
          frameon=False, loc="upper right", ncol=2, columnspacing=.6, handlelength=1.8, handletextpad=.3)
ax = axs[1]
for key, c in (("Kv", "keep_vt"), ("bridge", "keep_t")):
    st = [int(s) for s in F2[key]]
    ax.plot(st, [F2[key][str(s)]["vhat_rho_old_act"] for s in st], marker=C[c][1], color=C[c][0], mfc="white", ls="--", lw=.9, ms=2.8)
    ax.plot(st, [F2[key][str(s)]["vhat_rho_new_act"] for s in st], marker=C[c][1], color=C[c][0], mfc=C[c][0], ls="-", lw=.9, ms=2.8)
ax.set_xscale("log"); ax.set_xlabel("Step after the switch (log)"); ax.set_ylabel("Spearman with log activation energy"); ax.set_ylim(0, 1.18); ax.set_yticks(np.arange(0, 1.01, .2))
ax.legend(handles=[Line2D([], [], color=C["keep_vt"][0], marker="o", ls="-", label="keep $v,t$"), Line2D([], [], color=C["keep_t"][0], marker="v", ls="-", label="keep $t$"),
                   Line2D([], [], color="0.3", marker="o", ls="-", label="new-task activation"), Line2D([], [], color="0.3", marker="o", mfc="white", ls="--", label="old-task activation")],
          frameon=False, loc="upper left", ncol=2, columnspacing=.8, handlelength=1.8, handletextpad=.3)
title(ax, "(b)  Column log-RMS profile of $\\hat v$"); style(ax)
save(fig, "fig3_allocation_dynamics"); print("ok")
