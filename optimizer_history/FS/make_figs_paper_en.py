#!/usr/bin/env python3
"""Paper (doc 37) main figures, English, P5 house style:
panel titles '(a)  Short title' left-aligned; only element labels inside panels (statistics and explanations go to the caption);
legends frameless upper-left; top/right spines off; one colour + one marker per condition across all figures; dpi 220.
Outputs figs/paper_en/fig1_setup, fig6_format_mechanism and figA1_replay (.png/.pdf). Data entry points listed in doc 39.
10-16: the Fig 2-5 sections were removed; the paper versions of fig2-fig5 and figA2 are written by make_fig2_v13.py,
make_fig34_v13.py and make_fig5_v13.py, the preservation figure by make_fig_preserve_v13.py."""
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
# ===================== Fig 1 setup =====================
fig, ax = plt.subplots(figsize=(TW, 2.6)); ax.set_xlim(0, 11.5); ax.set_ylim(-.3, 3.7); ax.axis("off")
def box(x, y, w, h, t, fc, fs=6.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", fc=fc, ec="0.25", lw=.6)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
def arr(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=.6, color="0.25"))
box(.05, 1.35, 1.45, .8, "Base model\nQwen2.5-0.5B-\nInstruct", "#F2F2F2", 6.2)
rows = [(2.65, "MBPP start", "MBPP, lr 1e-5, 108 steps", "GSM8K\n125 steps", "core, data orders 17/31/47 (median calibration):\nkeep $v,t$ · log-RC $v$ · permuted $v$ · reset all\nalso: keep $m,v,t$ · keep $t$ · uncalibrated log-RC/permuted/constant $v$\n10% replay · same-task control"),
        (1.35, "GSM8K start", "GSM8K, lr 5e-6, 125 steps", "MBPP\n108 steps", "core, data orders 17/31/47 (median calibration):\nkeep $v,t$ · log-RC $v$ · permuted $v$ · reset all\nalso: log-RC $v$ (rms) · permuted $v$ (rms)\nsame-task control"),
        (.05, "PubMed start", "PubMed abstracts, lr 5e-6, 125 steps", "GSM8K\n125 steps", "core, data orders 17/31/47 (median calibration):\nkeep $v,t$ · log-RC $v$ · permuted $v$ · reset all\nalso: the first three with $t$ reset (order 17)\nsame-task control")]
for y, s_, tr, d, arms in rows:
    arr(1.55, 1.75, 2.35, y + .4); box(2.4, y + .05, 2.3, .7, s_, "#DCE6F2", 6.8, True); ax.text(3.55, y + .02 - .0, tr, ha="center", va="top", fontsize=5.0, color="0.3")
    arr(4.75, y + .4, 5.75, y + .4); ax.text(5.25, y + .48, d, ha="center", va="bottom", fontsize=5.4)
    box(5.8, y - .17, 5.65, 1.14, arms, "#FDF0E1", 5.4)
save(fig, "fig1_setup.png")
# ===================== Fig 6 format mechanism (PubMed -> GSM8K) =====================
IV = json.load(open(R / "f/diag25/f1/intervene_drift.json")); D1 = json.load(open(R / "g/diag25/g1/decomp.json")); UT = json.load(open(R / "g/diag25/g1/untie.json"))
fig = plt.figure(figsize=(TW, 7.3)); gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.05, .9], hspace=.66, wspace=.42, left=.235, right=.965, top=.97, bottom=.06)
ax = fig.add_subplot(gs[0, 0])
# v1.2 (10-10 I5): energy-matched random controls (three draws; bar = mean, whisker = range) from results/i/diag25/f1/intervene_energy.json
EN = json.load(open(R / "i/diag25/f1/intervene_energy.json"))
for key, ek in (("~ematch_rows", "rows_ematch"), ("~ematch_both", "rows+neurons_ematch")):
    v = [EN[f"{ek}{s}"]["unind"] for s in range(3)]; IV["E_vperm" + key] = {"unind": float(np.mean(v)), "rng": (min(v), max(v))}
ks = [("E_vperm", "none (permuted $v$)", "perm"), ("E_vperm~revert_neuron_both_used", "shared-use neurons", "perm"), ("E_vperm~revert_random0", "size-matched #1", "perm"), ("E_vperm~revert_random1", "size-matched #2", "perm"),
      ("E_vperm~ematch_rows", "energy-matched rows", "perm"), ("E_vperm~ematch_both", "energy-matched both", "perm"),
      ("E_vperm~revert_neuron_code_only", "code-specific neurons", "perm"), ("E_vperm~revert_vocab_indent+code", "592 vocabulary rows", "perm"), ("E_vperm~revert_both", "neurons + rows", "perm"),
      ("E_0", "none (reset all)", "reset"), ("E_0~transplant_random0_from_vperm", "size-matched → reset all", "reset"), ("E_0~transplant_both_from_vperm", "neurons + rows → reset all", "reset")]
for i, (k, lab, c) in enumerate(ks):
    mod = any(s in k for s in ("revert", "transplant", "ematch")); u = IV[k]["unind"]
    ax.barh(i, u, .7, color=C[c][0] if not mod else "white", edgecolor=C[c][0], hatch=None if not mod else ("xxxx" if "ematch" in k else "////"), lw=.9)
    if "rng" in IV[k]: ax.errorbar(u, i, xerr=[[u - IV[k]["rng"][0]], [IV[k]["rng"][1] - u]], fmt="none", ecolor="0.2", capsize=1.5, lw=.7)
    ax.text(max(u, IV[k].get("rng", (u, u))[1]) + .008, i, f"{u:.2f}", va="center", fontsize=6.4)
ax.set_yticks(range(len(ks))); ax.set_yticklabels([x[1] for x in ks], fontsize=6.8); ax.invert_yaxis(); ax.set_xlim(0, .6); ax.set_xlabel("Unindented-body rate")
ax.axhline(8.5, color="0.5", lw=.6, ls=":"); title(ax, "(a)  Revert to start / transplant"); style(ax)
ax = fig.add_subplot(gs[0, 1]); arms = [("E_vt", "keep_vt"), ("E_vperm", "perm"), ("E_0", "reset")]; combos = [("in_S|out_S", "trained", None), ("in_B_target|out_S", "input rows reverted", "...."), ("in_S|out_B_target", "output rows reverted", "////")]
for i, (a, c) in enumerate(arms):
    for j, (k, lab, h) in enumerate(combos):
        v = UT[a][k]["p_indent"]; ax.bar(i + (j - 1) * .27, v, .26, color=C[c][0] if h is None else "white", edgecolor=C[c][0], hatch=h, lw=.9); ax.text(i + (j - 1) * .27, v + .012, f"{v:.2f}", ha="center", fontsize=6.4, rotation=90, va="bottom")
ax.axhline(.514, color="0.3", lw=.8, ls=(0, (3, 2))); ax.set_xticks(range(3)); ax.set_xticklabels(["keep $v,t$", "permuted $v$", "reset all"]); ax.set_ylim(0, .9); ax.set_ylabel("First-token indent probability")
ax.legend(handles=[Patch(fc="0.45", label="trained"), Patch(fc="white", ec="0.45", hatch="....", label="input side reverted"), Patch(fc="white", ec="0.45", hatch="////", label="output side reverted"), Line2D([], [], color="0.3", ls=(0, (3, 2)), label="start")], frameon=False, loc="upper left", ncol=2, columnspacing=.5, handlelength=1.2)
title(ax, "(b)  Input use vs output use"); style(ax); ax.set_ylim(0, 1.0); ax.set_xticklabels(["keep $v,t$", "permuted $v$", "reset all"], fontsize=6.8)
ax = fig.add_subplot(gs[1, :]); order7 = [("E_vt", "keep_vt", False), ("E_vfac", "logrc", False), ("E_vperm", "perm", False), ("E_v", "keep_vt", True), ("E_vfac_t0", "logrc", True), ("E_vperm_t0", "perm", True), ("E_0", "reset", False)]
for i, (k, c, t0) in enumerate(order7):
    s = D1[k]["shapley_logp_indent"]; ax.barh(i - .18, s["representation"], .34, color=C[c][0], alpha=.95 if not t0 else .55); ax.barh(i + .18, s["readout"], .34, color="white", edgecolor=C[c][0], hatch="////", lw=.8)
    ax.scatter([s["total"]], [i], marker="|", s=90, color="k", zorder=4)
ax.axvline(0, color="0.3", lw=.8); ax.axhline(2.5, color="0.5", lw=.6, ls=":"); ax.axhline(5.5, color="0.5", lw=.6, ls=":")
ax.set_yticks(range(7)); ax.set_yticklabels(["keep $v,t$", "log-RC $v$", "permuted $v$", "keep $v$, reset $t$", "log-RC $v$, reset $t$", "permuted $v$, reset $t$", "reset all"], fontsize=6.6); ax.invert_yaxis()
ax.set_xlabel("$\\Delta\\log P$(indent) from start, step 125"); ax.set_xlim(-.3, .65)
ax.legend(handles=[Patch(fc="0.45", label="representation"), Patch(fc="white", ec="0.45", hatch="////", label="read-out"), Line2D([], [], color="k", marker="|", ls="", ms=10, label="sum (not an error bar)")], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=.8, handlelength=1.3)
title(ax, "(c)  Representation / read-out split"); style(ax)
steps = [0, 10, 25, 50, 75, 125]
for p, key, ttl in ((0, "readout", "(d)  Read-out over training"), (1, "representation", "(e)  Representation over training")):
    ax = fig.add_subplot(gs[2, p])
    for k, c, lab, ls, mf in (("E_vt", "keep_vt", "keep $v,t$", "-", True), ("E_v", "keep_vt", "keep $v$, reset $t$", (0, (4, 2)), False), ("E_0", "reset", "reset all", ":", True)):
        g2 = json.load(open(R / f"g/diag25/g2/{k}/decomp.json")); seq = [g2[f"s{s:04d}"]["shapley_logp_indent"][key] for s in (10, 25, 50, 75)] + [D1[k]["shapley_logp_indent"][key]]
        ax.plot(steps, [0] + seq, ls=ls, marker=C[c][1], color=C[c][0], mfc=C[c][0] if mf else "white", ms=3.6, lw=1.0, label=lab)
    ax.axhline(0, color="0.3", lw=.7); ax.set_xlabel("Steps after the switch"); ax.set_ylabel("$\\Delta\\log P$(indent)"); ax.set_xticks([0, 25, 50, 75, 125])
    if p == 0: ax.legend(frameon=False, loc="lower left")
    title(ax, ttl); style(ax)
save(fig, "fig6_format_mechanism.png")
# ===================== Fig A1 replay =====================
def last_eval(p):
    L = [json.loads(x) for x in open(p)]; e = [x for x in L if x["event"] == "eval_full"][-1]; return e["gsm8k"], e["mbpp"]
src = {"K_vt": R / "runs_b2/Kv/log.jsonl", "K_mvt": R / "t2/runs_t2/lr5e-6/G/log.jsonl", "K_0": R / "t3/runs_t3/lr5e-6/G/log.jsonl", "P_vt": R / "e/runs_e/P_vt/log.jsonl", "P_0": R / "e/runs_e/P_0/log.jsonl"}
EV = {k: last_eval(p) for k, p in src.items()}; M0 = [json.loads(x) for x in open(src["K_vt"])]; M0 = [x for x in M0 if x["event"] == "eval_full"][0]["mbpp"]
fig, ax = plt.subplots(figsize=(3.6, 2.7))
pts = [("keep $v,t$, 10% replay", "P_vt", "keep_vt", True), ("reset all, 10% replay", "P_0", "reset", True), ("keep $v,t$", "K_vt", "keep_vt", False), ("reset all", "K_0", "reset", False), ("keep $m,v,t$", "K_mvt", "keep_mvt", False)]
for lab, k, c, rep in pts:
    x, y = EV[k]; ax.scatter([x], [y], s=26, marker=C[c][1], facecolor=C[c][0] if rep else "white", edgecolor=C[c][0], lw=1.0, zorder=3); ax.annotate(lab, (x, y), fontsize=6, xytext=(6, -11) if lab.startswith("reset all, 10%") else (6, 3), textcoords="offset points")
ax.axhline(M0, ls=(0, (3, 2)), color="0.3", lw=.8); ax.set_xlabel("New-task probe NLL (GSM8K)"); ax.set_ylabel("Old-task probe NLL (MBPP)")
xs = [EV[k][0] for k in EV]; ax.set_xlim(min(xs) - .0015, max(xs) + .006)
ax.legend(handles=[Line2D([], [], color="0.3", marker="o", ls="", label="with replay"), Line2D([], [], color="0.3", marker="o", mfc="white", ls="", label="no replay"), Line2D([], [], color="0.3", ls=(0, (3, 2)), label="start (old task)")], frameon=False, loc="upper left")
title(ax, "Replay reference, MBPP → GSM8K"); style(ax)
save(fig, "figA1_replay.png")
