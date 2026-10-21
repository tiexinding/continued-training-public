#!/usr/bin/env python3
"""Paper figures 8 and 9 (format interventions and hidden-state / read-out decomposition, PubMed -> GSM8K), English, P5 house style:
panel titles '(a)  Short title' left-aligned, legends above the axes, no value labels on bars; top/right spines off;
one colour + one marker per condition across all figures. Outputs FIG/fig6a_format_interventions and
FIG/fig6b_format_decomposition (.png/.pdf).
10-19: the former single figure fig6_format_mechanism was split into two figures; the unused fig1_setup and figA1_replay
sections were removed.
10-21: light fills (tint of the condition colour, darker edge) instead of solid colour; no dot hatching (prints as grey);
Fig 8a colours only the unmodified models and the code groups, controls in neutral grey. The paper versions of the other figures are written by make_fig2_v13.py, make_fig34_v13.py,
make_fig5_v13.py and make_fig_preserve_v13.py."""
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
def tint(c, f):   # mix a colour with white; f = share of the colour (light fills print without dark blocks)
    r = np.array(matplotlib.colors.to_rgb(c)); return tuple(1 - (1 - r) * f)
plt.rcParams["hatch.linewidth"] = .55
jit = lambda n, w, seed=0: (np.random.default_rng(seed).random(n) - .5) * 2 * w
# ===================== Fig 8: format interventions (PubMed -> GSM8K) =====================
IV = json.load(open(R / "f/diag25/f1/intervene_drift.json")); D1 = json.load(open(R / "g/diag25/g1/decomp.json")); UT = json.load(open(R / "g/diag25/g1/untie.json"))
EN = json.load(open(R / "i/diag25/f1/intervene_energy.json"))   # energy-matched random controls (three draws; bar = mean, whisker = range)
for key, ek in (("~ematch_rows", "rows_ematch"), ("~ematch_both", "rows+neurons_ematch")):
    v = [EN[f"{ek}{s}"]["unind"] for s in range(3)]; IV["E_vperm" + key] = {"unind": float(np.mean(v)), "rng": (min(v), max(v))}
fig = plt.figure(figsize=(TW, 4.6)); gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=.34, left=.205, right=.985, top=.80, bottom=.10)
ax = fig.add_subplot(gs[0, 0])
ks = [("E_vperm", "none\n(permuted $v$)", "perm"), ("E_vperm~revert_neuron_both_used", "shared-use\nneurons", "perm"), ("E_vperm~revert_random0", "size-matched #1", "perm"), ("E_vperm~revert_random1", "size-matched #2", "perm"),
      ("E_vperm~ematch_rows", "energy-matched\nrows", "perm"), ("E_vperm~ematch_both", "energy-matched\nneurons + rows", "perm"),
      ("E_vperm~revert_neuron_code_only", "code-specific\nneurons", "perm"), ("E_vperm~revert_vocab_indent+code", "592 vocabulary\nrows", "perm"), ("E_vperm~revert_both", "code neurons\n+ 592 rows", "perm"),
      ("E_0", "none\n(reset all)", "reset"), ("E_0~transplant_random0_from_vperm", "size-matched\n→ reset all", "reset"), ("E_0~transplant_both_from_vperm", "code neurons + rows\n→ reset all", "reset")]
# colour = model only for the unmodified bars and the code groups; controls are neutral grey (10-21: less orange)
CODE = ("revert_neuron_code_only", "revert_vocab_indent+code", "revert_both", "transplant_both")
for i, (k, lab, c) in enumerate(ks):
    u = IV[k]["unind"]; ec = C[c][0]
    if "~" not in k: ax.barh(i, u, .68, color=tint(ec, .35), edgecolor=ec, lw=.8)                          # unmodified model
    elif any(t in k for t in CODE): ax.barh(i, u, .68, color="white", edgecolor=C["perm"][0], hatch="///", lw=.8)   # code groups (taken from permuted v)
    else: ax.barh(i, u, .68, color="white", edgecolor="0.45", lw=.9)                                        # control groups: open (no hatch, so grey print still separates them)
    if "rng" in IV[k]: ax.errorbar(u, i, xerr=[[u - IV[k]["rng"][0]], [IV[k]["rng"][1] - u]], fmt="none", ecolor="0.2", capsize=1.8, lw=.8)
ax.set_yticks(range(len(ks))); ax.set_yticklabels([x[1] for x in ks], fontsize=6.6, linespacing=.95); ax.invert_yaxis(); ax.set_xlim(0, .5); ax.set_xlabel("Unindented-body rate")
ax.axhline(8.5, color="0.5", lw=.6, ls=":"); style(ax)
ax.legend(handles=[Patch(fc=tint(C["perm"][0], .35), ec=C["perm"][0], lw=.8, label="permuted $v$, unmodified"), Patch(fc="white", ec=C["perm"][0], hatch="///", lw=.8, label="code groups restored\nor copied"),
                   Patch(fc=tint(C["reset"][0], .35), ec=C["reset"][0], lw=.8, label="reset all, unmodified"), Patch(fc="white", ec="0.45", lw=.9, label="control groups\n(energy-matched: mean, range)")],
          frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=2, columnspacing=.8, handlelength=1.6, handleheight=1.1, fontsize=6.3, borderaxespad=0)
ax.set_title("(a)  Restore to start / transplant", loc="left", pad=46)
ax = fig.add_subplot(gs[0, 1]); arms = [("E_vt", "keep_vt"), ("E_vperm", "perm"), ("E_0", "reset")]
combos = [("in_S|out_S", None), ("in_B_target|out_S", "--"), ("in_S|out_B_target", "///")]   # trained / input use restored / output use restored
for i, (a, c) in enumerate(arms):
    for j, (k, h) in enumerate(combos):
        v = UT[a][k]["p_indent"]; ax.bar(i + (j - 1) * .27, v, .24, color=tint(C[c][0], .35) if h is None else "white", edgecolor=C[c][0], hatch=h, lw=.8)
ax.axhline(.514, color="0.3", lw=.8, ls=(0, (3, 2))); ax.set_xticks(range(3)); ax.set_xticklabels(["keep $v,t$", "permuted $v$", "reset all"], fontsize=6.8)
ax.set_ylim(0, .75); ax.set_ylabel("First-token indent probability"); style(ax)
ax.legend(handles=[Patch(fc=tint("0.3", .35), ec="0.3", lw=.8, label="trained"), Patch(fc="white", ec="0.3", hatch="--", lw=.8, label="input use\nrestored"), Patch(fc="white", ec="0.3", hatch="///", lw=.8, label="output use\nrestored"), Line2D([], [], color="0.3", ls=(0, (3, 2)), label="start")],
          frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=2, columnspacing=.8, handlelength=1.6, handleheight=1.1, fontsize=6.3, borderaxespad=0)
ax.set_title("(b)  Input use vs output use", loc="left", pad=46)
save(fig, "fig6a_format_interventions.png")
# ===================== Fig 9: hidden-state / read-out decomposition =====================
fig = plt.figure(figsize=(TW, 6.0)); gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1], hspace=.62, wspace=.32, left=.17, right=.985, top=.89, bottom=.08)
ax = fig.add_subplot(gs[0, :]); order7 = [("E_vt", "keep_vt", False), ("E_vfac", "logrc", False), ("E_vperm", "perm", False), ("E_v", "keep_vt", True), ("E_vfac_t0", "logrc", True), ("E_vperm_t0", "perm", True), ("E_0", "reset", False)]
for i, (k, c, t0) in enumerate(order7):
    s = D1[k]["shapley_logp_indent"]; ax.barh(i - .19, s["representation"], .36, color=tint(C[c][0], .45 if not t0 else .2), edgecolor=C[c][0], lw=.8); ax.barh(i + .19, s["readout"], .36, color="white", edgecolor=C[c][0], hatch="///", lw=.8)
    ax.scatter([s["total"]], [i], marker="|", s=110, color="k", zorder=4)
ax.axvline(0, color="0.3", lw=.8, ls=(0, (3, 2))); ax.axhline(2.5, color="0.5", lw=.6, ls=":"); ax.axhline(5.5, color="0.5", lw=.6, ls=":")
ax.set_yticks(range(7)); ax.set_yticklabels(["keep $v,t$", "log-RC $v$", "permuted $v$", "keep $v$,\nreset $t$", "log-RC $v$,\nreset $t$", "permuted $v$,\nreset $t$", "reset all"], fontsize=6.8, linespacing=.95); ax.invert_yaxis()
ax.set_xlabel("$\\Delta\\log P$(indent) from start, step 125"); ax.set_xlim(-.3, .4); style(ax)
ax.legend(handles=[Patch(fc=tint("0.3", .45), ec="0.3", lw=.8, label="hidden-state term"), Patch(fc=tint("0.3", .2), ec="0.3", lw=.8, label="hidden-state term, $t$ reset"), Patch(fc="white", ec="0.3", hatch="///", lw=.8, label="read-out term"), Line2D([], [], color="k", marker="|", ls="", ms=10, label="sum of the two terms")],
          frameon=False, loc="lower left", bbox_to_anchor=(-.01, 1.01), ncol=4, columnspacing=1.0, handlelength=1.6, handleheight=1.1, fontsize=6.4, borderaxespad=0)
ax.set_title("(a)  Hidden-state / read-out split at step 125", loc="left", pad=22)
steps = [0, 10, 25, 50, 75, 125]
for p, key, ttl in ((0, "readout", "(b)  Read-out term over training"), (1, "representation", "(c)  Hidden-state term over training")):
    ax = fig.add_subplot(gs[1, p])
    for k, c, lab, ls, mf in (("E_vt", "keep_vt", "keep $v,t$", "-", True), ("E_v", "keep_vt", "keep $v$,\nreset $t$", (0, (4, 2)), False), ("E_0", "reset", "reset all", ":", True)):
        g2 = json.load(open(R / f"g/diag25/g2/{k}/decomp.json")); seq = [g2[f"s{s:04d}"]["shapley_logp_indent"][key] for s in (10, 25, 50, 75)] + [D1[k]["shapley_logp_indent"][key]]
        ax.plot(steps, [0] + seq, ls=ls, marker=C[c][1], color=C[c][0], mfc=C[c][0] if mf else "white", ms=3.6, lw=1.0, label=lab)
    ax.axhline(0, color="0.3", lw=.7, ls=(0, (3, 2))); ax.set_xlabel("Steps after the switch"); ax.set_ylabel("$\\Delta\\log P$(indent)"); ax.set_xticks([0, 25, 50, 75, 125]); style(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=3, columnspacing=.9, handlelength=2.2, fontsize=6.4, borderaxespad=0)
    ax.set_title(ttl, loc="left", pad=26)
save(fig, "fig6b_format_decomposition.png")
