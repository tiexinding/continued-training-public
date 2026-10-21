#!/usr/bin/env python3
"""Rebuild the paper's figures, tables and summary read-outs from the released read-outs and check them against the copies
in this directory.

    python reproduce.py            # tier-A steps only (released DATA/ is enough)
    python reproduce.py --tier_b   # also the eight read-outs regenerated from tier-B arrays (not released; only for a copy that has them)
    python reproduce.py --gsm8k_parquet <train.parquet> --tokenizer <Qwen2.5-0.5B-Instruct>   # also FS/tail_batch.py

The package is copied to a temporary directory and every step runs there, so nothing in this directory is overwritten.
Outputs are compared with the released copies: JSON by value (exact, or the largest relative difference), Markdown and
LaTeX byte for byte, PDF figures byte for byte after removing the CreationDate entry.
Requires Python 3.10+, numpy, scipy, matplotlib, pandas with pyarrow, and torch (tier-A steps read released .pt
records; no GPU and no training are needed). See requirements-analysis.txt."""
import argparse, json, math, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--tier_b", action="store_true")
ap.add_argument("--gsm8k_parquet", default=""); ap.add_argument("--tokenizer", default=""); ap.add_argument("--keep", action="store_true")
a = ap.parse_args()

# (step, script and arguments, outputs relative to the package root)
STEPS = [
    ("figures 8, 9", ["FS/make_figs_paper_en.py"], ["FIG/fig6a_format_interventions.pdf", "FIG/fig6b_format_decomposition.pdf"]),
    ("figures 2, 3", ["FS/make_fig2_v13.py"], ["FIG/fig2_core_allocation.pdf", "FIG/fig3_allocation_dynamics.pdf"]),
    ("figures 4, 5", ["FS/make_fig34_v13.py"], ["FIG/fig3_one_step_counterfactual.pdf", "FIG/fig4_cross_condition.pdf"]),
    ("figure 7", ["FS/make_fig5_v13.py"], ["FIG/fig5_functional_boundary.pdf"]),
    ("figure 6", ["FS/make_fig_preserve_v13.py"], ["FIG/fig_preserve.pdf"]),
    ("three-order summary (H)", ["FS/analyze_h.py"], ["DATA/h/summary_h.json", "DATA/h/tables_h.md"]),
    ("E-round summary", ["FS/analyze_e.py"], ["DATA/e/summary_e.json"]),
    ("C-round summary", ["FS/analyze_c.py"], ["DATA/c/summary_c.json"]),
    ("D-round summary", ["FS/analyze_d.py"], ["DATA/d/summary_d.json"]),
    ("K audit summary", ["FS/analyze_k.py", "--in", "DATA/k/k_M.json", "--out", "DATA/k/summary_k_M"], ["DATA/k/summary_k_M.json"]),
    ("applied step sizes (I)", ["FS/analyze_i2.py", "--runs_dir", "DATA/i/runs_i", "--out", "DATA/i/diag25/i/i2_summary.json"], ["DATA/i/diag25/i/i2_summary.json"]),
    ("order interventions (J)", ["FS/analyze_j.py"], ["DATA/underspec/j_verdict.json"]),
    ("format randomness", ["FS/underspec_c3c4.py"], ["DATA/underspec/c3c4.json", "DATA/underspec/c3c4_tables.md"]),
    ("HumanEval table (Holm)", ["FS/table_humaneval_holm.py"], ["DATA/release/table_humaneval_holm.json", "DATA/release/tab_humaneval_rows.tex"]),
    ("derived numbers", ["FS/derived_numbers.py"], ["DATA/release/derived_numbers.json"]),
    ("loss tables 4, 15", ["FS/table_losses.py"], ["DATA/release/table_losses.json"]),
    ("probe-loss intervals", ["FS/regen/gen_nll_two_calibers_e.py"], ["DATA/e/nll_two_calibers_e.json"]),
    ("HumanEval pairs and format", ["FS/regen/gen_humaneval_pairs_e_rescore.py"], ["DATA/e/humaneval_pairs_e_rescore.json"]),
    ("format flips (C.1)", ["FS/regen/gen_format_flip_means.py"], ["DATA/underspec/format_flip_means.json"]),
    ("t-reset format pairs", ["FS/regen/gen_unindent_paired_treset.py"], ["DATA/g/unindent_paired_treset.json"]),
    ("local restoration pairs", ["FS/regen/gen_intervene_v3_paired.py"], ["DATA/d/intervene_v3_paired.json"]),
]
TIER_B = [
    ("six-arm structure by kind", ["FS/regen/gen_structure_by_kind_six_arms.py"], ["DATA/b2/structure_by_kind_six_arms.json"]),
    ("six-arm neuron overlap", ["FS/regen/gen_neuron_overlap_sixarms.py"], ["DATA/b4/neuron_overlap_sixarms.json"]),
    ("early checkpoints", ["FS/regen/gen_early_checkpoint_W4_4.py"], ["DATA/b4/early_checkpoint_W4_4.json"]),
    ("predictor baselines (GM)", ["FS/regen/gen_predictor_baselines_dir2.py"], ["DATA/d/predictor_baselines_dir2.json"]),
    ("predictor baselines (MG)", ["FS/predictors_mg.py"], ["DATA/release/predictors_MG.json"]),
    ("first-step groups", ["FS/recompute_first_steps.py"], ["DATA/b4/first_steps_groups_v2.json"]),
    ("gate/up component read-outs", ["FS/gen_component_gateup.py"], ["DATA/b4/component_gateup.json"]),
    ("channel agreement (K)", ["FS/chan_agree_k.py", "--fields", "mvt=DATA/c/fields/K_mvt_rec_A_B_fields.npz", "vt=DATA/b2/fields/Kv_A_B_fields.npz",
      "t=DATA/b2/fields/bridge_A_B_fields.npz", "0=DATA/t3/fields/A_B_fields.npz", "vt31=DATA/c/fields/K_vt_s31_A_B_fields.npz",
      "--out", "DATA/k/chan_agree_components_MG.json"], ["DATA/k/chan_agree_components_MG.json"]),
]
if a.tier_b: STEPS += TIER_B
if a.gsm8k_parquet and a.tokenizer:
    STEPS.append(("tail-batch null", ["FS/tail_batch.py", "--parquet", a.gsm8k_parquet, "--tok", a.tokenizer, "--n_null", "2000"], ["DATA/underspec/tail_batch.json"]))


def jcmp(x, y, st):
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y): st["struct"] = True
        for k in set(x) & set(y): jcmp(x[k], y[k], st)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y): st["struct"] = True
        for u, v in zip(x, y): jcmp(u, v, st)
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
        if x != y and not (isinstance(x, float) and math.isnan(x) and math.isnan(y)):
            st["maxrel"] = max(st["maxrel"], abs(x - y) / max(abs(x), abs(y), 1e-300))
    elif x != y: st["struct"] = True


def compare(new, ref):
    if not new.exists(): return "NOT WRITTEN"
    if new.suffix == ".json":
        st = dict(maxrel=0.0, struct=False); jcmp(json.load(open(new)), json.load(open(ref)), st)
        return "DIFFERENT STRUCTURE" if st["struct"] else ("exact" if st["maxrel"] == 0 else f"max rel. diff. {st['maxrel']:.1e}")
    strip = lambda b: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", b) if new.suffix == ".pdf" else b
    return "identical" if strip(new.read_bytes()) == strip(ref.read_bytes()) else "BYTES DIFFER"


work = Path(tempfile.mkdtemp(prefix="oh_reproduce_"))
for sub in ("DATA", "FS", "train"): shutil.copytree(HERE / sub, work / sub)
(work / "FIG").mkdir()
env = dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
bad = 0
for name, cmd, outs in STEPS:
    for o in outs:                                    # outputs are regenerated, not read from the copied package
        p = work / o
        if p.exists(): p.unlink()
    r = subprocess.run([sys.executable, cmd[0]] + cmd[1:], cwd=work, env=env, capture_output=True, text=True)
    if r.returncode:
        bad += 1; print(f"[FAIL] {name}: exit {r.returncode}\n{r.stderr[-800:]}"); continue
    for o in outs:
        v = compare(work / o, HERE / o); ok = v in ("exact", "identical") or v.startswith("max rel. diff. ") and float(v.split()[-1]) < 1e-5
        bad += not ok; print(f"[{'ok' if ok else 'DIFF'}] {name:28} {o:52} {v}")
# Figure 1 (schematic): the script names its output after its own version; compared with FIG/fig0_mechanism.pdf
r = subprocess.run([sys.executable, "FS/draw_mechanism_v8.py"], cwd=work, env=dict(env, FIG_OUT=str(work / "FIG")), capture_output=True, text=True)
v = compare(work / "FIG/fig1_main_mechanism_v8_en.pdf", HERE / "FIG/fig0_mechanism.pdf") if not r.returncode else f"exit {r.returncode}"
bad += v != "identical"; print(f"[{'ok' if v == 'identical' else 'DIFF'}] {'figure 1 (schematic)':28} {'FIG/fig0_mechanism.pdf':52} {v}")
STEPS.append(("figure 1", [], []))
print(f"\n{len(STEPS)} steps, {'all outputs reproduced' if not bad else f'{bad} problem(s)'}; work dir {work}" + ("" if a.keep else " (removed)"))
if not a.keep: shutil.rmtree(work)
sys.exit(1 if bad else 0)
