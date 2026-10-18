# Data provenance

Paths are relative to this directory. `DATA/<x>` is the read-out stored in the working directory at `results/<x>`;
the layout is unchanged so that scripts and records refer to the same relative paths. Figure and table numbers follow
the arXiv version (labels in brackets, which are stable across versions).

Three conditions recur: *keep v,t* keeps the second moment and step counter and resets the first moment; *log-RC v*
replaces v by the row and column main effects of log v; *permuted v* permutes v's rows and columns. Switches: MG =
MBPP→GSM8K, GM = GSM8K→MBPP, PG = PubMed→GSM8K. Data orders 17, 31, 47.

## 1 Raw sources

| source | where | identity |
|---|---|---|
| base model | Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` | tokenizer files identical to the model's; `train/run_configs/setup_records.json` records `tok_mode manual` |
| stage-A starting states (weights + AdamW m, v, t) | GPU card only: `runs_t1/lr1e-5/M/state` (MBPP, 108 steps), `runs_d/A_g/state` (GSM8K, 125), `runs_e/B_g/state` (PubMed, 125) | weight SHA-256 prefixes `5d9825b68551462d` (M; `DATA/card_fields/card_state_fingerprints_261005.txt`), `938cdf4e74780d43` (A_g), `70f6268bf66fd3f7` (B_g) |
| second-stage runs | GPU card: `runs_*/<run>/`; the training log of every run is released (`DATA/**/log.jsonl`), endpoint snapshots are not | one setup record per run in `train/run_configs/setup_records.json`; endpoint hashes in `DATA/i/runs_i/*/endpoint_sha16.txt`, `DATA/h/runs_h/*/endpoint_sha16.txt` |
| training and probe subsets | rebuilt with Ren et al.'s `prepare_subsets.py` / `prepare_cpt_data.py` (commit `b4962cc`, seed 17) | IDs, statistics and SHA-256 in `train/data/` (see `train/README.md`) |
| HumanEval | `DATA/h/humaneval.parquet` (164 problems, official order, MIT) | — |

## 2 GPU-card extraction (`scripts/`)

These scripts read starting states, endpoint weights or per-step training records, so they ran on the card. Their
recorded invocations are in `scripts/run_records/run_<round>.sh` (paths replaced by `$OH_TRAIN` = `train/`,
`$OH_SUBSETS` = the prepared subsets, `$OH_RUNS` = the run root). Two invocations were not in the queue files and
were recovered from the session record:

```sh
python scripts/vfactor_test.py --state runs_t1/lr1e-5/M/state --act diag25/b4/neuron_act_M.npz --train $OH_SUBSETS/gsm8k/train.jsonl --new_key gsm8k --old_key mbpp --sel_sign 1 --out diag25/k/vfactor_M.json --deterministic
python scripts/vfactor_test.py --state runs_d/A_g/state --act diag25/b4/neuron_act_A_g.npz --train $OH_SUBSETS/mbpp/train.jsonl --new_key mbpp --old_key gsm8k --sel_sign -1 --out diag25/k/vfactor_Ag.json --deterministic
python FS/chan_agree_k.py --fields mvt=diag25/c3/pairs_K_mvt_rec/A_B_fields.npz vt=diag25/b2/pairs_Kv/A_B_fields.npz t=diag25/b2/pairs_bridge/A_B_fields.npz 0=diag25/t3/pairs/A_B_fields.npz vt31=diag25/c3/pairs_K_vt_s31/A_B_fields.npz --out diag25/k/chan_agree_components_MG.json
```

| script | reads | writes (released read-out) | used for |
|---|---|---|---|
| `train/fork_continue.py` | starting state, subsets | `runs_*/<run>/log.jsonl`, `records/` (per-step fields, `match_c.pt`, `upd_stats.pt`, `upd_energy.pt`), endpoint snapshot | every second-stage run |
| `train/state_mix_eval.py` | endpoint weights, probes | `DATA/*/diag25/*/segnll/*.json` (per-problem probe NLL) | probe losses and their intervals |
| `scripts/phase2_card.py` + `pairlib_t.py` / `pairlib.py` | start and endpoint weights | `DATA/**/pairs_*/A_B.json` (+ `A_B_fields.npz`, tier B or card only) | change structure, R², equal-change null |
| `scripts/neuron_act.py` | start weights, probes, `A_B_fields.npz` | `DATA/b4/neuron_act_M.{npz,json}`, `DATA/h/final/diag25/h/neuron_act_*_hall.json`, `DATA/{d,e}/diag25/b4/neuron_act_A_g_*.json`, `DATA/f/diag25/b4/neuron_act_B_g_*.json` | activations $A_j$ (Section 3.3), selectivity, overlap, ρ |
| `scripts/cross_order_channels.py` | 36 `A_B_fields.npz` (card only) | `DATA/i/diag25/i/cross_order_channels.json`, `DATA/j/diag25/j/cross_order_channels_j.json` | cross-order channel agreement |
| `scripts/h2p_energy.py` | `runs_h/*/records/upd_energy.pt` (card only) | `DATA/h/diag25/h/h2p_energy.json`, `DATA/i/diag25/i/h2p_energy_segments.json` | per-step energy (Appendix B) |
| `scripts/adam_offline_v3.py` (executes `adam_offline_v2.py`, both required) | starting states, subsets | `DATA/e/diag25/e1/adam_offline_v3_{M,Ag}.json` | one-step shares (Fig. 5, Tables 14, 16) |
| `scripts/vfactor_test.py` | starting states, activations | `DATA/k/vfactor_{M,Ag}.json` | factor association and energy formula (Tables 2, 3, 15) |
| `scripts/connect_audit.py` (+ `compare_legacy.py`) | MBPP start, activations, probes | `DATA/k/k_M.json`, `DATA/k/diag25/k/{trial_M_f32,trial_M_f64,legacy_check}.json`, `DATA/k/logs/k_smoke.log` | fixed-state output audit (Fig. 6, Table 17, Appendix A) |
| `scripts/calib_check.py` | starting states, `records/match_c.pt` | `DATA/i/diag25/i/calib/*.json` | calibration (Tables 12, 13), log v fits |
| `scripts/drift_diag.py` (`--stage groups/energy/logits/offline/interv/localize/heval`) | endpoint and start weights | `DATA/f/diag25/f1/*.json`, `DATA/i/diag25/f1/intervene_energy.json`, `DATA/h/diag25/f1/heval_h_*.json` | HumanEval generations and format (Figs. 7, 8) |
| `scripts/mech_diag.py` (`--stage decomp/untie/onestep`) | endpoint and intermediate weights | `DATA/g/diag25/g1/{decomp,untie,onestep}.json`, `DATA/g/diag25/g2/*/decomp.json` | format mechanism (Fig. 8, Appendix C) |
| `scripts/humaneval_eval.py`, `humaneval_diag.py` | endpoint weights | `DATA/e/diag25/e4/humaneval_e.json`, `DATA/e/diag25/e7/humaneval_diag.json` | HumanEval passes, prompt rewrites |
| `scripts/humaneval_rescore.py` (local; executes the generated programs) | saved generations | `DATA/e/humaneval_*_rescore.json` | raw assembly vs indent repair |
| `scripts/gen_audit.py` | endpoint weights | `DATA/**/*_max400.jsonl`, `*_mbpp.json` | GSM8K / MBPP generation checks |
| `scripts/intervene_v3.py` | endpoint weights, `A_B_fields.npz` | `DATA/d/diag25/d5/intervene_Gkeep_v3.json`, `DATA/e/diag25/e5/intervene_Gkeep_v3_confirm.json` | local restoration (Appendix B) |

## 3 Figures and tables (`FS/`)

Every row runs from `DATA/` alone unless marked *tier B*; `reproduce.py` runs them all and compares the output with the
released copy. Exact inputs of each script are listed in `FS/INPUTS.md` (recorded by tracing every file a script
opens).

| item | script | read-outs |
|---|---|---|
| Fig. 1 [fig:mechanism] | `FS/draw_mechanism_v8.py` | none (schematic; matrices drawn with `default_rng(7)`) |
| Fig. 2 [fig:ablation] (overlap per layer; nested paired differences computed in the script) | `FS/make_fig2_v13.py` | `b4/neuron_overlap_sixarms.json` |
| Fig. 3 [fig:dynamics] | `FS/make_fig2_v13.py` | `b4/first_steps_groups_v2.json` |
| Fig. 4 [fig:cross] (2×3, down columns per order) | `FS/make_fig34_v13.py` | `h/final/diag25/h/neuron_act_*_hall.json` |
| Fig. 5 [fig:onestep] | `FS/make_fig34_v13.py` | `e/diag25/e1/adam_offline_v3_{M,Ag}.json` (`matched`, median) |
| Fig. 6 [fig:preserve] | `FS/make_fig_preserve_v13.py` | `k/summary_k_M.json` |
| Fig. 7 [fig:boundary] | `FS/make_fig5_v13.py` | `e/humaneval_{e,d}_rescore.json`, `e/humaneval_pairs_e_rescore.json`, `e/nll_two_calibers_e.json`, `f/diag25/f1/{first_pos_probs,heval_f7,heval_f14}.json`, `h/diag25/f1/heval_h_PG_*.json`, `h/final/h_analysis/out/summary_h.json` |
| Fig. 8 [fig:format] | `FS/make_figs_paper_en.py` | `f/diag25/f1/intervene_drift.json`, `i/diag25/f1/intervene_energy.json`, `g/diag25/g1/{decomp,untie}.json`, `g/diag25/g2/*/decomp.json`; training logs of `runs_b2/Kv`, `t2`, `t3`, `e/runs_e/P_*` |
| Table 1 [tab:conditions], Tables 5a, 5b, 6 [tab:setup, tab:tasks, tab:streams] | — | `train/run_configs/setup_records.json`; log v fits in `i/diag25/i/calib/MG_vfac_med.json` |
| Table 2 [tab:factor] | (card: `vfactor_test.py`) | `k/vfactor_{M,Ag}.json` |
| Tables 3, 15 [tab:formula, tab:formulanew] | (card: `vfactor_test.py`) | `k/vfactor_{M,Ag}.json` (ranking: Spearman of 4-batch mean energies; shares: per-batch normalized, then averaged) |
| Table 4 [tab:losses], Table 21 [tab:heldout] | `FS/table_losses.py` | `h/summary_h.json`; starting-checkpoint segnll `diag25/b1/segnll/M.json`, `d/diag25/d4/segnll/A_g.json`, `e/diag25/e4/segnll/B_g.json` → `release/table_losses.json` |
| Table 7 [tab:measures] | — | summary of the read-outs in this table |
| Table 8 [tab:replication], Table 19 [tab:format] | `FS/analyze_h.py` | `h/summary_h.json`, `h/tables_h.md` |
| Table 9 [tab:rankstab] | (card: `cross_order_channels.py`) | `i/diag25/i/cross_order_channels.json` |
| Table 10 [tab:r2kinds] | `FS/regen/gen_structure_by_kind_six_arms.py` | `b2/structure_by_kind_six_arms.json` (from six `A_B.json`) |
| Table 11 [tab:gateup] | `FS/gen_component_gateup.py` (*tier B*) | `b4/component_gateup.json` |
| Tables 12, 13 [tab:calib, tab:applied] | `FS/analyze_i2.py` | `i/diag25/i/calib/*.json`, `i/diag25/i/i2_summary.json` |
| Tables 14, 16 [tab:onestep, tab:onestep_cos] | `FS/analyze_e.py` | `e/summary_e.json`; up columns and both starts from `e/diag25/e1/adam_offline_v3_{M,Ag}.json` |
| Table 17 [tab:kround] | `FS/analyze_k.py` | `k/k_M.json` → `k/summary_k_M.{json,md}` |
| Table 18 [tab:predictors] | `FS/predictors_mg.py` (MG, *tier B*), `FS/regen/gen_predictor_baselines_dir2.py` (GM, *tier B*) | `release/predictors_MG.json` (incl. MG new-task activation 0.28), `b4/early_checkpoint_W4_4.json`, `d/predictor_baselines_dir2.json` |
| Table 20 [tab:humaneval] | `FS/table_humaneval_holm.py` | `e/diag25/e4/humaneval_e.json`, `e/humaneval_e_rescore.json`, `h/diag25/f1/heval_h_PG_*_s{31,47}.json` → `release/tab_humaneval_rows.tex` |
| Table 22 [tab:probeloss] | `FS/regen/gen_nll_two_calibers_e.py` | `e/nll_two_calibers_e.json` |

Numbers quoted in the text:

| location | script | read-out |
|---|---|---|
| §4.1, Fig. 2 overlap, ρ, same-task | `FS/regen/gen_neuron_overlap_sixarms.py` (*tier B*) | `b4/neuron_overlap_sixarms.json` |
| §4.1 component agreement | `FS/chan_agree_k.py` (*tier B*) | `k/chan_agree_components_MG.json` |
| §4.2, Fig. 3 first steps (keep t: complete records to step 125, `b2/records/runs_b2/bridge/`) | `FS/recompute_first_steps.py` (*tier B*) | `b4/first_steps_groups_v2.json` |
| §4.1–4.2 gate and up rows | `FS/gen_component_gateup.py` (*tier B*) | `b4/component_gateup.json` |
| §4.3 45–48%, §5.3 retention, §5.4 √(2−2cos), Appendix B keep-t factor, Appendix C t-reset factor, §7 counts and β₂ᵏ, Appendix A precision checks | `FS/derived_numbers.py` | `release/derived_numbers.json` (each number asserted against its printed value) |
| Appendix B early first moment, Appendix C probe-loss intervals | `FS/regen/gen_nll_two_calibers_e.py` | `e/nll_two_calibers_e.json` |
| Appendix B local restoration 1.3–1.6× | `FS/regen/gen_intervene_v3_paired.py` | `d/intervene_v3_paired.json` |
| Appendix C replay 9.8% / 5.6% | `FS/regen/gen_replay_budget_audit.py` (needs the subsets and tokenizer) | `e/replay_budget_audit.json` |
| Appendix C t reset 13/14/15%, paired differences | `FS/regen/gen_unindent_paired_treset.py` | `g/unindent_paired_treset.json` |
| Appendix C format randomness, Multi-Bootstrap | `FS/underspec_c3c4.py` | `underspec/c3c4.json`, `underspec/c3c4_tables.md` |
| Appendix C order interventions | `FS/analyze_j.py`, `FS/tail_batch.py` (needs the GSM8K train parquet and tokenizer) | `underspec/j_verdict.json`, `underspec/tail_batch.json` |
| Appendix C 1 indentation token in 124,821 | `FS/count_indent_targets.py` (needs the subsets and tokenizer) | `r07/indent_in_gsm8k_targets.json` |
| Appendix A deterministic reruns | — | `i/status/verify_i.tsv`, `*/endpoint_sha16.txt` |

## 4 Release tiers

- **Tier A (this repository, ~105 MB):** every read-out above except the tier-B arrays. All figures and tables and all
  numbers except those marked *tier B* reproduce from tier A.
- **Tier B (~0.9 GB, not released):** eight per-neuron change arrays (`*_A_B_fields.npz`) and 35 per-step
  training records (`records/fields_*.pt`). They are inputs only to the eight steps marked *tier B*; their outputs are in
  tier A. They are not released because of their size; names, sizes and SHA-256 are in
  `DATA/TIER_B_sha256.json`.
- **Not released:** starting states, endpoint and intermediate snapshots, the remaining 29 `A_B_fields.npz` and the
  `upd_energy.pt` records (card only), and the training subsets themselves (§1).

## 5 Recovered one-off aggregation steps (`FS/regen/`)

Nine read-outs were first written by one-off commands in the analysis sessions. Their code was recovered verbatim from
the session record; each file states the original time, and the only changes are the working-directory line and, where
noted in the file, two inputs that lived in temporary folders (the HumanEval problem file, now read from
`DATA/h/humaneval.parquet`, and the replay training file, rebuilt with `train/replay_data.py`). All nine reproduce the
stored read-outs exactly; the rebuilt replay file also reproduces the recorded construction summary (107 old-task and
900 new-task samples, 18,252 old-task tokens).

| file | first written (UTC) | inputs |
|---|---|---|
| `gen_neuron_overlap_sixarms.py` | 2026-10-06 05:46 | activations, six `A_B_fields.npz` (tier B) |
| `gen_early_checkpoint_W4_4.py` | 2026-10-06 05:49 | activations, per-step records of keep v,t and keep t (tier B) |
| `gen_intervene_v3_paired.py` | 2026-10-07 03:11 | `d/diag25/d5/intervene_Gkeep_v3.json` |
| `gen_predictor_baselines_dir2.py` | 2026-10-07 03:12 | GSM8K-start activations, per-step records of GM keep v,t (tier B) |
| `gen_structure_by_kind_six_arms.py` | 2026-10-07 03:49 | six `A_B.json`; two `A_B_fields.npz` for a printed energy split only |
| `gen_replay_budget_audit.py` | 2026-10-07 06:26 | rebuilt replay file, tokenizer |
| `gen_nll_two_calibers_e.py` | 2026-10-07 06:30 | per-problem probe NLL (`segnll`) |
| `gen_humaneval_pairs_e_rescore.py` | 2026-10-07 07:03 | `e/humaneval_e_rescore.json`, `e/diag25/e4/humaneval_e.json`, HumanEval problems |
| `gen_unindent_paired_treset.py` | 2026-10-07 15:14 | `f/diag25/f1/heval_f{7,14}.json` |

## 6 What is not reproducible from this directory

- The card-side extraction (§2), because the starting states and snapshots are not released. The appendix reports the
  deterministic checks that were made instead: nine reruns of the I round, of which five match the original endpoint
  hash and four the original probe evaluation (`DATA/i/status/verify_i.tsv`); PG keep v,t order 17 rerun with the
  H-round code and matching endpoint hash; the equal-change null model reproducing 22,696 scalars bit for bit (recorded
  in the H-round report, comparison output not archived).
- Exact identity of the subsets used on the card with the copies listed in `train/data/`: the listed files are the
  copies retrieved from the card on 2026-10-04; later rounds used the same card paths, but their hashes were not
  recorded on the card (`train/README.md`).
