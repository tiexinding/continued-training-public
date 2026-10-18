# Training code, run records and data

## Code

| file | role |
|---|---|
| `fork_continue.py` | second-stage trainer: loads a stage-A state (weights + AdamW m, v, t) and applies the switch treatment (`--m_op`, `--v_op`, `--t_reset`, `--switch`); also runs stage A (`--mode segment`); records per-step fields, calibration constants (`records/match_c.pt`), applied-update statistics (`--upd_stats`) and per-step energy (`--upd_energy`) |
| `forklib.py` | decoupled AdamW (`DecompAdamW`, β = (0.9, 0.999), ε = 1e-8, decoupled weight decay 0.01, clipping 1.0), the v replacements (`v_ladder`: row–column factors of log v, single-axis factors, permutations, constant), calibration (`match_coeffs`), tokenization (`tok_mode manual`), data order (`epoch_order`) and the learning-rate schedule (`lr_at`: 5% linear warm-up, then constant) |
| `state_mix_eval.py` | per-problem probe NLL (`segnll`) |
| `replay_data.py` | old-task replay at a fixed token budget (10%) |
| `anchor_data.py` | self-anchored replay: the start's own answers on other prompts |
| `make_orders_j.py` | explicit data orders for the order interventions (`DATA/j/orders_j/*.json`) |

`run_configs/setup_records.json` holds the setup line of every training log (84 runs): mode, task, starting state,
learning rate, steps, warm-up, data-order seed and every command-line argument. The stage-A MBPP start is under the
key `t1/lr1e-5`.

### Code version

Results come from three successive versions of `fork_continue.py`; the released file is the latest
(SHA-256 prefix `5f5c34f035be94de`; `forklib.py` `c58936e80ebd0e6b`, `state_mix_eval.py` `c5559d84d0ef2f6e`). Each
later version only added options whose default reproduces the earlier behaviour:

| added | options | runs that use them |
|---|---|---|
| 2026-10-10 | `--upd_stats`, `--upd_rows` (read-only applied-update statistics) | the nine I-round reruns (`i/runs_i/*_s17`) |
| 2026-10-11 | `--order_file`, `--decay_last` (explicit data order; linear decay over the last k steps) | the six J-round runs (`j/runs_j/*`) |

The released file accepts every argument recorded in the 84 setup records. That the earlier runs are reproduced by the
released code was checked by deterministic reruns, not by reading the code: PG keep v,t order 17 rerun with the H-round
code gave the original endpoint hash, and of the nine I-round reruns five gave the original endpoint hash and four the
original probe evaluation (`DATA/i/status/verify_i.tsv`).

## Data

The training and probe subsets were built with the code of Ren et al. (`learning-dynamics-cl`, commit `b4962cc`):

```sh
python -m plasticity_loss_sft.prepare_subsets --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct \
    --train_size 1000 --probe_size 100 --allow_smaller            # seed 17 (default): gsm8k, mbpp, dolly_qa
python -m plasticity_loss_sft.prepare_cpt_data --tokenizer_name Qwen/Qwen2.5-0.5B-Instruct ...  # bio (PubMed) train and probe
```

Sources: `openai/gsm8k` (7,473 training problems), `google-research-datasets/mbpp` (974; 874 training items after
the code's filtering), `databricks/databricks-dolly-15k` (closed QA, 1,773), `slinusc/PubMedAbstractsSubset` (the
first 1,100 valid abstracts, reserved before the continued-pretraining stream). That repository has no license file,
so neither its code nor the subsets it produced are redistributed here. `data/<task>/selected_ids.json` lists the IDs
of the selected training and probe items, `data/<task>/stats.json` their token statistics, `data/manifest.json` the
preparation record, and `data/prepared_sha256.json` the SHA-256 and size of each prepared `train.jsonl` and
`probe.jsonl`, against which a rebuild can be checked.

The listed files are the copies retrieved from the GPU card on 2026-10-04. Later rounds used the same card paths, but
their hashes were not recorded on the card, so identity with the copies used there is assumed, not verified. One
indirect check: the replay file rebuilt from these copies reproduces the recorded construction summary of the replay
file used on the card (107 old-task and 900 new-task samples, 18,252 old-task tokens).

Derived training files: the replay file is rebuilt with
`python replay_data.py --new <subsets>/gsm8k/train.jsonl --old <subsets>/mbpp/train.jsonl --out train_gsm8k_replay10.jsonl --tok Qwen/Qwen2.5-0.5B-Instruct`;
the self-anchored replay file depends on the start's generations and is identified by its SHA-256 in
`data/prepared_sha256.json`.
