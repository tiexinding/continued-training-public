# Third-party sources

The license in the repository root covers the author's own code, read-outs and documentation. The following parts
come from or follow other sources, which keep their own terms.

## Code

- **Ren et al., `learning-dynamics-cl`** (https://github.com/Joshua-Ren/learning-dynamics-cl, commit `b4962cc`),
  accompanying *Learning Dynamics of Continual Learning* (arXiv:2609.33620).
  - We referenced their data formatting (`plasticity_loss_sft/data.py`) and probe-loss computation
    (`plasticity_loss_sft/probe_confidence_callback.py`) and rewrote them in condensed form in
    `train/forklib.py` (functions `_clean_message`, `to_messages`, `messages_to_prompt_completion`,
    `prompt_completion_rows`, `encode_probe`, `probe_nll`), so that probe losses are comparable with their protocol.
  - The trainer around them is our own: a decoupled AdamW written out explicitly, with moment and step-counter
    operations at the switch, the state replacements, calibration and per-step update records
    (`train/forklib.py`, `train/fork_continue.py`).
  - The training and probe subsets were built with their preparation scripts (`prepare_subsets`,
    `prepare_cpt_data`; commands in `train/README.md`). Their code and the subsets are not redistributed here; only
    IDs, statistics and SHA-256 of the prepared files are.

## Data

| source | used for | terms |
|---|---|---|
| GSM8K (`openai/gsm8k`) | training, probes, held-out test problems | MIT |
| MBPP (`google-research-datasets/mbpp`) | training, probes | CC BY 4.0 |
| HumanEval (`openai/openai_humaneval`) | evaluation only; problem file included for re-scoring | MIT |
| Dolly 15k (`databricks/databricks-dolly-15k`) | held-out loss reference | CC BY-SA 3.0 |
| PubMed abstracts (`slinusc/PubMedAbstractsSubset`, revision `111f2d4`) | stage-A continued pre-training, probes | CC0 1.0 |
| Qwen2.5-0.5B-Instruct | base model (weights not redistributed) | Apache 2.0 |

Released read-outs derived from these datasets (losses, pass marks, statistics) are covered by the root license;
the original items keep their own terms.
