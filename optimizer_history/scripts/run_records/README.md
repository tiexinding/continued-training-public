# Run records

These are the historical invocation records of the run queues on the GPU card, kept for audit. They are **not
portable launch scripts**: they still contain card paths (e.g. `/root/autodl-tmp/ren`), call helper files that are
not part of this package, follow the card's directory layout rather than this package's, and include `rm -rf`
clean-up of card directories. Do not run them as they are. The released trainer and extraction scripts, with the
setup record of every run in `train/run_configs/setup_records.json`, are the reference for re-running.
