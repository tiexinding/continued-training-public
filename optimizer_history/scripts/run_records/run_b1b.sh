# 发布包：卡上运行队列的原始记录，仅供审计，不能直接运行（见本目录 README.md）；部分路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2），仍含卡上路径与清理命令
#!/bin/bash
# 批次 1 补跑：grad_geom（与训练并行时 OOM，改在批次 4 之后 GPU 空闲时运行）
R=/root/autodl-tmp/ren; W=$OH_RUNS; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W; echo "== grad_geom $(date +%T)"; python grad_geom.py --state $M --out diag25/b1/grad_geom_M.json --deterministic > logs/b1_grad_geom.log 2>&1; rc=$?; echo rc=$rc; [ $rc -eq 0 ] && touch status/DONE_b1_grad_geom
