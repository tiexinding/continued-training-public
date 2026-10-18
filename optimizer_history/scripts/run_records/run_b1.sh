# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 批次 1 分析链（与批次 2 训练并行；GPU 显存紧，逐个运行）
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets; M=$W/runs_t1/lr1e-5/M/state
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W; mkdir -p diag25/b1 logs
echo "== diffD $(date +%T)"; /root/miniconda3/bin/python diffD.py --out diag25/b1/diffD.json > logs/b1_diffD.log 2>&1; echo rc=$?
echo "== gen_prefix $(date +%T)"; python gen_prefix.py --out diag25/b1 --deterministic --states W0=base M=$M Gkeep=$W/runs_t2/lr5e-6/G/state Gfresh=$W/runs_t3/lr5e-6/G/state A2e5=$W/runs3/lr2e-5/A/state > logs/b1_gen_prefix.log 2>&1; echo rc=$?
echo "== grad_geom $(date +%T)"; python grad_geom.py --state $M --out diag25/b1/grad_geom_M.json --deterministic > logs/b1_grad_geom.log 2>&1; echo rc=$?
echo "== seg_nll $(date +%T)"; cd $W/fork
printf '[{"name":"W0","base":"W0","swap_from":null,"groups":[]},{"name":"M","base":"A","swap_from":null,"groups":[]},{"name":"Gkeep","base":"B","swap_from":null,"groups":[]}]\n' > $W/diag25/b1/cfg_k.json
printf '[{"name":"Gfresh","base":"B","swap_from":null,"groups":[]}]\n' > $W/diag25/b1/cfg_f.json
printf '[{"name":"C","base":"B","swap_from":null,"groups":[]}]\n' > $W/diag25/b1/cfg_c.json
python state_mix_eval.py --stateA $M --stateB $W/runs_t2/lr5e-6/G/state --data $P --configs $W/diag25/b1/cfg_k.json --out $W/diag25/b1/segnll --nll > $W/logs/b1_segnll_k.log 2>&1; echo rc=$?
python state_mix_eval.py --stateA $M --stateB $W/runs_t3/lr5e-6/G/state --data $P --configs $W/diag25/b1/cfg_f.json --out $W/diag25/b1/segnll --nll > $W/logs/b1_segnll_f.log 2>&1; echo rc=$?
python state_mix_eval.py --stateA $M --stateB $W/runs_t2/lr5e-6/C/state --data $P --configs $W/diag25/b1/cfg_c.json --out $W/diag25/b1/segnll --nll > $W/logs/b1_segnll_c.log 2>&1; echo rc=$?
cd $W; echo "== vocab_rows $(date +%T)"; CUDA_VISIBLE_DEVICES="" python vocab_rows.py --out diag25/b1/vocab_rows.json > logs/b1_vocab.log 2>&1; echo rc=$?
echo "B1 all done $(date +%T)"; touch status/DONE_b1
