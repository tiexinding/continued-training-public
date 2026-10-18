# 发布包：卡上运行队列的原始记录；路径已换为 $OH_TRAIN / $OH_SUBSETS / $OH_RUNS（见 DATA_PROVENANCE §2）
#!/bin/bash
# 阶段 3 队列 v2（10-05 14:10）：训练与评估立即开跑（不等阶段 2），四层分析放到最后用 GPU 后端（等 $W/GPU_BACKEND_READY）。
# 每档 lr ∈ {2e-5, 5e-5}：A = base→GSM8K 216 步（fresh）；B = A→MBPP 216 步（keep）；C = A→GSM8K 再 216 步（keep，同起点对照）。
# 评估：state_mix_eval --nll --gen 于 W0 / A / B / C。可重入：status/DONE_<名> 存在即跳过。
set +e
R=/root/autodl-tmp/ren; W=$OH_RUNS; P=$R/code/data/prepared_subsets
export PATH=$R/venv/bin:$PATH PYTHONPATH=$R/code/plasticity/src HF_HOME=$R/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false FORK_THREADS=4
cd $W/fork; mkdir -p $W/runs3 $W/status $W/logs $W/diag25
PR="--probe gsm8k=$P/gsm8k/probe.jsonl --probe mbpp=$P/mbpp/probe.jsonl --probe dolly_qa=$P/dolly_qa/probe.jsonl"
C="--tok_mode manual --deterministic $PR"
go() { local n=$1; shift; [ -f $W/status/DONE_$n ] && { echo "skip $n"; return 0; }
  echo "== $n $(date +%T)"; local t0=$(date +%s)
  "$@" < /dev/null > $W/logs/$n.log 2>&1; local rc=$?
  echo -e "$n\t$rc\t$(( $(date +%s)-t0 ))\t$(date +%T)" >> $W/status/status.tsv
  [ $rc -eq 0 ] && touch $W/status/DONE_$n; return $rc; }

for LR in 2e-5 5e-5; do
  T=$W/runs3/lr$LR; mkdir -p $T
  go p3_A_$LR python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/A --init base    --switch fresh --lr $LR --eval_every 25 $C || continue
  go p3_B_$LR python fork_continue.py segment --task mbpp  --train $P/mbpp/train.jsonl  --out $T/B --init $T/A/state --switch keep --lr $LR --eval_every 25 $C
  go p3_C_$LR python fork_continue.py segment --task gsm8k --train $P/gsm8k/train.jsonl --out $T/C --init $T/A/state --switch keep --lr $LR --eval_every 25 $C
  printf '[{"name":"W0","base":"W0","swap_from":null,"groups":[]},{"name":"A","base":"A","swap_from":null,"groups":[]},{"name":"B","base":"B","swap_from":null,"groups":[]}]\n' > $T/cfg_AB.json
  printf '[{"name":"C","base":"B","swap_from":null,"groups":[]}]\n' > $T/cfg_C.json
  go p3_eval_AB_$LR python state_mix_eval.py --stateA $T/A/state --stateB $T/B/state --data $P --configs $T/cfg_AB.json --out $W/diag25/phase3_lr$LR/eval --nll --gen
  go p3_eval_C_$LR  python state_mix_eval.py --stateA $T/A/state --stateB $T/C/state --data $P --configs $T/cfg_C.json  --out $W/diag25/phase3_lr$LR/eval --nll --gen
done
echo "training+eval done $(date +%T)"
until [ -f $W/GPU_BACKEND_READY ]; do echo "waiting GPU backend $(date +%T)"; sleep 60; done
for LR in 2e-5 5e-5; do go p3_pair_$LR /root/miniconda3/bin/python $W/phase2_card.py --phase3 $W/runs3/lr$LR --out $W/diag25/phase3_lr$LR --device cuda; done
go p2_pair_gpu /root/miniconda3/bin/python $W/phase2_card.py --out $W/diag25_gpu --device cuda
for LR in 2e-5 5e-5; do rm -f $W/runs3/lr$LR/A/state/optim.pt $W/runs3/lr$LR/B/state/optim.pt $W/runs3/lr$LR/C/state/optim.pt; done
echo "phase3b all done $(date +%T)"; touch $W/status/DONE_phase3b
